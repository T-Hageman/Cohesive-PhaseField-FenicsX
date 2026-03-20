from dolfinx import fem
import ufl
import numpy as np
from typing import List, Tuple, Any

from mpi4py import MPI
from Utils.mpi_utils import mprint, comm, rank

from Models.BaseModel import BaseModel
from Utils.mpi_utils import mprint

from Models.ModelEnums import ModelType

class AppliedTorsion(BaseModel):
    
    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.BOUNDARY_TORSION
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.boundary_tag = self.mesh.get_boundary_tag(self.boundary)
        self.boundary_facets = self.mesh.get_boundary_facets(self.boundary)
        self.plane = self.params.Models[self.name]["plane"]
        self.rate = self.params.Models[self.name]["rate"]
        self.dummy = self.params.Models[self.name]["Dummy"]
        
        # Select vector components based on plane
        if self.plane == "xy":
            self.field_name = "u"
            self.component = [0, 1]
        elif self.plane == "xz":
            self.field_name = "u"
            self.component = [0, 2]
        else:
            self.field_name = "u"
            self.component = [1,2]

        self.step = params.Solution_Steps[self.field_name]
        self.ds = ufl.Measure("ds", domain=self.mesh.mesh, subdomain_data=self.mesh.facet_tags)
        self.cos_omega = fem.Constant(self.mesh.mesh, 1.0)
        self.sin_omega = fem.Constant(self.mesh.mesh, 0.0)
        
    def assemble_KF(self, step: int) -> None:
        
        if (step != self.step):
            return
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr = self.mesh.Get_Trial_Function(self.field_name, step)

        time = self.params.t
        dt = self.params.dt

        ds = self.ds(self.boundary_tag)

        # Update rotation angle constants
        omega = self.rate * (time + dt)
        self.cos_omega.value = np.cos(omega)
        self.sin_omega.value = np.sin(omega)

        # Spatial coordinates on the boundary
        x = ufl.SpatialCoordinate(self.mesh.mesh)
        c0 = x[self.component[0]]  # first in-plane coordinate
        c1 = x[self.component[1]]  # second in-plane coordinate

        # Target displacement from rigid-body rotation by omega
        #   u_target_0 = c0*(cos(w)-1) - c1*sin(w)
        #   u_target_1 = c0*sin(w)     + c1*(cos(w)-1)
        u_target_0 = c0 * (self.cos_omega - 1.0) - c1 * self.sin_omega
        u_target_1 = c0 * self.sin_omega + c1 * (self.cos_omega - 1.0)

        # Current displacement components
        u0 = field[self.component[0]]
        u1 = field[self.component[1]]

        # Test / trial components
        v0 = field_t[self.component[0]]
        v1 = field_t[self.component[1]]
        du0 = field_tr[self.component[0]]
        du1 = field_tr[self.component[1]]

        # Penalty residual  R = dummy * (u - u_target)
        F = (v0 * self.dummy * (u0 - u_target_0)
           + v1 * self.dummy * (u1 - u_target_1)) * ds

        # Stiffness (Jacobian w.r.t. u; target is independent of u)
        K = (v0 * self.dummy * du0
           + v1 * self.dummy * du1) * ds

        return K, F

    def PenaltyForces(self, time, dt, field):
        """Return the integrated penalty (torque) on the boundary.

        This is used by Update_Global_Measures-style callers that need
        a scalar force value.  It returns the sum of the two in-plane
        penalty-force components dotted with the displacement error.
        """
        omega = self.rate * (time + dt)
        self.cos_omega.value = np.cos(omega)
        self.sin_omega.value = np.sin(omega)

        x = ufl.SpatialCoordinate(self.mesh.mesh)
        c0 = x[self.component[0]]
        c1 = x[self.component[1]]

        u_target_0 = c0 * (self.cos_omega - 1.0) - c1 * self.sin_omega
        u_target_1 = c0 * self.sin_omega + c1 * (self.cos_omega - 1.0)

        u0 = field[self.component[0]]
        u1 = field[self.component[1]]

        force_0 = -self.dummy * (u0 - u_target_0)
        force_1 = -self.dummy * (u1 - u_target_1)
        return force_0, force_1
