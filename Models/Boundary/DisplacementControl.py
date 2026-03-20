from dolfinx import fem
import ufl
import numpy as np
from typing import List, Tuple, Any

from mpi4py import MPI
from Utils.mpi_utils import mprint, comm, rank

from Models.BaseModel import BaseModel
from Utils.mpi_utils import mprint

from Models.ModelEnums import ModelType

class DisplacementControl(BaseModel):
    
    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.BOUNDARY_DISPCONTROL
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.boundary_tag = self.mesh.get_boundary_tag(self.boundary)
        self.boundary_facets = self.mesh.get_boundary_facets(self.boundary)
        
        dof = self.params.Models[self.name]["field"]
        self.rate = self.params.Models[self.name]["rate"]
        self.dummy = self.params.Models[self.name]["Dummy"]
        self.F_Cutoff = self.params.Models[self.name].get("F_Cutoff", 0.05)
        
        # Check if Vector sub-component BC
        if "_x" in dof:
            self.field_name = dof[:-2]
            self.component = 0
        elif "_y" in dof:
            self.field_name = dof[:-2]
            self.component = 1
        else:
            self.field_name = dof
            self.component = None
            
        self.step = params.Solution_Steps[self.field_name]
        self.ds = ufl.Measure("ds", domain=self.mesh.mesh, subdomain_data=self.mesh.facet_tags)
        self.u_target = fem.Constant(self.mesh.mesh, 0.0)
        self.F_max = 0.0
        
        self.FOld = 0.0
        self.uOld = 0.0
        
        self.F0 = 1.00e6
        self.changed_tmax = False
        
    def assemble_KF(self, step: int) -> None:
        
        if (step != self.step):
            return
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr= self.mesh.Get_Trial_Function(self.field_name, step)
        
        if (self.component is not None):
            field = field[self.component]
            field_t = field_t[self.component]
            field_tr = field_tr[self.component]
            
        time = self.params.t
        dt = self.params.dt
        
        ds = self.ds(self.boundary_tag)
        
        force = self.PenaltyForces(time, dt, field)
        
        F = field_t * -1.0*force * ds
        K = field_t * self.dummy * field_tr * ds
        

        
        return K, F

    def PenaltyForces(self, time, dt, u):
        u_target = self.rate * (time + dt)
        self.u_target.value = u_target
        penalty_force = -self.dummy * (u - self.u_target)
        return penalty_force

    def Update_Global_Measures(self) -> None:
        field = self.mesh.Get_Field(self.field_name, self.step)
        if (self.component is not None):
            field = field[self.component]
            
        time = self.params.t
        dt = self.params.dt
        ds = self.ds(self.boundary_tag)
        force = self.PenaltyForces(time, dt, field)
        Fext = comm.allreduce(fem.assemble_scalar(fem.form(force*ds)), op=MPI.SUM)
        uExt = self.rate * (time + dt)
        self.params.Global_Measures[f"{self.name}_F"] = Fext
        self.params.Global_Measures[f"{self.name}_u"] = uExt
        
        self.ArcLen = np.sqrt((Fext-self.FOld)**2/(self.F0)**2 + (uExt - self.uOld)**2/(self.rate)**2)
        self.params.Global_Measures[f"{self.name}_ArcLen"] = self.ArcLen
        
        

    def commit(self) -> None:
        if abs(self.params.Global_Measures[f"{self.name}_F"]) > abs(self.F_max):
            self.F_max = self.params.Global_Measures[f"{self.name}_F"]
        self.params.Global_Measures[f"{self.name}_F_max"] = self.F_max
        #mprint(f"{self.name} - Current Force: {self.params.Global_Measures[f'{self.name}_F']:.6e}, Max Force: {self.F_max:.6e}")
        
        if abs(self.F0 - 1.00e6) < 1.0e-8: # base on lin-el stress changes
            self.F0 = abs(self.params.Global_Measures[f"{self.name}_F"])/np.abs((self.params.Global_Measures[f"{self.name}_u"] - self.uOld)/(self.rate))
        
        self.FOld = self.params.Global_Measures[f"{self.name}_F"]
        self.uOld = self.params.Global_Measures[f"{self.name}_u"]
        
        t = self.params.t + self.params.dt
        dt = self.params.dt
        
        if self.changed_tmax and abs(self.FOld/self.F_max)>self.F_Cutoff:
            self.params.end_time = self.original_tmax
            self.changed_tmax = False
        
        if t>100.0*dt and abs(self.FOld/self.F_max)<self.F_Cutoff and self.changed_tmax == False:
            self.original_tmax = self.params.end_time
            self.params.end_time = t + 50.0*dt
            self.changed_tmax = True
        
