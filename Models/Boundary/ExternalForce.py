from dolfinx import fem
import ufl
import numpy as np
from typing import List, Tuple, Any

from mpi4py import MPI
from Utils.mpi_utils import mprint, comm, rank

from Models.BaseModel import BaseModel
from Utils.mpi_utils import mprint

from Models.ModelEnums import ModelType

class ExternalForce(BaseModel):
    
    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.BOUNDARY_EXTERNALFORCE
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.boundary_tag = self.mesh.get_boundary_tag(self.boundary)
        self.boundary_facets = self.mesh.get_boundary_facets(self.boundary)
        
        
        self.DamagedForces = self.params.Models[self.name].get("DamagedForces", False)
        dof = self.params.Models[self.name]["field"]
        self.value = self.params.Models[self.name]["value"]
        
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
            
        if self.DamagedForces:
            self.step_dam = params.Solution_Steps["phasefield"]
            
        self.step = params.Solution_Steps[self.field_name]
        self.ds = ufl.Measure("ds", domain=self.mesh.mesh, subdomain_data=self.mesh.facet_tags)
        
    def assemble_KF(self, step: int) -> None:
        
        if (step != self.step):
            return
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr= self.mesh.Get_Trial_Function(self.field_name, step)
        
        dam = 1.0
        if self.DamagedForces:
            phi = self.mesh.Get_Field("phasefield", step)
            dam = (1.0-1e-6)*(1.0 - phi)**2 + 1e-6 
        
        if (self.component is not None):
            field = field[self.component]
            field_t = field_t[self.component]
            field_tr = field_tr[self.component]
        
        ds = self.ds(self.boundary_tag)
        
        F = field_t * dam * -1.0*self.value * ds
        K = None
        
        return K, F
