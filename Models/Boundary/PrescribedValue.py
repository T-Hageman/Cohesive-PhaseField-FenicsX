from dolfinx import fem
import numpy as np
from typing import List, Tuple, Any

from Models.BaseModel import BaseModel
from Utils.mpi_utils import mprint

from Models.ModelEnums import ModelType

class PrescribedValue(BaseModel):
    """
    Model for applying prescribed (Dirichlet) boundary conditions.
    """
    
    
    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.BOUNDARY_PRESCRIBED
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.boundary_tag = self.mesh.get_boundary_tag(self.boundary)
        self.boundary_facets = self.mesh.get_boundary_facets(self.boundary)
        
        dof = self.params.Models[self.name]["field"]
        self.value = self.params.Models[self.name]["value"]
        
        self.t_max = self.params.Models[self.name].get("tmax", None)
        
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
    
    def get_bcs_for_field(self, field_name: str) -> List:
        """
        Get boundary conditions for a specific field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            List of BCs associated with this field
        """
        
        if (field_name != self.field_name):
            return None
        
        if (self.t_max is not None) and (self.params.t+self.params.dt > self.t_max):
            return None
        
        self.bcs = []
        function_space, sub_index = self.mesh.Get_Field_Space_For_BC(field_name, self.component)
        if sub_index is not None:
            # Component BC (or field in mixed space)
            V_sub, sub_to_parent = function_space.sub(sub_index).collapse()
            
            bc_dofs = fem.locate_dofs_topological(
                (function_space.sub(sub_index), V_sub),
                self.mesh.mesh.topology.dim - 1,
                self.boundary_facets
            )
            bc_func = fem.Function(V_sub)
            field_info = self.mesh._get_field_info_in_mixed_space(field_name)
            if field_info is not None:
                mixed_info, _, _ = field_info
                current_sub = mixed_info["mixed_function"].sub(sub_index).collapse()
            else:
                current_sub = self.mesh.fields[field_name].sub(self.component).collapse()
            target = self.value
            if isinstance(self.value, (list, tuple)) and self.component is not None:
                target = self.value[self.component]
            bc_func.x.array[:] = target - current_sub.x.array
            
            bc = fem.dirichletbc(bc_func, bc_dofs, function_space.sub(sub_index))
            self.bcs.append(bc)

        elif self.component is not None:
            # Full vector BC
            bc_dofs = fem.locate_dofs_topological(
                function_space,
                self.mesh.mesh.topology.dim - 1,
                self.boundary_facets
            )
            bc_func = fem.Function(function_space)
            current_field = self.mesh.fields[field_name]
            # Set all components to the target increment.
            if function_space.element.value_shape:
                ncomp = function_space.element.value_shape[0]
                for i in range(ncomp):
                    if isinstance(self.value, (list, tuple)):
                        target = self.value[i]
                    else:
                        target = self.value
                    bc_func.sub(i).x.array[:] = target - current_field.sub(i).x.array
            else:
                bc_func.x.array[:] = self.value - current_field.x.array
            
            bc = fem.dirichletbc(bc_func, bc_dofs)
            self.bcs.append(bc)
        else:
            # Scalar BC
            bc_dofs = fem.locate_dofs_topological(
                function_space,
                self.mesh.mesh.topology.dim - 1,
                self.boundary_facets
            )
            bc_func = fem.Function(function_space)
            current_field = self.mesh.fields[field_name]
            bc_func.x.array[:] = self.value - current_field.x.array
            
            bc = fem.dirichletbc(bc_func, bc_dofs)
            self.bcs.append(bc)
        
        return self.bcs
