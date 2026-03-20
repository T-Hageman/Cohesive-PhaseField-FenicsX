from typing import Any, Dict, Tuple
from Utils.mpi_utils import mprint
from Mesh.Mesh import Mesh
from Params import Params

class BaseModel:
    REQUIRED_PARAMETERS = {}
    REQUIRED_MODEL_SETTINGS = {}
    
    def __init__(self, name, params: Params, mesh: Mesh):
        self.name = name
        self.params = params
        self.mesh = mesh
        
        pass
    
    def initialize_fields(self) -> None:
        """
        Initialize fields for the model.
        """
        pass
    
    def get_bcs_for_field(self, field_name: str) -> None:
        pass
    
    def assemble_KF(self, step) -> None:
        """
        Assemble stiffness matrix and force vector for the model at the given staggered solver step.
        """
        pass
    
    def commit(self) -> None:
        """
        Commit the current state of the model (e.g., update old fields).
        """
        pass
    
    def reset_step(self) -> None:
        """
        Reset the current step for the model (e.g., revert to old fields).
        """
        pass
