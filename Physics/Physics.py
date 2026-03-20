from typing import Any

from ._PhysicsBase import _PhysicsBase
from ._PhysicsModels import _PhysicsModels
from ._PhysicsKF import _PhysicsKF

class Physics(_PhysicsBase, _PhysicsModels, _PhysicsKF):
    """
    Managing physics models
    """
    
    def __init__(self, mesh: Any, params: Any) -> None:
        
        super().__init__(mesh, params)