from typing import Any


class _PhysicsBase:
    """
    Base class for physics models
    """

    def __init__(self, mesh: Any, params: Any) -> None:
        self.mesh = mesh
        self.params = params
        self.DEBUG_PRINT_TIMING = params.DEBUG_PRINT_TIMING if hasattr(params, 'DEBUG_PRINT_TIMING') else False
        
        super().__init__(mesh, params)