from typing import Any

from Models.ModelImports import MODEL_CLASS_MAP, ModelType
from Utils.mpi_utils import mprint

class _PhysicsModels:
    """
    Mixin class for managing physics models
    """

    def __init__(self, mesh: Any, params: Any) -> None:
        self._initialize_models()
        super().__init__(mesh, params)

    def _initialize_models(self) -> None:
        """
        Initialize physics models based on parameters.
        """
        
        mprint("Initializing physics models...")
        self.models = {}
        self.boundary_conditions = []  # Collect all boundary conditions
        
        for model_name in self.params.ModelNames:
            mprint(f"\tSetting up model: {model_name}")
            
            model_info = self.params.Models[model_name]
            try:
                model_type = ModelType(model_info["Type"])
            except ValueError as exc:
                raise ValueError(f"Unknown model type: {model_info['Type']}") from exc

            model_cls = MODEL_CLASS_MAP.get(model_type)
            if model_cls is None:
                raise ValueError(f"No class registered for model type: {model_type.value}")

            model = model_cls(model_name, self.params, self.mesh)
            self.models[model_name] = model
        
        # Finalize test functions after all models are initialized
        self.mesh.finalize_test_functions()
        
        # print number of DOFs per field
        self.mesh.print_dofs_info()
    
    def Reset_Step(self) -> None:
        """
        Reset the current step for all models (e.g., revert to old fields).
        """
        for model in self.models.values():
            model.reset_step()
    
    def Initialize_Fields(self) -> None:
        """
        Initialize fields for all models.
        """
        for model in self.models.values():
            model.initialize_fields()
            
    def commit(self) -> None:
        """
        Commit the current state of all models (e.g., update old fields).
        """
        for model in self.models.values():
            model.commit()