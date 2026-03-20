from enum import Enum
from typing import Dict, Type

class ModelType(str, Enum):
    """String enums representing the available physics models."""

    SOLID_LINEARELASTIC = "Solid:LinearElastic"
    SOLID_PHASE_FIELD = "Solid:PhaseField"
    BOUNDARY_PRESCRIBED = "Boundary:PrescribedValue"
    BOUNDARY_DISPCONTROL = "Boundary:DisplacementControl"
    BOUNDARY_EXTERNALFORCE = "Boundary:ExternalForce"
    BOUNDARY_TORSION = "Boundary:Torsion"
    SOLID_COHESIVE_LINEARELASTIC = "Solid:Cohesive_LinearElastic"
    SOLID_VISCOELASTIC = "Solid:ViscoElastic"
    SOLID_CRACK_DEPTH_MEASURE = "Solid:CrackDepthMeasure"
    
    