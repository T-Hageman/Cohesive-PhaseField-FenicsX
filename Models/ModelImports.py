from enum import Enum
from typing import Dict, Type

from .BaseModel import BaseModel
from .SolidModels.LinearElastic import LinearElastic
from .Boundary.PrescribedValue import PrescribedValue
from .SolidModels.PhaseField import PhaseField
from .SolidModels.Cohesive_LinearElastic import Cohesive_LinearElastic
from .Boundary.DisplacementControl import DisplacementControl
from .Boundary.Torsion import AppliedTorsion
from .Boundary.ExternalForce import ExternalForce
from .SolidModels.ViscoElastic import ViscoElastic
from .SolidModels.CrackDepthMeasure import CrackDepthMeasure

from .ModelEnums import ModelType

MODEL_CLASS_MAP: Dict[ModelType, Type[BaseModel]] = {
    ModelType.BOUNDARY_PRESCRIBED: PrescribedValue,
    ModelType.BOUNDARY_DISPCONTROL: DisplacementControl,
    ModelType.SOLID_LINEARELASTIC: LinearElastic,
    ModelType.SOLID_PHASE_FIELD: PhaseField,
    ModelType.SOLID_COHESIVE_LINEARELASTIC: Cohesive_LinearElastic,
    ModelType.SOLID_VISCOELASTIC: ViscoElastic,
    ModelType.BOUNDARY_TORSION: AppliedTorsion,
    ModelType.BOUNDARY_EXTERNALFORCE: ExternalForce,
    ModelType.SOLID_CRACK_DEPTH_MEASURE: CrackDepthMeasure,
}
