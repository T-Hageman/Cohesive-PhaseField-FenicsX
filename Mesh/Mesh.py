from typing import Any
from Utils.mpi_utils import mprint

from ._MeshBase import _MeshBase
from ._MeshGen import _MeshGen
from ._MeshDOFs import _MeshDOFs
from ._MeshGroups import _MeshGroups
from ._MeshPlotting import _MeshPlotting
from ._MeshEnd import _MeshEnd
from ._MeshOutputs import _MeshOutputs
from ._MeshLumpedOperators import _MeshLumpedOperators

class Mesh(_MeshBase, _MeshDOFs, _MeshGroups, _MeshGen, _MeshLumpedOperators, _MeshPlotting, _MeshOutputs, _MeshEnd):
    """
    Public mesh class assembling base behavior with mixins for generation,
    visualization, and DOF management.
    """

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        mprint("Mesh generation complete.")
