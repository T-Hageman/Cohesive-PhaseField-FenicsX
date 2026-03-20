from __future__ import annotations

from typing import Any, Dict
from dolfinx import fem


class _MeshBase:
    """
    Base mesh class holding shared initialization logic and mesh references.
    """

    def __init__(self, params: Any) -> None:
        """
        Initialize the mesh
        """

        self.params = params

        super().__init__(params)
