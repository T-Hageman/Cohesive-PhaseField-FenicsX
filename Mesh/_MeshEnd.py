from __future__ import annotations

from typing import Any, Dict
from dolfinx import fem


class _MeshEnd:
    """
    Base mesh class holding shared initialization logic and mesh references.
    """

    def __init__(self, params: Any) -> None:
        pass
