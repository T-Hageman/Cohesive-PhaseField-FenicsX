from enum import Enum


class LinearSolver(Enum):
    LU = "LU"
    GMRES = "GMRES"
