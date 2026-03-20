import time
from typing import Any

import numpy as np

# Safety-net: ensure BLAS/OpenMP threads are limited to 1 per MPI rank.
# The primary call lives in Main.py (before any heavy imports), but
# this guards against _LinSolvers being imported from tests or scripts
# that skip Main.py.
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

from petsc4py import PETSc

from Utils.mpi_utils import mprint, rank
from Solvers.SolverENUMS import LinearSolver


class _LinSolvers:
    def __init__(self, params: Any, physics: Any) -> None:
        self.params = params
        self.physics = physics

        self.ksp = None
        self.du = None
        self.F_neg = None
        self.mumps_configured = False  # Track if MUMPS options have been set
        self.asm_configured = False  # Track if ASM sub-solvers have been set

        self.linear_solver = self._normalize_solver_choice(
            getattr(params, "Linear_Solver", LinearSolver.LU)
        )

        self.gmres_restart = getattr(params, "GMRES_Restart", 50)
        self.gmres_max_it = getattr(params, "GMRES_Max_Iter", 200)
        self.gmres_rtol = getattr(params, "GMRES_RelTol", 1e-6)
        
        self.DEBUG_PRINT_TIMING = getattr(params, "DEBUG_PRINT_TIMING", False)

    def _normalize_solver_choice(self, choice):
        if isinstance(choice, LinearSolver):
            return choice.value
        return str(choice).upper()

    def _initialize_solver(self, F_vec, step):
        """
        Initialize KSP solver and work vectors for reuse across Newton iterations.
        Called once per staggered step.
        """
        pass

    def _cleanup_solver(self):
        """
        Destroy KSP solver and work vectors to free memory.
        Called after each staggered step completes.
        """
        if self.du is not None:
            self.du.destroy()
            self.du = None
        if self.F_neg is not None:
            self.F_neg.destroy()
            self.F_neg = None
        if self.ksp is not None:
            self.ksp.destroy()
            self.ksp = None

    def _solve_linear_system(self, K_mat, F_vec, step):
        """
        Solve the linear system K * du = -F using reusable KSP solver.

        Args:
            K_mat: Tangent matrix
            F_vec: Force vector (residual)
            step: Solution step index
        """
        t_start = time.time()
        
        self._cleanup_solver()

        self.du = F_vec.copy()
        self.F_neg = F_vec.copy()

        self.ksp = PETSc.KSP().create(F_vec.getComm())
        self.mumps_configured = False
        self.asm_configured = False
        self.fieldsplit_configured = False

        K_norm = K_mat.norm()
        F_norm = F_vec.norm()

        if not np.isfinite(F_norm):
            mprint(f"\t\t\tERROR: Force vector has NaN/Inf values! Vector norm = {F_norm}")
            raise RuntimeError("Aborting due to invalid force vector.")

        if not np.isfinite(K_norm):
            mprint(f"\t\t\tERROR: Matrix has NaN/Inf values! Matrix norm = {K_norm}")
            raise RuntimeError("Aborting due to invalid matrix.")

        if self.linear_solver == LinearSolver.GMRES.value:
            self.solve_ASM(K_mat, F_vec)
        else:
            self.solve_MUMPS(K_mat, F_vec)

        converged_reason = self.ksp.getConvergedReason()
        if converged_reason < 0:
            mprint(f"\t\t\t\tWarning: Solver did not converge (reason: {converged_reason})")
            mprint(
                f"\t\t\t\tIterations: {self.ksp.getIterationNumber()}, "
                f"Final residual: {self.ksp.getResidualNorm():.6e}"
            )

        t_end = time.time()
        if rank == 0 and self.DEBUG_PRINT_TIMING:
            mprint(f"\t\t\tSolver step duration: {t_end - t_start:.4f} s")

    def _configure_mumps_factor(self, pc):
        pc.setFactorSetUpSolverType()
        factor = pc.getFactorMatrix()
        factor.setMumpsIcntl(4, 1)    # Print level (1 = errors only)
        factor.setMumpsIcntl(7, 2)    # Ordering (2 = AMF, better for mixed systems)
        factor.setMumpsIcntl(8, 77)   # Scaling: automatic choice (row/col equilibration)
        factor.setMumpsIcntl(14, 100) # Memory relaxation (100% = more memory, less compression)
        factor.setMumpsIcntl(24, 1)   # Null pivot detection
        factor.setMumpsCntl(1, 0.01)  # Relative pivot threshold (larger = more stable)

    def _prepare_rhs(self, F_vec):
        self.F_neg.copy(F_vec)
        self.F_neg.scale(-1.0)

    def solve_MUMPS(self, K_mat, F_vec):
        """
        Direct solve using LU with MUMPS.
        """
        self.ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = self.ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        self.ksp.setOperators(K_mat)

        if not self.mumps_configured:
            self._configure_mumps_factor(pc)
            self.mumps_configured = True

        self._prepare_rhs(F_vec)
        self.ksp.solve(self.F_neg, self.du)

    def solve_ASM(self, K_mat, F_vec):
        """
        Solve using GMRES with ASM preconditioner and MUMPS on subdomains.
        """
        self.ksp.setType(PETSc.KSP.Type.GMRES)
        pc = self.ksp.getPC()
        pc.setType(PETSc.PC.Type.ASM)
        pc.setASMOverlap(0)
        self.ksp.setOperators(K_mat)

        if not self.asm_configured:
            pc.setUp()
            sub_ksps = pc.getASMSubKSP()
            for sub_ksp in sub_ksps:
                sub_ksp.setType(PETSc.KSP.Type.PREONLY)
                sub_pc = sub_ksp.getPC()
                sub_pc.setType(PETSc.PC.Type.LU)
                sub_pc.setFactorSolverType("mumps")
                self._configure_mumps_factor(sub_pc)
            self.asm_configured = True

        self._prepare_rhs(F_vec)
        self.ksp.solve(self.F_neg, self.du)

