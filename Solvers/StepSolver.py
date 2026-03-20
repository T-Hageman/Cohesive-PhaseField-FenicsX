from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, create_matrix
import numpy as np
import os
from Utils.mpi_utils import mprint
from Solvers._LinSolvers import _LinSolvers
import time


class StepSolver(_LinSolvers):
    def __init__(self, params, physics) -> None:
        super().__init__(params, physics)
        
        # Force single-threaded OpenMP for MUMPS
        os.environ['OMP_NUM_THREADS'] = '1'
        
        self.n_Staggered_Steps: int = max(set(params.Solution_Steps.values()))+1
        if self.n_Staggered_Steps > 1:
            self.n_passes = params.n_passes
        else:
            self.n_passes = 1
        
        # Solver parameters
        self.max_iterations = params.max_it
        self.relative_tolerance = params.rel_tol
        self.absolute_tolerance = params.abs_tol
        self.LineSearch = params.line_search
        self.LineSearch_Lims = params.line_search_lims
    
    def Solve_Step(self) -> None:
        """
        Solve the nonlinear system using Newton-Raphson iteration.
        """
        
        residual_scale = np.zeros(self.n_Staggered_Steps)
        for i in range(self.n_Staggered_Steps):
            residual_scale[i] = np.nan
        
        for passes in range(self.n_passes):
            if self.n_passes > 1:
                mprint(f" Pass {passes+1}/{self.n_passes}:")
            pass_converged = True
            last_du_per_step = {}
            for step in range(self.n_Staggered_Steps):
                if (self.n_Staggered_Steps > 1):
                    mprint(f"\tStaggered Step {step}...")
                    
                t_startStep = time.time()
                
                # Determine which fields to solve based on test functions registered for this step
                fields_to_solve = self.physics.get_fields_for_step(step)
                
                if not fields_to_solve:
                    mprint("Warning: No fields to solve for this step")
                    return

                # Assemble the force vector (residual)
                K, F = self.physics.Get_Kmat_Fvec(step)
                
                # Newton-Raphson iteration
                converged = False
                self._initialize_solver(F, step)
                for iteration in range(self.max_iterations):
                    t_startIt = time.time()
                    self._solve_linear_system(K, F, step)
                    
                    if (self.LineSearch):
                        self.Perform_LineSearch(F, self.du, fields_to_solve, step)
                    else:
                        self._update_solution(self.du, fields_to_solve, step)
                    
                    # Assemble the force vector (residual)
                    K, F = self.physics.Get_Kmat_Fvec(step)
                    
                    # Compute energy-based residual: sum(abs(F * du)) using PETSc operations
                    F_du = F.duplicate()
                    F_du.pointwiseMult(F, self.du)
                    F_du.abs()
                    residual_norm = F_du.sum()
                    F_du.destroy()
                    
                    if np.isnan(residual_scale[step]):
                        residual_scale[step] = residual_norm
                    
                    relative_residual = residual_norm / (residual_scale[step] + 1e-16)
                    mprint(f"\t\tIteration {iteration:2d}: Residual = {residual_norm:.6e}, Relative = {relative_residual:.6e}")
                    
                    if self.DEBUG_PRINT_TIMING:
                        t_endIt = time.time()
                        mprint(f"\t\t\t Iteration duration: {t_endIt - t_startIt:.4f} s")
                    
                    # Check convergence
                    if residual_norm < self.absolute_tolerance or relative_residual < self.relative_tolerance:
                        mprint(f"\t\tConverged in {iteration+1} iterations!")
                        converged = True
                        break
                
                if self.DEBUG_PRINT_TIMING:
                    t_endStep = time.time()
                    mprint(f"\t\t Staggered Step duration: {t_endStep - t_startStep:.4f} s")
                
                # Clean up solver after staggered step
                if self.du is not None:
                    last_du_per_step[step] = self.du.copy()
                self._cleanup_solver()
                
                if not converged:
                    mprint(f"\t\tWARNING: Did not converge in {self.max_iterations} iterations!")
                    mprint(f"\t\tFinal residual: {residual_norm:.6e}")

            # Recompute residuals after all steps to ensure cross-step consistency
            for step in range(self.n_Staggered_Steps):
                K_check, F_check = self.physics.Get_Kmat_Fvec(step)
                if step in last_du_per_step:
                    F_du = F_check.duplicate()
                    F_du.pointwiseMult(F_check, last_du_per_step[step])
                    F_du.abs()
                    residual_norm = F_du.sum()
                    F_du.destroy()
                else:
                    residual_norm = F_check.norm()
                if np.isnan(residual_scale[step]):
                    residual_scale[step] = residual_norm
                relative_residual = residual_norm / (residual_scale[step] + 1e-16)
                mprint(
                    f"\tPost-pass Step {step}: Residual = {residual_norm:.6e}, "
                    f"Relative = {relative_residual:.6e}"
                )
                if not (residual_norm < self.absolute_tolerance or relative_residual < self.relative_tolerance):
                    pass_converged = False

            for du in last_du_per_step.values():
                du.destroy()

            if self.n_passes > 1 and pass_converged:
                mprint(" All staggered steps converged after post-pass check; ending multi-pass early.")
                break
    
    def Perform_LineSearch(self, F_old, du, fields_to_solve, step):
        
        e_0 = F_old.dot(du)
        mprint(f"\t\tStarting Line Search: Initial energy = {e_0:.6e}")
        
        self._update_solution(du, fields_to_solve, step)
        _, F = self.physics.Get_Kmat_Fvec(step)
        e_1 = F.dot(du)
        mprint(f"\t\t\tTrial energy = {e_1:.6e}")
        
        if (abs(e_1 - e_0) < 1e-20):
            alpha = 1.0
        else:
            alpha = -e_0/(e_1 - e_0)
            alpha = max(self.LineSearch_Lims[0], min(self.LineSearch_Lims[1], alpha))
            
        mprint(f"\t\t\tComputed alpha = {alpha:.6f}")
        self._update_solution((alpha-1.0)*du, fields_to_solve, step)
        
    
    def _update_solution(self, du, fields_to_solve, step):
        """
        Update the solution fields with the increment.
        
        Args:
            du: Solution increment
            fields_to_solve: List of tuples (field_name, field_function)
            step: Solution step number
        """
        mesh = self.physics.mesh
        
        # Check if this step uses a mixed space
        if step in mesh.mixed_spaces and mesh.mixed_spaces[step] is not None:
            # Mixed formulation: update the mixed function directly
            mixed_info = mesh.mixed_spaces[step]
            mixed_func = mixed_info['mixed_function']
            
            # Update: w_new = w_old + du
            with mixed_func.x.petsc_vec.localForm() as loc_w:
                with du.localForm() as loc_du:
                    loc_w.array[:] += loc_du.array[:]
            
            # Synchronize ghost values after update
            mixed_func.x.scatter_forward()
        else:
            # Single field: direct update
            field_name, field_func = fields_to_solve[0]
            with field_func.x.petsc_vec.localForm() as loc_u:
                with du.localForm() as loc_du:
                    loc_u.array[:] += loc_du.array[:]
            
            # Synchronize ghost values after update
            field_func.x.scatter_forward()
