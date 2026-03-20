from typing import Any
from Models.ModelImports import MODEL_CLASS_MAP, ModelType
from Utils.mpi_utils import mprint, comm
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, create_matrix
import numpy as np
from petsc4py import PETSc
from ufl import derivative
import time

class _PhysicsKF:
    
    def __init__(self, mesh: Any, params: Any) -> None:
        # Store PETSc objects for explicit cleanup
        self.K_mat = None
        self.F_vec = None
    
    def Get_Kmat_Fvec(self, step: int) -> None:
        """
        Assemble global stiffness matrix and force vector for all models.
        """
        
        # Clean up previous PETSc objects before creating new ones
        t_start = time.time()
        if self.K_mat is not None:
            self.K_mat.destroy()
            self.K_mat = None
        if self.F_vec is not None:
            self.F_vec.destroy()
            self.F_vec = None
        
        # Get boundary conditions for this step
        bcs = self.get_boundary_conditions(step)
        
        t_end = time.time()
        if self.DEBUG_PRINT_TIMING:
            mprint(f"\t\t Getting BCs duration: {t_end - t_start:.4f} s")
        
        # Accumulate weak forms from all models
        F_form = None
        K_form = None
        
        for model in self.models.values():
            t_start = time.time()
            result = model.assemble_KF(step)
            if result is not None:
                K, F = result
            
                if F is not None:
                    F_form = F if F_form is None else F_form + F
                if K is not None:
                    K_form = K if K_form is None else K_form + K
            
            if self.DEBUG_PRINT_TIMING:
                comm.Barrier()
                t_end = time.time()
                mprint(f"\t\t '{model.name}': {t_end - t_start:.4f} s")
        
        t_start = time.time()
        # Compile and assemble force vector
        F_compiled = fem.form(F_form)
        self.F_vec = assemble_vector(F_compiled)
        
        # Compile and assemble stiffness matrix
        K_compiled = fem.form(K_form)
        self.K_mat = create_matrix(K_compiled)
        self.K_mat.zeroEntries()
        assemble_matrix(self.K_mat, K_compiled, bcs=None)
        self.K_mat.assemble()

        # Allow post-assembly additions to K/F from models
        for model in self.models.values():
            if hasattr(model, "Add_KF"):
                model.Add_KF(self.K_mat, self.F_vec, step)

        # Apply boundary conditions and sync after post-assembly edits
        self.F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.K_mat.assemble()
        if bcs:
            self._apply_bcs_to_matrix(self.K_mat, bcs)
            fem.petsc.set_bc(self.F_vec, bcs)

        self.F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.K_mat.assemble()

        t_end = time.time()
        if self.DEBUG_PRINT_TIMING:
            mprint(f"\t\t Assembling K/F total duration: {t_end - t_start:.4f} s")

        return self.K_mat, self.F_vec

    def Update_Global_Measures(self) -> None:
        """
        Update global measures from all models.
        """
        for model in self.models.values():
            if hasattr(model, "Update_Global_Measures"):
                model.Update_Global_Measures()

    def _apply_bcs_to_matrix(self, K_mat: PETSc.Mat, bcs) -> None:
        """
        Apply Dirichlet boundary conditions to a PETSc matrix after assembly.
        """
        rows = []
        for bc in bcs:
            dofs, pos = bc.dof_indices()
            if pos > 0:
                rows.append(dofs[:pos])

        if rows:
            rows = np.unique(np.concatenate(rows)).astype(PETSc.IntType)
        else:
            rows = np.empty(0, dtype=PETSc.IntType)
        if hasattr(K_mat, "zeroRowsColumnsLocal"):
            K_mat.zeroRowsColumnsLocal(rows, diag=1.0)
        else:
            K_mat.zeroRowsColumns(rows, diag=1.0)

    def get_fields_for_step(self, step):
        """
        Determine which fields need to be solved based on test functions 
        registered for this step in the mesh.
        
        Args:
            step: Solution step number
            
        Returns:
            List of tuples (field_name, field_function) for fields to solve
        """
        fields_to_solve = []
        
        # Check which test functions are registered for this step
        if not hasattr(self.mesh, 'test_functions'):
            mprint("Warning: No test functions registered in mesh")
            return fields_to_solve
        
        # Parse test function keys to find fields for this step
        # Keys are in format: "{field_name}_test_step{step}"
        for test_key in self.mesh.test_functions.keys():
            if test_key.endswith(f"_test_step{step}"):
                # Extract field name from key
                field_name = test_key.replace(f"_test_step{step}", "")
                
                # Get the corresponding field
                if field_name in self.mesh.fields:
                    field_func = self.mesh.fields[field_name]
                    fields_to_solve.append((field_name, field_func))
                    #mprint(f"  Found field '{field_name}' for step {step}")
                else:
                    mprint(f"  Warning: Test function for '{field_name}' found but field not in mesh.fields")
        return fields_to_solve
    
    def get_boundary_conditions(self, step: int = None):
        """
        Get boundary conditions for a given step.
        
        Args:
            step: Solution step (optional, returns all if None)
            
        Returns:
            List of boundary conditions
        """
        # Determine which fields are being solved in this step
        if step is not None and hasattr(self.mesh, 'step_fields'):
            fields_in_step = self.mesh.step_fields.get(step, [])
        else:
            fields_in_step = None
        
        bcs = []
        for model in self.models.values():
            if fields_in_step is not None and hasattr(model, 'get_bcs_for_field'):
                for field_name in fields_in_step:
                    field_bcs = model.get_bcs_for_field(field_name)
                    if field_bcs:
                        bcs.extend(field_bcs)

        return bcs
