from __future__ import annotations

from typing import Any, Dict
from dolfinx import fem
from Utils.mpi_utils import rank, size, comm, mprint

import os

import pyvista as pv
import numpy as np
from dolfinx.plot import vtk_mesh
from dolfinx import fem
from dolfinx.io import VTKFile
from basix.ufl import element
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc
from ufl import div, inner, dot, dx, TrialFunction, max_value

class _MeshLumpedOperators:
    """
    mesh class for output related functions
    """

    def __init__(self, params: Any) -> None:

        
        super().__init__(params)

    def Get_Weights_Dofs(self, field_names: list[str], step: int):
        """
        Get lumped mass matrix weights for specified fields at given step.
        """

        E_t  = self.Get_TestFunction(field_names[0], step=step)
        W_Vec = assemble_vector(fem.form(E_t*dx))
        W_Vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        W_Vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        dofs = {}
        local_size = None
        for name in field_names:
            field_output = self.Get_Field_For_Output(name)
            if isinstance(field_output, tuple):
                mixed_func, start_idx, comp_count = field_output
                parent_space = mixed_func.function_space
                if local_size is None:
                    index_map = parent_space.dofmap.index_map
                    bs = parent_space.dofmap.index_map_bs
                    local_size = index_map.size_local * bs
                field_dofs = []
                for offset in range(comp_count):
                    _, sub_dofs = parent_space.sub(start_idx + offset).collapse()
                    if local_size is not None:
                        sub_dofs = sub_dofs[sub_dofs < local_size]
                    field_dofs.append(sub_dofs)
                if field_dofs:
                    dofs[name] = np.concatenate(field_dofs).astype(np.int32, copy=False)
                else:
                    dofs[name] = np.empty(0, dtype=np.int32)
            else:
                field = field_output
                index_map = field.function_space.dofmap.index_map
                bs = field.function_space.dofmap.index_map_bs
                if local_size is None:
                    local_size = index_map.size_local * bs
                dofs[name] = np.arange(local_size, dtype=np.int32)

        if local_size is None:
            local_size = 0
        with W_Vec.localForm() as W_local:
            W_array = W_local.array[:local_size].copy()
        W = W_array[dofs[field_names[0]]]

        vals = {}
        for name in field_names:
            field_output = self.Get_Field_For_Output(name)
            if isinstance(field_output, tuple):
                mixed_func, start_idx, comp_count = field_output
                parent_space = mixed_func.function_space
                if local_size is None:
                    index_map = parent_space.dofmap.index_map
                    bs = parent_space.dofmap.index_map_bs
                    local_size = index_map.size_local * bs
                field_dofs = []
                for offset in range(comp_count):
                    _, sub_dofs = parent_space.sub(start_idx + offset).collapse()
                    if local_size is not None:
                        sub_dofs = sub_dofs[sub_dofs < local_size]
                    field_dofs.append(sub_dofs)
                if field_dofs:
                    dofs_local = np.concatenate(field_dofs).astype(np.int32, copy=False)
                else:
                    dofs_local = np.empty(0, dtype=np.int32)
                vals[name] = mixed_func.x.array[:local_size][dofs_local]
            else:
                field = field_output
                index_map = field.function_space.dofmap.index_map
                bs = field.function_space.dofmap.index_map_bs
                if local_size is None:
                    local_size = index_map.size_local * bs
                dofs_local = np.arange(local_size, dtype=np.int32)
                vals[name] = field.x.array[:local_size][dofs_local]

        return W, dofs, vals
    
    def Get_Old_For_Lumped(self, old_field_names: list[str], step: int):
        """
        Get old field values for specified old fields at given step.
        """

        valsOld = {}
        for name in old_field_names:
            field_output = self.Get_Field_For_Output(name)
            if isinstance(field_output, tuple):
                mixed_func, start_idx, comp_count = field_output
                parent_space = mixed_func.function_space
                index_map = parent_space.dofmap.index_map
                bs = parent_space.dofmap.index_map_bs
                local_size = index_map.size_local * bs
                field_dofs = []
                for offset in range(comp_count):
                    _, sub_dofs = parent_space.sub(start_idx + offset).collapse()
                    sub_dofs = sub_dofs[sub_dofs < local_size]
                    field_dofs.append(sub_dofs)
                if field_dofs:
                    dofs = np.concatenate(field_dofs).astype(np.int32, copy=False)
                else:
                    dofs = np.empty(0, dtype=np.int32)
                valsOld[name] = mixed_func.x.array[:local_size][dofs]
            else:
                field = field_output
                index_map = field.function_space.dofmap.index_map
                bs = field.function_space.dofmap.index_map_bs
                local_size = index_map.size_local * bs
                dofs = np.arange(local_size, dtype=np.int32)
                valsOld[name] = field.x.array[:local_size][dofs]

        return valsOld

        
