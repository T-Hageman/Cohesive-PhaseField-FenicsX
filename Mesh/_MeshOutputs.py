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

class _MeshOutputs:
    """
    mesh class for output related functions
    """

    def __init__(self, params: Any) -> None:
        
        self.output_fields: list = getattr(params, "output_fields", [])
        self.vtk: bool = getattr(params, "write_vtk", False)
        self.vtk_single_file: bool = getattr(params, "vtk_single_file", True)
        self.hdf5: bool = getattr(params, "write_hdf5", False)
        self.output_dir: str = getattr(params, "output_dir", None)
        if self.vtk or self.hdf5:
            self.output_freq: int = getattr(params, "output_interval", 10)
            self.save_Mesh_every_step: bool = getattr(params, "save_Mesh_every_step", False)
            if self.vtk:
                self._vtk_file = None  # Persistent VTKFile object for time series
            
        # Ensure folder is empty before outputs
        if rank == 0 and self.output_dir is not None:
            if os.path.exists(self.output_dir):
                import shutil
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Wait for rank 0 to finish directory setup
        if self.output_dir is not None:
            comm.Barrier()
        
        super().__init__(params)

    def Write_Outputs(self, step: int, time: float) -> None:
        """
        Write output files for the current time step.
        """

        if self.output_dir is None:
            return

        mprint(f"Writing output files for step {step}, time {time:.2f} s")
        
        if rank == 0:
            self._append_time_data(step, time)

        if self.vtk and (step % self.output_freq == 0):
            self._write_vtk(step, time)
            pass
        
        if self.hdf5 and (step % self.output_freq == 0):
            self._write_hdf5(step, time)
    
    def close_outputs(self) -> None:
        """
        Close any open output files (e.g., VTK file for time series).
        Call this at the end of the simulation.
        """
        if hasattr(self, '_vtk_file') and self._vtk_file is not None:
            self._vtk_file.close()
            mprint("Closed VTK time series file")
            self._vtk_file = None
            
    def _append_time_data(self, step: int, time: float) -> None:
        """
        Append time data to a master file for tracking simulation progress.
        """
        
        if rank != 0:
            return

        import h5py

        time_file = f"{self.output_dir}/time_data.hdf5"
        os.makedirs(self.output_dir, exist_ok=True)

        data_names = ["time", "dt"]
        if hasattr(self.params, "Global_Measures"):
            for name in sorted(self.params.Global_Measures.keys()):
                data_names.append(name)

        data_row = [float(time), float(self.params.dt)]
        if hasattr(self.params, "Global_Measures"):
            for name in sorted(self.params.Global_Measures.keys()):
                value = self.params.Global_Measures.get(name, np.nan)
                data_row.append(float(value))

        data_row = np.asarray(data_row, dtype=float)

        with h5py.File(time_file, "a") as h5f:
            if "data_names" not in h5f:
                names = np.asarray(data_names, dtype="S")
                h5f.create_dataset("data_names", data=names)
                data = data_row.reshape(1, -1)
                h5f.create_dataset("data", data=data, maxshape=(None, data.shape[1]))
                return

            existing_names = [
                name.decode("utf-8") for name in h5f["data_names"][()]
            ]
            if existing_names != data_names:
                raise ValueError(
                    "time_data.hdf5 data_names mismatch with current Global_Measures"
                )

            dataset = h5f["data"]
            dataset.resize((dataset.shape[0] + 1, dataset.shape[1]))
            dataset[-1, :] = data_row

    def _write_hdf5(self, step: int, time: float) -> None:
        """
        Write mesh and field data to an HDF5 file using DG1 discretization.
        """
        import h5py
        from mpi4py import MPI
        from basix.ufl import element

        os.makedirs(self.output_dir, exist_ok=True)
        mesh_path = f"{self.output_dir}/Mesh.hdf5"
        outputs_path = f"{self.output_dir}/Outputs_{step}.hdf5"

        if size > 1 and not getattr(h5py.get_config(), "mpi", False):
            raise RuntimeError("h5py MPI support is required for parallel HDF5 output.")

        file_kwargs = {}
        if size > 1:
            file_kwargs = {"driver": "mpio", "comm": comm}

        gdim = self.mesh.geometry.x.shape[1]
        tdim = self.mesh.topology.dim
        cell_map = self.mesh.topology.index_map(tdim)
        cell_start, cell_end = cell_map.local_range
        num_cells_local = cell_end - cell_start

        write_mesh = self.save_Mesh_every_step or (rank == 0 and not os.path.exists(mesh_path))
        write_mesh = comm.bcast(write_mesh, root=0)

        # Helper: build cell table for a given DG space
        def _build_cell_table(values, dm, npc):
            table = np.zeros((num_cells_local, npc), dtype=float)
            for cell in range(num_cells_local):
                dofs = dm.cell_dofs(cell)
                table[cell, :] = values[dofs]
            return table

        # Track which DG degrees have had their mesh coordinates written
        written_mesh_degrees = set()

        with h5py.File(outputs_path, "a", **file_kwargs) as h5f_out, \
             h5py.File(mesh_path, "a", **file_kwargs) as h5f_mesh:
            fields_group = h5f_out.require_group("Fields")

            for field_spec in self.output_fields:
                result = self.Process_Fields_For_Plotting(field_spec)
                local_has = 1 if result is not None else 0
                global_has = comm.allreduce(local_has, op=MPI.SUM)
                if global_has == 0:
                    continue

                if result is None:
                    # This rank has no data; other ranks do. Need DG space info.
                    # Use degree 1 as safe fallback for shape computation.
                    V_dg = self._get_dg_space(1)
                else:
                    field_values, _, V_dg = result

                dm = V_dg.dofmap
                npc = len(dm.cell_dofs(0)) if num_cells_local > 0 else 0
                if size > 1:
                    npc = comm.allreduce(npc, op=MPI.MAX)

                # Determine the DG degree from the space
                dg_degree = V_dg.element.basix_element.degree

                # Write mesh coordinates for this degree if not yet done
                if dg_degree not in written_mesh_degrees:
                    written_mesh_degrees.add(dg_degree)
                    mesh_group = h5f_mesh.require_group(f"Mesh_DG{dg_degree}")
                    mesh_group.require_dataset("X", shape=(cell_map.size_global, npc), dtype=float)
                    mesh_group.require_dataset("Y", shape=(cell_map.size_global, npc), dtype=float)
                    if gdim == 3:
                        mesh_group.require_dataset("Z", shape=(cell_map.size_global, npc), dtype=float)

                    if write_mesh:
                        dof_coords = V_dg.tabulate_dof_coordinates().reshape(-1, gdim)
                        cx = _build_cell_table(dof_coords[:, 0], dm, npc) if num_cells_local > 0 else np.zeros((0, npc))
                        cy = _build_cell_table(dof_coords[:, 1], dm, npc) if num_cells_local > 0 else np.zeros((0, npc))
                        mesh_group["X"][cell_start:cell_end, :] = cx
                        mesh_group["Y"][cell_start:cell_end, :] = cy
                        if gdim == 3:
                            cz = _build_cell_table(dof_coords[:, 2], dm, npc) if num_cells_local > 0 else np.zeros((0, npc))
                            mesh_group["Z"][cell_start:cell_end, :] = cz

                # Write field data
                if result is None:
                    dof_count = V_dg.dofmap.index_map.size_local
                    field_values = np.zeros(dof_count, dtype=float)

                cell_table = _build_cell_table(field_values, dm, npc) if num_cells_local > 0 else np.zeros((0, npc))

                field_ds_name = field_spec
                fields_group.require_dataset(
                    field_ds_name,
                    shape=(cell_map.size_global, npc),
                    dtype=float,
                )
                # Store the DG degree as an attribute so readers know which mesh group to use
                fields_group[field_ds_name].attrs["dg_degree"] = dg_degree
                fields_group[field_ds_name][cell_start:cell_end, :] = cell_table

    def _write_vtk(self, step: int, time: float) -> None:
        """
        Write VTK output for the current step using DG fields matching each field's degree.
        """

        functions = []
        for field_spec in self.output_fields:
            result = self.Process_Fields_For_Plotting(field_spec)
            if result is None:
                continue
            field_values, _, V_dg = result
            field_func = fem.Function(V_dg, name=field_spec)
            field_func.x.array[:] = field_values
            functions.append(field_func)
        
        if self.vtk_single_file:
            # Use persistent VTKFile object for time series
            if self._vtk_file is None:
                vtk_path = f"{self.output_dir}/Outputs.pvd"
                self._vtk_file = VTKFile(self.mesh.comm, vtk_path, "w")
            
            if functions:
                self._vtk_file.write_function(functions, time)
            else:
                self._vtk_file.write_mesh(self.mesh, time)
        else:
            # Separate file per step
            vtk_path = f"{self.output_dir}/Outputs_{step}.pvd"
            with VTKFile(self.mesh.comm, vtk_path, "w") as vtk:
                if functions:
                    vtk.write_function(functions, time)
                else:
                    vtk.write_mesh(self.mesh, time)
        
