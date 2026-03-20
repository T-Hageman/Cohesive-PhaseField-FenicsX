from __future__ import annotations

from typing import Any, Dict
from dolfinx import fem
from Utils.mpi_utils import rank, size, comm, mprint

import pyvista as pv
import numpy as np
from dolfinx.plot import vtk_mesh
from dolfinx import fem
from basix.ufl import element

class _MeshPlotting:
    """
    mesh class for plotting related functions
    """

    def __init__(self, params: Any) -> None:
        # Initialize storage for plotters and grids for field plotting
        self._field_plotters: Dict[str, Any] = {}
        self._field_grids: Dict[str, Any] = {}
        self._field_actors: Dict[str, Any] = {}
        self._measure_plots: Dict[tuple, Dict[str, Any]] = {}
        
        self.default_plot_size = (600, 500)
        
        super().__init__(params)
        
    def plot_mesh(self) -> None:
        """
        Plot the mesh using pyvista with colored boundaries and labels (non-blocking)
        MPI-aware: Each core creates DG1 mesh and sends data to rank 0 for plotting
        """
        import pyvista as pv
        import numpy as np
        from dolfinx.plot import vtk_mesh
        
        # Create DG1 function space on local mesh partition
        tdim = self.mesh.topology.dim
        dg1_element = element("DG", self.mesh.basix_cell(), 1)
        V_dg1 = fem.functionspace(self.mesh, dg1_element)
        
        # Get the DG1 mesh topology and geometry
        topology_dg1, cell_types_dg1, geometry_dg1 = vtk_mesh(V_dg1)
        
        # Gather boundary facet data from each rank
        local_boundary_data = self.Get_Boundary_For_Plotting(np, tdim)
        
        # Gather all data to rank 0 using gather (efficient for moderate-sized data)
        all_topologies = comm.gather(topology_dg1, root=0)
        all_cell_types = comm.gather(cell_types_dg1, root=0)
        all_geometries = comm.gather(geometry_dg1, root=0)
        all_boundary_data = comm.gather(local_boundary_data, root=0)
        
        # Only rank 0 performs plotting
        if rank != 0:
            return
        
        # Create a plotter
        plotter = pv.Plotter(window_size=self.default_plot_size)
        
        # Define colors for each rank's partition
        rank_colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", 
                      "plum", "peachpuff", "lightcyan", "pink"]
        
        # Plot each rank's mesh partition separately with different colors
        for rank_idx, (topology, cell_types, geometry) in enumerate(zip(all_topologies, all_cell_types, all_geometries)):
            rank_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
            rank_color = rank_colors[rank_idx % len(rank_colors)]
            if self.params.dim == 2:
                plotter.add_mesh(rank_grid, show_edges=True, color=rank_color, edge_color='black', 
                            opacity=0.5, line_width=1, label=f'Rank {rank_idx}')
            else:
                plotter.add_mesh(rank_grid, show_edges=True, color=rank_color, edge_color='black', 
                            opacity=1.0, line_width=1, label=f'Rank {rank_idx}')
        
        # Define boundary colors (will be indexed by boundary name)
        color_list = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "yellow"]
        boundary_colors = {}
        
        # Create color mapping from boundary_markers
        if hasattr(self, 'boundary_markers') and self.boundary_markers:
            for idx, (name, tag) in enumerate(self.boundary_markers.items()):
                boundary_colors[name] = color_list[idx % len(color_list)]
        
        # Plot boundaries with different colors - process each rank separately to avoid cross-rank connections
        if hasattr(self, 'boundary_markers') and self.boundary_markers:
            for name, tag in self.boundary_markers.items():
                color = boundary_colors.get(name, 'gray')
                all_centers = []
                
                # Process each rank's boundary data separately
                for rank_idx, rank_boundary_data in enumerate(all_boundary_data):
                    if name in rank_boundary_data:
                        data = rank_boundary_data[name]
                        points = data['points']
                        lines = data['lines']
                        faces = data['faces']
                        
                        if len(points) > 0 and (faces or lines):
                            # Create polydata for this rank's boundary portion
                            if faces:
                                # boundary_mesh = pv.PolyData(points, faces=np.hstack(faces))
                                # # Add boundary to plot (only label the first occurrence)
                                label = f"{name} (tag={tag})" if rank_idx == 0 else None
                                # plotter.add_mesh(boundary_mesh, color=color, line_width=1, label=label)
                            else:
                                boundary_mesh = pv.PolyData(points, lines=np.hstack(lines))
                                # Add boundary to plot (only label the first occurrence)
                                label = f"{name} (tag={tag})" if rank_idx == 0 else None
                                plotter.add_mesh(boundary_mesh, color=color, line_width=5, label=label)
                            
                            
                            
                            # Collect centers for label placement
                            all_centers.append(np.mean(points, axis=0))
                
                # Add text label at the average center of all boundary portions
                if all_centers:
                    center = np.mean(all_centers, axis=0)
                    plotter.add_point_labels(
                        [center], [f"{name} (tag={tag})"],
                        font_size=12, point_size=8, 
                        text_color=color, point_color=color,
                        always_visible=True, bold=True
                    )
        
        # Add domain label at mesh center
        all_geometry = np.concatenate(all_geometries)
        mesh_center = np.mean(all_geometry, axis=0)
        domain_label = "Domain"
        if hasattr(self, 'domain_markers') and self.domain_markers:
            # Get the first domain marker
            for name, tag in self.domain_markers.items():
                domain_label = f"{name} (tag={tag})"
                break
        
        plotter.add_point_labels(
            [mesh_center], [domain_label],
            font_size=14, point_size=10,
            text_color='darkgray', point_color='darkgray',
            always_visible=True, bold=True
        )
        
        plotter.show_axes()
        if self.mesh.topology.dim == 2:
            plotter.view_xy()
        else:
            plotter.view_isometric()
        plotter.add_legend()
        plotter.show(interactive_update=True) 

    def Get_Boundary_For_Plotting(self, np, tdim):
        local_boundary_data = {}
        if hasattr(self, 'facet_tags') and self.facet_tags is not None:
            fdim = tdim - 1
            self.mesh.topology.create_connectivity(fdim, tdim)
            self.mesh.topology.create_connectivity(tdim, 0)
            facet_to_cell = self.mesh.topology.connectivity(fdim, tdim)
            cell_to_vertex = self.mesh.topology.connectivity(tdim, 0)
            
            # Get geometry dofmap to properly map topology vertices to geometry coordinates
            geom_dofmap = self.mesh.geometry.dofmap
            
            if hasattr(self, 'boundary_markers') and self.boundary_markers:
                for name, tag in self.boundary_markers.items():
                    facets = self.facet_tags.find(tag)
                    
                    # Use a dictionary to track unique geometry points and their indices
                    geom_point_to_idx = {}
                    boundary_points = []
                    boundary_lines = []
                    boundary_faces = []
                    
                    for facet in facets:
                        # Get the cell attached to this facet
                        cells = facet_to_cell.links(facet)
                        if len(cells) == 0:
                            continue
                        cell = cells[0]
                        
                        # Get geometry dofs for this cell
                        geom_dofs = geom_dofmap[cell]
                        
                        # Get topology vertices for this cell
                        topo_vertices = cell_to_vertex.links(cell)
                        
                        # For the facet, we need to find which topology vertices belong to it
                        # Get facet topology vertices
                        self.mesh.topology.create_connectivity(fdim, 0)
                        facet_to_vertex = self.mesh.topology.connectivity(fdim, 0)
                        facet_topo_vertices = facet_to_vertex.links(facet)
                        
                        # Map facet topology vertices to geometry dofs
                        facet_indices = []
                        for topo_v in facet_topo_vertices:
                            # Find position of this topology vertex in the cell's topology vertices
                            local_pos = np.where(topo_vertices == topo_v)[0]
                            if len(local_pos) > 0:
                                # Get corresponding geometry dof
                                geom_dof = geom_dofs[local_pos[0]]
                                
                                if geom_dof not in geom_point_to_idx:
                                    # New geometry point - add it to the list
                                    geom_point_to_idx[geom_dof] = len(boundary_points)
                                    boundary_points.append(self.mesh.geometry.x[geom_dof].copy())
                                facet_indices.append(geom_point_to_idx[geom_dof])
                        
                        # Build connectivity using unique vertex indices
                        if len(facet_indices) == 2:  # 1D facet (line)
                            boundary_lines.append([2, facet_indices[0], facet_indices[1]])
                        elif len(facet_indices) >= 3:  # 2D facet (triangle or quad)
                            cell = [len(facet_indices)] + facet_indices
                            boundary_faces.append(cell)
                    
                    if boundary_points:
                        boundary_points_array = np.array(boundary_points)
                        local_boundary_data[name] = {
                            'points': boundary_points_array,
                            'lines': boundary_lines,
                            'faces': boundary_faces
                        }
                        
        return local_boundary_data # Non-blocking
        
    def Close_Plots(self):
        # Close all field plotters
        for plotter in self._field_plotters.values():
            try:
                plotter.close()
            except Exception:
                pass
        self._field_plotters.clear()
        self._field_grids.clear()
        self._field_actors.clear()
        
        # Close all measure plotters
        for plot_state in self._measure_plots.values():
            try:
                plot_state["plotter"].close()
            except Exception:
                pass
        self._measure_plots.clear()

    def Plot_Fields(self):
        """ Plot all fields defined in params.Plot_Fields 
        MPI-aware: Each core interpolates fields to DG1 mesh and sends data to rank 0 for plotting
        """

        # Get list of fields to plot from params
        if not hasattr(self.params, 'Plot_Fields'):
            mprint("No fields to plot (params.Plot_Fields not defined)")
            return
        
        plot_fields = self.params.Plot_Fields
        log_plot = getattr(self.params, "LogPlot", None)
        
        # Create DG1 function space for mesh topology (VTK grid structure)
        tdim = self.mesh.topology.dim
        dg1_element = element("DG", self.mesh.basix_cell(), 1)
        V_dg1 = fem.functionspace(self.mesh, dg1_element)
        
        # Get the DG1 mesh topology and geometry for this rank
        topology_dg1, cell_types_dg1, geometry_dg1 = vtk_mesh(V_dg1)
        
        # Process each field and gather data
        all_field_data = {}
        for field_idx, field_spec in enumerate(plot_fields):
            # Process field specification to get the field and title
            # For plotting, use DG1 so all fields share the same VTK grid
            result = self.Process_Fields_For_Plotting(field_spec, V_dg=V_dg1)
            if result is None:
                continue
            field_values, field_title, _ = result

            use_log = False
            if isinstance(log_plot, list) and field_idx < len(log_plot):
                use_log = bool(log_plot[field_idx])

            if use_log:
                field_title = f"{field_title} (log10)"
                field_values = np.log10(np.maximum(field_values, 1.0e-300))
            
            all_field_data[field_spec] = (field_values, field_title)
        
        # Gather all data to rank 0 using gather (efficient for moderate-sized data)
        all_topologies = comm.gather(topology_dg1, root=0)
        all_cell_types = comm.gather(cell_types_dg1, root=0)
        all_geometries = comm.gather(geometry_dg1, root=0)
        all_rank_field_data = comm.gather(all_field_data, root=0)
        
        # Only rank 0 performs plotting
        if rank != 0:
            return
        
        # Iterate over each field to plot
        for field_spec in plot_fields:
            # Gather field data from all ranks
            combined_field_values = []
            field_title = None
            
            for rank_field_data in all_rank_field_data:
                if field_spec in rank_field_data:
                    local_values, title = rank_field_data[field_spec]
                    combined_field_values.append(local_values)
                    if field_title is None:
                        field_title = title
            
            if not combined_field_values:
                continue
            
            # Concatenate field values from all ranks
            field_values = np.concatenate(combined_field_values)
            
            # Create combined grid from all ranks (similar to plot_mesh)
            if field_spec not in self._field_grids:
                # First time: create grids for each rank
                rank_grids = []
                for topology, cell_types, geometry, rank_values in zip(
                    all_topologies, all_cell_types, all_geometries, 
                    [rd.get(field_spec, (np.array([]), None))[0] for rd in all_rank_field_data]
                ):
                    if len(rank_values) > 0:
                        rank_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
                        rank_grid.point_data[field_spec] = rank_values
                        rank_grids.append(rank_grid)
                
                # Combine all rank grids into one
                if len(rank_grids) == 1:
                    plot_grid = rank_grids[0]
                else:
                    plot_grid = rank_grids[0].merge(rank_grids[1:])
                
                # Store the grid for later updates
                self._field_grids[field_spec] = plot_grid
                
                # Create a new plotter for this field (only on first call)
                if field_spec not in self._field_plotters:
                    plotter = pv.Plotter(window_size=self.default_plot_size)
                    plotter.add_text(field_title, position='upper_edge', font_size=14, color='black')
                    
                    # Add the mesh with the field as a surface/volume plot
                    cmap = 'viridis'
                    if field_spec.lower().find("phase") != -1:
                        cmap = 'inferno'
                    
                    is_3d = self.mesh.topology.dim == 3
                    actor = plotter.add_mesh(
                        plot_grid,
                        scalars=field_spec,
                        show_edges=False,
                        edge_color='black',
                        line_width=0.5,
                        opacity=1.0 if is_3d else 1.0,
                        cmap=cmap,
                        scalar_bar_args={
                            'title': field_title,
                            'title_font_size': 12,
                            'label_font_size': 10,
                            'n_labels': 5,
                            'italic': False,
                            'fmt': '%.2e',
                            'position_x': 0.15,
                            'position_y': 0.05,
                        }
                    )
                    
                    plotter.show_axes()
                    if is_3d:
                        plotter.view_isometric()
                    else:
                        plotter.view_xy()
                        plotter.enable_2d_style()
                    
                    # Store plotter and actor for later updates
                    self._field_plotters[field_spec] = plotter
                    self._field_actors[field_spec] = actor
                    
                    # Show non-blocking
                    plotter.show(interactive_update=True, auto_close=False, interactive=False)
                    
            else:
                # Subsequent call: update existing grid and plotter
                plot_grid = self._field_grids[field_spec]
                
                # Update field values in the grid
                # Recreate grids for each rank with new values
                rank_grids = []
                for topology, cell_types, geometry, rank_values in zip(
                    all_topologies, all_cell_types, all_geometries,
                    [rd.get(field_spec, (np.array([]), None))[0] for rd in all_rank_field_data]
                ):
                    if len(rank_values) > 0:
                        rank_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
                        rank_grid.point_data[field_spec] = rank_values
                        rank_grids.append(rank_grid)
                
                # Merge and update
                if len(rank_grids) > 0:
                    if len(rank_grids) == 1:
                        updated_grid = rank_grids[0]
                    else:
                        updated_grid = rank_grids[0].merge(rank_grids[1:])
                    
                    plotter = self._field_plotters[field_spec]
                    actor = self._field_actors[field_spec]
                    
                    # Update the existing grid in-place so the actor sees new scalars
                    new_scalars = updated_grid.point_data[field_spec]
                    if field_spec in plot_grid.point_data:
                        plot_grid.point_data[field_spec][:] = new_scalars
                    else:
                        plot_grid.point_data[field_spec] = new_scalars
                    plot_grid.Modified()
                    
                    # Update the scalar range (colorbar limits) based on new data
                    data_range = [new_scalars.min(), new_scalars.max()]
                    mapper = actor.GetMapper()
                    mapper.SetScalarRange(data_range)
                    
                    # Update the plotter; prefer update() for non-blocking refresh
                    try:
                        if hasattr(plotter, "update"):
                            plotter.update()
                        else:
                            plotter.render()
                    except Exception:
                        # If update fails, the window may have been closed
                        mprint(f"Warning: Could not update plot for '{field_spec}'. Window may have been closed.")
                        # Remove from storage to recreate on next call
                        del self._field_plotters[field_spec]
                        del self._field_grids[field_spec]
                        del self._field_actors[field_spec]
        
        #mprint(f"Plotted/Updated {len(plot_fields)} field(s)")

    def Plot_Measures(self):
        """
        Plot scalar measures defined in params.Plot_Measures as line plots.
        Each entry is a tuple (x_name, y_name) using Global_Measures or "t".
        """
        if not hasattr(self.params, "Plot_Measures"):
            return
        plot_measures = self.params.Plot_Measures
        if not plot_measures:
            return
        if rank != 0:
            return

        for x_name, y_name in plot_measures:
            x_val = None
            y_val = None
            if x_name == "t":
                x_val = float(getattr(self.params, "t", 0.0))
            elif hasattr(self.params, "Global_Measures"):
                x_val = self.params.Global_Measures.get(x_name, None)
            if hasattr(self.params, "Global_Measures"):
                y_val = self.params.Global_Measures.get(y_name, None)

            if x_val is None or y_val is None:
                continue

            key = (x_name, y_name)
            plot_state = self._measure_plots.get(key)
            if plot_state is None:
                plotter = pv.Plotter(window_size=self.default_plot_size)
                chart = pv.Chart2D()
                line = chart.line([x_val], [y_val], color="black")
                chart.x_label = x_name
                chart.y_label = y_name
                chart.title = f"{y_name} vs {x_name}"
                plotter.add_chart(chart)
                plotter.show(interactive_update=True, auto_close=False, interactive=False)
                self._measure_plots[key] = {
                    "plotter": plotter,
                    "chart": chart,
                    "line": line,
                    "x": [x_val],
                    "y": [y_val],
                }
                plot_state = self._measure_plots[key]
            else:
                plot_state["x"].append(x_val)
                plot_state["y"].append(y_val)
                line = plot_state["line"]
                if hasattr(line, "update"):
                    line.update(plot_state["x"], plot_state["y"])
                elif hasattr(line, "set_data"):
                    line.set_data(plot_state["x"], plot_state["y"])
                elif hasattr(line, "x") and hasattr(line, "y"):
                    try:
                        line.x = plot_state["x"]
                        line.y = plot_state["y"]
                    except AttributeError:
                        line = plot_state["chart"].line(
                            plot_state["x"], plot_state["y"], color="black"
                        )
                        plot_state["line"] = line
                else:
                    chart = plot_state["chart"]
                    if hasattr(chart, "clear"):
                        chart.clear()
                    line = chart.line(plot_state["x"], plot_state["y"], color="black")
                    plot_state["line"] = line
            plot_state["plotter"].render()


    def _get_dg_space(self, degree):
        """
        Get or create a cached DG function space of the given degree.
        """
        if not hasattr(self, '_dg_spaces'):
            self._dg_spaces = {}
        if degree not in self._dg_spaces:
            from basix.ufl import element as basix_element
            dg_elem = basix_element("DG", self.mesh.basix_cell(), degree)
            self._dg_spaces[degree] = fem.functionspace(self.mesh, dg_elem)
        return self._dg_spaces[degree]

    def _get_field_dg_degree(self, field_name):
        """
        Determine the appropriate DG degree for outputting a field.
        Falls back to 1 for quadrature fields or if order is not found.
        """
        order = self.field_order.get(field_name, 1)
        if order < 1:  # Quadrature fields have order -1
            return 1
        return order

    def Process_Fields_For_Plotting(self, field_spec, V_dg=None):
        """
        Process field specification and interpolate to a DG space matching the field's degree.
        
        Args:
            field_spec: Field specification string (e.g., "p", "v_x", "phase")
            V_dg: Optional DG function space override. If None, automatically
                  determines the degree from the field's function space.
            
        Returns:
            Tuple of (field_values, field_title, V_dg) or None if field cannot be processed.
            V_dg is the DG function space used for interpolation.
        """
        if "_" in field_spec:
            # Vector field component (e.g., "v_x" or "v_y")
            parts = field_spec.rsplit("_", 1)
            field_name = parts[0]
            component = parts[1]
            
            # Map component names to indices
            component_map = {"x": 0, "y": 1, "z": 2}
            if component not in component_map:
                mprint(f"Warning: Unknown component '{component}' in field '{field_spec}'. Skipping.")
                return None
            
            comp_idx = component_map[component]
            
            # Get the vector field for plotting (handles mixed elements)
            if field_name not in self.fields:
                mprint(f"Warning: Field '{field_name}' not found. Skipping '{field_spec}'.")
                return None
            
            field_output = self.Get_Field_For_Output(field_name)
            
            # Check if field is actually a vector field
            if "Vector" not in self.field_type.get(field_name, ""):
                mprint(f"Warning: Field '{field_name}' is not a vector field. Skipping '{field_spec}'.")
                return None
            
            # Determine DG space
            if V_dg is None:
                V_dg = self._get_dg_space(self._get_field_dg_degree(field_name))
            
            # Extract the specified component
            try:
                # Check if field is from mixed space (tuple) or regular field
                if isinstance(field_output, tuple):
                    # Mixed space: (mixed_func, start_idx, comp_count)
                    mixed_func, start_idx, comp_count = field_output
                    if comp_idx >= comp_count:
                        mprint(f"Warning: Component {component} out of range for field '{field_name}'. Skipping.")
                        return None
                    field_component = mixed_func.sub(start_idx + comp_idx).collapse()
                    field_title = f"{self.fields[field_name].name} - {component.upper()} component"
                else:
                    # Regular field
                    field_component = field_output.sub(comp_idx).collapse()
                    field_title = f"{field_output.name} - {component.upper()} component"
            except Exception as e:
                mprint(f"Warning: Could not extract component {component} from field '{field_name}': {e}. Skipping.")
                return None
            
            # Interpolate to DG space
            field_dg = fem.Function(V_dg)
            field_dg.interpolate(field_component)
            field_values = field_dg.x.array.copy()
                
        else:
            # Scalar field (e.g., "p")
            field_name = field_spec
            
            # Get the scalar field for plotting (handles mixed elements)
            if field_name not in self.fields:
                mprint(f"Warning: Field '{field_name}' not found. Skipping.")
                return None
            
            field_output = self.Get_Field_For_Output(field_name)
            
            # Check if field is a scalar field
            if "Vector" in self.field_type.get(field_name, ""):
                mprint(f"Warning: Field '{field_name}' is a vector field but no component specified. Skipping.")
                return None
            
            # Determine DG space
            if V_dg is None:
                V_dg = self._get_dg_space(self._get_field_dg_degree(field_name))
            
            # Handle mixed vs regular fields
            if isinstance(field_output, tuple):
                # Mixed space: (mixed_func, start_idx, comp_count)
                mixed_func, start_idx, comp_count = field_output
                field = mixed_func.sub(start_idx).collapse()
                field_title = self.fields[field_name].name
            else:
                # Regular field
                field = field_output
                field_title = field.name
            
            # Interpolate to DG space
            field_dg = fem.Function(V_dg)
            field_dg.interpolate(field)
            field_values = field_dg.x.array.copy()
        
        return field_values, field_title, V_dg
