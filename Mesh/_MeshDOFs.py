from basix.ufl import element, mixed_element, quadrature_element
from basix import LagrangeVariant
from dolfinx import fem
from ufl import TestFunction, TestFunctions, TrialFunction, TrialFunctions, Argument
from typing import Any, Dict, Optional, List, Tuple

from Utils.mpi_utils import mprint


class _MeshDOFs:
    """
    Mix-in offering helper routines for managing scalar/vector fields on the mesh.
    """
    
    def __init__(self, params: Any) -> None:
        self.fields: Dict[str, fem.Function] = {}
        self.field_order: Dict[str, int] = {}
        self.field_type: Dict[str, str] = {}
        
        self.test_functions = {}
        self.trial_functions = {}
        self.mixed_spaces = {}  # Store mixed spaces for each step
        self.step_fields = {}   # Track which fields belong to each step
        
        super().__init__(params)

    def Print_Fields(self) -> None:
        """
        Print information about all fields stored in the mesh.
        """
        mprint("Mesh fields:")
        for fname, fobj in self.fields.items():
            mprint(f"\t{fname}:  {self.field_type[fname]}{self.field_order[fname]} function ({fobj.name})")
    
    def _get_field_info_in_mixed_space(self, field_name: str, step: int = None) -> Optional[Tuple]:
        """
        Helper method to find field information in mixed spaces.
        
        Args:
            field_name: Name of the field to find
            step: Specific step to check, or None to check all steps
        
        Returns:
            Tuple of (mixed_info, start_idx, comp_count) if found, None otherwise
        """
        # Check if this field is part of a mixed space
        steps_to_check = [step] if step is not None else list(self.mixed_spaces.keys())
        
        for check_step in steps_to_check:
            if check_step not in self.mixed_spaces or self.mixed_spaces[check_step] is None:
                continue
                
            mixed_info = self.mixed_spaces[check_step]
            field_names = mixed_info['field_names']
            
            if field_name in field_names:
                field_component_info = mixed_info['field_component_info']
                
                # Find the index range for this field
                idx = 0
                for fname in field_names:
                    comp_count = sum(1 for f, _, _ in field_component_info if f == fname)
                    
                    if fname == field_name:
                        return (mixed_info, idx, comp_count)
                    
                    idx += comp_count
        
        return None

    def Get_Field(self, field_name: str, step: int = None):
        """
        Retrieve a field by name. If the field is part of a mixed space,
        automatically returns the symbolic split version suitable for UFL forms.
        Otherwise returns the original field function.
        
        Args:
            field_name: Name of the field to retrieve
            step: Solution step (if None, searches all mixed spaces)
        
        Returns:
            Field function or symbolic split field from mixed space
        """
        Field = self.fields.get(field_name, None)
        if Field is None:
            raise ValueError(f"Field '{field_name}' not found in mesh.")
        
        # Check if field is in a mixed space
        if not self.mixed_spaces:
            return Field
        
        field_info = self._get_field_info_in_mixed_space(field_name, step)
        if field_info is None:
            return Field
        
        # Field is in mixed space - extract symbolic version
        from ufl import split, as_vector
        
        mixed_info, start_idx, comp_count = field_info
        mixed_func = mixed_info['mixed_function']
        funcs = split(mixed_func)
        
        if comp_count == 1:
            # Scalar field
            return funcs[start_idx]
        else:
            # Vector field - combine components
            components = [funcs[start_idx + i] for i in range(comp_count)]
            return as_vector(components)

    def Get_Field_For_Output(self, field_name: str):
        """
        Retrieve a field suitable for plotting/output (collapsed from mixed space if needed).
        Unlike Get_Field which returns symbolic UFL expressions for forms, this returns
        actual Function objects that can be interpolated and plotted.
        
        For fields in mixed spaces, returns a tuple (mixed_func, start_idx, comp_count)
        For regular fields, returns just the field function.
        
        Args:
            field_name: Name of the field to retrieve
        
        Returns:
            Function object or tuple (mixed_func, start_idx, comp_count) for mixed fields
        """
        Field = self.fields.get(field_name, None)
        if Field is None:
            raise ValueError(f"Field '{field_name}' not found in mesh.")
        
        # Check if field is in a mixed space
        field_info = self._get_field_info_in_mixed_space(field_name)
        if field_info is None:
            return Field
        
        # Field is in mixed space - return tuple
        mixed_info, start_idx, comp_count = field_info
        return (mixed_info['mixed_function'], start_idx, comp_count)

    def Zero_Field(self, field_name: str) -> None:
        """
        Set the specified field to zero, regardless of whether it belongs to a mixed space.
        MPI-aware: Updates ghost values after zeroing.
        """
        field_output = self.Get_Field_For_Output(field_name)
        if isinstance(field_output, tuple):
            mixed_func, start_idx, comp_count = field_output
            parent_space = mixed_func.function_space
            parent_array = mixed_func.x.array
            for offset in range(comp_count):
                # Collapse sub-space to obtain mapping of its dofs into parent vector
                _, dofs = parent_space.sub(start_idx + offset).collapse()
                parent_array[dofs] = 0.0
            mixed_func.x.scatter_forward()
        else:
            field_output.x.array[:] = 0.0
            field_output.x.scatter_forward()
            
    def Constant_Field(self, field_name: str, value: float) -> None:
        """
        Set the specified field to a constant value, regardless of whether it belongs to a mixed space.
        MPI-aware: Updates ghost values after setting.
        """
        field_output = self.Get_Field_For_Output(field_name)
        if isinstance(field_output, tuple):
            mixed_func, start_idx, comp_count = field_output
            parent_space = mixed_func.function_space
            parent_array = mixed_func.x.array
            for offset in range(comp_count):
                # Collapse sub-space to obtain mapping of its dofs into parent vector
                _, dofs = parent_space.sub(start_idx + offset).collapse()
                parent_array[dofs] = value
            mixed_func.x.scatter_forward()
        else:
            field_output.x.array[:] = value
            field_output.x.scatter_forward()

    def Get_Field_Space_For_BC(self, field_name: str, component: int = None):
        """
        Get function space information for creating boundary conditions.
        
        For fields in mixed spaces, returns (mixed_space, sub_index)
        For regular fields, returns (field_space, component or None)
        
        Args:
            field_name: Name of the field
            component: Component index for vector fields (None for scalar or full vector)
        
        Returns:
            Tuple of (function_space, index) where index is the sub-space index or component
        """
        Field = self.fields.get(field_name, None)
        if Field is None:
            raise ValueError(f"Field '{field_name}' not found in mesh.")
        
        # Check if field is in a mixed space
        field_info = self._get_field_info_in_mixed_space(field_name)
        if field_info is None:
            return (Field.function_space, component)
        
        # Field is in mixed space
        mixed_info, start_idx, comp_count = field_info
        mixed_space = mixed_info['space']
        
        if component is not None:
            return (mixed_space, start_idx + component)
        else:
            return (mixed_space, start_idx)

    def Copy_Field(self, source_field_name: str, dest_field_name: str) -> None:
        """
        Copy data from one field to another, handling both mixed space and individual fields.
        MPI-aware: Updates ghost values after copying.
        
        Args:
            source_field_name: Name of the source field to copy from
            dest_field_name: Name of the destination field to copy to
        """
        # Get source field (handles mixed spaces)
        source_result = self.Get_Field_For_Output(source_field_name)
        
        # Get destination field (should not be in mixed space)
        dest_field = self.fields.get(dest_field_name, None)
        if dest_field is None:
            raise ValueError(f"Destination field '{dest_field_name}' not found in mesh.")
        
        # Check if source is in mixed space (returns tuple) or individual (returns Function)
        if isinstance(source_result, tuple):
            # Source field is in mixed space: (mixed_func, start_idx, comp_count)
            mixed_func, start_idx, comp_count = source_result
            
            # Check if destination is a vector or scalar field
            if comp_count > 1:
                # Vector field - interpolate each component
                for i in range(comp_count):
                    collapsed_component = mixed_func.sub(start_idx + i).collapse()
                    dest_field.sub(i).interpolate(collapsed_component)
            else:
                # Scalar field - interpolate directly
                collapsed_field = mixed_func.sub(start_idx).collapse()
                dest_field.interpolate(collapsed_field)
        else:
            # Source field is individual - use direct interpolation
            source_element = source_result.function_space.element
            dest_element = dest_field.function_space.element
            def _is_quadrature(element) -> bool:
                family = getattr(element, "family", None)
                if isinstance(family, str):
                    return family == "Quadrature"
                basix = None
                try:
                    basix = element.basix_element
                except RuntimeError:
                    basix = None
                if basix is not None:
                    basix_family = getattr(basix, "family", None)
                    family_name = getattr(basix_family, "name", None)
                    if isinstance(family_name, str):
                        return family_name.lower() == "quadrature"
                    if isinstance(basix_family, str):
                        return basix_family == "Quadrature"
                return False
            source_is_quad = _is_quadrature(source_element) or "Quadrature" in self.field_type.get(source_field_name, "")
            dest_is_quad = _is_quadrature(dest_element) or "Quadrature" in self.field_type.get(dest_field_name, "")
            if source_is_quad or dest_is_quad:
                if source_result.function_space == dest_field.function_space:
                    dest_field.x.array[:] = source_result.x.array
                else:
                    interp_points = dest_field.function_space.element.interpolation_points
                    expr = fem.Expression(source_result, interp_points)
                    dest_field.interpolate(expr)
            else:
                dest_field.interpolate(source_result)
        
        # Update ghost values
        dest_field.x.scatter_forward()

    def Set_From_Expression(self, field_name: str, expression) -> None:
        """
        Evaluate a UFL expression and set the field to the result.
        MPI-aware: Updates ghost values after interpolation.
        
        Args:
            field_name: Name of the field to set
            expression: UFL expression to evaluate
        
        Raises:
            ValueError: If field_name is not found
        """
        field = self.fields.get(field_name, None)
        if field is None:
            raise ValueError(f"Field '{field_name}' not found in mesh.")
        
        # Create a FEniCSx Expression and interpolate it to the field
        expr = fem.Expression(expression, field.function_space.element.interpolation_points)
        field.interpolate(expr)
        field.x.scatter_forward()

    def Get_TestFunction(self, field_name: str, step: int) -> Optional[fem.Function]:
        """
        Retrieve a test function for the specified field and step.
        """
        test_key = f"{field_name}_test_step{step}"
        test_func = self.test_functions.get(test_key, None)
        if test_func is None:
            raise ValueError(f"Test function for field '{field_name}' at step {step} not found.")
        
        return test_func

    def Get_Trial_Function(self, field_name: str, step: int):
        """
        Retrieve a trial function for the specified field and step.
        """
        trial_key = f"{field_name}_trial_step{step}"
        if trial_key in self.trial_functions:
            return self.trial_functions[trial_key]

        # Check if this field is part of a mixed space
        field_info = self._get_field_info_in_mixed_space(field_name, step)
        if field_info is not None:
            mixed_info, start_idx, comp_count = field_info
            trial_funcs = mixed_info['trial_functions']

            if comp_count == 1:
                trial_func = trial_funcs[start_idx]
            else:
                from ufl import as_vector
                components = [trial_funcs[start_idx + i] for i in range(comp_count)]
                trial_func = as_vector(components)

            self.trial_functions[trial_key] = trial_func
            return trial_func

        # Fall back to standalone field
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' not found. Define the field before creating a trial function.")

        field = self.fields[field_name]
        trial_func = TrialFunction(field.function_space)
        self.trial_functions[trial_key] = trial_func
        return trial_func

    def _create_element(self, space: str, order: int, shape: Optional[Tuple] = None):
        """
        Helper method to create a finite element.
        
        Args:
            space: "CG" or "DG"
            order: Polynomial degree
            shape: Optional shape for vector elements (e.g., (2,) for 2D vectors)
        
        Returns:
            Basix element
        """
        if space not in ["CG", "DG"]:
            raise ValueError(f"Unsupported function space: {space}")
        
        # Map deprecated names to new names
        element_name = "Lagrange" if space == "CG" else "DG"
        
        if (shape is None) and (element_name == "Lagrange"):
            return element(element_name, self.mesh.basix_cell(), degree=order, lagrange_variant=LagrangeVariant.bernstein)
        elif (element_name == "DG"):
            return element(element_name, self.mesh.basix_cell(), degree=order, shape=shape)
        else:
            return element(element_name, self.mesh.basix_cell(), degree=order, shape=shape, lagrange_variant=LagrangeVariant.bernstein)

    def Add_Field(self, field_name: str, field_description: str, space: str, order: int) -> fem.Function:
        """
        Add a scalar field to the mesh.
        """
        if field_name in self.fields:
            return self.fields[field_name]

        elem = self._create_element(space, order)
        V = fem.functionspace(self.mesh, elem)
        f = fem.Function(V, name=field_description)

        self.fields[field_name] = f
        self.field_order[field_name] = order
        self.field_type[field_name] = space
        return self.fields[field_name]

    def Add_Vector_Field(self, field_name: str, field_description: str, space: str, order: int, nDim: int) -> fem.Function:
        """
        Add a vector field to the mesh.
        """
        if field_name in self.fields:
            return self.fields[field_name]

        elem = self._create_element(space, order, shape=(nDim,))
        V = fem.functionspace(self.mesh, elem)
        f = fem.Function(V, name=field_description)

        self.fields[field_name] = f
        self.field_order[field_name] = order
        self.field_type[field_name] = f"{space} Vector"
        return self.fields[field_name]

    def Add_Quadrature_Field(self, field_name: str, field_description: str, size: Optional[Tuple[int, ...]] = None):
        
        if field_name in self.fields:
            return self.fields[field_name]
        
        quad_degree = getattr(self.params, "Quadrature_Degree", 2)
        value_shape = () if size is None else size
        q_el_tensor = quadrature_element(
            self.mesh.basix_cell(),
            degree=quad_degree,
            value_shape=value_shape,
        )

        self._Q_tensor = fem.functionspace(self.mesh, q_el_tensor)
        self.fields[field_name] = fem.Function(self._Q_tensor, name=field_description)
        self.field_order[field_name] = -1  # Indicate quadrature field
        self.field_type[field_name] = "Quadrature Tensor"
        return self.fields[field_name]

    def _add_test_function_internal(self, field_name: str, step: int, is_vector: bool) -> fem.Function:
        """
        Internal helper to add test functions for scalar or vector fields.
        
        Args:
            field_name: Name of the field
            step: Solution step identifier
            is_vector: True if this is a vector field
        
        Returns:
            Test function from the field's function space
        """
        # Check if test function already exists
        test_key = f"{field_name}_test_step{step}"
        if test_key in self.test_functions:
            return self.test_functions[test_key]
        
        # Check if field exists
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' not found. Define the field before creating a test function.")
        
        # Verify field type matches expectation
        is_field_vector = "Vector" in self.field_type.get(field_name, "")
        if is_vector != is_field_vector:
            expected_type = "vector" if is_vector else "scalar"
            actual_type = "vector" if is_field_vector else "scalar"
            method_name = "Add_Vector_TestFunction" if is_vector else "Add_TestFunction"
            other_method = "Add_TestFunction" if is_vector else "Add_Vector_TestFunction"
            raise ValueError(f"Field '{field_name}' is a {actual_type} field. Use {other_method} instead.")
        
        # Create test function from the field's function space
        field = self.fields[field_name]
        test_func = TestFunction(field.function_space)
        
        # Store test function
        self.test_functions[test_key] = test_func
        
        # Track field for this step
        if step not in self.step_fields:
            self.step_fields[step] = []
        if field_name not in self.step_fields[step]:
            self.step_fields[step].append(field_name)
        
        return test_func

    def Add_TestFunction(self, field_name: str, step: int) -> fem.Function:
        """
        Create a test function for the specified scalar field.
        
        Args:
            field_name: Name of the field to create a test function for
            step: Solution step identifier
            
        Returns:
            Test function from the field's function space
            
        Raises:
            ValueError: If field_name is not defined or is a vector field
        """
        return self._add_test_function_internal(field_name, step, is_vector=False)
    
    def Add_Vector_TestFunction(self, field_name: str, nDim: int, step: int) -> fem.Function:
        """
        Create a test function for the specified vector field.
        
        Args:
            field_name: Name of the vector field to create a test function for
            nDim: Number of dimensions (should match the field's dimension)
            step: Solution step identifier
            
        Returns:
            Test function from the field's function space
            
        Raises:
            ValueError: If field_name is not defined or is not a vector field
        """
        # Optionally verify dimension matches
        if field_name in self.fields:
            field = self.fields[field_name]
            element = getattr(getattr(field, "function_space", None), "element", None)
            value_shape = getattr(element, "value_shape", ())
            if len(value_shape) > 0 and value_shape[0] != nDim:
                mprint(f"Warning: Requested dimension {nDim} does not match field dimension {value_shape[0]}")

        return self._add_test_function_internal(field_name, step, is_vector=True)

    def _build_mixed_space_for_step(self, step: int) -> None:
        """
        Build a mixed function space for all fields registered for this step.
        This allows monolithic solving of multiple fields.
        
        Args:
            step: Solution step number
        """
        if step in self.mixed_spaces:
            return  # Already built
        
        # Get all fields for this step
        if step not in self.step_fields or len(self.step_fields[step]) == 0:
            return
        
        field_names = sorted(self.step_fields[step])  # Sort for consistency
        
        # If only one field, no mixed space needed
        if len(field_names) == 1:
            self.mixed_spaces[step] = None
            return
        
        mprint(f"\tBuilding mixed function space for step {step} with fields: {field_names}")
        
        # Collect elements for each field
        # For vector fields, we need to add one scalar element per component
        # to avoid "blocked element" issues in mixed formulations
        elements = []
        field_component_info = []  # Track which elements belong to which field
        
        for fname in field_names:
            field = self.fields[fname]
            ftype = self.field_type[fname]
            order = self.field_order[fname]
            
            # Determine base space type (CG or DG)
            space_type = ftype.replace(" Vector", "")
            
            # For vector fields, add one element per component
            if "Vector" in ftype:
                # Get dimension from the field
                shape = field.function_space.element.value_shape
                ndim = shape[0] if len(shape) > 0 else 1
                
                # Add one scalar element per component
                for comp in range(ndim):
                    elem = self._create_element(space_type, order)
                    elements.append(elem)
                    field_component_info.append((fname, comp, ndim))
            else:
                # Scalar field - add single element
                elem = self._create_element(space_type, order)
                elements.append(elem)
                field_component_info.append((fname, None, 1))
        
        # Create mixed element and function space
        mixed_el = mixed_element(elements)
        W = fem.functionspace(self.mesh, mixed_el)
        
        # Create a mixed function for the solution
        mixed_func = fem.Function(W)
        
        # Initialize to zero (individual fields will be used for initial values)
        mixed_func.x.array[:] = 0.0
        
        # Create test functions from mixed space
        test_funcs = TestFunctions(W)
        
        # Create trial functions from mixed space (for derivatives)
        trial_funcs = TrialFunctions(W)
        mixed_trial_function = TrialFunction(W)
        
        # Map test functions back to fields
        # For vector fields, we need to combine components
        from ufl import as_vector
        
        idx = 0
        for fname in field_names:
            test_key = f"{fname}_test_step{step}"
            
            # Find how many components this field has
            comp_count = sum(1 for f, _, _ in field_component_info if f == fname)
            
            if comp_count == 1:
                # Scalar field
                self.test_functions[test_key] = test_funcs[idx]
                idx += 1
            else:
                # Vector field - combine components using ufl.as_vector
                components = [test_funcs[idx + i] for i in range(comp_count)]
                self.test_functions[test_key] = as_vector(components)
                idx += comp_count
        
        # Store the mixed space info
        self.mixed_spaces[step] = {
            'space': W,
            'field_names': field_names,
            'field_component_info': field_component_info,
            'test_functions': test_funcs,
            'trial_functions': trial_funcs,
            'mixed_function': mixed_func,
            'mixed_trial_function': mixed_trial_function
        }
    
    def finalize_test_functions(self) -> None:
        """
        Finalize test function creation by building mixed spaces where needed.
        This should be called after all models have registered their test functions.
        """
        for step in self.step_fields.keys():
            self._build_mixed_space_for_step(step)
    
    def print_dofs_info(self) -> None:
        """
        Print the number of degrees of freedom being solved for in each step.
        MPI-aware: Shows local DOFs (rank 0) and total DOFs across all ranks.
        """
        from Utils.mpi_utils import comm, rank
        
        mprint("Degrees of freedom per solution step:")
        
        if not self.step_fields:
            mprint("\tNo solution steps defined.")
            return
        
        for step in sorted(self.step_fields.keys()):
            field_names = self.step_fields[step]
            
            if not field_names:
                mprint(f"\tStep {step}: No fields")
                continue
            
            # Count local DOFs for this step
            local_total_dofs = 0
            field_dof_info = []
            
            for field_name in field_names:
                if field_name not in self.fields:
                    continue
                
                field = self.fields[field_name]
                # Get local DOF count (owned by this rank)
                local_base_dofs = field.function_space.dofmap.index_map.size_local
                
                # For vector fields, multiply by number of components
                value_shape = field.function_space.element.value_shape
                num_components = 1 if len(value_shape) == 0 else value_shape[0]
                local_field_dofs = local_base_dofs * num_components
                
                # Get global DOF count (sum across all ranks)
                global_base_dofs = field.function_space.dofmap.index_map.size_global
                global_field_dofs = global_base_dofs * num_components
                
                local_total_dofs += local_field_dofs
                
                if num_components > 1:
                    field_dof_info.append({
                        'name': field_name,
                        'local': local_field_dofs,
                        'global': global_field_dofs,
                        'desc': f"{num_components} components × {local_base_dofs} local nodes ({global_base_dofs} global)"
                    })
                else:
                    field_dof_info.append({
                        'name': field_name,
                        'local': local_field_dofs,
                        'global': global_field_dofs,
                        'desc': None
                    })
            
            # Sum up total DOFs across all ranks
            global_total_dofs = comm.allreduce(local_total_dofs)
            
            # Print step information
            mprint(f"\tStep {step}: {local_total_dofs} DOFs (local rank {rank}), {global_total_dofs} DOFs (total)")
            for field_info in field_dof_info:
                if field_info['desc']:
                    mprint(f"\t\t{field_info['name']}: {field_info['local']} local, {field_info['global']} global ({field_info['desc']})")
                else:
                    mprint(f"\t\t{field_info['name']}: {field_info['local']} local, {field_info['global']} global")
