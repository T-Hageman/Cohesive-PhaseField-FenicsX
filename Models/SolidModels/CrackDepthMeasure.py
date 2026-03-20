import numpy as np
from ufl import div, inner, dot, dx, TrialFunction, max_value
from dolfinx import fem, mesh as dmesh

from Models.BaseModel import BaseModel
from Utils.maths_utils import nabla_s, macPlus, macMinus, dmacMinus
from Mesh.Mesh import Mesh
from Params import Params
from Models.ModelEnums import ModelType
from ufl import derivative, grad, conditional, lt
from mpi4py import MPI
from Utils.mpi_utils import mprint, comm, rank

class CrackDepthMeasure(BaseModel):
    
    def __init__(self, name, params: Params, mesh: Mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.SOLID_CRACK_DEPTH_MEASURE
        
        self.My_Step = params.Solution_Steps["phasefield"]
        
        self.params.Global_Measures[f"Crack_Depth"] = 0.0
    
    def assemble_KF(self, step: int) -> None:

        return
    
    def Evaluate_Crack_Depth(self, phasefield, Oldphasefield) -> float:
        Depth = np.inf
        
        # Get minimum y coordinate where phasefield is above 0.5
        field_output = phasefield
        if isinstance(field_output, tuple):
            mixed_func, start_idx, _ = field_output
            field_func = mixed_func.sub(start_idx).collapse()
        else:
            field_func = field_output
        field_func.x.scatter_forward()

        tdim = self.mesh.mesh.topology.dim
        self.mesh.mesh.topology.create_connectivity(0, tdim)
        v_to_c = self.mesh.mesh.topology.connectivity(0, tdim)
        num_vertices = self.mesh.mesh.topology.index_map(0).size_local
        if num_vertices > 0:
            vertices = np.arange(num_vertices, dtype=np.int32)
            points = dmesh.compute_midpoints(self.mesh.mesh, 0, vertices)
            cells = np.full(num_vertices, -1, dtype=np.int32)
            for i, v in enumerate(vertices):
                cell_links = v_to_c.links(v)
                if cell_links.size > 0:
                    cells[i] = cell_links[0]
            valid = cells >= 0
            if np.any(valid):
                pf_vals = field_func.eval(points[valid], cells[valid])
                if pf_vals.ndim > 1:
                    pf_vals = pf_vals[:, 0]
                mask = pf_vals > 0.5
                if np.any(mask):
                    Depth = float(np.min(points[valid][mask, 1]))
        
        # sync across processes
        min_Depth = comm.allreduce(Depth, op=MPI.MIN)
        max_y_local = float(np.max(self.mesh.mesh.geometry.x[:, 1]))
        max_y = comm.allreduce(max_y_local, op=MPI.MAX)
        
        crack_depth = 0.0 if not np.isfinite(min_Depth) else max(0.0, max_y - min_Depth)
        
        self.params.Global_Measures[f"Crack_Depth"] = crack_depth
        return crack_depth
    
    def Update_Global_Measures(self) -> None:
        phi = self.mesh.Get_Field_For_Output("phasefield")
        self.Evaluate_Crack_Depth(phi, None)
        return
