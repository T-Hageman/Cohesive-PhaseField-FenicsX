
# Run as:
# export OMP_NUM_THREADS=1
# mpirun -np 50 python3 ./Main.py


from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

from Params import Params
from Mesh.Mesh import Mesh
from Physics.Physics import Physics
from Solvers.TimeSolver import TimeSolver
from Utils.mpi_utils import rank, size, comm, mprint, isolate_jit_cache

# Give each mpirun invocation its own JIT cache so parallel sweeps
# do not fight over lock files.
isolate_jit_cache()

import time
t_start = time.time()

# Get Default Parameters
params = Params()

# Larger elements for quick example run
params.Materials["Iron"]["ell"] = 2.0*params.Materials["Iron"]["ell"]
params.dx = 2.0*params.dx 

# Generate Mesh
mesh = Mesh(params)
mesh.plot_mesh()

# Initialize Physics Models
physics = Physics(mesh, params)
physics.Initialize_Fields()

# Initialize solver
solver = TimeSolver(params, physics)

solver.run()

t_end = time.time()
mprint(f"Total simulation time: {t_end - t_start:.2f} seconds")

if rank == 0:
    input("Press Enter to exit...")
else:
    print("Process", rank, "complete.")
comm.Barrier()
print("Complete.")
