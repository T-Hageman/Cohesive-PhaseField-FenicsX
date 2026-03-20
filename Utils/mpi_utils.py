"""
MPI utilities for parallel computing with FEniCSx
"""

from mpi4py import MPI
from typing import Any
import os
from pathlib import Path

# Global MPI communicator
comm = MPI.COMM_WORLD

# Get MPI rank and size
rank: int = comm.Get_rank()
size: int = comm.Get_size()

# Check if running in parallel
is_parallel: bool = size > 1


def isolate_jit_cache(timeout: int = 120, per_process: bool = True) -> None:
    """Configure the DOLFINx JIT cache so concurrent runs do not collide.

    Call this **once** near the top of every entry-point script (Main.py,
    sweep scripts, test files, …) *before* the first ``fem.form()`` call.

    Parameters
    ----------
    timeout : int
        Lock-wait timeout in seconds for JIT compilation (default 120 s,
        the DOLFINx default is only 10 s).
    per_process : bool
        If *True* (default), each **OS-level process group** gets its own
        sub-directory under ``~/.cache/fenics/`` so that independent
        ``mpirun`` invocations never contend on the same lock file.
        MPI ranks within the *same* invocation still share a cache
        (rank 0's PID is broadcast).
        If *False*, only the timeout is increased — all runs share the
        default cache and rely on locking.
    """
    import dolfinx.jit as _jit

    # --- timeout (always applied) ---
    _jit.DOLFINX_DEFAULT_JIT_OPTIONS["timeout"] = (timeout, _jit.DOLFINX_DEFAULT_JIT_OPTIONS["timeout"][1])

    # --- per-process cache directory ---
    if per_process:
        # Use rank-0 PID so all ranks in the *same* mpirun share a cache
        pid = os.getpid() if rank == 0 else 0
        pid = comm.bcast(pid, root=0)

        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = base / "fenics" / f"run_{pid}"
        _jit.DOLFINX_DEFAULT_JIT_OPTIONS["cache_dir"] = (cache_dir, _jit.DOLFINX_DEFAULT_JIT_OPTIONS["cache_dir"][1])

def mprint(message: Any, root: int = 0) -> None:
    """
    Print message only from root process in parallel execution.
    
    Args:
        message: Message to print (any type that can be converted to string)
        root: MPI rank of the process that should print (default: 0)
    
    Returns:
        None
    """
    if rank == root:
        print(message, flush=True)
    else:
        #print("Message from rank", rank, "not printed.", flush=True)
        pass
