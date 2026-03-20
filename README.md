

# Cohesive Phase-Field Fracture in FEniCSx
![MIT License](https://img.shields.io/badge/license-MIT-green.svg)
<!-- If you have a DOI, add it here. Example: ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg) -->

## Why use this code?

- Implements a robust, mesh-independent cohesive phase-field fracture model
- Explicit, tunable strength surface and return-mapping formulation
- Open-source, reproducible, and well-documented
- Validated on standard benchmarks (tension, shear, dynamic fracture)
- Ready for extension and integration with FEniCSx workflows
## Installation

This code requires Python 3.10+ and an MPI-enabled environment. The main dependencies are:

- dolfinx
- petsc4py
- mpi4py
- gmsh
- numba

**Recommended (using conda):**

```bash
conda create -n fenicsx-cohesive python=3.10
conda activate fenicsx-cohesive
conda install -c conda-forge fenics-dolfinx petsc4py mpi4py gmsh numba
```

Or, using pip (ensure you have a working MPI installation):

```bash
python -m pip install --upgrade pip
python -m pip install dolfinx petsc4py mpi4py gmsh numba
```

For more details, see the [FEniCSx installation guide](https://docs.fenicsproject.org/dolfinx/main/python/installation.html).
## Quick Start

Run a default simulation (using 50 MPI processes):

```bash
export OMP_NUM_THREADS=1
mpirun -np 50 python3 Main.py
```

This will generate output files in the `Results/` directory (e.g., `Outputs_*.hdf5`).

To run all benchmark cases (warning: this is computationally intensive):

```bash
export OMP_NUM_THREADS=1
python3 Do_Sweep.py
```

You can also run individual cases by editing `Params.py` or passing arguments to `Do_Sweep.py`.
## Results Visualization

The code outputs results in HDF5 format in the `Results/` directory. To visualize fields (e.g., displacement, phase-field):

- Use the provided MATLAB scripts (e.g., `PlotFailSurfaces.m`, `PostProcessAnimations.m`)
- Or, use Python with `h5py` and `matplotlib` to read and plot the data

**Example (Python):**

```python
import h5py
import matplotlib.pyplot as plt
with h5py.File('Results/Outputs_1.hdf5', 'r') as f:
    u_x = f['u_x'][:]
    plt.plot(u_x)
    plt.title('Displacement field')
    plt.show()
```

See the MATLAB scripts for more advanced visualizations and animations.


---

This repository contains the source code used to generate the results for the paper:

> Tim Hageman, *Cohesive phase-field fracture with an explicit strength surface: an eigenstrain-based return-mapping formulation*.

It is provided as a **historic research snapshot** accompanying that manuscript. In other words, this repository is intended as a **reproducible archival copy of the code used in the paper**, not as an actively maintained or continuously developed software package.

The code remains openly available under the MIT license so that others can inspect, reproduce, adapt, and cite the implementation.

## What This Repository Contains

The implementation combines:

- a phase-field fracture formulation in `dolfinx` / FEniCSx,
- an eigenstrain-based cohesive constitutive update resolved locally at quadrature points,
- return-mapping style updates for the fracture eigenstrains, and
- benchmark drivers corresponding to the study discussed in the manuscript.

The benchmark set includes:

- a plate with a hole under tension and compression,
- a single-edge notched plate under shear, and
- a dynamically loaded notched plate.

## Main Entry Points

- `Main.py`: runs a single simulation using the default parameters in `Params.py`.
- `Params.py`: default problem definition, material properties, solver settings, and output controls.
- `Do_Sweep.py`: reproduces all the benchmark cases, for all the parameters used within the paper. This is provided for reference, do not try to run it yourself, it takes a long while to run.
- `Mesh/`: mesh generation, groups, plotting, outputs, and finite-element spaces.
- `Models/`: constitutive and boundary-condition models, including the cohesive linear elastic and phase-field models.
- `Physics/`: assembly and coupled problem definitions.
- `Solvers/`: staggered time-stepping and linear solver handling.
- `Utils/`: MPI and mathematical helper utilities.

## Representative Results

Below are representative results from the benchmark cases discussed in the paper.

### Plate with hole under compression

A unit square with a central hole (diameter 0.2 m) is loaded in compression. Due to the stress concentrations at the side of the hole, cracks nucleate and propagate along the direction of maximum shear stress at an approximate 45° angle. The fracture criterion correctly enforces a no-penetration condition, with the upper half of the domain slipping along the crack surface.

| Vertical displacement ($u_y$, deformations ×10) | Phase-field ($\phi$) |
|:---:|:---:|
| ![Compressive plate with hole – vertical displacement](DOWN_Gc100000_u_y.jpg) | ![Compressive plate with hole – phase field](DOWN_Gc100000_phasefield.jpg) |

*Results obtained using $G_\text{c} = 100 \; \text{kJ/m}^2$, $\ell = 0.05 \; \text{m}$, and the default material parameters.*

### Dynamic crack branching

A notched plate (1 m × 0.5 m, initial notch length 0.25 m) is subjected to a suddenly applied traction of $\sigma_\text{ext} = 10 \; \text{MPa}$ on the top and bottom edges. At low fracture energy ($G_\text{c} = 1 \; \text{kJ/m}^2$) the crack develops short branches as it propagates, eventually splitting into two distinct cracks—a hallmark of dynamic brittle fracture. Higher values of $G_\text{c}$ progressively suppress branching and slow crack propagation.

<p align="center">
  <img src="Dynamic_Gc1000_phasefield.jpg" alt="Dynamic crack branching – phase field for Gc = 1 kJ/m²" width="70%"/>
</p>

*Phase-field variable at $t = 8 \; \text{ms}$ for $G_\text{c} = 1 \; \text{kJ/m}^2$, $\ell = 0.01 \; \text{m}$, using the Drucker–Prager-like strength criterion.*


## Citation

If this repository contributes to your work, please cite the accompanying paper. Since the manuscript is currently under peer review, a conservative citation is:

```text
Hageman, T. Cohesive phase-field fracture with an explicit strength surface:
an eigenstrain-based return-mapping formulation. Submitted manuscript.
```

If a final bibliographic record, DOI, or archival release is added later, please use that version instead.
