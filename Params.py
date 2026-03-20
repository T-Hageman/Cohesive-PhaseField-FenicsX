from Models.ModelEnums import ModelType
from Solvers.SolverENUMS import LinearSolver

class Params:
    """
    Default parameters for example.
    """

    DEBUG_PRINT_TIMING: bool = False
    Global_Measures: dict = {}
    
    dim = 2
    
    ## Mesh parameters
    MeshType: str = "FullPlateWithHole"
    dx: float = 0.01          # Mesh resolution in meters
    Lx:float = 1.0
    Ly:float = 1.0
    Lz:float = dx
    R:float = 0.1
    ellipticity: float = 1.0
    rotation_angle: float = 0.0
    Quadrature_Degree: int = 3 
    
    ## physics parameters
    Materials: dict = {}
    Materials["Iron"] = {
        "Density": 8.0e3,        # density of iron (kg/m^3)
        "Youngs": 200e9,        # Young's modulus (Pa)
        "Poisson": 0.3,        # Poisson's ratio
        "LE_Damping": 1.0e6,
        "ell": 0.05,              # Length scale parameter (m)
        "f_t": 150e6,           # Tensile strength (Pa)
        "f_s": 150e6,           # Shear strength (Pa)
        "Gc": 1e5,
        "DP_eRef": 1e-5,
        "pf_damping": 0.0e-6,
        "PF_formulation": "AT2",
        "Inertia": True,
        "kappa": 1.0e-3       
    }
    
    Materials["Iron"]["Shear"] = Materials["Iron"]["Youngs"] / (2.0 * (1.0 + Materials["Iron"]["Poisson"]))
    Materials["Iron"]["Bulk"] = Materials["Iron"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Iron"]["Poisson"]))
    Materials["Iron"]["Lame"] = Materials["Iron"]["Shear"] * (2.0 * Materials["Iron"]["Poisson"]) / (1.0 - 2.0 * Materials["Iron"]["Poisson"])
    gravity: float = 0.0     # gravitational acceleration (m/s^2)
    
    Plot_Fields: list = ["u_x", "u_y", "phasefield"]
    Plot_Measures: list = [("t", "Crack_Length"), ("BC_Top_u", "BC_Top_F")]

    Solution_Steps = {"u": 0, "phasefield": 1}
    Field_Orders = {"u": 2, "phasefield": 2}
    
    Models = {}
    ModelNames = ["Solid", "pf", "BC_Bottom", "BC_Bottom2", "BC_Top"]

        
    if dim==3:
        ModelNames.append("BC_Back")
    Models["Solid"] = {
        "Type": ModelType.SOLID_COHESIVE_LINEARELASTIC,
        "Material": "Iron",
        "DamageModel": "PhaseField",
        "FailureType": "r1"
    }
    Models["pf"] = {
        "Type": ModelType.SOLID_PHASE_FIELD,
        "Material": "Iron",
        "CDF_field": "PsiField",
        "Irreversible_Method": "Hist"
    }
    Models["BC_Bottom"] = {
        "Type": ModelType.BOUNDARY_PRESCRIBED,
        "boundary": "bottom",
        "field": "u_y",
        "value": 0.0
    }
    Models["BC_Bottom2"] = {
        "Type": ModelType.BOUNDARY_PRESCRIBED,
        "boundary": "bottom",
        "field": "u_x",
        "value": 0.0
    }
    Models["BC_Top"] = {
        "Type": ModelType.BOUNDARY_DISPCONTROL,
        "boundary": "top",
        "field": "u_y",
        "rate": -3.0e-6,
        "Dummy": 1.0e16,
        "F_Cutoff": 0.05
    }
    Models["BC_Back"] = {
        "Type": ModelType.BOUNDARY_PRESCRIBED,
        "boundary": "back",
        "field": "u_z",
        "value": 0.0
    }
    
    ## Solver Parameters
    Linear_Solver: LinearSolver = LinearSolver.LU
    n_passes: int = 5
    max_it: int = 30
    line_search: bool = True
    line_search_lims: list = [0.1, 1.0]
    rel_tol: float = 1e-3
    abs_tol: float = 1e-6
    
    # Output parameters
    output_dir: str = "Results"   # directory to save output files
    output_interval: int = 1      # interval (in time steps) to write output files
    write_vtk: bool = False        # whether to write VTK files for visualization
    write_hdf5: bool = True      # whether to write HDF5 files for data storage
    save_Mesh_every_step: bool = False  # whether to save mesh at every time step or only once
    output_fields: list = ["u_x", "u_y", "phasefield"]  # list of specific fields to output
    
    ## Time parameters
    Newmark_Beta: float = 0.5625  # Newmark-beta parameter
    Newmark_Gamma: float = 1.0    # Newmark-gamma parameter
    
    start_time: float = 0.0      # start time (s)
    end_time: float = 200.0*670.0 # end time (s)
    dt: float = 10.0           # time step (s)
    TimeStepControl = {
        "Type": "Measure",
        "time_list": [0.0],
        "dt_list": [dt],
        "dt_min": 5.0e-5,
        "dt_max": dt,
        "Measure": "Crack_Length_Change",
        "Target_Change": Materials["Iron"]["ell"] * 0.05
    }
