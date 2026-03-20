# export OMP_NUM_THREADS=1

import argparse
import subprocess
import time

from Models.ModelEnums import ModelType
from Solvers.SolverENUMS import LinearSolver

from Mesh.Mesh import Mesh
from Physics.Physics import Physics
from Solvers.TimeSolver import TimeSolver
from Utils.mpi_utils import mprint, isolate_jit_cache

def Params_Cases(testCase = "FullPlate", include_plots: bool = False):
    class Params:
        DEBUG_PRINT_TIMING: bool = False
        Global_Measures: dict = {}
        
        dim = 2
        
        ## Mesh parameters
        if testCase == "FullPlate":
            MeshType: str = "FullPlateWithHole"
            dx: float = 0.01          # Mesh resolution in meters
            Lx:float = 1.0
            Ly:float = 1.0
            Lz:float = dx
            R:float = 0.1
            ellipticity: float = 1.0
            rotation_angle: float = 0.0
        elif testCase == "SingleEdgeNotched":
            MeshType: str = "SingleEdgeNotched"
            dx: float = 0.01          # Mesh resolution in meters
            Lx:float = 1.0
            Ly:float = 1.0
            Lz:float = dx
            LNotch: float = 0.5
            NotchHeight: float = 2*dx
        elif testCase == "Dynamic":
            MeshType: str = "SingleEdgeNotched"
            dx: float = 0.01          # Mesh resolution in meters
            Lx:float = 1.0
            Ly:float = 0.5
            Lz:float = dx
            LNotch: float = 0.25
            NotchHeight: float = 0.5*dx
        else:
            raise ValueError(f"Unknown test case: {testCase}")
        
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
        if testCase == "Dynamic":
            Materials["Iron"]["LE_Damping"] = 0.0
            #Materials["Iron"]["F_s"] = 4*150e6
        
        Materials["Iron"]["Shear"] = Materials["Iron"]["Youngs"] / (2.0 * (1.0 + Materials["Iron"]["Poisson"]))
        Materials["Iron"]["Bulk"] = Materials["Iron"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Iron"]["Poisson"]))
        Materials["Iron"]["Lame"] = Materials["Iron"]["Shear"] * (2.0 * Materials["Iron"]["Poisson"]) / (1.0 - 2.0 * Materials["Iron"]["Poisson"])
        gravity: float = 0.0     # gravitational acceleration (m/s^2)
        
        if include_plots:
            Plot_Fields: list = ["u_x", "u_y", "phasefield"]
            Plot_Measures: list = [("t", "Crack_Length"), ("BC_Top_u", "BC_Top_F")]
        else:
            Plot_Fields: list = []
            Plot_Measures: list = []

        Solution_Steps = {"u": 0, "phasefield": 1}
        Field_Orders = {"u": 2, "phasefield": 2}
        
        Models = {}
        if testCase == "FullPlate" or testCase == "SingleEdgeNotched":
            ModelNames = ["Solid", "pf", "BC_Bottom", "BC_Bottom2", "BC_Top"]
        else:
            ModelNames = ["Solid", "pf", "BC_Bottom", "BC_Top"]
            
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
        if testCase == "Dynamic":
            Models["BC_Bottom"] = {
                "Type": ModelType.BOUNDARY_EXTERNALFORCE,
                "boundary": "bottom",
                "field": "u_y",
                "value": -10.0e6,
                "DamagedForces": True
            }
            Models["BC_Top"] = {
                "Type": ModelType.BOUNDARY_EXTERNALFORCE,
                "boundary": "top",
                "field": "u_y",
                "value": 10.0e6,
                "DamagedForces": True
            }          
        else:
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
            if testCase == "FullPlate":
                Models["BC_Top"] = {
                    "Type": ModelType.BOUNDARY_DISPCONTROL,
                    "boundary": "top",
                    "field": "u_y",
                    "rate": -3.0e-6,
                    "Dummy": 1.0e16,
                    "F_Cutoff": 0.05
                }
            else:
                ModelNames.append("BC_Top2")
                Models["BC_Top"] = {
                    "Type": ModelType.BOUNDARY_DISPCONTROL,
                    "boundary": "top",
                    "field": "u_x",
                    "rate": 1.0e-6,
                    "Dummy": 1.0e14,
                    "F_Cutoff": 0.05
                }
                Models["BC_Top2"] = {
                    "Type": ModelType.BOUNDARY_PRESCRIBED,
                    "boundary": "top",
                    "field": "u_y",
                    "value": 0.0
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
        if testCase == "Dynamic":
            alpha = 0.1
            Newmark_Beta: float = 0.25*(1+alpha)**2  # Newmark-beta parameter
            Newmark_Gamma: float = 0.5+alpha    # Newmark-gamma parameter
            
            abs_tol = 1.0
            
            n_passes = 1
            
            start_time: float = 0.0      # start time (s)
            end_time: float = 1.0e-3 # end time (s)
            dt: float = 1e-5           # time step (s)
            TimeStepControl = {
                "Type": "Measure",
                "time_list": [0.0],
                "dt_list": [dt],
                "dt_min": 1.0e-8,
                "dt_max": dt,
                "Measure": "Crack_Length_Change",
                "Target_Change": Materials["Iron"]["ell"] * 0.05
            }
            
        else:
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
        
    return Params()

class TestPhaseField_Simple:
    
    def test_LDL_Phasefield(self, InputParams):
        mesh = Mesh(InputParams)
        physics = Physics(mesh, InputParams)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams, physics)
        solver.run()
   

if __name__ == "__main__":
    plts = True
    
    isolate_jit_cache()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--Gc", type=float, default=None, help="Run a single case with this Gc value.")
    parser.add_argument("--dir", type=int, choices=[1, -1], default=None, help="Run a single case with this loading direction.")
    parser.add_argument("--Geometry", type=str, choices=["PlateWithHole", "SENT","Dynamic"], default="PlateWithHole", help="Geometry type to run.")
    parser.add_argument("--dxRef", type=float, default=5.0, help="Reference dx for calculating element size.")
    parser.add_argument("--ell", type=float, default=None, help="Length scale parameter for the material.")
    parser.add_argument("--FailureMode", type=str, default="r1", help="Failure mode for Solid model (e.g. r1, DP).")
    parser.add_argument("--DP_ref", type=float, default=1.0, help="Reference DP strain parameter when using DP failure mode.")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory for this run.")
    args = parser.parse_args()

    if args.Gc is not None and args.dir is not None:
        if args.Geometry == "PlateWithHole":
            params = Params_Cases("FullPlate", plts)
            params.Models["BC_Top"]["rate"] = args.dir * abs(params.Models["BC_Top"]["rate"])
            params.Materials["Iron"]["Gc"] = args.Gc
        elif args.Geometry == "SENT":
            params = Params_Cases("SingleEdgeNotched", plts)
            if args.dir == 1:
                params.Models["BC_Top"]["field"] = "u_y"
            else:
                params.Models["BC_Top"]["field"] = "u_x"
            params.Materials["Iron"]["Gc"] = args.Gc
            params.end_time = 3000.0
        elif args.Geometry == "Dynamic":
            params = Params_Cases("Dynamic", plts)
            params.Materials["Iron"]["Gc"] = args.Gc
        else:
            raise ValueError(f"Unknown Geometry type: {args.Geometry}")

        if args.ell is not None:
            params.Materials["Iron"]["ell"] = args.ell
        params.dx = params.Materials["Iron"]["ell"] / args.dxRef
        params.Models["Solid"]["FailureType"] = args.FailureMode
        params.Materials["Iron"]["DP_eRef"] = args.DP_ref
        if args.output_dir is not None:
            params.output_dir = args.output_dir
        elif args.Geometry == "PlateWithHole":
            params.output_dir = (
                f"Geometry_{args.Geometry}/dir_{args.dir}/Gc_{args.Gc}_ell_{params.Materials['Iron']['ell']}_dxRef_{args.dxRef}"
            )
        elif args.Geometry == "SENT":
            params.output_dir = (
                f"Geometry_{args.Geometry}/dir_{args.dir}/Gc_{args.Gc}_ell_{params.Materials['Iron']['ell']}_FailureMode_{args.FailureMode}_DPref_{args.DP_ref}"
            )
        mprint(
            f"\n\n--- Running Phase-field test with Gc={args.Gc}, dir={args.dir}, Geometry={args.Geometry}, "
            f"ell={params.Materials['Iron']['ell']}, FailureMode={args.FailureMode}, DP_ref={args.DP_ref} ---"
        )
        TestPhaseField_Simple().test_LDL_Phasefield(params)
    else:
        cases = []
        
        Geometry = "PlateWithHole"
        FailureMode = "r1"
        for dxRef in [0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5]:
            for dir in [1, -1]:
                ell = 0.05
                output_dir = f"Geometry_PlateWithHole/dir_{dir}/Gc_{1e5}_dxRef_{dxRef}"
                cases.append((Geometry, 1e5, dir, dxRef, FailureMode, 1.0, ell, output_dir))
        for Gc in [1e4, 2.5e4, 5e4, 7.5e4, 2.5e5, 5e5, 7.5e5, 1e6]: #, 1e5
            for dir in [1, -1]:
                dxRef = 3.0
                ell = 0.05
                output_dir = f"Geometry_PlateWithHole/dir_{dir}/Gc_{Gc}_dxRef_{dxRef}"
                cases.append((Geometry, Gc, dir, dxRef, FailureMode, 1.0, ell, output_dir))

        Geometry = "PlateWithHole"
        FailureMode = "r1"
        for ell in [0.1, 0.075, 0.05, 0.025, 0.01]: 
            for dir in [1]: #, -1
                output_dir = (
                    f"Geometry_PlateWithHole_LSweep/dir_{dir}/ell_{ell}"
                )
                cases.append((Geometry, 1e5, dir, 3.0, FailureMode, 1.0, ell, output_dir))

        Geometry = "SENT"
        for dir in [1, -1]:
            FailureMode = "r1"
            ell = 0.05
            output_dir = f"Geometry_SENT/dir_{dir}/Gc_{1e5}_FailureMode_{FailureMode}_DPref_{1.0}"
            cases.append((Geometry, 1e5, dir, 3.0, FailureMode, 1.0, ell, output_dir))
            
            FailureMode = "DP"
            for DP_ref in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1e0]:
                output_dir = f"Geometry_SENT/dir_{dir}/Gc_{1e5}_FailureMode_{FailureMode}_DPref_{DP_ref}"
                cases.append((Geometry, 1e5, dir, 3.0, FailureMode, DP_ref, ell, output_dir))
                
        for dir in [1, -1]:
            FailureMode = "r1"
            ell = 0.05
            output_dir = f"Geometry_SENT/dir_{dir}/Gc_{1e4}_FailureMode_{FailureMode}_DPref_{1.0}"
            cases.append((Geometry, 1e4, dir, 3.0, FailureMode, 1.0, ell, output_dir))
            
            FailureMode = "DP"
            for DP_ref in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 1e-1, 1e0]:
                output_dir = f"Geometry_SENT/dir_{dir}/Gc_{1e4}_FailureMode_{FailureMode}_DPref_{DP_ref}"
                cases.append((Geometry, 1e4, dir, 3.0, FailureMode, DP_ref, ell, output_dir))
                
        Geometry = "Dynamic"
        for Gc in [2.5e4, 5e4, 7.5e4, 1e5]: #1e3, 2.5e3, 5e3, 7.5e3, 1e4, 
            output_dir = f"Geometry_Dynamic/Gc_{Gc}"
            l = 0.01
            cases.append((Geometry, Gc, 1, 3.0, "DP", 1.0, l, output_dir))        
   
                
        max_procs = 1
        active = []
        failed_cases = []
        while cases or active:
            while cases and len(active) < max_procs:
                Geometry, Gc, dir, dxRef, FailureMode, DP_ref, ell, output_dir = cases.pop(0)
                cmd = [
                    "mpirun", "-np", "50", "python3", __file__,
                    "--Gc", str(Gc),
                    "--dir", str(dir),
                    "--Geometry", Geometry,
                    "--dxRef", str(dxRef),
                    "--ell", str(ell),
                    "--FailureMode", str(FailureMode),
                    "--DP_ref", str(DP_ref),
                    "--output-dir", output_dir,
                ]
                mprint(
                    f"\n\n--- Spawning Phase-field test with Gc={Gc}, dir={dir}, Geometry={Geometry}, "
                    f"ell={ell}, FailureMode={FailureMode}, DP_ref={DP_ref}, output_dir={output_dir} ---"
                )
                active.append((Geometry, Gc, dir, dxRef, FailureMode, DP_ref, ell, output_dir, subprocess.Popen(cmd)))

            still_active = []
            for Geometry, Gc, dir, dxRef, FailureMode, DP_ref, ell, output_dir, proc in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((Geometry, Gc, dir, dxRef, FailureMode, DP_ref, ell, output_dir, proc))
                elif ret != 0:
                    failed_cases.append((Geometry, Gc, dir, dxRef, FailureMode, DP_ref, ell, output_dir, ret))
                    mprint(
                        f"Case failed (Gc={Gc}, dir={dir}, Geometry={Geometry}, dxRef={dxRef}, "
                        f"ell={ell}, FailureMode={FailureMode}, DP_ref={DP_ref}, output_dir={output_dir}) "
                        f"with exit code {ret}. Continuing to next case."
                    )
                    raise RuntimeError(f"Case failed with exit code {ret}")
                    
            active = still_active
            if active:
                time.sleep(0.2)

        if failed_cases:
            mprint("\n\n--- Sweep completed with failed cases ---")
            for Geometry, Gc, dir, dxRef, FailureMode, DP_ref, ell, output_dir, ret in failed_cases:
                mprint(
                    f"Failed: Geometry={Geometry}, Gc={Gc}, dir={dir}, dxRef={dxRef}, "
                    f"ell={ell}, FailureMode={FailureMode}, DP_ref={DP_ref}, output_dir={output_dir}, exit_code={ret}"
                )
        else:
            mprint("\n\n--- Sweep completed successfully (no failed cases) ---")
