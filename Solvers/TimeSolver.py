from Physics.Physics import Physics
from Params import Params
from Mesh.Mesh import Mesh
from Solvers.StepSolver import StepSolver
from Utils.mpi_utils import mprint

class TimeSolver(StepSolver):
    
    def __init__(self, params, physics) -> None:
        super().__init__(params, physics)
        
        self.t: float = params.start_time
        self.params.t = self.t
        
        # JIT warmup option
        self.jit_warmup = getattr(params, 'JIT_WARMUP', True)
        
    def warmup_jit(self) -> None:
        if self.jit_warmup:
            mprint("Performing JIT warmup (precompiling forms)...")
            for step in range(self.n_Staggered_Steps):
                K, F = self.physics.Get_Kmat_Fvec(step)
                if K is not None:
                    K.destroy()
                if F is not None:
                    F.destroy()
            mprint("JIT warmup complete.")
    
    def run(self) -> None:
        self.warmup_jit()
        
        Step: int = 0
        while self.t < self.params.end_time:
            self.Update_TimeIncrements()
            mprint(f"Step {Step}, time: {self.t:.4e} s, dt: {self.params.dt:.4e} s")
            
            # Solve for current Step
            StepValid: bool = False
            while StepValid == False:
                self.Solve_Step()
                self.Update_Measures()
                if Step > 1:
                    StepValid = self.Check_Step_Validity()
                    if StepValid == False:
                        self.physics.Reset_Step()
                else:
                    StepValid = True
            
            # Outputs
            self.physics.mesh.Plot_Fields()
            self.physics.mesh.Plot_Measures()
            
            # Commit Step
            self.t += self.params.dt
            self.params.t = self.t
            self.physics.commit()
            Step += 1
            
            self.physics.mesh.Write_Outputs(Step, self.t)
        
    def Update_Measures(self) -> None:
        self.physics.Update_Global_Measures()
    
    def Update_TimeIncrements(self) -> None:
        """
        Update time increments based on time step control settings.
        """
        if hasattr(self.params, "TimeStepControl"):
            tsc = self.params.TimeStepControl
        else:
            tsc = {"Type": "None"}
            
        if tsc["Type"] == "None":
            self.params.dt = self.params.dt
        if tsc["Type"] == "StepWise":
            time_list = tsc["time_list"]
            dt_list = tsc["dt_list"]
            for i in range(len(time_list)-1):
                if time_list[i] <= self.t < time_list[i+1]:
                    self.params.dt = dt_list[i]
                    break
            else:
                self.params.dt = dt_list[-1]
        if tsc["Type"] == "Measure":
            dt_old = self.params.dt
            self.params.dt = min(dt_old * 1.2, tsc.get("dt_max", 600.0))  # increase time step if below target
            if self.params.t < 0.0 and self.params.t + self.params.dt > 0.0:
                self.params.dt = -self.params.t  # ensure we hit t=0 exactly
            if self.params.t == 0.0:
                # check time_list for t=0, and then set dt, otherwise don't touch dt
                time_list = tsc["time_list"]
                dt_list = tsc["dt_list"]
                for i in range(len(time_list)):
                    if time_list[i] == 0.0:
                        self.params.dt = dt_list[i]
                        break
                
    
    def Check_Step_Validity(self) -> bool:
        """
        Check if the current step solution is valid.
        
        Returns:
            True if valid, False otherwise.
        """
        if hasattr(self.params, "TimeStepControl"):
            tsc = self.params.TimeStepControl
        else:
            tsc = {"Type": "None"}
        
        if tsc["Type"] == "Measure":
            measure_name = tsc["Measure"]
            target_change = tsc["Target_Change"]
            if self.params.dt <= 1.05*tsc.get("dt_min", 1.0e-8):
                mprint(f"  Time step {self.params.dt:.4e} s is at or below minimum. Accepting step to avoid excessively small time steps.")
                return True  # Already at minimum time step
            if measure_name in self.params.Global_Measures:
                measure_value = self.params.Global_Measures[measure_name]
                if measure_value > target_change:
                    self.params.dt = max(self.params.dt / 2.0, tsc.get("dt_min", 1.0e-8))
                    mprint(f"  Step invalid due to {measure_name} = {measure_value:.4e} exceeding target {target_change:.4e}")
                    mprint(f"  Reducing time step to {self.params.dt:.4e} s")
                    return False
                else:
                    mprint(f"  Step valid with {measure_name} = {measure_value:.4e} within target {target_change:.4e}")
        
        return True
