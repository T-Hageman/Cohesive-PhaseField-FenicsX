from ufl import div, inner, dot, dx, TrialFunction, max_value
from dolfinx import fem

from Models.BaseModel import BaseModel
from Utils.maths_utils import nabla_s, macPlus, macMinus, dmacMinus
from Mesh.Mesh import Mesh
from Params import Params
from Models.ModelEnums import ModelType
from ufl import derivative, grad, conditional, lt
from mpi4py import MPI
from Utils.mpi_utils import mprint, comm, rank

class PhaseField(BaseModel):

    REQUIRED_MODEL_SETTINGS = {
        "Type": "Solid:PhaseField", 
        "Material": "Ice",
    }
    REQUIRED_PARAMETERS = {
        "Field_Orders": {"phasefield": 2, "LMultiplier_PF": 1},
        "Solution_Steps": {"phasefield": 1, "LMultiplier_PF": 1},
        "dt": 1.0,
        "Ice":{
            "Gc": 100.0,
            "ell": 10.0
        },
    }
    
    def __init__(self, name, params: Params, mesh: Mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.SOLID_PHASE_FIELD
        
        self.Irreversible_Method = self.params.Models[self.name].get("Irreversible_Method", "AL") # AL or Hist
        
        mesh.Add_Field("phasefield", "Phasefield [-]", "CG", params.Field_Orders["phasefield"])
        mesh.Add_Field("phasefieldOld", "Old Phasefield [-]", "CG", params.Field_Orders["phasefield"])
        mesh.Add_Field("phasefieldOldOld", "Old Old Phasefield [-]", "CG", params.Field_Orders["phasefield"])
        if self.Irreversible_Method == "Hist":
            mesh.Add_Quadrature_Field("phasefield_Hist", "Phasefield History [-]", ())
            mesh.Add_Quadrature_Field("phasefield_HistOld", "Old Phasefield History [-]", ())
        else:
            mesh.Add_Field("LMultiplier_PF", "Lagrange Multiplier for Phasefield [-]", "CG", params.Field_Orders["phasefield"]-1)

        self.My_Step = params.Solution_Steps["phasefield"]
        
        # Register test functions (actual test functions will be retrieved when needed)
        mesh.Add_TestFunction("phasefield", self.My_Step)
        if self.Irreversible_Method == "AL":
            mesh.Add_TestFunction("LMultiplier_PF", self.My_Step)
        
        # crack driving force field
        self.CDF_Name = self.params.Models[self.name].get("CDF_field", "ElasticEnergy")
        
        # material parameters
        self.material = self.params.Models[self.name]["Material"]
        self.Gc = self.params.Materials[self.material]["Gc"]
        self.ell = self.params.Materials[self.material]["ell"]
        self.damping = self.params.Materials[self.material].get("pf_damping", 0.0)
        self.kappa = self.params.Materials[self.material].get("kappa", 1e-3)
        
        self.PF_Version = self.params.Materials[self.material].get("PF_formulation", "AT2")
    
        self.params.Global_Measures[f"Crack_Length"] = 0.0
        self.params.Global_Measures[f"Crack_Length_Change"] = 0.0
    
    def assemble_KF(self, step: int) -> None:
        """
        Assemble the global stiffness matrix and force vector for phase-field fractures.
        """
        
        if (step != self.My_Step):
            return

        # Get fields
        phi = self.mesh.Get_Field("phasefield", step=self.My_Step)
        phiOld = self.mesh.Get_Field("phasefieldOld", step=self.My_Step)
        phi_t = self.mesh.Get_TestFunction("phasefield", self.My_Step)
        phi_tr = self.mesh.Get_Trial_Function("phasefield", self.My_Step)
        
        if self.Irreversible_Method == "AL":
            L = self.mesh.Get_Field("LMultiplier_PF", step=self.My_Step)
            L_t = self.mesh.Get_TestFunction("LMultiplier_PF", self.My_Step)
            L_tr = self.mesh.Get_Trial_Function("LMultiplier_PF", self.My_Step)
        
        Psi_Elastic = self.mesh.Get_Field(self.CDF_Name)
        if (self.Irreversible_Method == "Hist"):
            if (self.params.t+self.params.dt>1e-9):
                Psi_Elastic_Hist = self.mesh.Get_Field("phasefield_HistOld")
                Psi_Elastic = max_value(Psi_Elastic, Psi_Elastic_Hist)
                self.mesh.Set_From_Expression("phasefield_Hist", Psi_Elastic)
            else:
                Psi_Elastic = self.mesh.Get_Field("phasefield_HistOld")
                self.mesh.Set_From_Expression("phasefield_Hist", Psi_Elastic)
            

        # Crack capacity terms, assuming AT-1 model
        if self.PF_Version == "AT1": # AT1
            F = (3*self.Gc/8/self.ell) * ( phi_t + self.ell**2 * inner(grad(phi_t), grad(phi))    ) * dx
            K = (3*self.Gc/8/self.ell) * (         self.ell**2 * inner(grad(phi_t), grad(phi_tr)) ) * dx
        else:    # AT2
            F = (self.Gc/2/self.ell) * ( 2.0*inner(phi_t, phi)    + self.ell**2 * inner(grad(phi_t), grad(phi))    ) * dx
            K = (self.Gc/2/self.ell) * ( 2.0*inner(phi_t, phi_tr) + self.ell**2 * inner(grad(phi_t), grad(phi_tr)) ) * dx
        
        # Crack driving force term
        F += -2.0*(1-self.kappa)*phi_t*(1-phi)*Psi_Elastic*dx
        K += -2.0*(1-self.kappa)*phi_t*(-phi_tr)*Psi_Elastic*dx
            
        # Damping term
        if self.damping > 0.0:
            F += self.damping*inner(phi_t, (phi - phiOld)/self.params.dt)*dx
            K += self.damping*inner(phi_t, phi_tr/self.params.dt)*dx

        if self.Irreversible_Method == "AL":
            # Irreversibility through augmented Lagrangian: E = 1/(2gamma)*(L+gamma*phiDot/dXdt)^2-1/2gamma * L^2   //see e.g. Geelen et al, A phase-field formulation for dynamic cohesive fracture
            oldVal =  conditional(lt(phiOld, 0.99), phiOld, 0.99)
            oldVal = conditional(lt(oldVal, 1.0e-9), 1.0e-9, oldVal)
            irrCondition = phi - oldVal
        
            dummy = 1.0e0
            AugLagr   = L + dummy*irrCondition
            AugMinus  = macMinus(AugLagr)
            dAug_dL   = dmacMinus(AugLagr)
            dAug_dhpi = dummy*dmacMinus(AugLagr)
            
            # Damage equation terms
            F += 1.0/dummy*inner(phi_t, dAug_dhpi*AugMinus)*dx
            K += 1.0/dummy*inner(phi_t, dAug_dhpi*dAug_dhpi*phi_tr + dAug_dhpi*dAug_dL*L_tr)*dx
            
            # Lagrange multiplier equation terms
            F += 1.0/dummy*inner(L_t, dAug_dL*AugMinus - L*(1.0+1.0e-10))*dx
            K += 1.0/dummy*inner(L_t, dAug_dL*dAug_dhpi*phi_tr + dAug_dL*dAug_dL*L_tr - L_tr*(1.0+1.0e-10))*dx

        return K, F
    
    def Evaluate_Crack_Length(self, phasefield, Oldphasefield) -> float:
        """
        Evaluate the total crack length in the domain based on the phase-field variable.
        
        Args:
            phasefield: The phase-field variable representing damage.
            
        Returns:
            Total crack length (float).
        """
        # Crack density function
        if self.PF_Version == "AT1":
            crack_density = macPlus(3/(8.0*self.ell) * (2.0*phasefield+ self.ell/2*inner(grad(phasefield), grad(phasefield))))
            old_crack_density = macPlus(3/(8.0*self.ell) * (2.0*Oldphasefield+ self.ell/2*inner(grad(Oldphasefield), grad(Oldphasefield))))
        else: # AT2
            crack_density =  macPlus(1/(2.0*self.ell) * (phasefield**2 + self.ell**2/2*inner(grad(phasefield), grad(phasefield))))
            old_crack_density = macPlus(1/(2.0*self.ell) * (Oldphasefield**2 + self.ell**2/2*inner(grad(Oldphasefield), grad(Oldphasefield))))
        
        total_crack_length = fem.assemble_scalar(fem.form(crack_density*dx(self.mesh.mesh)))
        old_crack_length = fem.assemble_scalar(fem.form(old_crack_density*dx(self.mesh.mesh)))
        
        # sync across processes
        total_crack_length = comm.allreduce(total_crack_length, op=MPI.SUM)
        old_crack_length = comm.allreduce(old_crack_length, op=MPI.SUM)
        
        self.params.Global_Measures[f"Crack_Length"] = total_crack_length
        self.params.Global_Measures[f"Crack_Length_Change"] = abs(total_crack_length - old_crack_length)
        
        #mprint(f"Total Crack Length: {total_crack_length:.6f}, Change in Crack Length: {total_crack_length - old_crack_length:.6f}")
        return total_crack_length
    
    def Update_Global_Measures(self) -> None:
        phi = self.mesh.Get_Field("phasefield", step=self.My_Step)
        phiOld = self.mesh.Get_Field("phasefieldOld", step=self.My_Step)
        self.Evaluate_Crack_Length(phi, phiOld)
    
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        for field_name in ("phasefield", "phasefieldOld", "phasefieldOldOld"):
            self.mesh.Zero_Field(field_name)
            
        if self.Irreversible_Method == "Hist":
            self.mesh.Zero_Field("phasefield_Hist")
            self.mesh.Zero_Field("phasefield_HistOld")
        else:
            self.mesh.Zero_Field("LMultiplier_PF")
            
            
    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """
        # Copy current fields to old fields
        self.mesh.Copy_Field("phasefieldOld", "phasefieldOldOld")
        self.mesh.Copy_Field("phasefield", "phasefieldOld")
        if self.Irreversible_Method == "Hist":
            self.mesh.Copy_Field("phasefield_Hist", "phasefield_HistOld")
        
        pass
    
    def reset_step(self):
        self.mesh.Copy_Field("phasefieldOld", "phasefield")
        
        pass
