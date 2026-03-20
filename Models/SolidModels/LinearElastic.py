from ufl import div, inner, dot, dx, TrialFunction, as_tensor, Identity, sqrt
from dolfinx import fem

from Models.BaseModel import BaseModel
from Utils.maths_utils import nabla_s, macPlus, macMinus, eig
from Mesh.Mesh import Mesh
from Params import Params
from Models.ModelEnums import ModelType
from ufl import derivative, tr, conditional, lt
import ufl

class LinearElastic(BaseModel):

    REQUIRED_MODEL_SETTINGS = {
        "Type": "Solid:LinearElastic", 
        "Material": "Ice",
        "DamageModel": None,
    }
    REQUIRED_PARAMETERS = {
        "Field_Orders": {"u": 2},
        "Solution_Steps": {"u": 0},
        "gravity": 9.81,
        "dt": 1.0,
        "Newmark_Beta": 0.5625,
        "Newmark_Gamma": 1.0,
        "Ice":{
            "Density": 910.0,
            "Lame": 5e9, 
            "Shear": 1e13,
        },
    }
    
    def __init__(self, name, params: Params, mesh: Mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.SOLID_LINEARELASTIC
        
        mesh.Add_Vector_Field("u", "Displacement [m]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("uOld", "Old Displacement [m]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("v", "Velocity [m/s]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("vOld", "Old Velocity [m/s]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("a", "Acceleration [m/s²]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("aOld", "Old Acceleration [m/s²]", "CG", params.Field_Orders["u"], params.dim)

        mesh.Add_Field("p", "Pressure [Pa]", "DG", params.Field_Orders["u"] - 1) # Pressure field for post-processing

        self.My_Step = params.Solution_Steps["u"]
        
        # Register test functions (actual test functions will be retrieved when needed)
        mesh.Add_Vector_TestFunction("u", params.dim, self.My_Step)
        
        # material parameters
        self.material = self.params.Models[self.name]["Material"]
        self.lame = self.params.Materials[self.material]["Lame"]
        self.shear = self.params.Materials[self.material]["Shear"]
        self.rho = self.params.Materials[self.material]["Density"]
        self.gravity = self.params.gravity
        self.damping = self.params.Materials[self.material].get("LE_Damping", 0.0)
        
        self.NM_Beta = self.params.Newmark_Beta
        self.NM_Gamma = self.params.Newmark_Gamma
        
        # Check if damage model is specified
        self.damage_model_name = self.params.Models[self.name].get("DamageModel", None)
        if self.damage_model_name:
            mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", ())
            self.DrivingForce_Type = self.params.Models[self.name].get("CrackDrivingForce", "Stress-Based")
            self.Gc = self.params.Materials[self.material]["Gc"]
            self.ell = self.params.Materials[self.material]["ell"]
            self.f_t = self.params.Materials[self.material]["f_t"]
            self.kappa = 1e-12 # residual stiffness
        else:
            pass
        self.d_e = 1.0  # No damage
        self.d_rho = 1.0  # No damage
    
    def assemble_KF(self, step: int) -> None:
        """
        Assemble the global stiffness matrix and force vector for the linear elastic model.
        """
        
        if (step != self.My_Step):
            return
        
        if self.damage_model_name == "PhaseField":
            self.Update_PhaseField_Fields()

        # Get fields - mesh automatically returns mixed space version if applicable
        u = self.mesh.Get_Field("u", step=self.My_Step)
        v = self.mesh.Get_Field("v")
        a = self.mesh.Get_Field("a")
        uOld = self.mesh.Get_Field("uOld")
        vOld = self.mesh.Get_Field("vOld")
        aOld = self.mesh.Get_Field("aOld")
        
        dv_du, da_du = self.Newmark_Update(u, v, a, uOld, vOld, aOld, self.params.dt)
        
        u_t = self.mesh.Get_TestFunction("u", self.My_Step)
        u_tr = self.mesh.Get_Trial_Function("u", self.My_Step)
        
        # Volumetric stresses (bulk response)
        F = self.d_e*self.lame * tr(nabla_s(u_t)) * tr(nabla_s(u)) * dx
        K = self.d_e*self.lame * tr(nabla_s(u_t)) * tr(nabla_s(u_tr)) * dx
        
        # Deviatoric stresses
        F += 2.0*self.d_e*self.shear * inner(nabla_s(u_t), nabla_s(u)) * dx
        K += 2.0*self.d_e*self.shear * inner(nabla_s(u_t), nabla_s(u_tr)) * dx
        
        # Inertia terms
        F += self.d_rho*self.rho * inner(u_t, a) * dx
        K += self.d_rho*self.rho * inner(u_t, da_du * u_tr) * dx
        
        # Damping terms
        if self.damping > 0.0:
            F += self.d_rho*self.damping * inner(u_t, v) * dx
            K += self.d_rho*self.damping * inner(u_t, dv_du * u_tr) * dx
        
        # Body forces (gravity)
        if (self.gravity != 0.0):
            F += -self.d_rho*self.rho*u_t[1]*self.gravity*dx #gravity
        
        return K, F
    
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        for field_name in ("u", "uOld", "v", "vOld", "a", "aOld", "p"):
            self.mesh.Zero_Field(field_name)
            
    def Newmark_Update(self, u, v, a, uOld, vOld, aOld, dt) -> None:
        """Update displacement, velocity, and acceleration using Newmark-beta method."""
        beta = self.NM_Beta
        gamma = self.NM_Gamma
        
        # Update velocity
        v.x.array[:] = gamma/beta/dt*(u.x.array - uOld.x.array) + (1 - gamma/beta)*vOld.x.array + dt*(1 - gamma/(2*beta))*aOld.x.array
        a.x.array[:] = 1/(beta*dt**2)*(u.x.array - uOld.x.array) - 1/(beta*dt)*vOld.x.array - (1/(2*beta) - 1)*aOld.x.array
        
        dv_du = gamma/(beta*dt)
        da_du = 1/(beta*dt**2)
        
        return dv_du, da_du
            

    def Update_PhaseField_Fields(self) -> None:
        if self.damage_model_name != "PhaseField":
            return
        
        # Damage functions
        phi_lim = conditional(lt(self.mesh.Get_Field("phasefield"), 1.0), self.mesh.Get_Field("phasefield"), 1.0)
        phi_lim = conditional(lt(phi_lim, 0.0), 0.0, phi_lim)
        self.d_e = self.kappa + (1.0-self.kappa)*(1.0 - phi_lim)**2
        self.d_rho = conditional(lt(self.mesh.Get_Field("phasefield"), 0.9), 1.0, 0.0)
        
        # Crack driving force
        strain = nabla_s(self.mesh.Get_Field("u", step=self.My_Step))
        
        if self.DrivingForce_Type == "FullEnergy":
            e_v = tr(strain)
            Psi_Elastic = 0.5 * self.lame * (macPlus(e_v))**2
            Psi_Elastic += self.shear * inner(nabla_s(self.mesh.Get_Field("u", step=self.My_Step)), nabla_s(self.mesh.Get_Field("u", step=self.My_Step)))
        elif self.DrivingForce_Type == "VolDev":
            e_v = tr(strain)
            Psi_Elastic = 0.5 * self.lame * (macPlus(e_v))**2
            Psi_Elastic += self.shear * inner(strain, strain)
        elif self.DrivingForce_Type == "VolOnly":
            e_v = tr(strain)
            Psi_Elastic = 0.5 * self.lame * (macPlus(e_v))**2
        elif self.DrivingForce_Type == "Stress-Based":
            # Extend strains to 3x3 if in 2D
            if self.params.dim == 2:
                strain_3D = as_tensor([[strain[0,0], strain[0,1], 0.0],
                                       [strain[1,0], strain[1,1], 0.0],
                                       [0.0,        0.0,        0.0]])
            else:
                strain_3D = strain
            
            sigma = self.lame * tr(strain_3D) * Identity(3) + 2.0 * self.shear * strain_3D
            eig1, eig2, eig3 = eig(sigma, 2)
            pf_form = self.params.Materials[self.material].get("PF_formulation", "AT2")
            if pf_form == "AT2":
                pre_fac = self.Gc/(2.0*self.ell)
                Psi_Elastic = pre_fac * macPlus((macPlus(eig1)**2 + macPlus(eig2)**2 + macPlus(eig3)**2) / self.f_t**2 - 1.0)
            else:  # AT1
                pre_fac = 3.0*self.Gc/(8.0*self.ell)
                Psi_Elastic = pre_fac * (macPlus(eig1)**2 + macPlus(eig2)**2 + macPlus(eig3)**2) / self.f_t**2
        else:
            raise ValueError(f"Unknown CrackDrivingForce type: {self.DrivingForce_Type}")
        
        self.mesh.Set_From_Expression("ElasticEnergy", Psi_Elastic)
        

    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """
        
        self.Newmark_Update(
            self.mesh.Get_Field("u", step=self.My_Step),
            self.mesh.Get_Field("v"),
            self.mesh.Get_Field("a"),
            self.mesh.Get_Field("uOld"),
            self.mesh.Get_Field("vOld"),
            self.mesh.Get_Field("aOld"),
            self.params.dt,
        )
        
        # Copy current fields to old fields using mesh utility
        self.mesh.Copy_Field("u", "uOld")
        self.mesh.Copy_Field("v", "vOld")
        self.mesh.Copy_Field("a", "aOld")
        
        # Update pressure field for post-processing
        # Evaluate pressure as p = -λ * tr(ε) where ε is the strain tensor
        u = self.mesh.Get_Field("u", step=self.My_Step)
        pressure_expr = -self.lame * tr(nabla_s(u))
        self.mesh.Set_From_Expression("p", pressure_expr)
        
    def reset_step(self):
        self.mesh.Copy_Field("uOld", "u")
        self.mesh.Copy_Field("vOld", "v")
        self.mesh.Copy_Field("aOld", "a")
        
        pass
