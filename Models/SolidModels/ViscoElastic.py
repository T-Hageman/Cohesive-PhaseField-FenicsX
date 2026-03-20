from ufl import div, inner, dot, dx, TrialFunction, as_tensor, Identity, sqrt
from dolfinx import fem
from basix.ufl import element, quadrature_element

from Models.BaseModel import BaseModel
from Utils.maths_utils import nabla_s, macPlus, macMinus, eig, StiffnessMatrix, J2Matrix, Tensor_To_Voight_Strain, Voight_To_Tensor_Strain
from Mesh.Mesh import Mesh
from Params import Params
from Models.ModelEnums import ModelType
from ufl import derivative, tr, conditional, lt
import ufl

class ViscoElastic(BaseModel):

    def __init__(self, name, params: Params, mesh: Mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.SOLID_VISCOELASTIC
        
        mesh.Add_Vector_Field("u", "Displacement [m]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("uOld", "Old Displacement [m]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("v", "Velocity [m/s]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("vOld", "Old Velocity [m/s]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("a", "Acceleration [m/s²]", "CG", params.Field_Orders["u"], params.dim)
        mesh.Add_Vector_Field("aOld", "Old Acceleration [m/s²]", "CG", params.Field_Orders["u"], params.dim)

        mesh.Add_Field("p", "Pressure [Pa]", "DG", params.Field_Orders["u"] - 1) # Pressure field for post-processing

        self.mesh.Add_Quadrature_Field("Plastic_Strain", "Plastic Strain Tensor", (self.params.dim, self.params.dim))
        self.mesh.Add_Quadrature_Field("Total_Strain", "Total Strain Tensor", (self.params.dim, self.params.dim))
        self.mesh.Add_Quadrature_Field("Plastic_Strain_Old", "Old Plastic Strain Tensor", (self.params.dim, self.params.dim))
        self.Totalstrain = self.mesh.Get_Field("Total_Strain")
        self.strain_pOld = self.mesh.Get_Field("Plastic_Strain_Old")
        self.strain_p = self.mesh.Get_Field("Plastic_Strain")

        self.My_Step = params.Solution_Steps["u"]
        
        # Register test functions (actual test functions will be retrieved when needed)
        mesh.Add_Vector_TestFunction("u", params.dim, self.My_Step)
        
        # material parameters
        self.material = self.params.Models[self.name]["Material"]
        self.lame = self.params.Materials[self.material]["Lame"]
        self.shear = self.params.Materials[self.material]["Shear"]
        self.bulk = self.params.Materials[self.material]["Bulk"]
        self.rho = self.params.Materials[self.material]["Density"]
        self.gravity = self.params.gravity
        self.damping = self.params.Materials[self.material].get("LE_Damping", 0.0)
        
        self.viscosity_type = self.params.Materials[self.material].get("Visc_Type", "Newtonian")
        if self.viscosity_type == "Newtonian":
            self.viscosity = self.params.Materials[self.material]["Viscosity"]
            self.plastic_n = 1.0
        elif self.viscosity_type == "Glen":
            self.Glen_A = self.params.Materials[self.material]["Glen_A"]
            self.plastic_n = self.params.Materials[self.material]["Glen_n"]
            self.viscosity = (1.0 / (self.Glen_A)) ** (1.0 / self.plastic_n)
        else:
            raise NotImplementedError(f"Viscosity type '{self.viscosity_type}' not implemented.")
        
        self.NM_Beta = self.params.Newmark_Beta
        self.NM_Gamma = self.params.Newmark_Gamma
        
        # Check if damage model is specified
        self.damage_model_name = self.params.Models[self.name].get("DamageModel", None)
        if self.damage_model_name:
            mesh.Add_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", "DG", 2*(params.Field_Orders["u"]-1))
            self.DrivingForce_Type = self.params.Models[self.name].get("CrackDrivingForce", "Stress-Based")
            self.Gc = self.params.Materials[self.material]["Gc"]
            self.ell = self.params.Materials[self.material]["ell"]
            self.f_t = self.params.Materials[self.material]["f_t"]
            self.kappa = 1e-12 # residual stiffness
        else:
            pass
        self.d_e = 1.0  # No damage
        self.d_rho = 1.0  # No damage

        self._setup_numba_kernel()
    
    def assemble_KF(self, step: int) -> None:
        """
        Assemble the global stiffness matrix and force vector for the linear elastic model.
        """
        
        if (step != self.My_Step):
            return
        


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
        
        if self.damage_model_name == "PhaseField":
            self.Update_PhaseField_Fields()
        
        # Update quadrature plastic strains
        self._update_plastic_fields(nabla_s(u))

        dx = ufl.Measure("dx", domain=self.mesh.mesh)

        # Volumetric stresses (bulk response)
        strain = nabla_s(u) - self.strain_p
        F = self.d_e*self.bulk * tr(nabla_s(u_t)) * tr(strain) * dx
        K = self.d_e*self.bulk * tr(nabla_s(u_t)) * tr(nabla_s(u_tr)) * dx
        
        # Deviatoric stresses
        strain_dev = strain - (1.0 / 3.0) * tr(strain) * Identity(self.params.dim)
        strain_dev_tr = nabla_s(u_tr) - (1.0 / 3.0) * tr(nabla_s(u_tr)) * Identity(self.params.dim)
        F += 2.0*self.d_e*self.shear * inner(nabla_s(u_t), strain_dev) * dx
        K += 2.0*self.d_e*self.shear * inner(nabla_s(u_t), strain_dev_tr) * dx
        
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

    def _setup_numba_kernel(self) -> None:
        import numba as nb
        import numpy as np

        dim = self.params.dim
        eta = float(self.viscosity)
        n_exp = float(self.plastic_n)
        D_el = StiffnessMatrix(self.lame, self.shear)
        J2Mat = J2Matrix("Stress")

        @nb.njit(cache=True)
        def ReturnMapping_KF(state, e, e_pOld, dt):
            elastic_strain = np.ascontiguousarray(state[0:6, 0])
            stress = D_el @ elastic_strain
            lMult = state[6, 0]

            J2 = 0.5 * (stress.T @ J2Mat @ stress)
            if J2 < 1.0e-20:
                J2 = 1.0e-20
            dJ2 = J2Mat @ stress
            ddJ2 = J2Mat

            # Yield function
            strain_rate = lMult / dt
            f = np.sqrt(3.0) * np.sqrt(J2) - eta * strain_rate ** (1.0 / n_exp)
            df_ds =np.sqrt(3.0) / (2.0 * np.sqrt(J2)) * dJ2
            ddf_ds = np.sqrt(3.0) / (2.0 * np.sqrt(J2)) * ddJ2 - np.sqrt(3.0) / (4.0 * J2 * np.sqrt(J2)) * np.outer(dJ2, dJ2)
            df_dl = - eta / n_exp * strain_rate ** (1.0 / n_exp - 1.0) * (1.0 / dt)
            ddf_ds = ddf_ds  # Neglect second derivative for simplicity
            
            F = np.zeros((7,))
            K = np.zeros((7, 7))
            # Residuals
            for i in range(6):
                F[i] = state[i, 0] - e[i, 0] + e_pOld[i, 0] + lMult * df_ds[i] # strains: e_el - (e - e_pOld - lMult*df_ds) = 0
            F[6] = f # yield function, f=0
            # Tangent
            ddf_de = D_el @ ddf_ds
            df_de = D_el @ df_ds
            for i in range(6):
                for j in range(6):
                    K[i, j] = (1.0 if i == j else 0.0) + lMult * ddf_de[i, j]
                K[i, 6] = df_ds[i]
                K[6, i] = df_de[i]
            K[6, 6] = df_dl
            
            ep_new = np.zeros((6, 1))
            for i in range(6):
                ep_new[i, 0] = e_pOld[i, 0] + lMult * df_ds[i]
            return F, K, ep_new

        @nb.njit(cache=True)
        def _update_plastic_strain_ip(strain_vals, eps_p_old, dt):
            strain_voight = Tensor_To_Voight_Strain(strain_vals.reshape((dim,dim)), dim)
            strain_p_old = Tensor_To_Voight_Strain(eps_p_old.reshape((dim,dim)), dim)
            
            state = np.zeros((7, 1))
            for i in range(6):
                state[i, 0] = strain_voight[i, 0] - strain_p_old[i, 0]  # initial guess: elastic predictor
            
            strainTrial = np.ascontiguousarray(state[0:6, 0])
            stressTrial = D_el @ strainTrial
            J2_Trial = 0.5 * (stressTrial.T @ J2Mat @ stressTrial)
            if J2_Trial> 1.0e-20:
                state[6, 0] = 1.0e-10
            else:
                state[6, 0] = 0.0
            
            converged = False
            max_iters = 250
            tol = 1.0e-3
            tol0 = None
            iters = 0
            while (not converged):
                F, K, _ = ReturnMapping_KF(state, strain_voight, strain_p_old, dt)
                if (not np.isfinite(F).all()) or (not np.isfinite(K).all()):
                    return eps_p_old.copy()
                # Solve for update
                delta = np.linalg.solve(K, -F)
                state[:, 0] += delta
                res_norm = np.linalg.norm(F)
                if tol0 is None:
                    tol0 = res_norm
                res_norm /= tol0
                if res_norm < tol:
                    converged = True
                iters += 1
                if iters >= max_iters:
                    print("Warning: Return mapping did not converge within max iterations.")
                    break
                
            _, _, ep_new = ReturnMapping_KF(state, strain_voight, strain_p_old, dt)
            ep_new_tensor = Voight_To_Tensor_Strain(ep_new, dim)
            return ep_new_tensor

        @nb.njit(cache=True)
        def _update_plastic_strain(strain_vals, eps_p_old, eps_p_new, dt):
            ncomp = dim * dim
            n_ips = strain_vals.shape[0] // ncomp
            for ip in range(n_ips):
                strain_ip = np.zeros((dim, dim))
                eps_p_old_ip = np.zeros((dim, dim))
                base = ip * ncomp
                for i in range(dim):
                    for j in range(dim):
                        idx = base + i * dim + j
                        strain_ip[i, j] = strain_vals[idx]
                        eps_p_old_ip[i, j] = eps_p_old[idx]
                ep_new_tensor = _update_plastic_strain_ip(strain_ip, eps_p_old_ip, dt)
                for i in range(dim):
                    for j in range(dim):
                        idx = base + i * dim + j
                        eps_p_new[idx] = ep_new_tensor[i, j]

        self._numba_update_plastic = _update_plastic_strain

    def _update_plastic_fields(self, strain_expr) -> None:
        interp_points = self.Totalstrain.function_space.element.interpolation_points
        expr = fem.Expression(strain_expr, interp_points)
        self.Totalstrain.interpolate(expr)
        self.Totalstrain.x.scatter_forward()
        self._numba_update_plastic(
            self.Totalstrain.x.array,
            self.strain_pOld.x.array,
            self.strain_p.x.array,
            float(self.params.dt),
        )
        self.strain_p.x.scatter_forward()
    
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
        v.x.scatter_forward()
        a.x.scatter_forward()
        
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
        self.d_rho = conditional(lt(self.mesh.Get_Field("phasefield"), 0.5), 1.0, 0.0)
        
        # Crack driving force
        strain = nabla_s(self.mesh.Get_Field("u", step=self.My_Step)) - self.strain_p
        
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
        
        # Update plastic strain history
        self.strain_pOld.x.array[:] = self.strain_p.x.array[:]
        self.strain_pOld.x.scatter_forward()
        
        # Update pressure field for post-processing
        # Evaluate pressure as p = -λ * tr(ε) where ε is the strain tensor
        u = self.mesh.Get_Field("u", step=self.My_Step)
        pressure_expr = -self.lame * tr(nabla_s(u)-self.strain_p)
        self.mesh.Set_From_Expression("p", pressure_expr)
        
    def reset_step(self):
        self.mesh.Copy_Field("uOld", "u")
        self.mesh.Copy_Field("vOld", "v")
        self.mesh.Copy_Field("aOld", "a")
        
        pass
