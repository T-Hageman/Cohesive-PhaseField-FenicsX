from ufl import div, inner, dot, dx, TrialFunction, as_tensor, Identity, sqrt
from dolfinx import fem

from Models.BaseModel import BaseModel
from Utils.maths_utils import nabla_s, macPlus, macMinus, eig, Tensor_To_Mandel_UFL, J2Matrix
from Utils.maths_utils import StiffnessMatrix, DevMat, Tensor_To_Mandel, sign_nb
from Mesh.Mesh import Mesh
from Params import Params
from Models.ModelEnums import ModelType
from ufl import derivative, tr, conditional, lt
import ufl
from Utils.mpi_utils import mprint

class Cohesive_LinearElastic(BaseModel):

    def __init__(self, name, params: Params, mesh: Mesh):
        super().__init__(name, params, mesh)
        self.type = ModelType.SOLID_COHESIVE_LINEARELASTIC
        self.dim = params.dim
        
        mesh.Add_Vector_Field("u", "Displacement [m]", "CG", params.Field_Orders["u"], self.dim)
        mesh.Add_Vector_Field("uOld", "Old Displacement [m]", "CG", params.Field_Orders["u"], self.dim)
        mesh.Add_Vector_Field("v", "Velocity [m/s]", "CG", params.Field_Orders["u"], self.dim)
        mesh.Add_Vector_Field("vOld", "Old Velocity [m/s]", "CG", params.Field_Orders["u"], self.dim)
        mesh.Add_Vector_Field("a", "Acceleration [m/s²]", "CG", params.Field_Orders["u"], self.dim)
        mesh.Add_Vector_Field("aOld", "Old Acceleration [m/s²]", "CG", params.Field_Orders["u"], self.dim)
        
        mesh.Add_Quadrature_Field("eta", "eta field [-]", (6,1))
        mesh.Add_Quadrature_Field("etaOld", "Old eta field [-]", (6,1))
        mesh.Add_Quadrature_Field("sigma", "Stress field [Pa]", (6,1))
        mesh.Add_Quadrature_Field("dsigma_dStrainVoigt", "dsigma/dStrainVoigt field [-]", (6,6))

        self.Step_u = params.Solution_Steps["u"]
        
        # Register test functions (actual test functions will be retrieved when needed)
        mesh.Add_Vector_TestFunction("u", params.dim, self.Step_u)
        
        # material parameters
        self.material = self.params.Models[self.name]["Material"]
        self.bulk = self.params.Materials[self.material]["Bulk"]
        self.lame = self.params.Materials[self.material]["Lame"]
        self.shear = self.params.Materials[self.material]["Shear"]
        self.rho = self.params.Materials[self.material]["Density"]
        self.gravity = self.params.gravity
        self.damping = self.params.Materials[self.material].get("LE_Damping", 0.0)
        
        self.NM_Beta = self.params.Newmark_Beta
        self.NM_Gamma = self.params.Newmark_Gamma
        
        # Check if damage model is specified
        mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", ())
        mesh.Add_Quadrature_Field("PsiField", "Crack_Damage_Field [J/m³]", ())
        mesh.Add_Quadrature_Field("total_strains", "Total Strains [-]", (6,1))
        mesh.Add_Quadrature_Field("dam_elastic", "Elastic damage function [-]", ())
        mesh.Add_Quadrature_Field("LambdaField", "Internal variable field [-]", (2,1))
        mesh.Add_Quadrature_Field("LambdaFieldOld", "Old Internal variable field [-]", (2,1))

        self.DrivingForce_Type = self.params.Models[self.name].get("CrackDrivingForce", "Stress-Based")
        self.Gc = self.params.Materials[self.material]["Gc"]
        self.ell = self.params.Materials[self.material]["ell"]
        self.f_t = self.params.Materials[self.material]["f_t"]
        self.f_s = self.params.Materials[self.material]["f_s"]
        self.kappa = self.params.Materials[self.material].get("kappa", 1e-3)
        self.failureType = self.params.Models[self.name].get("FailureType", "r1")
        self.Inertia = self.params.Materials[self.material].get("Inertia", True)
        self.DP_eRef = self.params.Materials[self.material].get("DP_eRef", 1e12)

        self.d_e = 1.0  # No damage
        self.d_rho = 1.0  # No damage
        
        self.trace_i = ufl.as_matrix([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        self.devStrain_mat = ufl.as_matrix(J2Matrix("strain"))
        
        # Cache for compiled expressions (will be initialized on first use)
        self._strain_expr = None
        self._damage_expr = None
        
        self._setup_numba_kernel()
        
    
    def assemble_KF(self, step: int) -> None:
        """
        Assemble the global stiffness matrix and force vector for the linear elastic model.
        """
        
        if (step != self.Step_u):
            return
        
        self.Update_PhaseField_Fields()

        # Get fields - mesh automatically returns mixed space version if applicable
        u = self.mesh.Get_Field("u", step=self.Step_u)
        v = self.mesh.Get_Field("v")
        a = self.mesh.Get_Field("a")
        uOld = self.mesh.Get_Field("uOld")
        vOld = self.mesh.Get_Field("vOld")
        aOld = self.mesh.Get_Field("aOld")
        
        dv_du, da_du = self.Newmark_Update(self.mesh.fields["u"], v, a, uOld, vOld, aOld, self.params.dt)
        
        u_t = self.mesh.Get_TestFunction("u", self.Step_u)
        u_tr = self.mesh.Get_Trial_Function("u", self.Step_u)
        
        strain0_t = Tensor_To_Mandel_UFL(nabla_s(u_t), self.dim)
        strain0_tr = Tensor_To_Mandel_UFL(nabla_s(u_tr), self.dim)
        
        # Get quadrature fields
        stress_q = self.mesh.Get_Field("sigma")
        dsigma_dStrainVoigt_q = self.mesh.Get_Field("dsigma_dStrainVoigt")

        # Momentum Balance: Stresses
        stress_vec = ufl.as_vector([stress_q[i, 0] for i in range(6)])
        F = inner(strain0_t, stress_vec) * dx
        K = dot(strain0_t, dot(dsigma_dStrainVoigt_q, strain0_tr)) * dx
        
        # Inertia terms
        if (self.rho > 0.0 and self.Inertia):
            F += self.rho * inner(u_t, a) * dx
            K += self.rho * inner(u_t, da_du * u_tr) * dx
        
        # Damping terms
        if self.damping > 0.0 and self.rho > 0.0 and self.Inertia:
            F += self.damping * self.rho * inner(u_t, v) * dx
            K += self.damping * self.rho * inner(u_t, dv_du * u_tr) * dx
            
        # Body forces (gravity)
        if (self.gravity != 0.0 and self.rho > 0.0):
            F += -self.rho*u_t[1]*self.gravity*dx #gravity

        return K, F
    
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        for field_name in ("u", "uOld", "v", "vOld", "a", "aOld", "eta", "etaOld", "ElasticEnergy", "PsiField", "LambdaField", "LambdaFieldOld"):
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
        # Damage functions
        phi_lim = conditional(lt(self.mesh.Get_Field("phasefield"), 1.0), self.mesh.Get_Field("phasefield"), 1.0)
        phi_lim = conditional(lt(phi_lim, 0.0), 0.0, phi_lim)
        self.d_e = (1-self.kappa)*(1.0 - phi_lim)**2 + self.kappa
        self.d_rho = conditional(lt(self.mesh.Get_Field("phasefield"), 0.5), 1.0, 0.0)
        
        # Pi Field
        self.Update_Pi_Fields()
        
    def Update_Pi_Fields(self) -> None:
        
        # strains to ip's
        Totalstrain = self.mesh.Get_Field("total_strains")
        strain = Tensor_To_Mandel_UFL(nabla_s(self.mesh.Get_Field("u", step=self.Step_u)), self.dim)
        
        # Create cached expression on first call
        if self._strain_expr is None:
            interp_points = Totalstrain.function_space.element.interpolation_points
            self._strain_expr = fem.Expression(strain, interp_points)
        
        Totalstrain.interpolate(self._strain_expr)
        Totalstrain.x.scatter_forward()

        # damage field to ip's
        dam = self.mesh.Get_Field("dam_elastic")

        if self._damage_expr is None:
            interp_points = dam.function_space.element.interpolation_points
            self._damage_expr = fem.Expression(self.d_e, interp_points)
        
        dam.interpolate(self._damage_expr)
        dam.x.scatter_forward()

        self._numba_update_state(
            Totalstrain.x.array,
            self.params.dt,
            dam.x.array,
            self.mesh.Get_Field("sigma").x.array,
            self.mesh.Get_Field("eta").x.array,
            self.mesh.Get_Field("etaOld").x.array,
            self.mesh.Get_Field("dsigma_dStrainVoigt").x.array,
            self.mesh.Get_Field("ElasticEnergy").x.array,
            self.mesh.Get_Field("PsiField").x.array,
            self.mesh.Get_Field("LambdaField").x.array,
            self.mesh.Get_Field("LambdaFieldOld").x.array
        )
        
        self.mesh.Get_Field("sigma").x.scatter_forward()
        self.mesh.Get_Field("eta").x.scatter_forward()
        self.mesh.Get_Field("dsigma_dStrainVoigt").x.scatter_forward()
        self.mesh.Get_Field("ElasticEnergy").x.scatter_forward()
        self.mesh.Get_Field("PsiField").x.scatter_forward()
        self.mesh.Get_Field("LambdaField").x.scatter_forward()
    
    def _setup_numba_kernel(self):
        import numba as nb
        import numpy as np
        import warnings
        from numba.core.errors import NumbaPerformanceWarning
        
        nb.config.DISABLE_PERFORMANCE_WARNINGS = True
        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
        
        bulk = self.bulk
        lame = self.lame
        shear = self.shear
        f_t = self.f_t
        f_s = self.f_s
        DP_eRef = self.DP_eRef
        failureType = self.failureType
        if failureType == "r1" or failureType == "r1TO":# or failureType == "DP":
            numLambdas = 2
        else:
            numLambdas = 1
        
        kappa = self.kappa
        visc_eta = 1e-6*bulk*kappa
        
        I1Vec = np.zeros((1,6), dtype=np.float64)
        I1Vec[0,0] = 1.0
        I1Vec[0,1] = 1.0
        I1Vec[0,2] = 1.0
        
        devMat = DevMat()
        D = StiffnessMatrix(lame, shear)
        lambda_min = 1.0e-9
        
        @nb.njit(cache=True)
        def Get_Energy(strain):
            """
            Args:
                strain: (6, 1)

            Returns
            -------
                E : float, Strain energy.
                dE_dstrain : (6, 1) Vector, Energy gradient (stress).
                ddE_dstrain2 : (6, 6) Matrix, Energy Hessian (lin. el. stiffness matrix).
            """

            # Volumetric
            vol_strain = (I1Vec @ strain)[0, 0]
            E = 0.5 * bulk * vol_strain**2
            dE_dstrain = bulk * I1Vec.T * vol_strain
            ddE_dstrain2 = bulk * I1Vec.T @ I1Vec
            
            # Deviatoric
            E += shear * (strain.T @ devMat @ strain)[0,0]
            dE_dstrain += 2.0 * shear * (devMat @ strain)
            ddE_dstrain2 += 2.0 * shear * devMat
            
            return E, dE_dstrain, ddE_dstrain2
        
        @nb.njit(cache=True)
        def Get_eta(e, l):
            """
            Args:
                e : (6,1) Vector, total strain
                l : (2,1) Vector, internal variables

            Returns
            -------
                eta : (6,1) Vector, fracture strain.
                deta_de : (6,6) Matrix, derivatives of eta w.r.t. strain components.
                deta_dl : (2,6,1) array, derivatives of eta w.r.t. internal variables.
            """

            

            vol_e = 1.0/3.0 * I1Vec.T @ I1Vec @ e
            vol_e_sq = (vol_e @ vol_e.T)[0,0]
            vol_e_norm = np.sqrt(vol_e_sq + 1.0e-20)
            vol_dir = vol_e / vol_e_norm
            dvol_e_de = 1.0/3.0 * I1Vec.T @ I1Vec
            dvol_dir_de = (np.eye(6, dtype=np.float64) / vol_e_norm - (vol_e @ vol_e.T) / (vol_e_norm**3)) @ dvol_e_de
            
            dev_e = devMat @ e
            dev_e_sq = (dev_e.T @ dev_e)[0,0]
            dev_e_norm = np.sqrt(dev_e_sq + 1.0e-20)
            dev_dir = dev_e / dev_e_norm
            ddev_dir_de = devMat / dev_e_norm - (dev_e @ dev_e.T) / (dev_e_norm**3)
            
            e_sq = (e.T @ e)[0,0]
            e_norm = np.sqrt(e_sq + 1.0e-20)
            e_dir = e / e_norm
            de_dir_de = np.eye(6, dtype=np.float64) / e_norm - (e @ e.T) / (e_norm**3)
            
            if numLambdas == 2:
                eta = l[0,0] * vol_dir + l[1,0] * dev_dir
                
                deta_dl = np.empty((2, 6, 1), dtype=np.float64)
                deta_dl[0, :, :] = vol_dir
                deta_dl[1, :, :] = dev_dir
                
                deta_de = 0.0*np.eye(6, dtype=np.float64)
                deta_de += l[0,0] * dvol_dir_de
                deta_de += l[1,0] * ddev_dir_de
                
                d2eta_dlde = np.empty((2,6,6), dtype=np.float64)
                d2eta_dlde[0,:,:] = dvol_dir_de
                d2eta_dlde[1,:,:] = ddev_dir_de
            else:
                if ((I1Vec @ e) >= 0.0):
                    e_dir = e_dir
                    de_dir_de = de_dir_de
                else:
                    e_dir = dev_dir
                    de_dir_de = ddev_dir_de

                eta = l[0,0] * e_dir
                deta_dl = np.empty((1, 6, 1), dtype=np.float64)
                deta_dl[0,:,:] = e_dir
                deta_de = l[0,0] * de_dir_de
                
                d2eta_dlde = np.empty((1,6,6), dtype=np.float64)
                d2eta_dlde[0,:,:] = de_dir_de
            
            return eta, deta_de, deta_dl, d2eta_dlde
        
        @nb.njit(cache=True)
        def Get_FailureSurf(eta, dam, strain):
            """
            Args:
                eta : (6,1) Vector, fracture strain
                dam : float, damage variable

            Returns
            -------
                F : float, failure function value.
                dF_deta : (1,6) Vector, derivative of failure function w.r.t. eta.
            """
            eta_vol = (I1Vec @ eta)[0,0]
            eta_dev_vec = devMat @ eta
            eta_dev_sq = (eta.T @ devMat @ eta)[0,0]
            eta_dev = np.sqrt(eta_dev_sq + 1.0e-20)
            
            F1 = f_t/3.0 * eta_vol
            dF1 = f_t/3.0 * I1Vec
            
            F2 = f_s * np.sqrt(0.5) * eta_dev
            dF2 = f_s * np.sqrt(0.5) * (eta.T @ devMat) / eta_dev
            ddF2 = f_s * np.sqrt(0.5) * (devMat / eta_dev - (eta_dev_vec @ eta_dev_vec.T) / (eta_dev**3))
            
            F1S = (I1Vec @ strain)[0,0]
            
            F = 0.0
            dF_deta = np.zeros((1,6), dtype=np.float64)
            ddF_deta2 = np.zeros((6,6), dtype=np.float64)
            if (failureType == "r1"): 
                if (F1S>=0.0):
                    F       = dam * F1
                    dF_deta = dam * dF1
                    FDamage = F1
                else:
                    F       = -1e6 * F1
                    dF_deta = -1e6 * dF1
                    FDamage = 0.0
                    
                F += dam * F2
                dF_deta += dam * dF2
                ddF_deta2 = dam * ddF2
                FDamage += F2
            elif failureType == "r1TO":
                if (F1S>=0.0):
                    F       = dam * F1   + dam * F2
                    dF_deta = dam * dF1  + dam * dF2
                    ddF_deta2 = dam * ddF2
                    FDamage = F1         + F2
                else:
                    F       = -1e12 * F1   + 1e12*dam*F2
                    dF_deta = -1e12 * dF1  + 1e12*dam*dF2
                    ddF_deta2 =              1e12*dam*ddF2
                    FDamage = 0.0          + 1e12*F2
            elif failureType == "r2":
                if (F1S > 0.0):
                    F       = dam*np.sqrt(F1**2 + F2**2)
                    FDamage =     np.sqrt(F1**2 + F2**2)
                    dF_deta = dam*(F1*dF1 + F2*dF2) / np.sqrt(F1**2 + F2**2)
                    ddF_deta2 = dam*(dF1.T @ dF1 + dF2.T @ dF2 + F2*ddF2 - (F1*dF1 + F2*dF2).T @ (F1*dF1 + F2*dF2) / (F1**2 + F2**2)) / np.sqrt(F1**2 + F2**2)
                else:
                    F = dam*F2
                    FDamage = F2
                    dF_deta = dam*dF2
                    ddF_deta2 = dam*ddF2           
            elif failureType == "DP":
                if (F1S > 0.0):
                    F       = dam*np.sqrt(F1**2 + F2**2)
                    FDamage =     np.sqrt(F1**2 + F2**2)
                    dF_deta = dam*(F1*dF1 + F2*dF2) / np.sqrt(F1**2 + F2**2)
                    ddF_deta2 = dam*(dF1.T @ dF1 + dF2.T @ dF2 + F2*ddF2 - (F1*dF1 + F2*dF2).T @ (F1*dF1 + F2*dF2) / (F1**2 + F2**2)) / np.sqrt(F1**2 + F2**2)
                elif (F1S > -99.0*DP_eRef):
                    F = -1e12*F1 + dam*F2*(1-F1S/DP_eRef)
                    FDamage = F2*(1-F1S/DP_eRef)
                    dF_deta = -1e12*dF1 + dam*dF2*(1-F1S/DP_eRef)
                    ddF_deta2 = dam*ddF2*(1-F1S/DP_eRef)
                else:
                    F = -1e12*F1 + 100.0*dam*F2
                    FDamage = 100.0*F2
                    dF_deta = -1e12*dF1 + 100.0*dam*dF2
                    ddF_deta2 = 100.0*dam*ddF2
            elif failureType == "ShearOnly":
                F = dam * F2 + 1e6*F1
                FDamage = f_s * eta_dev
                dF_deta = dam * dF2 + 1e6*dF1
                ddF_deta2 = dam * ddF2
            else:
                raise ValueError("Failure type not recognized: {}".format(failureType))
                
            return F, dF_deta, ddF_deta2, FDamage
        
        @nb.njit(cache=True)
        def schur_solve(K, F):
            """Solve K @ x = F using Schur complement, exploiting A = K[0:6,0:6] = I.
            Returns x = K^{-1} F  (caller negates as needed).
            """
            B    = K[0:6, 6:6+numLambdas]              # (6, nL)
            Cblk = K[6:6+numLambdas, 0:6]              # (nL, 6)
            Dblk = K[6:6+numLambdas, 6:6+numLambdas]  # (nL, nL)
            F_sig = F[0:6, :]                           # (6, 1)
            F_lam = F[6:6+numLambdas, :]               # (nL, 1)

            S   = Dblk - Cblk @ B       # (nL, nL)
            rhs = F_lam - Cblk @ F_sig  # (nL, 1)

            x = np.zeros((6 + numLambdas, 1), dtype=np.float64)
            if numLambdas == 1:
                x[6, 0] = rhs[0, 0] / S[0, 0]
            else:  # numLambdas == 2
                det_S = S[0,0]*S[1,1] - S[0,1]*S[1,0]
                x[6, 0] = ( S[1,1]*rhs[0,0] - S[0,1]*rhs[1,0]) / det_S
                x[7, 0] = (-S[1,0]*rhs[0,0] + S[0,0]*rhs[1,0]) / det_S

            # x_sigma = F_sigma - B @ x_lambda
            x_lam = x[6:6+numLambdas, :]
            for i in range(6):
                x[i, 0] = F_sig[i, 0]
                for j in range(numLambdas):
                    x[i, 0] -= B[i, j] * x_lam[j, 0]
            return x

        @nb.njit(cache=True)
        def schur_inv_block_00(K):
            """Compute K^{-1}[0:6, 0:6] using Schur complement, exploiting A = I.
            K^{-1}_{00} = I + B @ S^{-1} @ C_blk.
            """
            B    = K[0:6, 6:6+numLambdas]
            Cblk = K[6:6+numLambdas, 0:6]
            Dblk = K[6:6+numLambdas, 6:6+numLambdas]

            S = Dblk - Cblk @ B  # (nL, nL)

            if numLambdas == 1:
                Sinv_C = Cblk / S[0, 0]           # (1, 6)
            else:  # numLambdas == 2
                det_S = S[0,0]*S[1,1] - S[0,1]*S[1,0]
                Sinv = np.zeros((2, 2), dtype=np.float64)
                Sinv[0, 0] =  S[1,1] / det_S
                Sinv[0, 1] = -S[0,1] / det_S
                Sinv[1, 0] = -S[1,0] / det_S
                Sinv[1, 1] =  S[0,0] / det_S
                Sinv_C = Sinv @ Cblk              # (2, 6)

            return np.eye(6, dtype=np.float64) + B @ Sinv_C

        @nb.njit(cache=True)
        def ReturnMapping_KF(state, e, dt, dam, lOld, visc_extra=0.0):
            l = np.zeros((numLambdas,1), dtype=np.float64)
            stress = state[0:6,0]
            for i in range(numLambdas):
                l[i,0] = state[6 + i, 0]

            eta, deta_de, deta_dl, d2eta_dlde = Get_eta(e, l)
            F, dF_deta, ddF_deta2, _ = Get_FailureSurf(eta, dam, e)
            strain = e - eta
            E, dE_dstrain, ddE_dstrain2 = Get_Energy(strain)
            
            
            F = np.zeros((6 + numLambdas,1), dtype=np.float64)    # 0-5: stresses, 6-7: eta multipliers
            K = np.zeros((6 + numLambdas,6 + numLambdas), dtype=np.float64)
            C = np.zeros((6,6), dtype=np.float64)
            
            # stress - dPsi/dstrain = 0
            F[0:6,0] = state[0:6,0] - dE_dstrain[0:6,0]
            K[0:6,0:6] = np.eye(6, dtype=np.float64)
            C = - ddE_dstrain2 @ (np.eye(6, dtype=np.float64))
            for i in range(numLambdas):
                K[0:6,6+i] = (ddE_dstrain2 @ (deta_dl[i,:,:]))[:, 0]
            C += ddE_dstrain2 @ deta_de
            
            sc = bulk/dam
            # strength surface, dPsi/deta + dPi/deta = 0
            for i in range(numLambdas):
                # Derivatives
                deta_dli = deta_dl[i, :, :]
                dF_dl = (dF_deta @ deta_dli)[0, 0]
                ddF_dll = (deta_dli.T @ ddF_deta2 @ deta_dli)[0, 0]
                dE_dl = (-stress.T @ deta_dli)[0]
                dE_dldstress = -deta_dli.T
            
                if (dF_dl + dE_dl >= 0.0): # l_1 = 0
                    F[6+i,0]   = sc*(state[6 + i,0]-lambda_min)
                    K[6+i,6+i] = sc*1.0
                else: 
                    ve = visc_eta + visc_extra
                    F[6+i,0]   = sc*(dF_dl + dE_dl  + ve * state[6 + i,0] * sign_nb(state[6 + i,0]))
                    K[6+i,0:6] = sc*         dE_dldstress[0, :]
                    K[6+i,6+i] = sc*(ddF_dll        + ve * sign_nb(state[6 + i,0]))
                
            return K, F, C
        
        @nb.njit(cache=True)
        def Compute_Results(strains, state, dt, dam, lOld, visc_extra=0.0):
            K, _, C = ReturnMapping_KF(state, strains, dt, dam, lOld, visc_extra)
            l = np.zeros((numLambdas, 1), dtype=np.float64)
            for i in range(numLambdas):
                l[i, 0] = state[6 + i, 0]
            eta, _, _, _ = Get_eta(strains, l)
            Kinv_00 = schur_inv_block_00(K)
            
            # Stresses
            stress = np.zeros((6,1), dtype=np.float64)
            for i in range(6):
                stress[i,0] = state[i,0]
                
            # Tangent stiffness
            dsigma_dstrain = -Kinv_00 @ C
            
            # Elastic Energy
            strain_el = strains - eta
            E_el, _, _ = Get_Energy(strain_el)
            
            # Pi-Field
            _, _, _, F = Get_FailureSurf(eta, dam, strains)
            E_Psi = F
            
            return stress, eta, dsigma_dstrain, E_el, E_Psi, l
            
            
        
        @nb.njit(cache=True)
        def _update_ip_state(strains, eta, eta_old, dt, dam, LambdaField, LOld):
            
            LS_Max = 1.0
            visc_extra = 0.0
            for attempt in range(8):
                try:
                    stress, eta_out, dsigma_dstrain, E_el, E_Psi, L = _update_ip_state_internal(
                        strains, eta, eta_old, dt, dam, LambdaField, LOld, LS_Max, visc_extra)
                except:
                    visc_extra = max(visc_extra * 10.0, 1.0e-6 * bulk)
                    continue
                
                # Check for NaN/Inf in stress and tangent
                has_nan = False
                for i in range(6):
                    if np.isnan(stress[i, 0]) or np.isinf(stress[i, 0]):
                        has_nan = True
                        break
                if not has_nan:
                    for i in range(6):
                        for j in range(6):
                            if np.isnan(dsigma_dstrain[i, j]) or np.isinf(dsigma_dstrain[i, j]):
                                has_nan = True
                                break
                        if has_nan:
                            break
                
                if not has_nan:
                    return stress, eta_out, dsigma_dstrain, E_el, E_Psi, L
                
                # NaN detected — increase viscosity and retry
                visc_extra = max(visc_extra * 10.0, 1.0e-6 * bulk)
            
            # All attempts exhausted — fall back to previous converged state
            F, dF, _, _ = Get_FailureSurf(eta, dam, strains)
            E, dE, ddE = Get_Energy(strains - eta)
            print("WARNING: Return mapping produced NaN after all retries, using previous time step values. dam =", dam)
            print("\tStrains:", strains.T)
            print("\tPrevious eta:", eta_old.T)
            print("\tLambdaField:", LambdaField.T)
            print("\tF:", F, "dF:", dF)
            print("\tstress:", dE)
            
            
            
            stress_fb = D @ (strains - eta_old)
            strain_el_fb = strains - eta_old
            E_el_fb, _, _ = Get_Energy(strain_el_fb)
            _, _, _, F_fb = Get_FailureSurf(eta_old, dam, strains)
            return stress_fb, eta_old.copy(), D.copy(), E_el_fb, F_fb, LOld.copy()
        
        @nb.njit(cache=True)
        def _update_ip_state_internal(strains, eta, eta_old, dt, dam, LambdaField, LOld, LS_Max, visc_extra=0.0):
            # Start from elastic trial state: eta=0, sigma = D @ strains
            stress0 = D @ strains
            
            state = np.zeros((6 + numLambdas,1), dtype=np.float64)
            for i in range(6):
                state[i,0] = stress0[i,0]
            for i in range(numLambdas):
                state[6 + i,0] = lambda_min
                
            converged = False
            max_iters = 100
            tol = 1e-9
            tol0 = None
            iter = 0
            K, F, _ = ReturnMapping_KF(state, strains, dt, dam, LOld, visc_extra)
            while (not converged):
                delta = -schur_solve(K, F)
                if False:
                    state[:, 0] += delta[:, 0]
                else:
                    e0 = (delta.T @ F)[0, 0]
                    state[:, 0] += delta[:, 0]
                    _, F, _ = ReturnMapping_KF(state, strains, dt, dam, LOld, visc_extra)
                    e1 = (delta.T @ F)[0, 0]
                    if (abs(e0-e1) < 1.0e-20):
                        ls = 1.0
                    else:
                        ls = -e0/(e1 - e0)
                        ls = min(max(ls, 0.1), LS_Max)
                    state[:, 0] += (ls-1.0)*delta[:, 0]
                
                for i in range(numLambdas):
                    if (state[6 + i,0] < lambda_min):
                        state[6 + i,0] = lambda_min
                
                K, F, _ = ReturnMapping_KF(state, strains, dt, dam, LOld, visc_extra)
                res_norm = np.linalg.norm(F)
                if tol0 is None:
                    tol0 = res_norm + 1.0e0/tol
                res_norm /= tol0
                if res_norm < tol:
                    converged = True
                
                iter += 1   
                if (iter > max_iters):
                    if (res_norm >= 1e-3):
                        if numLambdas == 2:
                            if (state[6,0] > 10.0*lambda_min or state[7,0] > 10.0*lambda_min):
                                # print(
                                #     "Warning: Pi Return Mapping did not converge, l1 =", state[6,0],
                                #     "l2 =", state[7,0], "res_norm =", res_norm
                                # )
                                pass
                        else:
                            if (state[6,0] > 10.0*lambda_min):
                                # print(
                                #     "Warning: Pi Return Mapping did not converge, l =", state[6,0],
                                #     "res_norm =", res_norm
                                # )
                                pass
                    break
                
            stress, eta, dsigma_dstrain, E_el, E_Psi, L = Compute_Results(strains, state, dt, dam, LOld, visc_extra)
            return stress, eta, dsigma_dstrain, E_el, E_Psi, L
            
        @nb.njit(cache=True)   
        def _update_state(strains, dt, dam, stress, eta, eta_old, dsigma_dstrain, E_el, E_Psi, LambdaField, LambdaFieldOld):
            n_ips = strains.shape[0] // 6
            for ip in range(n_ips):
                strain_ip = np.zeros((6, 1))
                eta_ip = np.zeros((6,1))  
                eta_old_ip = np.zeros((6,1))     
                base_voight = ip * 6
                L = np.zeros((numLambdas,1))
                LOld = np.zeros((numLambdas,1))
                for i in range(6):
                    strain_ip[i, 0]  = strains[base_voight + i]
                    eta_ip[i, 0] = eta[base_voight + i]
                    eta_old_ip[i, 0] = eta_old[base_voight + i]
                d = dam[ip]
                if (d < kappa):
                    d = kappa
                if (d > 1.0):
                    d = 1.0
                for i in range(numLambdas):
                    L[i,0] = LambdaField[ip * 2 + i]
                    LOld[i,0] = LambdaFieldOld[ip * 2 + i]
            
                # Confirm None of the inputs are NaN or Inf before calling the update function
                has_nan = False
                for i in range(6):
                    if np.isnan(strain_ip[i, 0]) or np.isinf(strain_ip[i, 0]):
                        has_nan = True
                        break
                    if np.isnan(eta_ip[i, 0]) or np.isinf(eta_ip[i, 0]):
                        has_nan = True
                        break
                    if np.isnan(eta_old_ip[i, 0]) or np.isinf(eta_old_ip[i, 0]):
                        has_nan = True
                        break
                for i in range(numLambdas):
                    if np.isnan(L[i, 0]) or np.isinf(L[i, 0]):
                        has_nan = True
                        break
                    if np.isnan(LOld[i, 0]) or np.isinf(LOld[i, 0]):
                        has_nan = True
                        break
                
                if has_nan:
                    print("WARNING: NaN detected in input fields, skipping update for this ip.")
                    print("strain_ip:", strain_ip.flatten())
                    print("eta_ip:", eta_ip.flatten())
                    print("eta_old_ip:", eta_old_ip.flatten())
                    print("L:", L.flatten())
                    raise ValueError("NaN detected in input fields of return-mapping scheme.")
            
                stress_ip, eta_ip, dsigma_dstrain_ip, E_el_ip, E_Psi_ip, L = _update_ip_state(strain_ip, eta_ip, eta_old_ip, dt, d, L, LOld)

                E_el[ip] = E_el_ip
                E_Psi[ip] = E_Psi_ip
                for i in range(6):
                    stress[base_voight + i] = stress_ip[i, 0]
                    eta[base_voight + i] = eta_ip[i, 0]
                    for j in range(6):
                        dsigma_dstrain[ip * 36 + i * 6 + j] = dsigma_dstrain_ip[i, j]
                for i in range(numLambdas):
                    LambdaField[ip * 2 + i] = L[i,0]
        
        self._numba_update_state = _update_state


    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """
        self.mesh.Copy_Field("u","u")
        self.Newmark_Update(
            self.mesh.fields["u"],
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
        self.mesh.Copy_Field("eta", "etaOld")
        self.mesh.Copy_Field("LambdaField", "LambdaFieldOld")
        
    def reset_step(self):
        self.mesh.Copy_Field("uOld", "u")
        self.mesh.Copy_Field("vOld", "v")
        self.mesh.Copy_Field("aOld", "a")
        self.mesh.Copy_Field("etaOld", "eta")
        self.mesh.Copy_Field("LambdaFieldOld", "LambdaField")
        
        pass
