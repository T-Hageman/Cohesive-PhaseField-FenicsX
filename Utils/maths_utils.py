import numba as nb
import numpy as np

def nabla_s(field):
    """Symetric gradient operator."""
    from ufl import sym, nabla_grad
    return sym(nabla_grad(field))
    
def macPlus(a):
    """Macaulay Plus operator."""
    from ufl import conditional, gt
    return (a+abs(a))/2
    #return conditional(gt(a, 0.0), a, 0.0)

def dmacPlus(a):
    """Derivative of Macaulay Plus operator."""
    from ufl import conditional, gt
    return conditional(gt(a, 0.0), 1.0, 0.0)

def macMinus(a):
    """Macaulay Minus operator."""
    from ufl import conditional, lt
    return (a - abs(a))/2
    #return conditional(lt(a, 0.0), a, 0.0)

def dmacMinus(a):
    """Derivative of Macaulay Minus operator."""
    from ufl import conditional, lt
    return conditional(lt(a, 0.0), 1.0, 0.0)

def eig(A, dim):
    from ufl import sqrt
    
    if dim == 2:
        s11, s22, s12 = A[0,0], A[1,1], A[0,1]
        s33 = A[2,2]  # Out-of-plane stress (non-zero in plane strain)
        
        # In-plane eigenvalues
        trace_2d = s11 + s22
        det_2d = s11*s22 - s12**2
        discriminant = sqrt(trace_2d**2/4 - det_2d)
        
        eig1 = trace_2d/2 + discriminant
        eig2 = trace_2d/2 - discriminant
        eig3 = s33  # Out-of-plane eigenvalue
        return eig1, eig2, eig3
    else:
        raise NotImplementedError("3D eigenvalues not yet implemented")
    
def ft_to_Gc(f_t, E, ell, PF_formulation="AT2"):
    """Convert tensile strength to fracture toughness."""
    if PF_formulation == "AT2":
        Gc = f_t**2*ell / E * 2.0
    elif PF_formulation == "AT1":
        Gc = f_t**2*ell / E * 8.0/3.0
    else:
        raise ValueError(f"Unknown Phase Field formulation: {PF_formulation}")
    return Gc

def StiffnessMatrix(Lame, Shear):
    import numpy as np
    D_el = np.zeros((6,6))
    D_el[0,0] = Lame + 2.0*Shear
    D_el[0,1] = Lame
    D_el[0,2] = Lame
    D_el[1,0] = Lame
    D_el[1,1] = Lame + 2.0*Shear
    D_el[1,2] = Lame
    D_el[2,0] = Lame
    D_el[2,1] = Lame
    D_el[2,2] = Lame + 2.0*Shear
    D_el[3,3] = Shear
    D_el[4,4] = Shear
    D_el[5,5] = Shear
    return D_el

def J2Matrix(stress_or_strain: str):
    J2Mat = np.zeros((6,6))
    if stress_or_strain == "strain" or stress_or_strain == "Strain":
        J2Mat = np.zeros((6,6))
        J2Mat[0,0] = 2.0/3.0
        J2Mat[0,1] = -1.0/3.0
        J2Mat[0,2] = -1.0/3.0
        J2Mat[1,0] = -1.0/3.0
        J2Mat[1,1] = 2.0/3.0
        J2Mat[1,2] = -1.0/3.0
        J2Mat[2,0] = -1.0/3.0
        J2Mat[2,1] = -1.0/3.0
        J2Mat[2,2] = 2.0/3.0
        J2Mat[3,3] = 1.0/2.0
        J2Mat[4,4] = 1.0/2.0
        J2Mat[5,5] = 1.0/2.0
    elif stress_or_strain == "stress" or stress_or_strain == "Stress":
        J2Mat = np.zeros((6,6))
        J2Mat[0,0] = 2.0/3.0
        J2Mat[0,1] = -1.0/3.0
        J2Mat[0,2] = -1.0/3.0
        J2Mat[1,0] = -1.0/3.0
        J2Mat[1,1] = 2.0/3.0
        J2Mat[1,2] = -1.0/3.0
        J2Mat[2,0] = -1.0/3.0
        J2Mat[2,1] = -1.0/3.0
        J2Mat[2,2] = 2.0/3.0
        J2Mat[3,3] = 2.0
        J2Mat[4,4] = 2.0
        J2Mat[5,5] = 2.0
    else:
        raise ValueError(f"Unknown stress_or_strain type: {stress_or_strain}")
    
    return J2Mat

def DevMat():
    Dev = np.zeros((6,6))
    Dev[0,0] = 2.0/3.0
    Dev[0,1] = -1.0/3.0
    Dev[0,2] = -1.0/3.0
    Dev[1,0] = -1.0/3.0
    Dev[1,1] = 2.0/3.0
    Dev[1,2] = -1.0/3.0
    Dev[2,0] = -1.0/3.0
    Dev[2,1] = -1.0/3.0
    Dev[2,2] = 2.0/3.0
    Dev[3,3] = 1.0
    Dev[4,4] = 1.0
    Dev[5,5] = 1.0
    return Dev

def VoigtCorrectMat():
    # used for contractions of strains:strains
    C = np.zeros((6,6))
    C[0,0] = 1.0
    C[1,1] = 1.0
    C[2,2] = 1.0
    C[3,3] = 0.5
    C[4,4] = 0.5
    C[5,5] = 0.5
    return C

@nb.njit(cache=True)
def Tensor_To_Mandel(tensor, dim):
    mandel = np.zeros((6,1))
    mandel[0] = tensor[0,0]
    mandel[1] = tensor[1,1]
    if dim == 3:
        mandel[2] = tensor[2,2]
        mandel[3] = np.sqrt(2.0) * tensor[1,2]
        mandel[4] = np.sqrt(2.0) * tensor[0,2]
    mandel[5] = np.sqrt(2.0) * tensor[0,1]
    return mandel

@nb.njit(cache=True)
def Mandel_To_Tensor(mandel, dim):
    tensor = np.zeros((dim,dim))
    tensor[0,0] = mandel[0,0]
    tensor[1,1] = mandel[1,0]
    if dim == 3:
        tensor[2,2] = mandel[2,0]
        tensor[1,2] = mandel[3,0] / np.sqrt(2.0)
        tensor[2,1] = mandel[3,0] / np.sqrt(2.0)
        tensor[0,2] = mandel[4,0] / np.sqrt(2.0)
        tensor[2,0] = mandel[4,0] / np.sqrt(2.0)
    tensor[0,1] = mandel[5,0] / np.sqrt(2.0)
    tensor[1,0] = mandel[5,0] / np.sqrt(2.0)
    return tensor

def Tensor_To_Mandel_UFL(tensor, dim):
    from ufl import as_vector, sqrt
    
    if dim == 3:
        mandel = [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            sqrt(2.0) * tensor[1, 2],
            sqrt(2.0) * tensor[0, 2],
            sqrt(2.0) * tensor[0, 1],
        ]
    else:
        mandel = [
            tensor[0, 0],
            tensor[1, 1],
            0.0,
            0.0,
            0.0,
            sqrt(2.0) * tensor[0, 1],
        ]
    return as_vector(mandel)

def Mandel_To_Tensor_UFL(mandel, dim):
    from ufl import as_tensor, sqrt
    
    if dim == 3:
        tensor = [
            [mandel[0]      , mandel[5] / sqrt(2.0), mandel[4] / sqrt(2.0)],
            [mandel[5] / sqrt(2.0), mandel[1]      , mandel[3] / sqrt(2.0)],
            [mandel[4] / sqrt(2.0), mandel[3] / sqrt(2.0), mandel[2]      ],
        ]
    else:
        tensor = [
            [mandel[0]      , mandel[5] / sqrt(2.0)],
            [mandel[5] / sqrt(2.0), mandel[1]      ],
        ]
    return as_tensor(tensor)

@nb.njit(cache=True)
def Tensor_To_Voight_Strain(strain_tensor, dim):
    strain_voight = np.zeros((6,1))
    strain_voight[0] = strain_tensor[0,0]
    strain_voight[1] = strain_tensor[1,1]
    if dim == 3:
        strain_voight[2] = strain_tensor[2,2]
        strain_voight[3] = 2.0*strain_tensor[1,2]
        strain_voight[4] = 2.0*strain_tensor[0,2]
    strain_voight[5] = 2.0*strain_tensor[0,1]
    return strain_voight

@nb.njit(cache=True)
def Voight_To_Tensor_Strain(strain_voight, dim):
    strain_tensor = np.zeros((dim,dim))
    strain_tensor[0,0] = strain_voight[0,0]
    strain_tensor[1,1] = strain_voight[1,0]
    if dim == 3:
        strain_tensor[2,2] = strain_voight[2,0]
        strain_tensor[1,2] = strain_voight[3,0] / 2.0
        strain_tensor[2,1] = strain_voight[3,0] / 2.0
        strain_tensor[0,2] = strain_voight[4,0] / 2.0
        strain_tensor[2,0] = strain_voight[4,0] / 2.0
    strain_tensor[0,1] = strain_voight[5,0] / 2.0
    strain_tensor[1,0] = strain_voight[5,0] / 2.0
    return strain_tensor

def Tensor_To_Voight_Strain_UFL(strain_tensor, dim):
    from ufl import as_vector
    
    if dim == 3:
        strain_voight = [
            strain_tensor[0, 0],
            strain_tensor[1, 1],
            strain_tensor[2, 2],
            2.0 * strain_tensor[1, 2],
            2.0 * strain_tensor[0, 2],
            2.0 * strain_tensor[0, 1],
        ]
    else:
        strain_voight = [
            strain_tensor[0, 0],
            strain_tensor[1, 1],
            0.0,
            0.0,
            0.0,
            2.0 * strain_tensor[0, 1],
        ]
    return as_vector(strain_voight)

def Voight_To_Tensor_Strain_UFL(strain_voight, dim):
    from ufl import as_tensor
    
    if dim == 3:
        strain_tensor = [
            [strain_voight[0]      , strain_voight[5] / 2.0, strain_voight[4] / 2.0],
            [strain_voight[5] / 2.0, strain_voight[1]      , strain_voight[3] / 2.0],
            [strain_voight[4] / 2.0, strain_voight[3] / 2.0, strain_voight[2]      ],
        ]
    else:
        strain_tensor = [
            [strain_voight[0]      , strain_voight[5] / 2.0],
            [strain_voight[5] / 2.0, strain_voight[1]      ],
        ]
    return as_tensor(strain_tensor)

@nb.njit(cache=True)
def Tensor_To_Voight_Stress(stress_tensor, dim):
    stress_voight = np.zeros((6,1))
    stress_voight[0] = stress_tensor[0,0]
    stress_voight[1] = stress_tensor[1,1]
    if dim == 3:
        stress_voight[2] = stress_tensor[2,2]
        stress_voight[3] = stress_tensor[1,2]
        stress_voight[4] = stress_tensor[0,2]
    stress_voight[5] = stress_tensor[0,1]
    return stress_voight

@nb.njit(cache=True)
def sign_nb(a):
    if a > 0.0:
        return 1.0
    elif a < 0.0:
        return -1.0
    else:
        return 0.0

@nb.njit(cache=True)
def Voight_To_Tensor_Stress(stress_voight, dim):
    stress_tensor = np.zeros((dim,dim))
    stress_tensor[0,0] = stress_voight[0,0]
    stress_tensor[1,1] = stress_voight[1,0]
    if dim == 3:
        stress_tensor[2,2] = stress_voight[2,0]
        stress_tensor[1,2] = stress_voight[3,0]
        stress_tensor[2,1] = stress_voight[3,0]
        stress_tensor[0,2] = stress_voight[4,0]
        stress_tensor[2,0] = stress_voight[4,0]
    stress_tensor[0,1] = stress_voight[5,0]
    stress_tensor[1,0] = stress_voight[5,0]
    return stress_tensor

def Tensor_To_Voight_Stress_UFL(stress_tensor, dim):
    from ufl import as_vector
    
    if dim == 3:
        stress_voight = [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
            stress_tensor[0, 1],
        ]
    else:
        stress_voight = [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            0.0,
            0.0,
            0.0,
            stress_tensor[0, 1],
        ]
    return as_vector(stress_voight)

def Voight_To_Tensor_Stress_UFL(stress_voight, dim):
    from ufl import as_tensor
    
    if dim == 3:
        stress_tensor = [
            [stress_voight[0]      , stress_voight[5] , stress_voight[4] ],
            [stress_voight[5]      , stress_voight[1] , stress_voight[3] ],
            [stress_voight[4]      , stress_voight[3] , stress_voight[2] ],
        ]
    else:
        stress_tensor = [
            [stress_voight[0]      , stress_voight[5] ],
            [stress_voight[5]      , stress_voight[1] ],
        ]
    return as_tensor(stress_tensor)