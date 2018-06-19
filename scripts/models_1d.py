"""
"""
import numpy as np
import scipy.integrate


def U0_sym(x):
    return 5.0 * (x**2 - 1.)**2


def dU0_dx_sym(x):
    return 20.0 * x * (x**2 - 1.)


def U0_asym(x):
    return 5.0*x**4 - 10.0*x**2 + 3*x


def dU0_dx_asym(x):
    return 20.0*x**3 - 20.0*x + 3.0


#-----


def V(x, ks, L):
    return 0.5 * ks * (x - L)**2


def dV_dx(x, ks, L):
    return ks * (x - L)


def V_MMTK(x, ks, L):
    """
    unfortunately MMTK harmonic potenital is defined as k*(x-l)**2 (without a factor 1/2)
    :param x:
    :param ks:
    :param L:
    :return:
    """
    return ks * (x - L) ** 2

#-----

def U_sym(x, ks, L):
    return U0_sym(x) + V(x, ks, L)


def dU_dx_sym(x, ks, L):
    return dU0_dx_sym(x) + dV_dx(x, ks, L)


def U_asym(x, ks, L):
    return U0_asym(x) + V(x, ks, L)


def dU_dx_asym(x, ks, L):
    return dU0_dx_asym(x) + dV_dx(x, ks, L)

#-------

def numerical_df_t(U, ks, lambda_F, limit=5.):
    """
    :param U: function, potential energy function
    :param ks: float, force constant passed to U
    :param lambda_F: 1d ndarray, scheduled protocol
    :param limit: float, integration limit
    :return: ndarray, free energy
    """
    steps = lambda_F.shape[0]
    df_t_num = np.zeros([steps], dtype=float)

    for step in range(steps):
        tmp_func = lambda x: np.exp( -U(x, ks, lambda_F[step]) )
        df_t_num[step], err = -np.log(scipy.integrate.quad(tmp_func, lambda_F[step] - limit, lambda_F[step] + limit ) )

    df_t_num -= df_t_num[0]
    return df_t_num

#---------


def symmetrize_lambda(lambda_t):
    """
    :param lambda_t: 1d ndarray, the scheduled protocol
    :return: 1d ndarray
    """
    r_lambda = lambda_t[::-1]
    return np.append(lambda_t, r_lambda[1:])


def switching(ks, lambda_t, equil_steps, niterations, dt, U, dU_dx):
    """
    :param ks: float, force constant
    :param lambda_t: 1d ndarray, the scheduled protocol
    :param equil_steps: int, number of equilibration steps
    :param niterations: int, number of production steps
    :param dt: float, time step
    :param U: function, the potential energy function
    :param dU_dx: function, the derivative of potential energy
    :return: (ndarray, ndarray)
            the coordinates and work
    """
    steps_per_iteration = lambda_t.shape[0]

    Xs = np.zeros( shape = (niterations, steps_per_iteration) )  # positions
    Ws = np.zeros( shape = (niterations, steps_per_iteration) )  # accumulated work

    stddev = np.sqrt(2 * dt)

    # Equilibration
    for t in range(equil_steps):
        Xs[:, 0] = Xs[:, 0] + np.random.randn(niterations) * stddev - dU_dx( Xs[:, 0], ks, lambda_t[0] ) * dt

    # Switching
    for t in range(1, steps_per_iteration):
        Xs[:, t] = Xs[:, t-1] + np.random.randn(niterations) * stddev - dU_dx( Xs[:, t-1], ks, lambda_t[t-1] ) * dt
        Ws[:, t] = Ws[:, t-1] + U(Xs[:, t], ks, lambda_t[t]) - U( Xs[:, t], ks, lambda_t[t-1] )

    return Xs, Ws

