import pytest
import numpy as np
import math

import matplotlib.pyplot as plt

from mumaxplus import Grid, World, Ferromagnet


SRTOL = 1e-3

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


def make_long_magnet(d_comp):
    """Makes a world with a magnet of the appropriate size according to the
    direction of the double derivative.
    """
    gridsize, pbc_repetitions = [0, 0, 0], [0, 0, 0]
    gridsize[d_comp], pbc_repetitions[d_comp] = N1, 1  # set for PBC grid
    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)

    gridsize[(d_comp + 1)%3], gridsize[(d_comp + 2)%3] = n1, n2  # set for magnet
    magnet =  Ferromagnet(world, Grid(gridsize))
    magnet.enable_elastodynamics = True

    return magnet


def set_and_check_sine_force(magnet, d_comp, u_comp, C):
    """Sets given component of the displacement of the given magnet to a sine
    wave, the periodicity of which depends on the direction of the derivative.
    Then checks if the simulated force corresponds to the expected analitical force.
    You need to set C yourself.
    """
    magnet.enable_elastodynamics = True  # just in case

    L = N1*cellsize[d_comp]
    k = P * 2*math.pi/L
    def displacement_func(x, y, z):
        u = [0., 0., 0.]
        relevant_coord = (x, y, z)[d_comp]
        u[u_comp] = A * math.sin(k * relevant_coord)
        return tuple(u)

    magnet.elastic_displacement = displacement_func
    
    force_num = magnet.elastic_force.eval()
    force_anal = - k**2 * C * magnet.elastic_displacement.eval()

    assert max_semirelative_error(force_num, force_anal) < SRTOL


cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
N1, N2, n1, n2 = 128, 64, 1, 4
P = 2  # sinus periods
A = 1e-15  # displacement amplitude
c11 = 283e9
c44 = 58e9


# ==================================================
# Tests for double derivate along the corresponding direction
# c11 is constant TODO: vary c11 as well

def test_dx_dx_ux():
    magnet = make_long_magnet(d_comp=0)
    magnet.c11 = c11
    set_and_check_sine_force(magnet, d_comp=0, u_comp=0, C=c11)

def test_dy_dy_uy():
    magnet = make_long_magnet(d_comp=1)
    magnet.c11 = c11
    set_and_check_sine_force(magnet, d_comp=1, u_comp=1, C=c11)

def test_dz_dz_uz():
    magnet = make_long_magnet(d_comp=2)
    magnet.c11 = c11
    set_and_check_sine_force(magnet, d_comp=2, u_comp=2, C=c11)

# ==================================================
# Tests for double derivate along the different direction
# c44 is constant TODO: vary c44 as well (careful of mixed derivatives!)

def test_dy_dy_ux():
    magnet = make_long_magnet(d_comp=1)
    magnet.c44 = c44
    magnet.c12 = -c44  # to remove mixed derivate
    set_and_check_sine_force(magnet, d_comp=1, u_comp=0, C=c44)

def test_dz_dz_ux():
    magnet = make_long_magnet(d_comp=2)
    magnet.c44 = c44
    magnet.c12 = -c44  # to remove mixed derivate
    set_and_check_sine_force(magnet, d_comp=2, u_comp=0, C=c44)


def test_dx_dx_uy():
    magnet = make_long_magnet(d_comp=0)
    magnet.c44 = c44
    magnet.c12 = -c44  # to remove mixed derivate
    set_and_check_sine_force(magnet, d_comp=0, u_comp=1, C=c44)

def test_dz_dz_uy():
    magnet = make_long_magnet(d_comp=2)
    magnet.c44 = c44
    magnet.c12 = -c44  # to remove mixed derivate
    set_and_check_sine_force(magnet, d_comp=2, u_comp=1, C=c44)


def test_dx_dx_uz():
    magnet = make_long_magnet(d_comp=0)
    magnet.c44 = c44
    magnet.c12 = -c44  # to remove mixed derivate
    set_and_check_sine_force(magnet, d_comp=0, u_comp=2, C=c44)

def test_dy_dy_uz():
    magnet = make_long_magnet(d_comp=1)
    magnet.c44 = c44
    magnet.c12 = -c44  # to remove mixed derivate
    set_and_check_sine_force(magnet, d_comp=1, u_comp=2, C=c44)


# ==================================================
# Tests for mixed derivates
# f_i += (c12+c44) ∂j(∂i(u_j))
# c12+c44 is constant TODO: vary c12+c44 as well (careful of double derivative!)
# TODO: make the tests


if __name__ == "__main__":
    # for manual testing
    pass
