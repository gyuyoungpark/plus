import pytest
import numpy as np
import math

import matplotlib.pyplot as plt

from mumaxplus import Grid, World, Ferromagnet


def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


cx, cy, cz = 2e-9, 2e-9, 2e-9  # TODO: make different
cellsize = (cx, cy, cz)
P = 2  # sinus periods
A = 1e-15  # displacement amplitude


# ==================================================
# Tests for double derivate along the corresponding direction
# c11 is constant TODO: vary c11 as well

def test_dx_dx_ux():
    nx, ny, nz = 128, 1, 1
    length, width, thickness = nx*cx, ny*cy, nz*cz

    world = World(cellsize, mastergrid=Grid((nx, 0, 0)), pbc_repetitions=(1,0,0))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))
    
    magnet.enable_elastodynamics = True
    c11 = 283e9
    magnet.c11 = c11

    k = P * 2*math.pi/length
    magnet.elastic_displacement = lambda x,y,z: (A * math.sin(k * x), 0, 0)

    force_num = magnet.elastic_force.eval()
    force_anal = - k**2 * c11 * magnet.elastic_displacement.eval()

    assert max_semirelative_error(force_num, force_anal) < 1e-3


def test_dy_dy_uy():
    nx, ny, nz = 1, 128, 1
    length, width, thickness = nx*cx, ny*cy, nz*cz

    world = World(cellsize, mastergrid=Grid((0, ny, 0)), pbc_repetitions=(0,1,0))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))
    
    magnet.enable_elastodynamics = True
    c11 = 283e9
    magnet.c11 = c11

    k = P * 2*math.pi/width
    magnet.elastic_displacement = lambda x,y,z: (0, A * math.sin(k * y), 0)

    force_num = magnet.elastic_force.eval()
    force_anal = - k**2 * c11 * magnet.elastic_displacement.eval()

    assert max_semirelative_error(force_num, force_anal) < 1e-3


def test_dz_dz_uz():
    nx, ny, nz = 1, 1, 128
    length, width, thickness = nx*cx, ny*cy, nz*cz

    world = World(cellsize, mastergrid=Grid((0, 0, nz)), pbc_repetitions=(0,0,1))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))
    
    magnet.enable_elastodynamics = True
    c11 = 283e9
    magnet.c11 = c11

    k = P * 2*math.pi/thickness
    magnet.elastic_displacement = lambda x,y,z: (0, 0, A * math.sin(k * z))

    force_num = magnet.elastic_force.eval()
    force_anal = - k**2 * c11 * magnet.elastic_displacement.eval()

    assert max_semirelative_error(force_num, force_anal) < 1e-3


# ==================================================
# Tests for double derivate along the different direction
# c44 is constant TODO: vary c44 as well (careful of mixed derivatives!)


def test_dy_dy_ux():
    nx, ny, nz = 1, 128, 1
    length, width, thickness = nx*cx, ny*cy, nz*cz

    world = World(cellsize, mastergrid=Grid((0, ny, 0)), pbc_repetitions=(0,1,0))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))
    
    magnet.enable_elastodynamics = True
    c44 = 58e9
    magnet.c44 = c44
    magnet.c12 = -c44  # to remove mixed derivate

    k = P * 2*math.pi/width
    magnet.elastic_displacement = lambda x,y,z: (A * math.sin(k * y), 0, 0)

    force_num = magnet.elastic_force.eval()
    force_anal = - k**2 * c44 * magnet.elastic_displacement.eval()

    assert max_semirelative_error(force_num, force_anal) < 1e-3


def test_dz_dz_ux():
    nx, ny, nz = 1, 1, 128
    length, width, thickness = nx*cx, ny*cy, nz*cz

    world = World(cellsize, mastergrid=Grid((0, 0, nz)), pbc_repetitions=(0,0,1))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))
    
    magnet.enable_elastodynamics = True
    c44 = 58e9
    magnet.c44 = c44
    magnet.c12 = -c44  # to remove mixed derivate

    k = P * 2*math.pi/thickness
    magnet.elastic_displacement = lambda x,y,z: (A * math.sin(k * z), 0, 0)

    force_num = magnet.elastic_force.eval()
    force_anal = - k**2 * c44 * magnet.elastic_displacement.eval()

    assert max_semirelative_error(force_num, force_anal) < 1e-3


# TODO: test much more!


if __name__ == "__main__":
    # for manual testing
    test_dy_dy_ux()
    test_dz_dz_ux()
