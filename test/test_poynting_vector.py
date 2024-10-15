import pytest
import numpy as np
import math

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
P = 2  # periods
A = 4  # amplitude
N = 128  # number of 1D cells
msat = 800e3
c11, c12, c44 = 283e9, 58e9, 166e9
vel = 4

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


def create_magnet(d_comp, m_comp):
    """Makes a world with a 1D magnet in the d_comp 
    and a magnetization in the m_comp direction.
    """
    gridsize, gridsize_magnet, pbc_repetitions = [0, 0, 0], [1, 1, 1], [0, 0, 0]
    m = [0, 0, 0]
    m[m_comp] = 1
    gridsize[d_comp], pbc_repetitions[d_comp] = N, 1  # set for PBC grid
    gridsize_magnet[d_comp] = N

    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    
    magnet =  Ferromagnet(world, Grid(gridsize_magnet))
    magnet.enable_elastodynamics = True

    magnet.magnetization = m
    magnet.msat = msat

    return magnet


def sine_displacement(magnet, i_comp, j_comp, v):
    """Creates the magnetization following a sine in the i_comp direction
    depending on the j_comp.
    """
    magnet.enable_elastodynamics = True  # just in case

    magnet.c11 = c11
    magnet.c12 = c12
    magnet.c44 = c44 

    L = N*cellsize[i_comp]
    k = P*2*math.pi/L

    def displacement_func(x, y, z):
        u = [0,0,0]
        comp = (x,y,z)[i_comp]
        u[j_comp] = A * math.sin(k * comp)
        return tuple(u)

    magnet.elastic_displacement = displacement_func
    magnet.elastic_velocity = v
    
    stress = magnet.stress_tensor.eval()

    poynting_num = magnet.poynting_vector.eval()
    poynting_anal = np.zeros(shape=poynting_num.shape)

    poynting_anal[0,...] = stress[0,...] * v[0] + stress[3,...] * v[1] + stress[4,...] * v[2]
    poynting_anal[1,...] = stress[3,...] * v[0] + stress[1,...] * v[1] + stress[5,...] * v[2]
    poynting_anal[2,...] = stress[4,...] * v[0] + stress[5,...] * v[1] + stress[2,...] * v[2]

    assert max_semirelative_error(poynting_num, poynting_anal) < RTOL


def test_x_Exx_vx():
    m_comp, i, j = 0, 0, 0
    v = (vel, 0, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_y_Eyy_vx():
    m_comp, i, j = 1, 1, 1
    v = (vel, 0, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_z_Ezz_vx():
    m_comp, i, j = 2, 2, 2
    v = (vel, 0, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_y_Exy_vx():
    m_comp, i, j = 1, 0, 1
    v = (vel, 0, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_z_Exz_vx():
    m_comp, i, j = 2, 0, 2
    v = (vel, 0, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_x_Exy_vx():
    m_comp, i, j = 0, 0, 1
    v = (vel, 0, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_x_Exz_vx():
    m_comp, i, j = 0, 0, 2
    v = (vel, 0, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_x_Exx_vy():
    m_comp, i, j = 0, 0, 0
    v = (0, vel, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_y_Eyy_vy():
    m_comp, i, j = 1, 1, 1
    v = (0, vel, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_z_Ezz_vy():
    m_comp, i, j = 2, 2, 2
    v = (0, vel, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_y_Exy_vy():
    m_comp, i, j = 1, 0, 1
    v = (0, vel, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_x_Exy_vy():
    m_comp, i, j = 0, 0, 1
    v = (0, vel, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_z_Eyz_vy():
    m_comp, i, j = 2, 1, 2
    v = (0, vel, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_y_Eyz_vy():
    m_comp, i, j = 1, 1, 2
    v = (0, vel, 0)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_x_Exx_vz():
    m_comp, i, j = 0, 0, 0
    v = (0, 0, vel)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_y_Eyy_vz():
    m_comp, i, j = 1, 1, 1
    v = (0, 0, vel)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_z_Ezz_vz():
    m_comp, i, j = 2, 2, 2
    v = (0, 0, vel)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_z_Exz_vz():
    m_comp, i, j = 2, 0, 2
    v = (0, 0, vel)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_z_Eyz_vz():
    m_comp, i, j = 2, 1, 2
    v = (0, 0, vel)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_x_Exz_vz():
    m_comp, i, j = 0, 0, 2
    v = (0, 0, vel)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)

def test_y_Eyz_vz():
    m_comp, i, j = 1, 1, 2
    v = (0, 0, vel)
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, v)