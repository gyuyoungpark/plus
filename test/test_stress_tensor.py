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
B = -8.8e6

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


def sine_displacement(magnet, i_comp, j_comp, c11, c12, c44):
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
    strain_num = magnet.strain_tensor.eval()
    
    strain_anal = np.zeros(shape=strain_num.shape)

    # derivative of sine is cosine
    cos = np.zeros(shape=strain_anal[0,...].shape)
    index = [0, 0, 0]
    index[i_comp] = slice(None)  # index with length N
    
    cos[index[2], index[1], index[0]] = A * np.cos(k * cellsize[i_comp] * np.arange(0,N))

    if i_comp == j_comp:
        strain_anal[j_comp,...] = k * cos
    else:
        strain_anal[2 + i_comp + j_comp,...] = 0.5 * k * cos
    
    stress_num = magnet.stress_tensor.eval()
    stress_anal = np.zeros(shape=stress_num.shape)

    for i in range(3):
        ip1 = (i+1)%3
        ip2 = (i+2)%3

        stress_anal[i,...] = c11 * strain_anal[i,...] + c12 * strain_anal[ip1,...] + c12 * strain_anal[ip2,...]
        stress_anal[i+3,...] = c44 * strain_anal[i+3,...]

    assert max_semirelative_error(stress_num, stress_anal) < RTOL


def test_x_Exx():
    m_comp, i, j, c11, c12, c44 = 0, 0, 0, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)

def test_y_Eyy():
    m_comp, i, j, c11, c12, c44 = 1, 1, 1, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)

def test_z_Ezz():
    m_comp, i, j, c11, c12, c44 = 2, 2, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)

def test_y_Exy():
    m_comp, i, j, c11, c12, c44 = 1, 0, 1, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)

def test_z_Exz():
    m_comp, i, j, c11, c12, c44 = 2, 0, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)

def test_x_Exy():
    m_comp, i, j, c11, c12, c44 = 0, 0, 1, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)

def test_z_Eyz():
    m_comp, i, j, c11, c12, c44 = 2, 1, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)

def test_x_Exz():
    m_comp, i, j, c11, c12, c44 = 0, 0, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)

def test_y_Eyz():
    m_comp, i, j, c11, c12, c44 = 1, 1, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    sine_displacement(magnet, i, j, c11, c12, c44)