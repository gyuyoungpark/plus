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
    m = [0,0,0]
    m[m_comp] = 1
    gridsize[d_comp], pbc_repetitions[d_comp] = N, 1  # set for PBC grid
    gridsize_magnet[d_comp] = N

    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    
    magnet =  Ferromagnet(world, Grid(gridsize_magnet))
    magnet.enable_elastodynamics = True

    magnet.magnetization = m
    magnet.msat = msat

    return magnet

def elastic_energy(magnet, i_comp, j_comp, c11, c12, c44):
    """Creates the displacement following a sine in the i_comp direction
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
    stress_num = magnet.stress_tensor.eval()
    
    E_el_num = magnet.elastic_energy_density.eval()
    
    E_el_anal = np.zeros(shape=E_el_num.shape)
    for i in range(3):
        E_el_anal += 0.5 * stress_num[i,...] * strain_num[i,...] + stress_num[i+3,...] * strain_num[i+3,...]

    assert max_semirelative_error(E_el_num, E_el_anal) < RTOL


def test_elastic_x_Exx():
    m_comp, i, j, c11, c12, c44 = 0, 0, 0, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)

def test_elastic_y_Eyy():
    m_comp, i, j, c11, c12, c44 = 1, 1, 1, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)

def test_elastic_z_Ezz():
    m_comp, i, j, c11, c12, c44 = 2, 2, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)

def test_elastic_y_Exy():
    m_comp, i, j, c11, c12, c44 = 1, 0, 1, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)

def test_elastic_z_Exz():
    m_comp, i, j, c11, c12, c44 = 2, 0, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)

def test_elastic_x_Exy():
    m_comp, i, j, c11, c12, c44 = 0, 0, 1, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)

def test_elastic_z_Eyz():
    m_comp, i, j, c11, c12, c44 = 2, 1, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)

def test_elastic_x_Exz():
    m_comp, i, j, c11, c12, c44 = 0, 0, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)

def test_elastic_y_Eyz():
    m_comp, i, j, c11, c12, c44 = 1, 1, 2, 283e9, 58e9, 166e9
    magnet = create_magnet(i, m_comp)
    elastic_energy(magnet, i, j, c11, c12, c44)