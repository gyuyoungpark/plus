import pytest
import numpy as np
import math

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
P = 2  # periods
A = 1  # amplitude
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

    return world, magnet


def sine_displacement(magnet, d_comp, cos_comp, B1, B2):
    """Creates the magnetization following a sine in the d_comp direction
    and cosine in another direction it then calculates the magnetoelasticforce.
    """
    magnet.enable_elastodynamics = True  # just in case

    magnet.B1 = B1
    magnet.B2 = B2

    L = N*cellsize[d_comp]
    k = P*2*math.pi/L

    def displacement_func(x, y, z):
        u = [0,0,0]
        comp = (x,y,z)[d_comp]
        u[d_comp] = A * math.sin(k * comp)
        u[cos_comp] = A * math.cos(k * comp)
        return tuple(u)

    magnet.elastic_displacement = displacement_func
    strain_num = magnet.strain_tensor.eval()
    
    strain_anal = np.zeros(shape=strain_num.shape)
    strain_anal[d_comp,...] = k * magnet.elastic_displacement.eval()[cos_comp,...]
    strain_anal[2 + d_comp + cos_comp,...] = - 0.5 * k * magnet.elastic_displacement.eval()[d_comp,...]

    h_num = magnet.magnetoelastic_field.eval()
    h_anal = np.zeros(shape=h_num.shape)

    m = magnet.magnetization.eval()
    for i in range(3):
        ip1 = (i+1)%3
        ip2 = (i+2)%3

        h_anal[i,...] = - 2 * (B1 * strain_anal[i,...] * m[i,...] + 
                               B2 * (strain_num[i+ip1+2,...] * m[ip1,...] + 
                               strain_num[i+ip2+2,...] * m[ip2,...])) / msat

    assert max_semirelative_error(h_num, h_anal) < RTOL

def test_x_sinx_cosx_0_B1():
    d_comp, m_comp = 0, 0
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_x_sinx_0_cosx_B1():
    d_comp, m_comp = 0, 0
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_x_sinx_cosx_0_B2():
    d_comp, m_comp = 0, 0
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_x_cosz_0_sinz_B2():
    d_comp, m_comp = 2, 0
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_x_sinx_0_cosx_B2():
    d_comp, m_comp = 0, 0
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_x_cosy_siny_0_B2():
    d_comp, m_comp = 1, 0
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_y_0_siny_cosy_B1():
    d_comp, m_comp = 1, 1
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_y_cosy_siny_0_B1():
    d_comp, m_comp = 1, 1
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_y_sinx_cosx_0_B2():
    d_comp, m_comp = 0, 1
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_y_0_siny_cosy_B2():
    d_comp, m_comp = 1, 1
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_y_cosy_siny_0_B2():
    d_comp, m_comp = 1, 1
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_y_0_cosz_sinz_B2():
    d_comp, m_comp = 2, 1
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_z_cosz_0_sinz_B1():
    d_comp, m_comp = 2, 2
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_z_0_cosz_sinz_B1():
    d_comp, m_comp = 2, 2
    B1, B2 = B, 0
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_z_0_siny_cosy_B2():
    d_comp, m_comp = 1, 2
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_z_cosz_0_sinz_B2():
    d_comp, m_comp = 2, 2
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+1)%3, B1, B2)

def test_z_sinx_0_cosx_B2():
    d_comp, m_comp = 0, 2
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)

def test_z_0_cosz_sinz_B2():
    d_comp, m_comp = 2, 2
    B1, B2 = 0, B
    world, magnet = create_magnet(d_comp, m_comp)
    sine_displacement(magnet, d_comp, (d_comp+2)%3, B1, B2)
