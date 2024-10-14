import pytest
import numpy as np

from mumaxplus import Grid, World, Ferromagnet

RTOL = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
N = 128  # number of 1D cells
msat = 800e3
rho = 8e3
vel = 2

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

def create_magnet(d_comp):
    """Makes a world with a 1D magnet in the d_comp 
    and a magnetization in the m_comp direction.
    """
    gridsize, gridsize_magnet, pbc_repetitions = [0, 0, 0], [1, 1, 1], [0, 0, 0]
    gridsize[d_comp], pbc_repetitions[d_comp] = N, 1  # set for PBC grid
    gridsize_magnet[d_comp] = N

    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    
    magnet =  Ferromagnet(world, Grid(gridsize_magnet))
    magnet.enable_elastodynamics = True

    magnet.msat = msat

    return magnet

def kinetic_energy(magnet, v):
    """Calculates the kinetic energy with a velocity and mass density.
    """
    magnet.enable_elastodynamics = True  # just in case

    magnet.elastic_velocity = v
    magnet.rho = rho
    
    E_kin_num = magnet.kinetic_energy_density.eval()
    E_kin_anal = rho * np.linalg.norm(v)**2 / 2

    assert max_semirelative_error(E_kin_num, E_kin_anal) < RTOL

def test_kinetic_xx():
    d_comp = 0
    v = (vel,0,0)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)

def test_kinetic_xy():
    d_comp = 0
    v = (0,vel,0)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)

def test_kinetic_xz():
    d_comp = 0
    v = (0,0,vel)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)

def test_kinetic_yx():
    d_comp = 1
    v = (vel,0,0)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)

def test_kinetic_yy():
    d_comp = 1
    v = (0,vel,0)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)

def test_kinetic_yz():
    d_comp = 1
    v = (0,0,vel)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)

def test_kinetic_zx():
    d_comp = 2
    v = (vel,0,0)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)

def test_kinetic_zy():
    d_comp = 2
    v = (0,vel,0)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)

def test_kinetic_zz():
    d_comp = 2
    v = (0,0,vel)
    magnet = create_magnet(d_comp)
    kinetic_energy(magnet, v)