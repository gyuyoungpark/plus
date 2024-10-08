import pytest
import numpy as np
import math

import matplotlib.pyplot as plt

from mumaxplus.util import show

from mumaxplus import Grid, World, Ferromagnet

RTOL = 2e-2

cx, cy, cz = 1e-9, 1e-9, 1e-9
cellsize = (cx, cy, cz)
n1, n2 = 64, 1
P = 2  # sine periods
B1 = 1
B2 = 2

def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


def create_magnet(gridsize, pbc_repetitions):
    """Makes a world with a magnet
    """
    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)
    magnet =  Ferromagnet(world, Grid((n1, n2, 1)))
    magnet.enable_elastodynamics = True

    magnet.enable_elastodynamics = True
    magnet.B1 = B1
    magnet.B2 = B2

    return world, magnet

def sine_force(magnet):
    """Creates a magnetization of the form (cos(x), sin(x), 0) and calculates the magnetoelastic force
    """
    magnet.enable_elastodynamics = True  # just in case

    L = n1*cellsize[0]
    k = P*2*math.pi/L
    def magnetization_func(x, y, z):
        m = (math.cos(k * x), math.sin(k * x), 0)
        return m

    magnet.magnetization = magnetization_func

    force_num = magnet.magnetoelastic_force.eval()

    force_anal_x = -2 * B1 * k * magnet.magnetization.eval()[0,...] * magnet.magnetization.eval()[1,...]
    force_anal_y = B2 * k * (magnet.magnetization.eval()[0,...]**2 - magnet.magnetization.eval()[1,...]**2)
    force_anal_z = np.zeros(shape=magnet.magnetization.eval()[0,...].shape)
    
    return force_num, [force_anal_x, force_anal_y, force_anal_z]


class TestMagnetoElasticForce:
    def setup_class(self):
        world, magnet = create_magnet((0,0,0), (0,0,0))
        self.force_num, self.force_anal = sine_force(magnet)
        
        world_pbc, magnet_pbc = create_magnet((n1,0,0), (1,0,0))
        self.force_num_pbc, self.force_anal_pbc = sine_force(magnet_pbc)

    def test_x(self):
        assert max_semirelative_error(self.force_num_pbc[0,...], self.force_anal_pbc[0]) < RTOL
    
    def test_y(self):
        assert max_semirelative_error(self.force_num_pbc[1,...], self.force_anal_pbc[1]) < RTOL

    def test_x_PBC(self):
        assert max_semirelative_error(self.force_num_pbc[0,...], self.force_anal_pbc[0]) < RTOL
    
    def test_y_PBC(self):
        assert max_semirelative_error(self.force_num_pbc[1,...], self.force_anal_pbc[1]) < RTOL