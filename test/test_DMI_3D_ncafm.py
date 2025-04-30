"""This test is based on the 3D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c
   It compares the final magnetization of all sublattices in an NCAFM
   with that of a ferromagnet. All NCAFM exchanges are set to 0 for this test."""

import pytest
import numpy as np
from mumaxplus import NCAFM, Ferromagnet, Grid, World
from mumaxplus.util.shape import Cylinder
from mumaxplus.util.config import blochskyrmion


ATOL = 1e-4
def max_absolute_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    return np.max(err)

def simulations(openBC):
    """This simulates a 3D cylinder with bulk DMI and a bloch skyrmion
       in a ferromagnet and an NCAFM."""
    
    # constants
    A = 8.78e-12
    D = 1.58e-3
    Ms = 0.384e6
    Bz = 0.4

    # diameter and thickness of the skyrmion
    diam, thickness = 183e-9, 21e-9
    skyrmion_radius = 36e-9

    # charge and polarization of the skyrmion
    charge, pol = 1, -1

    nx, ny, nz = 183, 183, 21
    dx, dy, dz = 1e-9, 1e-9, 1e-9

    gridsize = (nx, ny, nz)
    cellsize = (dx, dy, dz)

    # ferromagnet simulation
    world = World(cellsize=cellsize)
    geo = Cylinder(diam, thickness).translate((nx*dx-dx)/2, (ny*dy-dy)/2, (nz*dz-dz)/2)
    magnet = Ferromagnet(world, Grid(gridsize), geometry=geo)

    magnet.enable_demag = False
    magnet.enable_openbc = openBC
    magnet.msat = Ms
    magnet.aex = A
    magnet.dmi_tensor.set_bulk_dmi(D)

    magnet.bias_magnetic_field = (0,0,Bz)

    magnet.magnetization = blochskyrmion(magnet.center, skyrmion_radius, charge, pol)

    magnet.minimize()

    # antiferromagnet simulation
    world_NCAFM = World(cellsize=cellsize)
    geo = Cylinder(diam, thickness).translate((nx*dx-dx)/2, (ny*dy-dy)/2, (nz*dz-dz)/2)
    magnet_NCAFM = NCAFM(world_NCAFM, Grid(gridsize), geometry=geo)

    magnet_NCAFM.enable_demag = False
    magnet_NCAFM.enable_openbc = openBC
    magnet_NCAFM.msat = Ms
    magnet_NCAFM.aex = A
    magnet_NCAFM.ncafmex_cell = 0
    magnet_NCAFM.ncafmex_nn = 0

    for sub in magnet_NCAFM.sublattices:
        sub.dmi_tensor.set_bulk_dmi(D)

    magnet_NCAFM.bias_magnetic_field = (0,0,Bz)

    magnet_NCAFM.magnetization = blochskyrmion(magnet.center, skyrmion_radius, charge, pol)

    magnet_NCAFM.minimize()


    return  magnet, magnet_NCAFM


@pytest.mark.slow
class TestDMI3D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test_closed(self):
        magnet, magnet_NCAFM = simulations(False)
        for i in range(3):
            sub = magnet_NCAFM.sublattices[i]
            err = max_absolute_error(magnet.magnetization.eval(), sub.magnetization.eval())
            assert err < ATOL
    
    def test_open(self):
        magnet, magnet_NCAFM = simulations(True)
        for i in range(3):
            sub = magnet_NCAFM.sublattices[i]
            err = max_absolute_error(magnet.magnetization.eval(), sub.magnetization.eval())
            assert err < ATOL