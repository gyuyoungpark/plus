"""This test is based on the 3D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c"""

import pytest
import numpy as np
from mumaxplus import Ferromagnet, Grid, World
from mumax3 import Mumax3Simulation
from mumaxplus.util.shape import Cylinder
from mumaxplus.util.config import blochskyrmion


ATOL = 2e-3
def max_absolute_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    return np.max(err)

def simulations():
    """This simulates a 3D cylinder with bulk DMI and a bloch skyrmion
       in both mumax³ and mumax⁺."""
    
    # constants
    A = 8.78e-12
    D = 1.5e-3
    Ms = 0.384e6
    Bz = 0.4

    # diameter and thickness of the skyrmion
    diam, thickness = 183e-9, 21e-9

    # charge and polarization of the skyrmion
    charge, pol = 1, -1

    gridsize = (183, 183, 21)
    cellsize = (1e-9, 1e-9, 1e-9)

    # mumax⁺ simulation
    world = World(cellsize=cellsize)
    geo = Cylinder(diam, thickness).translate(91.5e-9 - 0.5e-9, 91.5e-9 - 0.5e-9, 10.5e-9 - 0.5e-9)
    magnet = Ferromagnet(world, Grid(gridsize), geometry=geo)

    magnet.enable_demag = False
    magnet.msat = Ms
    magnet.aex = A
    magnet.dmi_tensor.set_bulk_dmi(D)

    magnet.bias_magnetic_field = (0,0,Bz)

    magnet.magnetization = blochskyrmion(magnet.center, 91.5e-9, charge, pol)

    tolerance = 1e-6
    magnet.minimize(tolerance)

    # mumax³ simulation
    mumax3sim = Mumax3Simulation(
        f"""
            nx := 183
            ny := 183
            nz := 21

            lx := 183e-9
            ly := 183e-9
            lz := 21e-9

            dx := lx / nx
            dy := ly / ny
            dz := lz / nz

            SetGridSize(nx, ny, nz)
            SetCellSize(dx, dy, dz)

            // Define the cylinder
            SetGeom(Circle(lx))

            Msat         = {Ms}
            Aex          = {A}
            Dbulk        = {D}

            // External field in T
            B_ext = vector(0, 0, {Bz})

            // No Demag
            NoDemagSpins = 1

            m = BlochSkyrmion(1, -1)

            OutputFormat = OVF2_TEXT

            // Relax with conjugate gradient:
            MinimizerStop = {tolerance}
            minimize();
            SaveAs(m, "m")
        """
    )

    return  magnet, mumax3sim


@pytest.mark.mumax3
@pytest.mark.slow
class TestDMI3D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test(self):
        magnet, mumax3sim = simulations()
        err = max_relative_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < ATOL