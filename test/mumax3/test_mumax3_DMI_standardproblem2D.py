"""This test is based on the 2D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c"""

import pytest
import numpy as np
from mumaxplus import Ferromagnet, Grid, World
from mumax3 import Mumax3Simulation
from mumaxplus.util.shape import *
from mumaxplus.util.config import *


ATOL = 1e-3
def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    return np.max(err)

def simulations():
    """This simulates a 2D circle with interfacial DMI and a Neél skyrmion
       in both mumax³ and mumax⁺."""
    
    # constants
    A = 13e-12
    D = 3e-3
    Ku = 0.4e6
    anisU = (0,0,1)
    Ms = 0.86e6

    # charge and polarization of the skyrmion
    charge, pol = -1, 1

    # diameter and thickness of the skyrmion
    diam = 100e-9
    thickness = 2e-9

    cellsize = (2e-9, 2e-9, 2e-9)
    gridsize = (50, 50, 1)

    # mumax⁺ simulation
    world = World(cellsize=cellsize)
    geo = Cylinder(diam, thickness).translate(50e-9 - 1e-9, 50e-9 - 1e-9, 0)
    magnet = Ferromagnet(world, Grid(gridsize), geometry=geo)

    magnet.enable_demag = False

    magnet.msat = Ms
    magnet.aex = A
    magnet.ku1 = Ku
    magnet.anisU = anisU

    magnet.dmi_tensor.set_interfacial_dmi(D)
    magnet.magnetization = neelskyrmion(magnet.center, 50e-9, charge, pol)
    magnet.minimize()

    # mumax³ simulation
    mumax3sim = Mumax3Simulation(
        f"""
            // Disk radius
            R  := 50e-9

            // Number of cells
            nx := 50
            ny := 50
            nz := 1

            // Mesh discretisation
            dx := 2e-9
            dy := 2e-9
            dz := 2e-9

            // Set mesh and disk geometry
            SetGridSize(nx, ny, nz)
            SetCellSize(dx, dy, dz)
            SetGeom(Circle(2 * R))

            Msat        = {Ms}
            Aex         = {A}
            Ku1         = {Ku}
            anisU       = vector{tuple(anisU)}
            Dind        = {D}
            // No Demag:
            NoDemagSpins = 1

            // Initial state
            m = Neelskyrmion({charge}, {pol})

            OutputFormat = OVF2_TEXT

            minimize()
            SaveAs(m, "m.ovf")
        """
    )

    return  magnet, mumax3sim


@pytest.mark.mumax3
class TestDMI2D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test(self):
        magnet, mumax3sim = simulations()
        err = max_relative_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < ATOL