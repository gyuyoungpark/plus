"""This test is based on the 1D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c"""

import pytest
import numpy as np
from mumaxplus import Ferromagnet, Grid, World
from mumax3 import Mumax3Simulation

RTOL = 1e-5
def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

@pytest.fixture(scope="class", params=[True, False])
def simulations(request):
    """This simulates a 1D wire with bulk or interfacial DMI in both
       mumax³ and mumax⁺ with open boundary conditions."""
    
    # test the interfacial and bulk DMI
    interfacial = request.param

    # constants
    A = 13e-12
    D = 3e-3
    Ku = 0.4e6
    anisU = (0,0,1)
    Ms = 0.86e6

    cellsize = (1e-9, 1e-9, 1e-9)
    gridsize = (100, 1, 1)
    magnetization = (0,0,1)

    # mumax⁺ simulation
    world = World(cellsize=cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))
    magnet.enable_demag = False
    magnet.enable_openbc = True

    magnet.msat = Ms
    magnet.aex = A
    magnet.ku1 = Ku
    magnet.anisU = anisU

    if interfacial:
        DMI = "Dind"
        D = -D
        magnet.dmi_tensor.set_interfacial_dmi(D)
    else:
        DMI = "Dbulk"
        magnet.dmi_tensor.set_bulk_dmi(D)
    
    magnet.magnetization = magnetization
    magnet.minimize()
    
    # simulation mumax³
    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{cellsize}
            setgridsize{gridsize}
            msat = {Ms}
            aex = {A}
            Ku1 = {Ku}
            anisU = vector{anisU}
            {DMI} = {D}
            enabledemag = False
            SetPBC(0,2,0)

            m = Uniform{tuple(magnetization)}
            minimize()
            saveas(m, "m.ovf")
        """
    )

    return  magnet, mumax3sim


@pytest.mark.mumax3
class TestDMI1D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test(self, simulations):
        magnet, mumax3sim = simulations
        err = max_relative_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < RTOL