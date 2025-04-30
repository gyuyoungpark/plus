import numpy as np

from mumaxplus import Antiferromagnet, NCAFM, Grid, World

def compute_local_dmi_field(magnet, sub, symmetry_factor):
    D = np.array(magnet.dmi_vector.average())
    return symmetry_factor / sub.msat.average()[0] * np.cross(D[:, None, None, None],
                                                              sub.magnetization(), axis=0)
class TestLocalDMI:
    def test_local_dmi_afm(self):
        world = World((1, 1, 1))
        magnet = Antiferromagnet(world, Grid((32, 32, 5)))
        magnet.msat = 5
        magnet.dmi_vector = (8, 42, 3e-1)

        for i, sub in enumerate(magnet.sublattices):
            result = sub.homogeneous_dmi_field()
            wanted = compute_local_dmi_field(magnet, magnet.other_sublattice(sub), (-1)**i)

            relative_error = np.abs(result - wanted) / np.abs(wanted)
            max_relative_error = np.max(relative_error)
            assert max_relative_error < 3e-2

    def test_local_dmi_ncafm(self):
        world = World((1, 1, 1))
        magnet = NCAFM(world, Grid((32, 32, 5)))
        magnet.msat = 5
        magnet.dmi_vector = (8, 42, 3e-1)

        for i, sub in enumerate(magnet.sublattices):
            result = sub.homogeneous_dmi_field()
            sub2, sub3 = [s for j, s in enumerate(magnet.sublattices) if j != i]
            
            wanted = compute_local_dmi_field(magnet, sub2, (-1)**(i)) + \
                     compute_local_dmi_field(magnet, sub3, (-1)**(i+1))

            relative_error = np.abs(result - wanted) / np.abs(wanted)
            max_relative_error = np.max(relative_error)
            assert max_relative_error < 3e-2
