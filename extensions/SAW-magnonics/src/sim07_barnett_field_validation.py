"""Simulation 7: Barnett (spin-rotation) field validation.

Validates the spin-rotation coupling (Barnett effect) implementation by
comparing the numerically evaluated Barnett effective field against the
analytical formula:

    H_Barnett = omega / gamma  (in code units, Tesla)

where omega = 1/2 curl(v) is the lattice angular velocity.

The Barnett effect couples lattice rotation to spin: a rotating lattice
generates an effective magnetic field along the rotation axis, analogous
to the Einstein-de Haas effect in reverse.

Protocol:
  1. Set up a 1D chain with elastodynamics enabled.
  2. Apply sinusoidal velocity v_z(x) = V0 sin(kx) analytically.
  3. The resulting angular velocity: omega_y = 1/2 dvz/dx = 1/2 V0 k cos(kx).
  4. Enable Barnett coupling (enable_barnett = True).
  5. Compare numerical spin_rotation_field with analytical H = omega/gamma.
  6. Test multiple parameter sets: vary gamma, velocity amplitude, wavelength.

References:
  - Barnett, S. J. (1915), Phys. Rev. 6, 239.
  - Matsuo, M. et al. (2020), PRB 102, 174413.
"""

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import os

import numpy as np
import matplotlib.pyplot as plt

import plot_style as ps

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim07_data.npy")


# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
MSAT = 1.0e6          # A/m
GAMMA_DEFAULT = 1.7595e11  # rad/(T.s) - default gyromagnetic ratio
NX, NY, NZ = 256, 1, 1
CX, CY, CZ = 5e-9, 5e-9, 5e-9
V_AMP = 1.0           # velocity amplitude (m/s)
RHO = 8000            # kg/m^3
C11 = 283e9           # Pa (needed for elastodynamics)
C44 = 80e9            # Pa
C12 = C11 - 2 * C44


def analytical_barnett_field(omega_vec, gamma_val):
    """Compute analytical Barnett field per cell.

    H_Barnett = omega / gamma (Tesla, in mumax+ code units)

    Parameters
    ----------
    omega_vec : ndarray, shape (3, nz, ny, nx)
        Angular velocity pseudovector.
    gamma_val : float
        Gyromagnetic ratio (rad/(T.s)).

    Returns
    -------
    H : ndarray, shape (3, nz, ny, nx)
        Analytical effective field (T).
    """
    return omega_vec / gamma_val


def run_validation(v_amp, gamma_val, wavelength_factor=4, label=""):
    """Run a single Barnett field validation case.

    Parameters
    ----------
    v_amp : float
        Velocity amplitude (m/s).
    gamma_val : float
        Gyromagnetic ratio (rad/(T.s)).
    wavelength_factor : int
        Wavelength = wavelength_factor * NX * CX / (2 pi).
    label : str
        Label for this test case.

    Returns
    -------
    dict with x, H_num, H_ana arrays.
    """
    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize, mastergrid=Grid((NX, 0, 0)),
                  pbc_repetitions=(2, 0, 0))
    magnet = Ferromagnet(world, grid)

    # Material
    magnet.msat = MSAT
    magnet.aex = 1e-11
    magnet.alpha = 0
    magnet.gamma = gamma_val
    magnet.magnetization = (0, 0, 1)

    # Enable elastodynamics and Barnett effect
    magnet.enable_elastodynamics = True
    magnet.enable_barnett = True
    magnet.rho = RHO
    magnet.C11 = C11
    magnet.C44 = C44
    magnet.C12 = C12

    # Set sinusoidal velocity: v_z(x) = V0 sin(kx), v_x = v_y = 0
    Lx = NX * CX
    k = 2 * np.pi * wavelength_factor / Lx
    xs = np.arange(NX) * CX
    vz = v_amp * np.sin(k * xs)
    v_field = np.zeros((3, NZ, NY, NX))
    v_field[2, 0, 0, :] = vz
    magnet.elastic_velocity = v_field

    # Evaluate numerical quantities
    omega_num = magnet.angular_velocity.eval()       # (3, nz, ny, nx)
    H_num = magnet.spin_rotation_field.eval()        # (3, nz, ny, nx)

    # Analytical angular velocity: omega_y = 1/2 (dvx/dz - dvz/dx) = -1/2 V0 k cos(kx)
    omega_ana = np.zeros((3, NZ, NY, NX))
    omega_ana[1, 0, 0, :] = -0.5 * v_amp * k * np.cos(k * xs)

    # Analytical Barnett field
    H_ana = analytical_barnett_field(omega_ana, gamma_val)

    # Compute error
    diff = np.sqrt(np.sum((H_num - H_ana)**2, axis=0))
    H_mag = np.sqrt(np.sum(H_ana**2, axis=0))
    H_mag_max = np.max(H_mag)
    if H_mag_max > 0:
        rel_err = np.max(diff) / H_mag_max
    else:
        rel_err = np.max(np.abs(H_num))

    print(f"  {label}: max relative error = {rel_err:.2e}")

    return {
        'x': xs * 1e9,
        'H_num': H_num,
        'H_ana': H_ana,
        'omega_num': omega_num,
        'omega_ana': omega_ana,
        'rel_err': rel_err,
        'label': label,
    }


def plot_validation(results, filename="fig_barnett_field_validation.pdf"):
    """Plot numerical vs analytical Barnett field."""
    ps.apply_style()
    n = len(results)
    labels = ['(a)', '(b)', '(c)', '(d)']
    fig, axes = plt.subplots(n, 1,
                             figsize=(ps.SINGLE_COL, 2.2 * n * ps.CM_TO_INCH * 2.54),
                             sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, res) in enumerate(zip(axes, results)):
        x = res['x']
        for c, clabel in enumerate(['x', 'y', 'z']):
            h_num = res['H_num'][c, 0, 0, :]
            h_ana = res['H_ana'][c, 0, 0, :]
            if np.max(np.abs(h_ana)) < 1e-30 and np.max(np.abs(h_num)) < 1e-30:
                continue
            ax.plot(x, h_ana, '-', label=f'$H_{clabel}$ (ana)', alpha=0.8)
            ax.plot(x, h_num, '--', label=f'$H_{clabel}$ (num)', alpha=0.8)

        ax.set_ylabel(r'$H_\mathrm{Barnett}$ (T)')
        ax.legend(fontsize=6.5, ncol=3, loc='lower right')
        ps.add_panel_label(ax, labels[i])

    axes[-1].set_xlabel(r'$x$ (nm)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename.replace(".pdf", ".png")), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=== Sim 07: Barnett (spin-rotation) field validation ===")

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        results = np.load(CACHE_FILE, allow_pickle=True).item()['results']
    else:
        results = []

        # Case 1: default gamma, moderate velocity
        results.append(run_validation(
            v_amp=1.0, gamma_val=GAMMA_DEFAULT, wavelength_factor=4,
            label="V=1 m/s, gamma=default"))

        # Case 2: higher velocity amplitude
        results.append(run_validation(
            v_amp=10.0, gamma_val=GAMMA_DEFAULT, wavelength_factor=4,
            label="V=10 m/s, gamma=default"))

        # Case 3: modified gamma (half default)
        results.append(run_validation(
            v_amp=1.0, gamma_val=GAMMA_DEFAULT / 2, wavelength_factor=4,
            label="V=1 m/s, gamma=half"))

        # Case 4: shorter wavelength
        results.append(run_validation(
            v_amp=1.0, gamma_val=GAMMA_DEFAULT, wavelength_factor=8,
            label="V=1 m/s, short wavelength"))

        np.save(CACHE_FILE, {'results': results})
        print(f"  Data saved to {CACHE_FILE}")

    plot_validation(results)

    # Summary
    all_pass = all(r['rel_err'] < 1e-2 for r in results)
    if all_pass:
        print("  ALL CASES PASSED (relative error < 1%)")
    else:
        print("  WARNING: some cases have large errors")


if __name__ == "__main__":
    main()
