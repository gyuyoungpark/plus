"""Simulation 5: Magneto-rotation field validation.

Validates the magneto-rotation coupling implementation by comparing the
numerically evaluated magneto-rotation effective field against the analytical
formula:

    H_mr = (Kmr / Ms) [u_hat (m . (Omega x u_hat))
                       + (m . u_hat) (Omega x u_hat)]

where Omega = 1/2 curl(u) is the rotation pseudovector.

Protocol:
  1. Set up a 1D chain with elastodynamics enabled.
  2. Apply sinusoidal displacement u_z(x) = A sin(kx) analytically.
  3. The resulting rotation: Omega_y = 1/2 duz/dx = 1/2 A k cos(kx).
  4. Set PMA with anisU = (0, 0, 1) and uniform Kmr.
  5. Compare numerical magneto_rotation_field with analytical formula.
  6. Test multiple parameter sets: vary Kmr, m direction, wavelength.

Material: model PMA thin film
    Msat = 1.0 MA/m
    Kmr  = varied (0.5 -- 2.0 MJ/m^3)
    anisU = (0, 0, 1)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

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

CACHE_FILE = os.path.join(DATA_DIR, "sim05_data.npy")


# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
MSAT = 1.0e6          # A/m
NX, NY, NZ = 256, 1, 1
CX, CY, CZ = 5e-9, 5e-9, 5e-9
A_DISP = 1e-12        # displacement amplitude (m)
RHO = 8000            # kg/m^3
C11 = 283e9           # Pa (needed for elastodynamics)
C44 = 80e9            # Pa
C12 = C11 - 2 * C44


def analytical_mr_field(m_vec, omega_vec, uhat, Kmr, Msat):
    """Compute analytical magneto-rotation field per cell.

    Parameters
    ----------
    m_vec : ndarray, shape (3, nz, ny, nx)
        Normalized magnetization.
    omega_vec : ndarray, shape (3, nz, ny, nx)
        Rotation pseudovector.
    uhat : ndarray, shape (3,)
        Anisotropy unit direction.
    Kmr : float
        Magneto-rotation coupling constant (J/m^3).
    Msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    H : ndarray, shape (3, nz, ny, nx)
        Analytical effective field (T).
    """
    # Broadcast uhat to field shape
    uh = np.array(uhat).reshape(3, 1, 1, 1)

    # Omega x uhat
    cross = np.zeros_like(omega_vec)
    cross[0] = omega_vec[1] * uh[2] - omega_vec[2] * uh[1]
    cross[1] = omega_vec[2] * uh[0] - omega_vec[0] * uh[2]
    cross[2] = omega_vec[0] * uh[1] - omega_vec[1] * uh[0]

    # m . (Omega x uhat)
    a = np.sum(m_vec * cross, axis=0, keepdims=True)
    # m . uhat
    b = np.sum(m_vec * uh, axis=0, keepdims=True)

    coeff = Kmr / Msat
    H = coeff * (a * uh + b * cross)
    return H


def run_validation(Kmr_val, m_dir, wavelength_factor=4, label=""):
    """Run a single validation case.

    Parameters
    ----------
    Kmr_val : float
        Magneto-rotation coupling constant (J/m^3).
    m_dir : tuple
        Magnetization direction (will be normalized).
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
    m_arr = np.array(m_dir, dtype=float)
    m_arr /= np.linalg.norm(m_arr)
    magnet.magnetization = tuple(m_arr)

    # PMA axis
    uhat = np.array([0, 0, 1], dtype=float)
    magnet.anisU = tuple(uhat)
    magnet.Kmr = Kmr_val

    # Enable elastodynamics to have displacement field
    magnet.enable_elastodynamics = True
    magnet.rho = RHO
    magnet.C11 = C11
    magnet.C44 = C44
    magnet.C12 = C12

    # Set sinusoidal displacement: u_z(x) = A sin(kx), u_x = u_y = 0
    Lx = NX * CX
    k = 2 * np.pi * wavelength_factor / Lx
    xs = np.arange(NX) * CX
    uz = A_DISP * np.sin(k * xs)
    u_field = np.zeros((3, NZ, NY, NX))
    u_field[2, 0, 0, :] = uz
    magnet.elastic_displacement = u_field

    # Evaluate numerical quantities
    rot_num = magnet.rotation_vector.eval()   # (3, nz, ny, nx)
    H_num = magnet.magneto_rotation_field.eval()  # (3, nz, ny, nx)

    # Analytical rotation: Omega_y = 1/2 (dux/dz - duz/dx) = -1/2 A k cos(kx)
    omega_ana = np.zeros((3, NZ, NY, NX))
    omega_ana[1, 0, 0, :] = -0.5 * A_DISP * k * np.cos(k * xs)

    # Analytical magneto-rotation field
    m_field = magnet.magnetization.eval()
    H_ana = analytical_mr_field(m_field, omega_ana, uhat, Kmr_val, MSAT)

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
        'rot_num': rot_num,
        'omega_ana': omega_ana,
        'rel_err': rel_err,
        'label': label,
    }


def plot_validation(results, filename="fig_rotation_field_validation.pdf"):
    """Plot numerical vs analytical magneto-rotation field."""
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

        ax.set_ylabel(r'$H_\mathrm{mr}$ (T)')
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
    print("=== Sim 05: Magneto-rotation field validation ===")

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        results = np.load(CACHE_FILE, allow_pickle=True).item()['results']
    else:
        results = []

        # Case 1: m along z (parallel to uhat), Kmr = 1 MJ/m^3
        results.append(run_validation(
            Kmr_val=1.0e6, m_dir=(0, 0, 1), wavelength_factor=4,
            label="m//z, Kmr=1 MJ/m3"))

        # Case 2: m tilted 45 deg in xz plane
        results.append(run_validation(
            Kmr_val=1.0e6,
            m_dir=(1, 0, 1), wavelength_factor=4,
            label="m=(1,0,1)/sqrt2, Kmr=1 MJ/m3"))

        # Case 3: m along x (perpendicular to uhat)
        results.append(run_validation(
            Kmr_val=2.0e6, m_dir=(1, 0, 0), wavelength_factor=8,
            label="m//x, Kmr=2 MJ/m3"))

        # Case 4: larger Kmr, shorter wavelength
        results.append(run_validation(
            Kmr_val=0.5e6,
            m_dir=(0.5, 0.5, np.sqrt(0.5)), wavelength_factor=2,
            label="m tilted, Kmr=0.5 MJ/m3, short wave"))

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
