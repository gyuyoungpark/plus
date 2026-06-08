"""Simulation 1: SAW-driven domain wall motion.

A Rayleigh SAW propagating along x drives a Neel domain wall through
magnetoelastic coupling.  The DW velocity scales quadratically with
strain amplitude:

    v_DW ~ epsilon_0^2

because the magnetoelastic force on the DW arises from a product of
the strain and the DW magnetization gradient, both of which carry the
SAW spatial profile.

Protocol:
  - CoFeB PMA thin film with interfacial DMI (Neel wall)
  - Two-domain initial state with DW at x = L/3
  - Minimize to relax the DW structure
  - Apply SAW at f = 200 MHz, v_SAW = 4000 m/s
  - Sweep strain amplitude epsilon_0
  - Record DW position vs time, extract velocity

Material: CoFeB PMA
    Msat  = 600 kA/m
    Aex   = 10 pJ/m
    Ku1   = 800 kJ/m^3
    alpha = 0.01
    B1    = -8.8 MJ/m^3
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

import plot_style as ps
from saw import SurfaceAcousticWave

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import twodomain

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim01_data.npy")


# -----------------------------------------------------------------------
# Material and geometry parameters
# -----------------------------------------------------------------------
MSAT = 6e5           # A/m
AEX = 1e-11          # J/m
ALPHA = 0.01
KU1 = 8e5            # J/m^3
DMI_D = 1e-3         # J/m^2 (interfacial DMI)
B1 = -8.8e6          # J/m^3  (first magnetoelastic coupling)
B2 = 0.0

# Grid
NX, NY, NZ = 256, 8, 1
CX, CY, CZ = 2.4e-9, 4 * 2.4e-9, 1e-9

# SAW
F_SAW = 200e6        # Hz
V_SAW = 4000.0       # m/s  (LiNbO3-like substrate)
LAMBDA_SAW = V_SAW / F_SAW   # ~ 20 um

# Simulation
T_RUN = 40e-9        # s  (longer run for steady-state)
N_STEPS = 4000


def setup_cofeb_dw(epsilon_0):
    """Create a CoFeB PMA film with a Neel DW and SAW driving.

    Parameters
    ----------
    epsilon_0 : float
        Peak SAW strain amplitude.

    Returns
    -------
    world, magnet, saw : World, Ferromagnet, SurfaceAcousticWave
    """
    cellsize = (CX, CY, CZ)
    mastergrid = Grid((0, NY, 0))
    pbc_reps = (0, 4, 0)

    world = World(cellsize, mastergrid=mastergrid, pbc_repetitions=pbc_reps)
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    # Material parameters
    magnet.msat = MSAT
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.ku1 = KU1
    magnet.anisU = (0, 0, 1)

    # Interfacial DMI for Neel wall stabilization
    magnet.dmi_tensor.set_interfacial_dmi(DMI_D)

    # Two-domain initial state: DW at x = L/3
    L = NX * CX
    magnet.magnetization = twodomain((0, 0, 1), (-1, 0, 0), (0, 0, -1),
                                     L / 3, 5 * CX)

    # Minimize to relax DW
    print(f"    Minimizing (eps0={epsilon_0:.1e})...")
    magnet.minimize()

    # Magnetoelastic coupling
    magnet.B1 = B1
    magnet.B2 = B2

    # Apply SAW
    saw = SurfaceAcousticWave(F_SAW, LAMBDA_SAW, epsilon_0, direction='x')
    saw.apply(magnet)

    return world, magnet, saw


def dw_position(magnet):
    """Estimate DW position from spatially averaged mz.

    For a two-domain state along x with mz=+1 left, mz=-1 right:
        <mz> = (2*x_DW - L) / L
        x_DW = L/2 * (1 + <mz>)
    """
    mz_avg = magnet.magnetization.average()[2]
    L = NX * CX
    return 0.5 * L * (1 + mz_avg)


def run_dw_sweep():
    """Sweep SAW amplitude and record DW position vs time.

    Returns
    -------
    results : list of dict
        Each dict contains 'epsilon_0', 'time', 'dw_pos', 'velocity'.
    """
    amplitudes = [1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 4e-3]
    results = []

    for eps0 in amplitudes:
        print(f"  Running eps0 = {eps0:.1e} ...")
        world, magnet, saw = setup_cofeb_dw(eps0)

        time_arr = np.linspace(0, T_RUN, N_STEPS + 1)
        quantity_dict = {"dw_pos": lambda: dw_position(magnet)}

        output = world.timesolver.solve(time_arr, quantity_dict)

        t = np.array(output["time"])
        pos = np.array(output["dw_pos"])

        # Extract velocity from peak of smoothed numerical derivative
        # (avoids artifacts from DW saturation at late times)
        dt_arr = np.diff(t)
        dx_arr = np.diff(pos)
        v_inst = dx_arr / dt_arr
        # Smooth with moving average (window ~ 5% of total points)
        win = max(10, len(v_inst) // 20)
        kernel = np.ones(win) / win
        v_smooth = np.convolve(np.abs(v_inst), kernel, mode='valid')
        velocity = np.max(v_smooth)
        r_value = 0.0  # not used with this method

        results.append({
            'epsilon_0': eps0,
            'time': t,
            'dw_pos': pos,
            'velocity': velocity,
            'r_squared': r_value**2,
        })

        print(f"    v_DW = {velocity:.2f} m/s, R^2 = {r_value**2:.6f}")

    return results


def plot_results(results, filename="fig_saw_domain_wall.pdf"):
    """Create the SAW DW motion figure.

    Panel (a): DW position vs time for each amplitude.
    Panel (b): DW velocity vs epsilon_0^2 with linear fit.
    """
    fig, axes = ps.double_panel_v(height_cm=12.0, hspace=0.35)

    colors = ps.COLORS_6

    # (a) DW position vs time
    for i, res in enumerate(results):
        t_ns = res['time'] * 1e9
        pos_nm = res['dw_pos'] * 1e9
        label = rf"$\varepsilon_0 = {res['epsilon_0']*1e3:g}\times10^{{-3}}$"
        axes[0].plot(t_ns, pos_nm, color=colors[i], linewidth=1.0, label=label)

    axes[0].set_xlabel(r'$t$ (ns)')
    axes[0].set_ylabel(r'$x_\mathrm{DW}$ (nm)')
    axes[0].legend(fontsize=6.5, loc='center right')
    ps.add_panel_label(axes[0], '(a)')

    # (b) Velocity vs epsilon_0^2 (data points only — no fit)
    eps_arr = np.array([r['epsilon_0'] for r in results])
    vel_arr = np.array([r['velocity'] for r in results])
    eps2 = eps_arr**2

    axes[1].plot(eps2 * 1e6, vel_arr, 'o', color=ps.BLUE,
                 markersize=5, markeredgecolor='white', markeredgewidth=0.4,
                 zorder=10)

    axes[1].set_xlabel(r'$\varepsilon_0^2$ ($\times 10^{-6}$)')
    axes[1].set_ylabel(r'$v_\mathrm{DW}$ (m/s)')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    ps.add_panel_label(axes[1], '(b)')

    fig.savefig(os.path.join(FIG_DIR, filename.replace(".pdf", ".png")), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


def main():
    print("=== Sim 01: SAW-driven domain wall motion ===")

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        results = np.load(CACHE_FILE, allow_pickle=True).item()['results']
    else:
        results = run_dw_sweep()
        np.save(CACHE_FILE, {'results': results})
        print(f"  Data saved to {CACHE_FILE}")

    plot_results(results)


if __name__ == "__main__":
    main()
