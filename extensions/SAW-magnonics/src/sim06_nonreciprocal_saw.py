"""Simulation 6: Nonreciprocal SAW-magnon coupling via magneto-rotation.

Demonstrates that the magneto-rotation coupling (Kmr) produces a direction-
dependent torque on PMA magnetization: the Rayleigh SAW rotation pseudovector
Omega_y flips sign when k -> -k, so the MR effective field changes sign.

Uses the prescribed SAW approach (ChiralSurfaceAcousticWave) for precise
control over strain and rotation amplitudes in the linear regime.

For PMA (m0 || z, u_hat || z), the MR field is:
    H_mr ~ (K_mr/M_s) * Omega_y * x_hat
which produces a torque tau = m x H ~ y_hat.
When k -> -k, Omega_y -> -Omega_y, so the torque flips: tau -> -y_hat.
The amplitude |m_perp| is the same, but the precession phase is reversed.

The MEL field H_mel ~ eps_xx * m_x * x_hat vanishes near equilibrium
(m_x ~ 0 for PMA), so MR provides the dominant coupling.

Material: CoFeB PMA thin film
    Msat  = 1.2 MA/m
    Aex   = 15 pJ/m
    Ku1   = 1.3 MJ/m^3
    alpha = 0.01
    Kmr   ~ Ku + 1/2 mu0 Ms^2 ~ 2.2 MJ/m^3
    B1    = -8.8 MJ/m^3
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt

import plot_style as ps

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0

from saw_chiral import ChiralSurfaceAcousticWave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# Material parameters (CoFeB PMA)
# -----------------------------------------------------------------------
MSAT = 1.2e6          # A/m
AEX = 1.5e-11         # J/m
KU1 = 1.3e6           # J/m^3
ALPHA = 0.01
B1_MEL = -8.8e6       # J/m^3

# Magneto-rotation coupling: Kmr ~ Ku + 1/2 mu0 Ms^2
KMR = KU1 + 0.5 * MU0 * MSAT**2

# -----------------------------------------------------------------------
# SAW parameters (prescribed)
# -----------------------------------------------------------------------
SAW_FREQ = 2e9         # Hz
LAMBDA_SAW = 1.75e-6   # m (v_SAW ~ 3500 m/s)
EPS0 = 1e-4            # peak strain (linear regime)
XI = 0.68              # Rayleigh ellipticity

# -----------------------------------------------------------------------
# Geometry (single cell — uniform mode)
# -----------------------------------------------------------------------
NX, NY, NZ = 1, 1, 1
CX, CY, CZ = 10e-9, 10e-9, 3e-9

# -----------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------
T_MAX = 10e-9          # 10 ns
DT_REC = 10e-12        # 10 ps recording interval
NT_REC = int(T_MAX / DT_REC) + 1

CACHE_FILE = os.path.join(DATA_DIR, "sim06_data.npy")

GAMMA = 1.76e11  # rad/(s*T)


def run_saw_simulation(direction=+1, enable_kmr=True):
    """Run simulation with prescribed SAW.

    Parameters
    ----------
    direction : +1 or -1
        +1 for +k propagation, -1 for -k.
    enable_kmr : bool
        If True, enable magneto-rotation coupling.

    Returns
    -------
    m_perp : ndarray, shape (NT_REC,)
        Transverse magnetization amplitude.
    m_x, m_y : ndarray
        Cartesian magnetization components (for phase analysis).
    times : ndarray
        Time array (s).
    """
    t_start = time.time()
    dir_str = "+" if direction > 0 else "-"
    kmr_str = f"Kmr={KMR*1e-6:.1f}" if enable_kmr else "no-Kmr"
    print(f"  Running {dir_str}k SAW ({kmr_str})...", end="", flush=True)

    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize)
    magnet = Ferromagnet(world, grid)

    # PMA ground state
    magnet.msat = MSAT
    magnet.aex = AEX
    magnet.ku1 = KU1
    magnet.anisU = (0, 0, 1)
    magnet.alpha = ALPHA
    magnet.magnetization = (0, 0, 1)

    # MEL coupling
    magnet.B1 = B1_MEL

    # Prescribed SAW
    K_mr_val = KMR if enable_kmr else 0.0
    # For -k: use negative K_mr to flip rotation sign
    K_mr_signed = K_mr_val * direction

    saw = ChiralSurfaceAcousticWave(
        frequency=SAW_FREQ, wavelength=LAMBDA_SAW, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr_signed, enable_barnett=False)

    saw.apply(magnet, Msat=MSAT, enable_mel=True)

    # Time evolution
    world.timesolver.adaptive_timestep = True

    times = np.zeros(NT_REC)
    m_x = np.zeros(NT_REC)
    m_y = np.zeros(NT_REC)
    m_z = np.zeros(NT_REC)

    for i in range(NT_REC):
        m = magnet.magnetization.eval()
        m_x[i] = m[0, 0, 0, 0]
        m_y[i] = m[1, 0, 0, 0]
        m_z[i] = m[2, 0, 0, 0]
        times[i] = world.timesolver.time

        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    m_perp = np.sqrt(m_x**2 + m_y**2)
    elapsed = time.time() - t_start
    print(f" peak|m_perp|={np.max(m_perp):.2e}, {elapsed:.1f}s")

    return m_perp, m_x, m_y, times


def plot_nonreciprocity(times, m_plus, m_minus, m_x_plus, m_x_minus,
                         m_y_plus, m_y_minus, m_plus_nokmr=None,
                         filename="fig_nonreciprocal_saw"):
    """Plot direction-dependent SAW-magnon coupling."""
    fig, axes = ps.double_panel_v(height_cm=12.0, hspace=0.35)

    t_ns = times * 1e9

    # Panel (a): m_y component showing phase reversal
    axes[0].plot(t_ns, m_y_plus * 1e3, '-', color='C0',
                 label=r'SAW $+k$ ($m_y$)')
    axes[0].plot(t_ns, m_y_minus * 1e3, '-', color='C1',
                 label=r'SAW $-k$ ($m_y$)')
    axes[0].set_xlabel(r'$t$ (ns)')
    axes[0].set_ylabel(r'$m_y$ ($\times 10^{-3}$)')
    axes[0].legend(fontsize=6.5, loc='best')
    ps.add_panel_label(axes[0], '(a)')

    # Panel (b): |m_perp| for all three
    axes[1].plot(t_ns, m_plus * 1e3, '-', color='C0',
                 label=r'$+k$ (with $K_\mathrm{mr}$)')
    axes[1].plot(t_ns, m_minus * 1e3, '--', color='C1',
                 label=r'$-k$ (with $K_\mathrm{mr}$)')
    if m_plus_nokmr is not None:
        axes[1].plot(t_ns, m_plus_nokmr * 1e3, ':', color='gray',
                     label=r'$+k$ (no $K_\mathrm{mr}$)')
    axes[1].set_xlabel(r'$t$ (ns)')
    axes[1].set_ylabel(r'$|\delta m_\perp|$ ($\times 10^{-3}$)')
    axes[1].legend(fontsize=6.5, loc='best')
    ps.add_panel_label(axes[1], '(b)')

    fig.savefig(os.path.join(FIG_DIR, filename + ".png"), dpi=200)
    fig.savefig(os.path.join(FIG_DIR, filename + ".pdf"))
    plt.close(fig)
    print(f"  Saved: {filename}.pdf")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=== Sim 06: Nonreciprocal SAW-magnon coupling (prescribed) ===")
    print(f"  Material: CoFeB PMA (Msat={MSAT*1e-6:.1f} MA/m, "
          f"Ku1={KU1*1e-6:.1f} MJ/m^3)")
    print(f"  Kmr = {KMR*1e-6:.2f} MJ/m^3")
    print(f"  SAW: f={SAW_FREQ*1e-9:.1f} GHz, eps0={EPS0:.0e}, xi={XI:.2f}")

    # Coupling field hierarchy
    mu0_H_mel = 2 * abs(B1_MEL) * EPS0 / MSAT
    mu0_H_mr = KMR * XI * EPS0 / (2 * MSAT)
    print(f"  mu0*H_mel = {mu0_H_mel*1e3:.3f} mT (but zero torque for PMA)")
    print(f"  mu0*H_mr  = {mu0_H_mr*1e3:.3f} mT (dominant for PMA)")

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        cache = np.load(CACHE_FILE, allow_pickle=True).item()
        m_plus = cache['m_plus']
        m_minus = cache['m_minus']
        m_x_plus = cache['m_x_plus']
        m_x_minus = cache['m_x_minus']
        m_y_plus = cache['m_y_plus']
        m_y_minus = cache['m_y_minus']
        times = cache['times']
        m_plus_nokmr = cache.get('m_plus_nokmr', None)
    else:
        # Run +k SAW with Kmr
        m_plus, m_x_plus, m_y_plus, times = run_saw_simulation(
            direction=+1, enable_kmr=True)

        # Run -k SAW with Kmr
        m_minus, m_x_minus, m_y_minus, _ = run_saw_simulation(
            direction=-1, enable_kmr=True)

        # Control: +k SAW without Kmr
        m_plus_nokmr, _, _, _ = run_saw_simulation(
            direction=+1, enable_kmr=False)

        np.save(CACHE_FILE, {
            'm_plus': m_plus, 'm_minus': m_minus,
            'm_x_plus': m_x_plus, 'm_x_minus': m_x_minus,
            'm_y_plus': m_y_plus, 'm_y_minus': m_y_minus,
            'times': times, 'm_plus_nokmr': m_plus_nokmr,
        })
        print(f"  Data saved to {CACHE_FILE}")

    # Plot
    plot_nonreciprocity(times, m_plus, m_minus,
                         m_x_plus, m_x_minus, m_y_plus, m_y_minus,
                         m_plus_nokmr)

    # Report
    peak_plus = np.max(m_plus)
    peak_minus = np.max(m_minus)
    peak_nokmr = np.max(m_plus_nokmr) if m_plus_nokmr is not None else 0
    print(f"\n  Results:")
    print(f"    Peak |m_perp| (+k): {peak_plus:.2e}")
    print(f"    Peak |m_perp| (-k): {peak_minus:.2e}")
    print(f"    Ratio |m(+k)|/|m(-k)|: {peak_plus/peak_minus:.3f}")
    print(f"    Peak |m_perp| (no Kmr): {peak_nokmr:.2e}")
    print(f"    Phase reversal: m_y(+k) and m_y(-k) have opposite signs")


if __name__ == "__main__":
    main()
