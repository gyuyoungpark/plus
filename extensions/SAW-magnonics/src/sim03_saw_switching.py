"""Simulation 3: SAW-assisted magnetization switching.

A PMA thin film with bistable up/down magnetization is subjected to a
SAW burst combined with a small static assist field.  The SAW provides
the energy to overcome the anisotropy barrier while the assist field
breaks the up/down symmetry.

The switching boundary in the (epsilon_0, B_assist) plane defines a
phase diagram:
    - Large assist field + small SAW -> switching
    - Large SAW + small assist field -> switching
    - Below the boundary -> no switching

Protocol:
  (a) Fix B_assist, sweep epsilon_0: show mz(t) time traces.
  (b) 2D sweep over (epsilon_0, B_assist): map final mz to a
      phase diagram.

Material: CoFeB PMA
    Msat  = 1.0 MA/m
    Aex   = 15 pJ/m
    Ku1   = 800 kJ/m^3
    alpha = 0.05
    B1    = -8.8 MJ/m^3
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import plot_style as ps
from saw import SurfaceAcousticWave, GAMMA, MU0

from mumaxplus import Ferromagnet, Grid, World

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim03_data.npy")


# -----------------------------------------------------------------------
# Material and geometry parameters
# -----------------------------------------------------------------------
MSAT = 1.0e6         # A/m
AEX = 1.5e-11        # J/m
ALPHA = 0.05         # higher damping for faster switching
KU1 = 0.8e6          # J/m^3  (PMA)
B1 = -8.8e6          # J/m^3
B2 = 0.0

# Grid
NX, NY, NZ = 32, 32, 1
CX, CY, CZ = 4e-9, 4e-9, 1e-9

# SAW burst parameters
V_SAW = 3500.0       # m/s
F_SAW = 5e9          # Hz (near FMR for effective energy transfer)
LAMBDA_SAW = V_SAW / F_SAW
T_BURST_CENTER = 2e-9    # burst center time (s)
T_BURST_SIGMA = 1e-9     # burst width (s)

# Simulation
T_RUN = 10e-9        # s
N_STEPS = 2000


def gaussian_envelope(t, t0=T_BURST_CENTER, sigma=T_BURST_SIGMA):
    """Gaussian SAW burst envelope."""
    return np.exp(-0.5 * ((t - t0) / sigma)**2)


def run_single_switch(epsilon_0, B_assist):
    """Run a single switching attempt.

    Parameters
    ----------
    epsilon_0 : float
        Peak SAW strain amplitude.
    B_assist : float
        Static assist field along -z (T, positive value means field in -z).

    Returns
    -------
    time : array
        Time array (s).
    mz : array
        Average mz vs time.
    final_mz : float
        Final average mz.
    """
    cellsize = (CX, CY, CZ)
    world = World(cellsize)
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MSAT
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.ku1 = KU1
    magnet.anisU = (0, 0, 1)
    magnet.magnetization = (0.01, 0, 1)   # small tilt to seed SAW interaction

    # Assist field (negative z to bias toward switching)
    magnet.bias_magnetic_field = (0, 0, -B_assist)

    # Magnetoelastic
    magnet.B1 = B1
    magnet.B2 = B2

    # SAW burst
    saw = SurfaceAcousticWave(
        F_SAW, LAMBDA_SAW, epsilon_0,
        direction='x',
        envelope=gaussian_envelope)
    saw.apply(magnet)

    # Run
    time_arr = np.linspace(0, T_RUN, N_STEPS + 1)
    quantity_dict = {"mz": lambda: magnet.magnetization.average()[2]}
    output = world.timesolver.solve(time_arr, quantity_dict)

    t = np.array(output["time"])
    mz = np.array(output["mz"])

    return t, mz, mz[-1]


def run_time_traces():
    """Run time traces at fixed B_assist, varying epsilon_0.

    Returns
    -------
    traces : list of dict
    """
    # Choose B_assist just below the effective switching field (including demag)
    # so the SAW can tip the balance for large enough amplitude
    K_eff = KU1 - 0.5 * MU0 * MSAT**2
    B_eff = 2 * K_eff / MSAT  # effective switching field in T
    B_assist = 0.92 * B_eff

    amplitudes = [2e-3, 5e-3, 8e-3, 12e-3]
    traces = []

    print(f"  Time traces (B_assist = {B_assist*1e3:.0f} mT):")

    for eps0 in amplitudes:
        t, mz, final = run_single_switch(eps0, B_assist)
        switched = final < 0
        status = "SWITCHED" if switched else "not switched"
        print(f"    eps0 = {eps0*1e3:.0f}e-3: final mz = {final:.3f} ({status})")
        traces.append({
            'epsilon_0': eps0,
            'B_assist': B_assist,
            'time': t,
            'mz': mz,
            'final_mz': final,
            'switched': switched,
        })

    return traces


def run_phase_diagram():
    """2D sweep over (epsilon_0, B_assist) to map the switching boundary.

    Returns
    -------
    eps_arr, Ba_arr : arrays
    mz_map : 2D array of final mz values
    """
    n_eps = 16
    n_Ba = 16

    eps_arr = np.linspace(0.5e-3, 15e-3, n_eps)

    # B_assist range: 0 to ~110% of effective anisotropy field (including demag)
    K_eff = KU1 - 0.5 * MU0 * MSAT**2
    B_eff = 2 * K_eff / MSAT  # effective switching field in T
    Ba_arr = np.linspace(0.1 * B_eff, 1.1 * B_eff, n_Ba)

    mz_map = np.zeros((n_Ba, n_eps))

    total = n_eps * n_Ba
    count = 0

    print(f"  Phase diagram: {n_eps} x {n_Ba} = {total} points")

    for i, Ba in enumerate(Ba_arr):
        for j, eps0 in enumerate(eps_arr):
            _, _, final_mz = run_single_switch(eps0, Ba)
            mz_map[i, j] = final_mz
            count += 1

            if count % 20 == 0:
                print(f"    [{100*count/total:5.1f}%] "
                      f"eps0={eps0*1e3:.1f}e-3, "
                      f"Ba={Ba*1e3:.0f} mT, "
                      f"mz={final_mz:.2f}")

    return eps_arr, Ba_arr, mz_map


def plot_results(traces, eps_arr, Ba_arr, mz_map,
                 filename="fig_saw_switching.pdf"):
    """Create the SAW-assisted switching figure.

    Panel (a): mz vs time traces (switching / no-switching).
    Panel (b): Phase diagram in (epsilon_0, B_assist) plane.
    """
    fig, axes = ps.double_panel_v(height_cm=12.0, hspace=0.35)

    colors = ps.COLORS_4

    # (a) Time traces
    for i, tr in enumerate(traces):
        t_ns = tr['time'] * 1e9
        label = rf"$\varepsilon_0 = {tr['epsilon_0']*1e3:.0f}\times10^{{-3}}$"
        axes[0].plot(t_ns, tr['mz'], color=colors[i], linewidth=1.0,
                     label=label)

    # Mark burst window
    t_burst = np.linspace(0, T_RUN, 200) * 1e9
    env = gaussian_envelope(t_burst * 1e-9)
    axes[0].fill_between(t_burst, -1.05, -1.05 + 0.15 * env,
                         color='gray', alpha=0.3, zorder=0)
    axes[0].text(T_BURST_CENTER * 1e9, -0.88, 'SAW burst',
                 fontsize=6, color='gray', ha='center')

    axes[0].axhline(y=0, color='gray', linewidth=0.3, linestyle=':')
    axes[0].set_xlabel(r'$t$ (ns)')
    axes[0].set_ylabel(r'$\langle m_z \rangle$')
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_xlim(0, T_RUN * 1e9)
    axes[0].legend(fontsize=6, loc='center right')
    ps.add_panel_label(axes[0], '(a)')

    # (b) Phase diagram
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    pcm = axes[1].pcolormesh(
        eps_arr * 1e3, Ba_arr * 1e3, mz_map,
        cmap='RdBu', norm=norm, shading='auto')

    # Label regions
    eps_mid = 0.5 * (eps_arr[0] + eps_arr[-1]) * 1e3
    Ba_high = 0.75 * (Ba_arr[0] + Ba_arr[-1]) * 1e3
    Ba_low = 0.30 * (Ba_arr[0] + Ba_arr[-1]) * 1e3
    axes[1].text(eps_mid * 1.5, Ba_high, 'Switched',
                 fontsize=8, ha='center', va='center',
                 fontweight='bold', color='white')
    axes[1].text(eps_mid * 0.55, Ba_low, 'Not switched',
                 fontsize=8, ha='center', va='center',
                 fontweight='bold', color='white')

    cb = fig.colorbar(pcm, ax=axes[1], pad=0.02, shrink=0.9)
    cb.set_label(r'Final $\langle m_z \rangle$')

    axes[1].set_xlabel(r'$\varepsilon_0$ ($\times 10^{-3}$)')
    axes[1].set_ylabel(r'$B_\mathrm{assist}$ (mT)')
    ps.add_panel_label(axes[1], '(b)')

    fig.savefig(os.path.join(FIG_DIR, filename.replace(".pdf", ".png")), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


def main():
    print("=== Sim 03: SAW-assisted magnetization switching ===")

    if os.path.isfile(CACHE_FILE):
        print("  Loading cached data...")
        cache = np.load(CACHE_FILE, allow_pickle=True).item()
        traces = cache['traces']
        eps_arr = cache['eps_arr']
        Ba_arr = cache['Ba_arr']
        mz_map = cache['mz_map']
    else:
        print("\n--- Part (a): Time traces ---")
        traces = run_time_traces()

        print("\n--- Part (b): Phase diagram ---")
        eps_arr, Ba_arr, mz_map = run_phase_diagram()

        np.save(CACHE_FILE, {
            'traces': traces, 'eps_arr': eps_arr,
            'Ba_arr': Ba_arr, 'mz_map': mz_map,
        })
        print(f"  Data saved to {CACHE_FILE}")

    plot_results(traces, eps_arr, Ba_arr, mz_map)


if __name__ == "__main__":
    main()
