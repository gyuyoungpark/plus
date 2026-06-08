"""Simulation 30: Thermal noise contribution to parametric threshold.

Tests the hypothesis that the factor ~30 mismatch between the analytic
zero-temperature Mathieu threshold eps_th^(0) and the observed value
in the main text comes from thermal seeding of the parametric
instability. Compares parametric m_perp(eps0) at four temperatures:

  T in {0, 20, 100, 300} K

at f_SAW = 2 f_K (MEL only, K_mr=0). Predicts:
  - T=0:   no spontaneous seeding -> threshold close to analytic eps_th^(0)
  - T>0:   threshold reduced ~ sqrt(k_B T) (noise floor)

v2 NOTE: v1 (eps=1e-4..3e-3) was FULLY SATURATED for all T, masking
the threshold transition. v2 spans 1e-6..1e-4 to bracket the observed
threshold eps_th^obs ~ 3e-5 from sim21.

Grid: 128x16x1 at 10 nm (even ncells required for thermal noise).
T_RUN = 24 ns, dt_rec = 100 ps. Total runs: 20.
Estimated runtime: ~35 minutes on a single GPU.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0
from saw_chiral import ChiralSurfaceAcousticWave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CACHE_FILE = os.path.join(DATA_DIR, "sim30_thermal_v2.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim30_v2_checkpoint.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 0.0
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 128, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

B0 = 50e-3
T_RUN = 24e-9
DT_REC = 100e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13

T_VALUES = np.array([0.0, 20.0, 100.0, 300.0])
# v2: probe BELOW threshold (eps_th^obs ~ 3e-5) and ABOVE.
# v1 used 1e-4..3e-3 which is fully saturated -> no T-dependence visible.
EPS_VALUES = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4])


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_single(T_K, eps0):
    f_saw = 2 * F_K
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.B1 = B1
    magnet.magnetization = (1, 0, 0.005)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

    magnet.temperature = T_K  # 0 disables noise, >0 activates Langevin

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=eps0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=KMR, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=True)

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    m_perp_arr = np.zeros(NT_REC)
    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        m_perp_arr[i] = np.sqrt(avg[1] ** 2 + avg[2] ** 2)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    n_half = NT_REC // 2
    return float(np.max(m_perp_arr[n_half:]))


def main():
    t_total = time.time()
    n_t = len(T_VALUES)
    n_e = len(EPS_VALUES)

    print("=" * 72)
    print("Sim 30: Thermal noise contribution to parametric threshold")
    print(f"  B0={B0*1e3:.0f}mT, f_K={F_K*1e-9:.2f}GHz")
    print(f"  Grid {NX}x{NY}x{NZ} @ {CX*1e9:.0f}nm, K_mr=0 (MEL only)")
    print(f"  T (K): {T_VALUES}")
    print(f"  eps0:  {EPS_VALUES}")
    print(f"  Total runs: {n_t * n_e}")
    print("=" * 72)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading...")
        data = dict(np.load(CACHE_FILE))
        plot_results(data)
        return

    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        n_done = sum(1 for k in ckpt if k.startswith('m_'))
        print(f"  Checkpoint: {n_done}/{n_t * n_e} runs done")

    m_grid = np.zeros((n_t, n_e))
    for ti, T_K in enumerate(T_VALUES):
        for ei, eps0 in enumerate(EPS_VALUES):
            key = f'm_t{ti}_e{ei}'
            if key in ckpt:
                m_grid[ti, ei] = float(ckpt[key])
                print(f"  T={T_K:.0f}K, eps={eps0:.0e}: "
                      f"|m_perp|={m_grid[ti, ei]:.3e} [ckpt]")
                continue
            t0 = time.time()
            m_grid[ti, ei] = run_single(T_K, eps0)
            ckpt[key] = m_grid[ti, ei]
            np.savez(CHECKPOINT, **ckpt)
            print(f"  T={T_K:.0f}K, eps={eps0:.0e}: "
                  f"|m_perp|={m_grid[ti, ei]:.3e}  ({time.time()-t0:.0f}s)")

    save_dict = {
        'T_values': T_VALUES,
        'eps_values': EPS_VALUES,
        'm_grid': m_grid,
        'B0': B0, 'f_K': F_K, 'MS': MS, 'alpha': ALPHA,
    }
    np.savez(CACHE_FILE, **save_dict)
    if os.path.isfile(CHECKPOINT):
        os.remove(CHECKPOINT)
    print(f"\n  Total: {(time.time()-t_total)/60:.1f} min")
    print(f"  Saved: {CACHE_FILE}")
    plot_results(save_dict)


def plot_results(data):
    try:
        from plot_style import (apply_style, label_panels, DOUBLE_COL,
                                SKY_BLUE, VERMILION, TEAL, BLACK, ORANGE, PINK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        ORANGE, PINK = '#E69F00', '#CC79A7'
        DOUBLE_COL = 7.0

    T_vals = data['T_values']
    eps_vals = data['eps_values']
    m = data['m_grid']

    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL / 2, DOUBLE_COL / 2.4))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.18)

    colors = [BLACK, SKY_BLUE, ORANGE, VERMILION]
    for ti, T_K in enumerate(T_vals):
        label = 'T = 0' if T_K == 0 else f'T = {T_K:.0f} K'
        ax.loglog(eps_vals, m[ti], 'o-', color=colors[ti % len(colors)],
                  ms=4, mec='k', mew=0.3, label=label)
    ax.set_xlabel(r'$\varepsilon_0$')
    ax.set_ylabel(r'$|m_\perp|$ at $2f_K$')
    ax.legend(fontsize=7)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim30_thermal.{ext}'), dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_sim30_thermal.pdf/png")

    print("\n  Threshold separation:")
    for ti, T_K in enumerate(T_vals):
        m0 = m[ti, 0]
        m1 = m[ti, -1]
        print(f"  T={T_K:5.0f}K  m(eps_min)={m0:.2e}  m(eps_max)={m1:.2e}  "
              f"ratio={m1/m0:.2f}")


if __name__ == "__main__":
    main()
