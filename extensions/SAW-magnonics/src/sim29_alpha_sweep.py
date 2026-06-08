"""Simulation 29: Damping scaling of parametric MEL threshold.

Tests Mathieu-equation finite-damping prediction

   eps_th(alpha) ~ sqrt(alpha)        (linear in alpha for power)

at f_SAW = 2 f_K (parametric MEL pump, MR disabled). Confirms that
the factor ~30 mismatch between analytic eps_th^(0) and observed value
in the main text scales with damping as expected.

Sweeps:
  alpha in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}     (5 values)
  eps0  in {3e-4, 1e-3, 3e-3, 1e-2}            (4 values, span threshold)

Grid: 128x16x1 at 10 nm. Material: YIG (sim20 params, K_mr=0).
T_RUN = 16 ns, dt_rec = 100 ps. Total runs: 20.
Estimated runtime: ~30 minutes on a single GPU.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim29_alpha_sweep.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim29_checkpoint.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
B1 = -8.8e6
KMR = 0.0
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 128, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

B0 = 50e-3
T_RUN = 16e-9
DT_REC = 100e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13

ALPHA_VALUES = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
EPS_VALUES = np.array([3e-4, 1e-3, 3e-3, 1e-2])


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_single(alpha, eps0):
    f_saw = 2 * F_K
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = alpha
    magnet.B1 = B1
    magnet.magnetization = (1, 0, 0.005)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

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
    n_a = len(ALPHA_VALUES)
    n_e = len(EPS_VALUES)

    print("=" * 72)
    print("Sim 29: Damping scaling of parametric MEL threshold")
    print(f"  B0={B0*1e3:.0f}mT, f_K={F_K*1e-9:.2f}GHz, 2f_K={2*F_K*1e-9:.2f}GHz")
    print(f"  Grid {NX}x{NY}x{NZ} @ {CX*1e9:.0f}nm, K_mr=0 (MEL only)")
    print(f"  alpha: {ALPHA_VALUES}")
    print(f"  eps0:  {EPS_VALUES}")
    print(f"  Total runs: {n_a * n_e}")
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
        print(f"  Checkpoint: {n_done}/{n_a * n_e} runs done")

    m_grid = np.zeros((n_a, n_e))
    for ai, alpha in enumerate(ALPHA_VALUES):
        for ei, eps0 in enumerate(EPS_VALUES):
            key = f'm_a{ai}_e{ei}'
            if key in ckpt:
                m_grid[ai, ei] = float(ckpt[key])
                print(f"  alpha={alpha:.0e}, eps={eps0:.0e}: "
                      f"|m_perp|={m_grid[ai, ei]:.3e} [ckpt]")
                continue
            t0 = time.time()
            m_grid[ai, ei] = run_single(alpha, eps0)
            ckpt[key] = m_grid[ai, ei]
            np.savez(CHECKPOINT, **ckpt)
            print(f"  alpha={alpha:.0e}, eps={eps0:.0e}: "
                  f"|m_perp|={m_grid[ai, ei]:.3e}  ({time.time()-t0:.0f}s)")

    save_dict = {
        'alpha_values': ALPHA_VALUES,
        'eps_values': EPS_VALUES,
        'm_grid': m_grid,
        'B0': B0, 'f_K': F_K, 'MS': MS,
    }
    np.savez(CACHE_FILE, **save_dict)
    if os.path.isfile(CHECKPOINT):
        os.remove(CHECKPOINT)
    print(f"\n  Total: {(time.time()-t_total)/60:.1f} min")
    print(f"  Saved: {CACHE_FILE}")
    plot_results(save_dict)


def estimate_threshold(eps, m_perp, baseline_quantile=0.0):
    """Crude threshold: smallest eps for which m_perp > 3 x baseline."""
    baseline = max(np.min(m_perp), 1e-8)
    above = np.where(m_perp > 3 * baseline)[0]
    if len(above) == 0:
        return np.nan
    if above[0] == 0:
        return eps[0]
    # Interpolate in log-log
    i = above[0]
    log_eps = np.log10([eps[i - 1], eps[i]])
    log_m = np.log10([m_perp[i - 1], m_perp[i]])
    log_target = np.log10(3 * baseline)
    f = (log_target - log_m[0]) / (log_m[1] - log_m[0] + 1e-30)
    return 10 ** (log_eps[0] + f * (log_eps[1] - log_eps[0]))


def plot_results(data):
    try:
        from plot_style import (apply_style, label_panels, DOUBLE_COL,
                                SKY_BLUE, VERMILION, TEAL, BLACK, ORANGE, PINK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        ORANGE, PINK = '#E69F00', '#CC79A7'
        DOUBLE_COL = 7.0

    alpha_vals = data['alpha_values']
    eps_vals = data['eps_values']
    m = data['m_grid']

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.6))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97,
                        top=0.92, bottom=0.18)

    colors = [SKY_BLUE, TEAL, ORANGE, VERMILION, PINK]

    # (a) |m_perp| vs eps0 for each alpha
    ax = axes[0]
    for ai, a_val in enumerate(alpha_vals):
        ax.loglog(eps_vals, m[ai], 'o-', color=colors[ai % len(colors)],
                  ms=4, mec='k', mew=0.3,
                  label=f'$\\alpha=$ {a_val:.0e}')
    ax.set_xlabel(r'$\varepsilon_0$')
    ax.set_ylabel(r'$|m_\perp|$ at $2f_K$')
    ax.legend(fontsize=6, loc='lower right')

    # (b) eps_th vs alpha
    ax = axes[1]
    eps_th = np.array([estimate_threshold(eps_vals, m[ai])
                       for ai in range(len(alpha_vals))])
    valid = ~np.isnan(eps_th)
    ax.loglog(alpha_vals[valid], eps_th[valid], 'o', color=BLACK,
              ms=5, mec='k', mew=0.3, label='simulation')
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(np.log10(alpha_vals[valid]),
                                       np.log10(eps_th[valid]), 1)
        a_fit = np.array([alpha_vals[valid][0], alpha_vals[valid][-1]])
        ax.plot(a_fit, 10 ** (slope * np.log10(a_fit) + intercept),
                '--', color=VERMILION, lw=0.8,
                label=f'slope = {slope:.2f}')
    a_ref = alpha_vals[valid] if valid.sum() else alpha_vals
    e0 = eps_th[valid][0] if valid.sum() else 1e-3
    ax.plot(a_ref, e0 * (a_ref / a_ref[0]),
            ':', color=TEAL, lw=0.7,
            label=r'analytic $\varepsilon_0^\mathrm{asy}\propto\alpha$')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\varepsilon_\mathrm{th}$')
    ax.legend(fontsize=7, loc='lower right')

    try:
        label_panels(axes)
    except Exception:
        pass

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim29_alpha_sweep.{ext}'), dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_sim29_alpha_sweep.pdf/png")
    if valid.sum() >= 2:
        print(f"  eps_th ~ alpha^{slope:.2f}  (Mathieu predicts 0.5)")


if __name__ == "__main__":
    main()
