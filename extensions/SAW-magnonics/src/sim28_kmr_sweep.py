"""Simulation 28: K_mr universal scaling for direct MR excitation.

Defends K_mr=10^6 J/m^3 choice against the "phenomenological parameter"
critique. Predicts and verifies universal linear scaling

  |m_perp|(K_mr, eps0) ~ (K_mr * eps0) / (alpha * MS * omega_K)

at f_SAW = f_K (MR-only direct excitation, MEL disabled).

Sweeps:
  K_mr in {3e4, 1e5, 3e5, 1e6, 3e6}  J/m^3   (literature 10^4-10^7)
  eps0 in {3e-5, 1e-4, 3e-4, 1e-3}            (overlap with sim21)

Grid: 128x16x1 at 10 nm. Material: YIG (sim20 params, alpha=5e-4).
T_RUN = 8 ns, dt_rec = 100 ps. Total runs: 20.
Estimated runtime: ~12 minutes on a single GPU.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim28_kmr_sweep.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim28_checkpoint.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 128, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

B0 = 50e-3
T_RUN = 8e-9
DT_REC = 100e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13

KMR_VALUES = np.array([3e4, 1e5, 3e5, 1e6, 3e6])
EPS_VALUES = np.array([3e-5, 1e-4, 3e-4, 1e-3])


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_single(K_mr, eps0):
    f_saw = F_K
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0.005)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=eps0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=False)

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
    n_k = len(KMR_VALUES)
    n_e = len(EPS_VALUES)

    print("=" * 72)
    print("Sim 28: K_mr universal scaling for direct MR excitation")
    print(f"  B0={B0*1e3:.0f}mT, f_K={F_K*1e-9:.2f}GHz")
    print(f"  Grid {NX}x{NY}x{NZ} @ {CX*1e9:.0f}nm, alpha={ALPHA}")
    print(f"  K_mr: {KMR_VALUES}")
    print(f"  eps0: {EPS_VALUES}")
    print(f"  Total runs: {n_k * n_e}")
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
        print(f"  Checkpoint: {n_done}/{n_k * n_e} runs done")

    m_grid = np.zeros((n_k, n_e))
    for ki, K_mr in enumerate(KMR_VALUES):
        for ei, eps0 in enumerate(EPS_VALUES):
            key = f'm_k{ki}_e{ei}'
            if key in ckpt:
                m_grid[ki, ei] = float(ckpt[key])
                print(f"  K_mr={K_mr:.0e}, eps={eps0:.0e}: "
                      f"|m_perp|={m_grid[ki, ei]:.3e} [ckpt]")
                continue

            t0 = time.time()
            m_grid[ki, ei] = run_single(K_mr, eps0)
            ckpt[key] = m_grid[ki, ei]
            np.savez(CHECKPOINT, **ckpt)
            print(f"  K_mr={K_mr:.0e}, eps={eps0:.0e}: "
                  f"|m_perp|={m_grid[ki, ei]:.3e}  ({time.time()-t0:.0f}s)")

    save_dict = {
        'kmr_values': KMR_VALUES,
        'eps_values': EPS_VALUES,
        'm_grid': m_grid,
        'B0': B0, 'f_K': F_K,
        'alpha': ALPHA, 'MS': MS,
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

    kmr = data['kmr_values']
    eps = data['eps_values']
    m = data['m_grid']

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.6))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97,
                        top=0.92, bottom=0.18)

    colors = [SKY_BLUE, TEAL, ORANGE, VERMILION, PINK]

    # (a) |m_perp| vs eps0 for each K_mr
    ax = axes[0]
    for ki, K_mr in enumerate(kmr):
        ax.loglog(eps, m[ki], 'o-', color=colors[ki % len(colors)],
                  ms=4, mec='k', mew=0.3,
                  label=f'$K_\\mathrm{{mr}}=$ {K_mr:.0e}')
    eps_ref = np.array([eps[0], eps[-1]])
    y_ref = m[0, 0] * (eps_ref / eps[0])
    ax.plot(eps_ref, y_ref, 'k:', lw=0.5, alpha=0.5)
    ax.text(eps_ref[-1] * 0.5, y_ref[-1] * 1.5,
            r'$\propto \varepsilon_0$', fontsize=7, color='gray')
    ax.set_xlabel(r'$\varepsilon_0$')
    ax.set_ylabel(r'$|m_\perp|$')
    ax.legend(fontsize=6, loc='lower right')

    # (b) Universal collapse: |m_perp| vs K_mr*eps0
    ax = axes[1]
    KK, EE = np.meshgrid(kmr, eps, indexing='ij')
    x_universal = (KK * EE).ravel()
    y_universal = m.ravel()
    order = np.argsort(x_universal)
    ax.loglog(x_universal[order], y_universal[order], 'o',
              color=BLACK, ms=4, mec='k', mew=0.3,
              label='simulation')
    # Linear fit slope=1
    log_x = np.log10(x_universal[y_universal > 0])
    log_y = np.log10(y_universal[y_universal > 0])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    x_fit = np.array([x_universal[order][0], x_universal[order][-1]])
    ax.plot(x_fit, 10 ** (slope * np.log10(x_fit) + intercept),
            '--', color=VERMILION, lw=0.8,
            label=f'slope = {slope:.2f}')
    ax.set_xlabel(r'$K_\mathrm{mr}\cdot\varepsilon_0$ (J m$^{-3}$)')
    ax.set_ylabel(r'$|m_\perp|$')
    ax.legend(fontsize=7, loc='upper left')

    try:
        label_panels(axes)
    except Exception:
        pass

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim28_kmr_sweep.{ext}'), dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_sim28_kmr_sweep.pdf/png  (collapse slope={slope:.3f})")


if __name__ == "__main__":
    main()
