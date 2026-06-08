"""Simulation 24: Grid convergence test.

Verifies that the parametric 2f_K peak and MR f_K peak are not
numerical artifacts by comparing three grid sizes:
  128x16x1, 256x16x1 (reference), 512x16x1

Runs at B0=50 mT with:
  - f_SAW = f_K (MR direct resonance)
  - f_SAW = 2f_K (MEL parametric pump)
for Full and MR-only channels.

Expected runtime: ~1 hour total.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim24_grid_convergence.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
V_SAW = 3500.0

CX, CY, CZ = 10e-9, 10e-9, 20e-9
NY, NZ = 16, 1
EPS0 = 1e-4
B0 = 50e-3

T_RUN = 10e-9
DT_REC = 100e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13

GRID_SIZES = [128, 256, 512]


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)

CONFIGS = [
    ('full_fk',    F_K,     True,  KMR,  'Full @ f_K'),
    ('full_2fk',   2 * F_K, True,  KMR,  'Full @ 2f_K'),
    ('mr_fk',      F_K,     False, KMR,  'MR @ f_K'),
    ('mel_2fk',    2 * F_K, True,  0.0,  'MEL @ 2f_K'),
]


def run_single(nx, f_saw, enable_mel, K_mr):
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((nx, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((nx, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0.01)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

    if enable_mel:
        magnet.B1 = B1

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    m_perp_arr = np.zeros(NT_REC)
    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        m_perp_arr[i] = np.sqrt(avg[1]**2 + avg[2]**2)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    n_half = NT_REC // 2
    return float(np.max(m_perp_arr[n_half:]))


def main():
    total_start = time.time()
    print("=" * 60)
    print("Sim 24: Grid convergence test")
    print(f"  B0 = {B0*1e3:.0f} mT, f_K = {F_K*1e-9:.2f} GHz")
    print(f"  Grids: {GRID_SIZES}")
    print(f"  Total runs: {len(GRID_SIZES) * len(CONFIGS)}")
    print("=" * 60)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading...")
        data = dict(np.load(CACHE_FILE))
        plot_results(data)
        return

    results = {}
    for nx in GRID_SIZES:
        for cfg_key, f_saw, enable_mel, K_mr, label in CONFIGS:
            key = f'{cfg_key}_nx{nx}'
            t0 = time.time()
            print(f"  {nx}x{NY}x{NZ}  {label:16s}...", end="", flush=True)
            peak = run_single(nx, f_saw, enable_mel, K_mr)
            elapsed = time.time() - t0
            results[key] = peak
            print(f"  peak={peak:.2e}  ({elapsed:.0f}s)")

    results['grid_sizes'] = np.array(GRID_SIZES)
    results['f_K'] = F_K
    np.savez(CACHE_FILE, **results)
    total = time.time() - total_start
    print(f"\n  Total: {total/60:.1f} min. Saved: {CACHE_FILE}")
    plot_results(results)


def plot_results(data):
    try:
        from plot_style import (apply_style, axis_label, SINGLE_COL,
                                SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        SINGLE_COL = 3.4

    grids = data['grid_sizes']

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    for cfg_key, _, _, _, label in CONFIGS:
        vals = [float(data[f'{cfg_key}_nx{nx}']) for nx in grids]
        color = {'full_fk': BLACK, 'full_2fk': VERMILION,
                 'mr_fk': SKY_BLUE, 'mel_2fk': TEAL}[cfg_key]
        marker = {'full_fk': 'o', 'full_2fk': '^',
                  'mr_fk': 's', 'mel_2fk': 'D'}[cfg_key]
        ax.plot(grids, vals, f'{marker}-', color=color, ms=6,
                mec='k', mew=0.3, label=label)

    ax.set_xlabel('$N_x$')
    ax.set_ylabel('Peak $|m_\\perp|$')
    ax.set_xticks(grids)
    ax.legend(fontsize=6)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim24_convergence.{ext}'),
                    dpi=200)
    plt.close(fig)
    print("  Saved: fig_sim24_convergence.pdf/png")

    # Print table
    print(f"\n  Grid convergence:")
    print(f"  {'Config':>16s}  {'128':>10s}  {'256':>10s}  {'512':>10s}  {'256/128':>8s}  {'512/256':>8s}")
    for cfg_key, _, _, _, label in CONFIGS:
        vals = [float(data[f'{cfg_key}_nx{nx}']) for nx in grids]
        r1 = vals[1] / vals[0] if vals[0] > 0 else 0
        r2 = vals[2] / vals[1] if vals[1] > 0 else 0
        print(f"  {label:>16s}  {vals[0]:>10.2e}  {vals[1]:>10.2e}  "
              f"{vals[2]:>10.2e}  {r1:>8.3f}  {r2:>8.3f}")


if __name__ == "__main__":
    main()
