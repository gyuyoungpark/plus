"""Simulation 22: Fine angle sweep near crossover angle theta_c.

High-resolution angle dependence to resolve the destructive
MEL-MR interference dip near theta_c ~ 1.1 deg.

Grid: 256x16x1 at 10 nm (full micromagnetic)
Material: YIG (unified PRL parameters)
B0 at Kittel resonance for f = 3 GHz, f_SAW = 3 GHz
Angles: 20 values with fine sampling near theta_c + coarse at large angles
Full coupling (MEL+MR)

Estimated runtime: ~2-3 hours on a single GPU.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim22_fine_angle.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim22_checkpoint.npz")

# ==========================================================================
# Material: YIG (unified)
# ==========================================================================
GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68

# ==========================================================================
# Geometry
# ==========================================================================
NX, NY, NZ = 256, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

# ==========================================================================
# SAW parameters
# ==========================================================================
F_SAW = 3.0e9
LAMBDA_SAW = 1.16e-6   # V_SAW / F_SAW ~ 3500/3e9
EPS0 = 1e-4

T_RUN = 10e-9
DT_REC = 100e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13        # smaller timestep for oblique angles

# Fine near theta_c, coarser at large angles
ANGLES = np.array([
    0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1, 1.2, 1.5,
    2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 45.0,
])


def find_resonance_field(f_target):
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * MS, -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4 * a * c)) / 2


B0_RES = find_resonance_field(F_SAW)


def run_oblique(theta_deg, B0, enable_mel=True, K_mr=KMR):
    """Run full coupling at oblique angle. Returns peak |m_perp|."""
    theta = np.radians(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (ct, st, 0)
    magnet.bias_magnetic_field = (B0 * ct, B0 * st, 0)
    magnet.enable_demag = True

    if enable_mel:
        magnet.B1 = B1

    saw = ChiralSurfaceAcousticWave(
        frequency=F_SAW, wavelength=LAMBDA_SAW, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    m_perp_arr = np.zeros(NT_REC)
    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        dm_y = avg[1] - st
        m_perp_arr[i] = np.sqrt(dm_y**2 + avg[2]**2)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    n_quarter = NT_REC // 4
    return float(np.max(m_perp_arr[n_quarter:]))


def main():
    total_start = time.time()
    theta_c = np.degrees(np.arcsin(KMR * XI / (4 * abs(B1))))

    print("=" * 72)
    print("Sim 22: Fine angle sweep near theta_c")
    print(f"  Grid: {NX}x{NY}x{NZ}, PBC=(4,4,0), demag=ON")
    print(f"  B0_res = {B0_RES*1e3:.2f} mT, f_SAW = {F_SAW*1e-9:.1f} GHz")
    print(f"  theta_c = {theta_c:.2f} deg")
    print(f"  Angles: {len(ANGLES)} values, 0-{ANGLES[-1]:.0f} deg")
    print("=" * 72)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading and plotting...")
        data = dict(np.load(CACHE_FILE))
        plot_results(data)
        return

    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        n_done = sum(1 for k in ckpt if k.startswith('peak_'))
        print(f"  Checkpoint loaded ({n_done}/{len(ANGLES)} done)")

    peaks = np.zeros(len(ANGLES))

    for i, theta in enumerate(ANGLES):
        key = f'peak_{theta:.1f}'
        if key in ckpt:
            peaks[i] = float(ckpt[key])
            print(f"  [{i+1:2d}/{len(ANGLES)}] theta={theta:5.1f} deg  "
                  f"[checkpoint] peak={peaks[i]:.2e}")
            continue

        t0 = time.time()
        peaks[i] = run_oblique(theta, B0_RES)
        dt = time.time() - t0
        print(f"  [{i+1:2d}/{len(ANGLES)}] theta={theta:5.1f} deg  "
              f"peak={peaks[i]:.2e}  ({dt:.1f}s)")

        ckpt[key] = np.array(peaks[i])
        np.savez(CHECKPOINT, **ckpt)

    np.savez(CACHE_FILE, angles=ANGLES, peaks=peaks,
             B0_res=B0_RES, theta_c=theta_c)
    if os.path.isfile(CHECKPOINT):
        os.remove(CHECKPOINT)

    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Saved: {CACHE_FILE}")
    plot_results({'angles': ANGLES, 'peaks': peaks, 'theta_c': theta_c})


def plot_results(data):
    try:
        from plot_style import (apply_style, label_panels, axis_label,
                                SINGLE_COL, CM_TO_INCH,
                                SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, BLACK = '#56B4E9', '#D55E00', '#000000'
        SINGLE_COL = 3.4

    angles = data['angles']
    peaks = data['peaks']
    theta_c = float(data['theta_c'])
    norm = peaks[0] if peaks[0] > 0 else 1

    # Analytic coupling rate
    theta_an = np.linspace(0, np.pi / 2, 500)
    g_mel = GAMMA * abs(B1) * EPS0 * np.abs(np.sin(2 * theta_an)) / MS
    g_mr = GAMMA * KMR * XI * EPS0 * np.abs(np.cos(theta_an)) / (2 * MS)
    g_total = np.sqrt(g_mel**2 + g_mr**2)
    g_norm = g_total / g_total[0] if g_total[0] > 0 else g_total

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    ax.plot(angles, peaks / norm, 'o-', color=SKY_BLUE, ms=5,
            mec='k', mew=0.3, lw=1.0, label='Micromagnetic', zorder=5)
    ax.plot(np.degrees(theta_an), g_norm, 'k:', lw=0.6, alpha=0.5,
            label=r'Linear $g_\mathrm{total}$')

    ax.axvline(theta_c, ls=':', color='gray', lw=0.5)
    ax.text(theta_c + 0.3, 0.05, f'$\\theta_c$={theta_c:.1f}°',
            fontsize=7, color='gray', va='bottom')

    # Mark dip
    dip_mask = angles < 3
    if np.sum(dip_mask) > 2:
        dip_idx = np.argmin(peaks[dip_mask])
        if peaks[dip_mask][dip_idx] < peaks[0] * 0.5:
            ax.annotate('', xy=(angles[dip_mask][dip_idx],
                        peaks[dip_mask][dip_idx] / norm),
                        xytext=(angles[dip_mask][dip_idx], 0.8),
                        arrowprops=dict(arrowstyle='->', color=VERMILION,
                                        lw=1.0))

    ax.set_xlabel(r'$\theta$ (deg)')
    ax.set_ylabel('Normalized $|m_\\perp|$')
    ax.set_xlim(-0.5, 47)
    ax.legend(fontsize=7)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim22_fine_angle.{ext}'), dpi=200)
    plt.close(fig)
    print("  Saved: fig_sim22_fine_angle.pdf/png")

    # Summary
    print(f"\n  Angle   |  peak |m_perp|  |  normalized")
    for i, theta in enumerate(angles):
        print(f"  {theta:6.1f}  |  {peaks[i]:.3e}     |  {peaks[i]/norm:.4f}")


if __name__ == "__main__":
    main()
