"""Simulation 26b: Fine 19-point angular sweep at f_SAW = 2 f_K.

Extends sim26 (7 angles) to 19 angles (5 deg spacing) for quantitative
Floquet comparison.  The Floquet analysis in Sec.~S15 of the
supplemental predicts a destructive-interference dip at

   sin^2(theta_c) = xi * cos(theta_c)        (xi = 0.68)
   -> theta_c approx 45.1 deg

between the two MEL pump channels e_xx (sin^2 theta) and e_xz
(xi cos theta).  This script provides the dense angular profile
needed to test that prediction.

Sweeps theta in {0, 5, 10, ..., 90} deg at f_SAW = 2 f_K.
Records peak |m_perp| for MEL-only (clean parametric) and Full
(MEL+MR, realistic) configurations.

Grid: 256x16x1 at 10 nm. Material: YIG.
T_RUN = 12 ns, dt_rec = 100 ps. Total runs: 38.
Estimated runtime: ~50 minutes on a single GPU.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim26b_angular_fine.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim26b_checkpoint.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 256, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9
EPS0 = 1e-4
B0 = 50e-3

T_RUN = 12e-9
DT_REC = 100e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13

THETAS_DEG = np.arange(0.0, 91.0, 5.0)


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_one(theta_deg, enable_mel, K_mr):
    f_saw = 2 * F_K
    wavelength = V_SAW / f_saw
    th = np.radians(theta_deg)
    mx0, my0 = np.cos(th), np.sin(th)
    Bx, By = B0 * np.cos(th), B0 * np.sin(th)

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (mx0, my0, 0.005)
    magnet.bias_magnetic_field = (Bx, By, 0)
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
        m_perp_in = -avg[0] * my0 + avg[1] * mx0
        m_perp_arr[i] = np.sqrt(m_perp_in ** 2 + avg[2] ** 2)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    n_half = NT_REC // 2
    return float(np.max(m_perp_arr[n_half:]))


def main():
    t_total = time.time()
    n_th = len(THETAS_DEG)

    print("=" * 72)
    print(f"Sim 26b: Fine angular sweep at 2 f_K  ({n_th} angles)")
    print(f"  B0={B0*1e3:.0f}mT, f_K={F_K*1e-9:.2f}GHz")
    print(f"  Predict: dip at theta_c where sin^2(th)=xi*cos(th)")
    print("=" * 72)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading...")
        data = dict(np.load(CACHE_FILE))
        plot_angular(data)
        return

    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        n_done = sum(1 for k in ckpt if k.startswith('p_'))
        print(f"  Checkpoint: {n_done}/{2*n_th} runs done")

    peaks_mel = np.zeros(n_th)
    peaks_full = np.zeros(n_th)

    for i, th in enumerate(THETAS_DEG):
        for label, enable_mel, K_mr, store in [
                ('MEL',  True, 0,   peaks_mel),
                ('Full', True, KMR, peaks_full)]:
            key = f'p_{label}_{i:02d}'
            if key in ckpt:
                store[i] = float(ckpt[key])
                print(f"  [{label}] th={th:5.1f} peak={store[i]:.3e} [ckpt]")
                continue
            t0 = time.time()
            store[i] = run_one(th, enable_mel, K_mr)
            ckpt[key] = store[i]
            np.savez(CHECKPOINT, **ckpt)
            print(f"  [{label}] th={th:5.1f} peak={store[i]:.3e} "
                  f"({time.time()-t0:.0f}s)")

    save_dict = {
        'thetas_deg': THETAS_DEG,
        'peaks_mel': peaks_mel,
        'peaks_full': peaks_full,
        'B0': B0, 'f_K': F_K, 'XI': XI,
    }
    np.savez(CACHE_FILE, **save_dict)
    if os.path.isfile(CHECKPOINT):
        os.remove(CHECKPOINT)
    print(f"\n  Total: {(time.time()-t_total)/60:.1f} min")
    print(f"  Saved: {CACHE_FILE}")
    plot_angular(save_dict)


def floquet_angular(theta_rad, xi):
    """Analytic Floquet pump amplitude with destructive interference.

    h(theta)/h_0 = | sin^2(theta) - xi * cos(theta) |
    Zero at sin^2(th) = xi*cos(th) -> cos(th) = (-xi+sqrt(xi^2+4))/2.
    """
    return np.abs(np.sin(theta_rad) ** 2 - xi * np.cos(theta_rad))


def plot_angular(data):
    try:
        from plot_style import (apply_style, label_panels, DOUBLE_COL,
                                SINGLE_COL, SKY_BLUE, VERMILION, TEAL,
                                BLACK, ORANGE)
        apply_style()
    except ImportError:
        SINGLE_COL, DOUBLE_COL = 3.4, 7.0
        SKY_BLUE, VERMILION, TEAL, BLACK, ORANGE = \
            '#56B4E9', '#D55E00', '#009E73', '#000', '#E69F00'

    thetas = data['thetas_deg']
    p_mel = data['peaks_mel']
    p_full = data['peaks_full']
    xi = float(data['XI']) if 'XI' in data else 0.68

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.6))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97,
                        top=0.92, bottom=0.18)

    # (a) Sim data + Eq. (4)-based MEL coupling-element reference
    ax = axes[0]
    ax.plot(thetas, p_mel / p_mel[0], 'o', color=VERMILION, ms=4,
            mec='k', mew=0.3, label='MEL only (sim)')
    ax.plot(thetas, p_full / p_full[0], 's', color=BLACK, ms=4,
            mec='k', mew=0.3, label='Full (sim)', alpha=0.6)

    # Eq. (4) reference: A |cos 2theta|^2 + B cos^4 theta with A, B fitted
    # to the MEL-only data (positive least squares).
    th_dense = np.linspace(0, np.pi / 2, 361)
    w_in = np.cos(2 * th_dense) ** 2
    w_out = np.cos(th_dense) ** 4
    th_data = np.radians(thetas)
    W = np.vstack([np.cos(2 * th_data) ** 2,
                   np.cos(th_data) ** 4]).T
    coeffs, *_ = np.linalg.lstsq(W, p_mel / p_mel[0], rcond=None)
    Acoef, Bcoef = coeffs[0], coeffs[1]
    ax.plot(np.degrees(th_dense), Acoef * w_in + Bcoef * w_out,
            '-', color=TEAL, lw=1.2,
            label=r'$A|\cos 2\theta|^2 + B\cos^4\theta$ reference')

    ax.axvline(45.0, color=TEAL, ls=':', lw=0.6, alpha=0.5)
    ax.text(45.0, 1.05, r'$\theta=45^\circ$',
            fontsize=7, color=TEAL, ha='center')

    ax.set_xlabel(r'$\theta$ (deg)')
    ax.set_ylabel(r'Normalized peak $|m_\perp|$ at $2f_K$')
    ax.set_xlim(-2, 92)
    ax.set_ylim(-0.05, 1.2)
    ax.legend(fontsize=6, loc='upper right')

    # (b) Polar view (data only; no theory overlay)
    ax = axes[1]
    ax.remove()
    ax = fig.add_subplot(1, 2, 2, projection='polar')
    th_rad = np.radians(thetas)
    ax.plot(th_rad, p_mel / p_mel.max(), 'o-', color=VERMILION,
            ms=3, lw=0.8, label='MEL')
    ax.plot(th_rad, p_full / p_full.max(), 's--', color=BLACK,
            ms=3, lw=0.6, alpha=0.6, label='Full')
    ax.set_thetalim(0, np.pi / 2)
    ax.set_thetagrids([0, 30, 45, 60, 90])
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='lower right',
              bbox_to_anchor=(1.25, -0.15))

    try:
        label_panels([axes[0], ax])
    except Exception:
        pass

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim26b_angular_fine.{ext}'),
                    dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_sim26b_angular_fine.pdf/png")
    print(f"  Eq.(4) reference fit: A={Acoef:.3f}, B={Bcoef:.3f}")


if __name__ == "__main__":
    main()
