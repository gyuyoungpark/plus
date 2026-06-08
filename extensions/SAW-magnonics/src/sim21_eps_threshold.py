"""Simulation 21: Strain amplitude threshold for parametric MEL pump.

Tests the scaling of the 2omega_K parametric peak with strain amplitude:
  - Direct FMR (MR):  |m_perp| ~ eps0  (linear)
  - Parametric (MEL):  |m_perp| ~ eps0^2 or threshold behavior

Grid: 256x16x1 at 10 nm (full micromagnetic)
Material: YIG (unified PRL parameters)
B0 fixed at Kittel resonance for f = 3 GHz
eps0 sweep: 6 values from 1e-5 to 1e-3
f_SAW: 40 points covering both f_K and 2f_K regions
Channels: Full (MEL+MR), MR-only

Estimated runtime: ~8-10 hours on a single GPU.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim21_eps_threshold.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim21_checkpoint.npz")

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
V_SAW = 3500.0

# ==========================================================================
# Geometry
# ==========================================================================
NX, NY, NZ = 256, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9

# ==========================================================================
# Sweep parameters
# ==========================================================================
F_REF = 3.0e9   # reference frequency for B0 calculation

T_RUN = 10e-9
DT_REC = 100e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13

EPS_VALUES = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
F_SAW_VALUES = np.linspace(1e9, 8e9, 40)

CHANNELS = [
    ('full',    True,  KMR),
    ('mr_only', False, KMR),
]


def find_resonance_field(f_target):
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * MS, -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4 * a * c)) / 2


def kittel_freq_hz(B0):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * MS)) / (2 * np.pi)


B0_RES = find_resonance_field(F_REF)


def run_single(B0, f_saw, enable_mel, K_mr, eps0):
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0.01)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

    if enable_mel:
        magnet.B1 = B1

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=eps0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

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
    n_eps = len(EPS_VALUES)
    n_f = len(F_SAW_VALUES)
    n_ch = len(CHANNELS)

    print("=" * 72)
    print("Sim 21: Strain amplitude threshold for parametric MEL pump")
    print(f"  Grid: {NX}x{NY}x{NZ}, PBC=(4,4,0), demag=ON")
    print(f"  B0_res = {B0_RES*1e3:.2f} mT (f_K = {F_REF*1e-9:.1f} GHz)")
    print(f"  eps0: {EPS_VALUES}")
    print(f"  f_SAW: {F_SAW_VALUES[0]*1e-9:.1f}-{F_SAW_VALUES[-1]*1e-9:.1f} GHz "
          f"({n_f} points)")
    print(f"  Total runs: {n_eps * n_f * n_ch}")
    print("=" * 72)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading and plotting...")
        data = dict(np.load(CACHE_FILE))
        plot_results(data)
        return

    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        n_done = sum(1 for k in ckpt if k.startswith('spec_'))
        print(f"  Checkpoint loaded ({n_done}/{n_eps * n_ch} segments)")

    spectra = {}
    for ch_name, _, _ in CHANNELS:
        spectra[ch_name] = np.zeros((n_eps, n_f))

    for ei, eps0 in enumerate(EPS_VALUES):
        for ci, (ch_name, enable_mel, K_mr) in enumerate(CHANNELS):
            key = f'spec_{ch_name}_eps{eps0:.0e}'

            if key in ckpt:
                spectra[ch_name][ei] = ckpt[key]
                print(f"\n  eps0={eps0:.0e}, {ch_name:8s} [checkpoint]")
                continue

            print(f"\n  eps0={eps0:.0e}, {ch_name:8s}")
            seg_start = time.time()

            for j, f_saw in enumerate(F_SAW_VALUES):
                m_ss = run_single(B0_RES, f_saw, enable_mel, K_mr, eps0)
                spectra[ch_name][ei, j] = m_ss

                if (j + 1) % 10 == 0:
                    pct = 100 * (j + 1) / n_f
                    print(f"    [{pct:5.1f}%] f={f_saw*1e-9:.2f} GHz  "
                          f"|m_perp|={m_ss:.2e}  ({time.time()-seg_start:.0f}s)")

            ckpt[key] = spectra[ch_name][ei]
            np.savez(CHECKPOINT, **ckpt)
            print(f"    Segment done ({time.time()-seg_start:.0f}s)")

    save_dict = {
        'B0_res': B0_RES,
        'eps_values': EPS_VALUES,
        'f_saw_values': F_SAW_VALUES,
    }
    for ch_name in [c[0] for c in CHANNELS]:
        save_dict[f'spectra_{ch_name}'] = spectra[ch_name]
    np.savez(CACHE_FILE, **save_dict)
    if os.path.isfile(CHECKPOINT):
        os.remove(CHECKPOINT)

    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time/3600:.1f} hours")
    print(f"  Saved: {CACHE_FILE}")
    plot_results(save_dict)


def plot_results(data):
    """Plot eps0-dependent spectra and peak scaling."""
    try:
        from plot_style import (apply_style, label_panels, axis_label,
                                DOUBLE_COL, CM_TO_INCH,
                                SKY_BLUE, VERMILION, TEAL, BLACK, ORANGE)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK, ORANGE = \
            '#56B4E9', '#D55E00', '#009E73', '#000000', '#E69F00'
        DOUBLE_COL = 7.0

    from scipy.signal import find_peaks

    eps_vals = data['eps_values']
    f_ghz = data['f_saw_values'] * 1e-9
    spec_full = data['spectra_full']
    spec_mr = data['spectra_mr_only']
    f_K = F_REF * 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.5))
    fig.subplots_adjust(wspace=0.35)

    # ---- (a) Spectra at different eps0 (Full coupling) ----
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(eps_vals)))
    for ei, eps0 in enumerate(eps_vals):
        s = spec_full[ei]
        if np.max(s) > 0:
            ax.plot(f_ghz, s, '-', color=colors[ei], lw=0.8,
                    label=f'$\\varepsilon_0={eps0:.0e}$')
    ax.axvline(f_K, color='gray', ls=':', lw=0.5)
    ax.axvline(2 * f_K, color=VERMILION, ls=':', lw=0.5)
    ax.set_xlabel('$f_\\mathrm{SAW}$ (GHz)')
    ax.set_ylabel('Peak $|m_\\perp|$')
    ax.set_yscale('log')
    ax.legend(fontsize=5.5, ncol=2)

    # ---- (b) Peak value at f_K and 2f_K vs eps0 ----
    ax = axes[1]
    # Find peak near f_K and 2f_K for each eps0
    peak_fK_full = []
    peak_2fK_full = []
    peak_fK_mr = []

    f_saw = data['f_saw_values']
    mask_fK = (f_saw > (F_REF - 1e9)) & (f_saw < (F_REF + 1e9))
    mask_2fK = (f_saw > (2 * F_REF - 1e9)) & (f_saw < (2 * F_REF + 1e9))

    for ei in range(len(eps_vals)):
        sf = spec_full[ei]
        sm = spec_mr[ei]
        peak_fK_full.append(np.max(sf[mask_fK]) if np.any(mask_fK) else 0)
        peak_2fK_full.append(np.max(sf[mask_2fK]) if np.any(mask_2fK) else 0)
        peak_fK_mr.append(np.max(sm[mask_fK]) if np.any(mask_fK) else 0)

    ax.loglog(eps_vals, peak_fK_mr, 's-', color=SKY_BLUE, ms=5,
              mec='k', mew=0.3, label=r'MR only @ $\omega_K$')
    ax.loglog(eps_vals, peak_fK_full, 'o-', color=BLACK, ms=5,
              mec='k', mew=0.3, label=r'Full @ $\omega_K$')
    ax.loglog(eps_vals, peak_2fK_full, '^-', color=VERMILION, ms=5,
              mec='k', mew=0.3, label=r'Full @ $2\omega_K$')

    # Reference slopes
    eps_ref = np.array([eps_vals[0], eps_vals[-1]])
    y1 = peak_fK_mr[0] * (eps_ref / eps_ref[0])
    y2 = peak_2fK_full[0] * (eps_ref / eps_ref[0])**2
    ax.plot(eps_ref, y1, 'k:', lw=0.5, alpha=0.5)
    ax.plot(eps_ref, y2, ':', color=VERMILION, lw=0.5, alpha=0.5)
    ax.text(eps_ref[-1]*0.5, y1[-1]*1.5, r'$\propto \varepsilon_0$',
            fontsize=6, color='gray')
    ax.text(eps_ref[-1]*0.5, y2[-1]*0.3, r'$\propto \varepsilon_0^2$',
            fontsize=6, color=VERMILION)

    ax.set_xlabel(r'$\varepsilon_0$')
    ax.set_ylabel('Peak $|m_\\perp|$')
    ax.legend(fontsize=6)

    try:
        label_panels(axes)
    except Exception:
        pass

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim21_threshold.{ext}'), dpi=200)
    plt.close(fig)
    print("  Saved: fig_sim21_threshold.pdf/png")

    # Summary
    print(f"\n  Peak scaling:")
    print(f"  {'eps0':>10s}  {'MR@fK':>10s}  {'Full@fK':>10s}  {'Full@2fK':>10s}")
    for ei, eps0 in enumerate(eps_vals):
        print(f"  {eps0:>10.0e}  {peak_fK_mr[ei]:>10.2e}  "
              f"{peak_fK_full[ei]:>10.2e}  {peak_2fK_full[ei]:>10.2e}")


if __name__ == "__main__":
    main()
