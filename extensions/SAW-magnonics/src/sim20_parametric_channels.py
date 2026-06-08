"""Simulation 20: Three-channel FMR spectroscopy — parametric MEL pump proof.

Core evidence for PRL: MEL-only channel shows absorption at f_SAW = 2f_K
(parametric pump), while MR-only peaks at f_K (direct resonance).

Grid: 256x16x1 at 10 nm (full micromagnetic, exchange + demag)
Material: YIG (MS=140 kA/m, Aex=3.65 pJ/m, alpha=5e-4, B1=-8.8 MJ/m^3)
SAW: Chiral Rayleigh, xi=0.68, V_SAW=3500 m/s, eps0=1e-4

Sweep: 5 B0 x 50 f_SAW x 3 channels = 750 runs
Channels: Full (MEL+MR), MR-only, MEL-only
Checkpoint: per (B0, channel) pair

Estimated runtime: ~15-20 hours on a single GPU.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim20_parametric_channels.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim20_checkpoint.npz")

# ==========================================================================
# Material: YIG (unified across all PRL simulations)
# ==========================================================================
GAMMA = 1.76e11       # rad/s/T
MS = 140e3            # A/m  [Serga2010]
AEX = 3.65e-12        # J/m  [Serga2010]
ALPHA = 5e-4          # Gilbert damping (simulation value)
B1 = -8.8e6           # J/m^3 magnetoelastic constant [Dreher2012]
KMR = 1.0e6           # J/m^3 magneto-rotation [Centala2025]
XI = 0.68             # Rayleigh ellipticity (nu=0.3) [Auld1973]
V_SAW = 3500.0        # m/s SAW phase velocity

# ==========================================================================
# Geometry: full micromagnetic
# ==========================================================================
NX, NY, NZ = 256, 16, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9   # cellsize < exchange length (17 nm)

# ==========================================================================
# Sweep parameters
# ==========================================================================
EPS0 = 1e-4
T_RUN = 10e-9         # 10 ns per point
DT_REC = 100e-12      # 100 ps recording interval
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 5e-13       # fixed timestep (safe, 13x below CFL limit)

B0_VALUES = np.array([10e-3, 20e-3, 30e-3, 40e-3, 50e-3, 60e-3, 70e-3,
                      80e-3, 90e-3, 100e-3, 110e-3, 120e-3, 130e-3])  # 13 values
F_SAW_VALUES = np.linspace(0.5e9, 10e9, 50)                  # 50 points

CHANNELS = [
    ('full',     True,  KMR),    # MEL + MR
    ('mr_only',  False, KMR),    # MR only
    ('mel_only', True,  0.0),    # MEL only (parametric pump)
]


def kittel_freq_hz(B0):
    """Kittel FMR frequency for in-plane magnetized thin film."""
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * MS)) / (2 * np.pi)


# ==========================================================================
# Single simulation point
# ==========================================================================
def run_single(B0, f_saw, enable_mel, K_mr, eps0=EPS0):
    """Run one SAW-FMR point and return peak |m_perp|."""
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(4, 4, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0.01)    # small z-tilt as parametric seed
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

    # Peak in second half (steady-state region)
    n_half = NT_REC // 2
    return float(np.max(m_perp_arr[n_half:]))


# ==========================================================================
# Main sweep with checkpointing
# ==========================================================================
def main():
    total_start = time.time()
    n_b = len(B0_VALUES)
    n_f = len(F_SAW_VALUES)
    n_ch = len(CHANNELS)

    print("=" * 72)
    print("Sim 20: Three-channel FMR spectroscopy (parametric MEL proof)")
    print(f"  Grid: {NX}x{NY}x{NZ}, cellsize=({CX*1e9:.0f},{CY*1e9:.0f},{CZ*1e9:.0f}) nm")
    print(f"  PBC=(4,4,0), demag=ON, dt={DT_STEP:.0e}")
    print(f"  B0: {B0_VALUES*1e3} mT  ({n_b} values)")
    print(f"  f_SAW: {F_SAW_VALUES[0]*1e-9:.1f}-{F_SAW_VALUES[-1]*1e-9:.1f} GHz "
          f"({n_f} points)")
    print(f"  Channels: {[c[0] for c in CHANNELS]}")
    print(f"  Total runs: {n_b * n_f * n_ch}")
    print("=" * 72)

    if os.path.isfile(CACHE_FILE):
        cached = dict(np.load(CACHE_FILE))
        cached_B0 = cached.get('B0_values', np.array([]))
        match = (cached_B0.shape == B0_VALUES.shape
                 and np.allclose(cached_B0, B0_VALUES))
        if match:
            print("  Results already cached. Loading and plotting...")
            plot_results(cached)
            return
        print(f"  Cache exists but B0 grid mismatch "
              f"(cached {cached_B0.size}-pt vs current "
              f"{B0_VALUES.size}-pt). Ignoring cache; continuing sweep "
              f"from checkpoint.")

    # Load checkpoint
    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        idx_keys = {k for k in ckpt if k.startswith('idx_')}
        spec_keys = {k for k in ckpt if k.startswith('spec_')}
        partial_specs = {k for k in spec_keys
                         if k.replace('spec_', 'idx_') in idx_keys}
        n_done    = len(spec_keys - partial_specs)
        n_partial = len(partial_specs)
        msg = f"  Checkpoint loaded ({n_done}/{n_b * n_ch} segments done"
        if n_partial:
            msg += f", {n_partial} partial — will resume mid-segment"
        msg += ")"
        print(msg)

    # Allocate result arrays
    spectra = {}
    for ch_name, _, _ in CHANNELS:
        spectra[ch_name] = np.zeros((n_b, n_f))

    # Sweep
    for bi, B0 in enumerate(B0_VALUES):
        B0_mT = B0 * 1e3
        f_K = kittel_freq_hz(B0)

        for ci, (ch_name, enable_mel, K_mr) in enumerate(CHANNELS):
            key_spec = f'spec_{ch_name}_{B0_mT:.0f}'
            key_idx  = f'idx_{ch_name}_{B0_mT:.0f}'

            # Three cases per (B0, channel) segment:
            #   complete: spec present, idx absent  -> skip
            #   partial : spec present, idx present -> resume from idx
            #   fresh   : spec absent               -> start at j=0
            if key_spec in ckpt and key_idx not in ckpt:
                spectra[ch_name][bi] = ckpt[key_spec]
                print(f"\n  B0={B0_mT:.0f} mT, {ch_name:8s} [done, cached]  "
                      f"f_K={f_K*1e-9:.2f} GHz")
                continue

            if key_spec in ckpt and key_idx in ckpt:
                spectra[ch_name][bi] = np.array(ckpt[key_spec], copy=True)
                start_j = int(ckpt[key_idx])
                print(f"\n  B0={B0_mT:.0f} mT, {ch_name:8s} "
                      f"[resume j={start_j}/{n_f}]  "
                      f"(f_K={f_K*1e-9:.2f}, 2f_K={2*f_K*1e-9:.2f} GHz)")
            else:
                start_j = 0
                print(f"\n  B0={B0_mT:.0f} mT, {ch_name:8s}  "
                      f"(f_K={f_K*1e-9:.2f}, 2f_K={2*f_K*1e-9:.2f} GHz)")

            seg_start = time.time()

            for j in range(start_j, n_f):
                f_saw = F_SAW_VALUES[j]
                m_ss = run_single(B0, f_saw, enable_mel, K_mr)
                spectra[ch_name][bi, j] = m_ss

                # Per-point checkpoint (atomic): partial array + next index.
                ckpt[key_spec] = np.array(spectra[ch_name][bi], copy=True)
                ckpt[key_idx]  = np.array(j + 1)
                tmp = CHECKPOINT + ".tmp"
                np.savez(tmp, **ckpt)
                os.replace(tmp, CHECKPOINT)

                if (j + 1) % 10 == 0:
                    pct = 100 * (j + 1) / n_f
                    print(f"    [{pct:5.1f}%] f={f_saw*1e-9:.2f} GHz  "
                          f"|m_perp|={m_ss:.2e}  "
                          f"({time.time()-seg_start:.0f}s)")

            # Segment complete: keep spec, drop idx so subsequent runs skip.
            ckpt[key_spec] = np.array(spectra[ch_name][bi], copy=True)
            if key_idx in ckpt:
                del ckpt[key_idx]
            tmp = CHECKPOINT + ".tmp"
            np.savez(tmp, **ckpt)
            os.replace(tmp, CHECKPOINT)
            seg_time = time.time() - seg_start
            print(f"    Segment done ({seg_time:.0f}s). Checkpoint saved.")

    # Save final results
    save_dict = {
        'B0_values': B0_VALUES,
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


# ==========================================================================
# Plotting
# ==========================================================================
def plot_results(data):
    """Generate publication figure from results."""
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

    B0_vals = data['B0_values']
    f_ghz = data['f_saw_values'] * 1e-9
    spec_full = data['spectra_full']
    spec_mr = data['spectra_mr_only']
    spec_mel = data['spectra_mel_only']

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3))
    fig.subplots_adjust(wspace=0.4)

    # ---- (a) Spectra at one B0 (full / MR / MEL) ----
    ax = axes[0]
    bi = 1  # B0=30 mT
    f_K = kittel_freq_hz(B0_vals[bi]) * 1e-9
    for spec, label, color, ls in [
            (spec_full[bi], 'Full', BLACK, '-'),
            (spec_mr[bi], 'MR only', SKY_BLUE, '--'),
            (spec_mel[bi], 'MEL only', VERMILION, ':')]:
        if np.max(spec) > 0:
            ax.plot(f_ghz, spec / np.max(spec_full[bi]), ls,
                    color=color, lw=1.0, label=label)
    ax.axvline(f_K, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.axvline(2 * f_K, color=VERMILION, ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel('$f_\\mathrm{SAW}$ (GHz)')
    ax.set_ylabel('$|m_\\perp|$ (norm.)')
    ax.set_yticks([])
    ax.legend(fontsize=7)

    # ---- (b) Peak frequency vs B0 (3 channels) ----
    ax = axes[1]
    for spec_arr, ch_label, color, marker in [
            (spec_full, 'Full', VERMILION, 'o'),
            (spec_mr, 'MR only', SKY_BLUE, 's'),
            (spec_mel, 'MEL only', TEAL, '^')]:
        peaks = []
        b0_valid = []
        for bi in range(len(B0_vals)):
            s = spec_arr[bi]
            if np.max(s) > 1e-7:
                pks, _ = find_peaks(s, height=0.3 * np.max(s))
                if len(pks) > 0:
                    peaks.append(f_ghz[pks[np.argmax(s[pks])]])
                    b0_valid.append(B0_vals[bi] * 1e3)
        if peaks:
            ax.plot(b0_valid, peaks, marker, color=color, ms=6,
                    mec='k', mew=0.3, label=ch_label, zorder=5)

    B_fit = np.linspace(5, 150, 200) * 1e-3
    f_K_fit = np.array([kittel_freq_hz(b) * 1e-9 for b in B_fit])
    ax.plot(B_fit * 1e3, f_K_fit, 'k--', lw=0.7, label=r'$\omega_K$')
    ax.plot(B_fit * 1e3, 2 * f_K_fit, '--', color=VERMILION, lw=0.7,
            alpha=0.5, label=r'$2\omega_K$')
    ax.set_xlabel('$B_0$ (mT)')
    ax.set_ylabel('$f_\\mathrm{peak}$ (GHz)')
    ax.legend(fontsize=6)

    # ---- (c) Spectra at two B0 (full vs MR, stacked) ----
    ax = axes[2]
    for bi, (B0_idx, offset) in enumerate([(0, 0), (2, 1.3)]):
        if B0_idx >= len(B0_vals):
            continue
        sf = spec_full[B0_idx]
        sm = spec_mr[B0_idx]
        norm = max(np.max(sf), np.max(sm), 1e-10)
        ax.fill_between(f_ghz, offset, offset + sf / norm * 0.9,
                        color=BLACK, alpha=0.2)
        ax.plot(f_ghz, offset + sf / norm * 0.9, '-', color=BLACK, lw=0.7)
        ax.fill_between(f_ghz, offset, offset + sm / norm * 0.9,
                        color=SKY_BLUE, alpha=0.15)
        ax.plot(f_ghz, offset + sm / norm * 0.9, '--', color=SKY_BLUE, lw=0.6)
        f_K = kittel_freq_hz(B0_vals[B0_idx]) * 1e-9
        ax.axvline(f_K, color='gray', ls=':', lw=0.4, alpha=0.5)
        ax.axvline(2 * f_K, color=VERMILION, ls=':', lw=0.4, alpha=0.5)
    ax.set_xlabel('$f_\\mathrm{SAW}$ (GHz)')
    ax.set_ylabel('$|m_\\perp|$ (a.u.)')
    ax.set_yticks([])

    try:
        label_panels(axes)
    except Exception:
        pass

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim20_parametric.{ext}'), dpi=200)
    plt.close(fig)
    print("  Saved: fig_sim20_parametric.pdf/png")

    # Summary table
    print(f"\n  Peak locations (GHz):")
    print(f"  {'B0 (mT)':>10s}  {'f_K':>8s}  {'2f_K':>8s}  "
          f"{'Full':>8s}  {'MR':>8s}  {'MEL':>8s}")
    from scipy.signal import find_peaks
    for bi, B0 in enumerate(B0_vals):
        f_K = kittel_freq_hz(B0) * 1e-9
        row = f"  {B0*1e3:>8.0f}  {f_K:>8.2f}  {2*f_K:>8.2f}"
        for spec in [spec_full[bi], spec_mr[bi], spec_mel[bi]]:
            if np.max(spec) > 1e-7:
                pks, _ = find_peaks(spec, height=0.3 * np.max(spec))
                if len(pks):
                    row += f"  {f_ghz[pks[np.argmax(spec[pks])]]:.2f}"
                else:
                    row += "      ---"
            else:
                row += "      ---"
        print(row)


if __name__ == "__main__":
    main()
