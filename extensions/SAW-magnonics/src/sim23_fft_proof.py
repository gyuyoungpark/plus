"""Simulation 23: FFT proof of parametric magnon generation.

Direct proof that MEL drives magnons at f_K when SAW is at 2f_K:
  - Run MEL-only at f_SAW = 2f_K, save high-resolution time trace
  - FFT of m_y(t) should show peak at f_K (not 2f_K)
  - Compare with MR-only at f_SAW = f_K (direct FMR, peak at f_K)
  - Compare with Full at f_SAW = 2f_K

Grid: 256x16x1 at 10 nm (full micromagnetic)
Material: YIG (unified PRL parameters)
Expected runtime: ~15 minutes total.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim23_fft_proof.npz")

# Material: YIG
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
B0 = 50e-3  # 50 mT
T_RUN = 10e-9
DT_REC = 10e-12   # 10 ps (Nyquist up to 50 GHz, 1001 points)
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_trace(f_saw, enable_mel, K_mr, label=""):
    """Run one SAW-FMR simulation and return full time trace."""
    t0 = time.time()
    wavelength = V_SAW / f_saw
    print(f"  [{label}] f_SAW={f_saw*1e-9:.2f} GHz, "
          f"{'MEL' if enable_mel else 'no-MEL'}, "
          f"K_mr={K_mr*1e-6:.0f}...", end="", flush=True)

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

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    times = np.zeros(NT_REC)
    my = np.zeros(NT_REC)
    mz = np.zeros(NT_REC)

    for i in range(NT_REC):
        avg = magnet.magnetization.average()
        times[i] = world.timesolver.time
        my[i] = avg[1]
        mz[i] = avg[2]
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    elapsed = time.time() - t0
    print(f" {elapsed:.0f}s")
    return times, my, mz


def main():
    print("=" * 60)
    print("Sim 23: FFT proof of parametric magnon generation")
    print(f"  B0 = {B0*1e3:.0f} mT, f_K = {F_K*1e-9:.2f} GHz")
    print(f"  2f_K = {2*F_K*1e-9:.2f} GHz")
    print("=" * 60)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading...")
        data = dict(np.load(CACHE_FILE))
    else:
        results = {}

        # 1. MEL-only at f_SAW = 2f_K (parametric pump)
        t, my, mz = run_trace(2 * F_K, enable_mel=True, K_mr=0,
                              label="MEL-only @ 2f_K")
        results['mel_2fk_times'] = t
        results['mel_2fk_my'] = my
        results['mel_2fk_mz'] = mz

        # 2. MR-only at f_SAW = f_K (direct FMR)
        t, my, mz = run_trace(F_K, enable_mel=False, K_mr=KMR,
                              label="MR-only @ f_K")
        results['mr_fk_times'] = t
        results['mr_fk_my'] = my
        results['mr_fk_mz'] = mz

        # 3. Full at f_SAW = 2f_K
        t, my, mz = run_trace(2 * F_K, enable_mel=True, K_mr=KMR,
                              label="Full @ 2f_K")
        results['full_2fk_times'] = t
        results['full_2fk_my'] = my
        results['full_2fk_mz'] = mz

        # 4. Baseline (no SAW)
        t, my, mz = run_trace(F_K, enable_mel=False, K_mr=0,
                              label="No SAW (baseline)")
        results['baseline_times'] = t
        results['baseline_my'] = my
        results['baseline_mz'] = mz

        results['B0'] = B0
        results['f_K'] = F_K
        np.savez(CACHE_FILE, **results)
        print(f"  Saved: {CACHE_FILE}")
        data = results

    plot_results(data)


def plot_results(data):
    try:
        from plot_style import (apply_style, label_panels, axis_label,
                                DOUBLE_COL, SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        DOUBLE_COL = 7.0

    f_K_ghz = float(data['f_K']) * 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.5))
    fig.subplots_adjust(wspace=0.35)

    configs = [
        ('mel_2fk', f'MEL-only @ $2f_K$', VERMILION),
        ('mr_fk', f'MR-only @ $f_K$', SKY_BLUE),
        ('full_2fk', f'Full @ $2f_K$', BLACK),
    ]

    # (a) Time traces (m_y)
    ax = axes[0]
    for key, label, color in configs:
        t = data[f'{key}_times'] * 1e9
        my = data[f'{key}_my']
        ax.plot(t, my, '-', color=color, lw=0.5, label=label, alpha=0.8)
    ax.set_xlabel('$t$ (ns)')
    ax.set_ylabel('$m_y$')
    ax.legend(fontsize=7)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    # (b) FFT power spectra
    ax = axes[1]
    for key, label, color in configs:
        my = data[f'{key}_my']
        dt = float(data[f'{key}_times'][1] - data[f'{key}_times'][0])
        # Use second half for FFT (more developed signal)
        n_half = len(my) // 2
        signal = my[n_half:] - np.mean(my[n_half:])
        N = len(signal)
        freqs = np.fft.rfftfreq(N, d=dt) * 1e-9  # GHz
        power = np.abs(np.fft.rfft(signal))**2
        power /= np.max(power) if np.max(power) > 0 else 1
        ax.plot(freqs, power, '-', color=color, lw=0.8, label=label)

    ax.axvline(f_K_ghz, color='gray', ls=':', lw=0.5, alpha=0.6)
    ax.axvline(2 * f_K_ghz, color=VERMILION, ls=':', lw=0.5, alpha=0.6)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('FFT power (norm.)')
    ax.set_xlim(0, 3 * f_K_ghz)
    ax.legend(fontsize=7)

    label_panels = lambda axes: [ax.text(-0.12, 1.05, f'({chr(97+i)})',
        transform=ax.transAxes, fontsize=10, fontweight='bold')
        for i, ax in enumerate(axes)]
    label_panels(axes)

    fig.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim23_fft.{ext}'), dpi=200)
    plt.close(fig)
    print("  Saved: fig_sim23_fft.pdf/png")

    # Print peak frequencies
    print(f"\n  FFT peak frequencies (f_K = {f_K_ghz:.2f} GHz):")
    for key, label, _ in configs:
        my = data[f'{key}_my']
        dt = float(data[f'{key}_times'][1] - data[f'{key}_times'][0])
        n_half = len(my) // 2
        signal = my[n_half:] - np.mean(my[n_half:])
        freqs = np.fft.rfftfreq(len(signal), d=dt) * 1e-9
        power = np.abs(np.fft.rfft(signal))**2
        peak_idx = np.argmax(power[1:]) + 1
        print(f"    {label:25s}: {freqs[peak_idx]:.2f} GHz")


if __name__ == "__main__":
    main()
