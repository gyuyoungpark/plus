"""Simulation 25: Spatially resolved magnon spectrum (k-space FFT).

Answers referee question: "Provide spatial-resolved Fourier analysis"
of the parametrically pumped magnons at SAW frequency 2 f_K.

Records m_y(x, t) along the SAW propagation direction and produces a
2D omega-k spectrum |M_y(k, omega)|^2. Predictions:
  - MR-only at f_SAW = f_K:   peak at (k = k_SAW, omega = omega_K) — direct
  - MEL-only at f_SAW = 2f_K: peak at (k = k_SAW/2, omega = omega_K)
                              — parametric subharmonic with k=k_pump/2

Grid: 1024 x 8 x 1 at 5 nm cellsize (longer x for finer k resolution).
Material: YIG (sim20 params). T = 20 ns, dt_rec = 20 ps.
Estimated runtime: ~20 minutes.
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

CACHE_FILE = os.path.join(DATA_DIR, "sim25_kspectrum.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
V_SAW = 3500.0

NX, NY, NZ = 2048, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9

EPS0 = 1e-4
B0 = 50e-3

T_RUN = 20e-9
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13


def kittel_freq_hz(B0_val):
    return GAMMA * np.sqrt(B0_val * (B0_val + MU0 * MS)) / (2 * np.pi)


F_K = kittel_freq_hz(B0)


def run_spatial(f_saw, enable_mel, K_mr, label=""):
    t0 = time.time()
    wavelength = V_SAW / f_saw
    print(f"  [{label}] f_SAW={f_saw*1e-9:.2f} GHz, "
          f"lambda_SAW={wavelength*1e9:.0f} nm, "
          f"k_SAW={2*np.pi/wavelength*1e-6:.2f} um^-1")

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0.005)
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

    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    for i in range(NT_REC):
        m = magnet.magnetization.eval()
        # m shape: (3, NZ, NY, NX) — average over y, take z=0
        my_xt[i] = m[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    elapsed = time.time() - t0
    print(f"    done {elapsed:.0f}s")
    return my_xt, wavelength


def main():
    print("=" * 60)
    print("Sim 25: Spatial k-spectrum of parametric magnon emission")
    print(f"  B0={B0*1e3:.0f}mT, f_K={F_K*1e-9:.2f}GHz, 2f_K={2*F_K*1e-9:.2f}GHz")
    print(f"  Grid: {NX}x{NY}x{NZ} @ {CX*1e9:.0f}nm")
    print(f"  T={T_RUN*1e9:.0f}ns, dt_rec={DT_REC*1e12:.0f}ps -> NT={NT_REC}")
    print("=" * 60)

    if os.path.isfile(CACHE_FILE):
        print("  Cached. Loading...")
        data = dict(np.load(CACHE_FILE))
    else:
        results = {}

        my, lam = run_spatial(F_K, enable_mel=False, K_mr=KMR,
                              label="MR-only @ f_K")
        results['mr_fk_my_xt'] = my
        results['mr_fk_kSAW'] = 2 * np.pi / lam

        my, lam = run_spatial(2 * F_K, enable_mel=True, K_mr=0,
                              label="MEL-only @ 2f_K")
        results['mel_2fk_my_xt'] = my
        results['mel_2fk_kSAW'] = 2 * np.pi / lam

        results['B0'] = B0
        results['f_K'] = F_K
        results['NX'] = NX
        results['CX'] = CX
        results['DT_REC'] = DT_REC
        np.savez(CACHE_FILE, **results)
        print(f"  Saved: {CACHE_FILE}")
        data = results

    plot_spectrum(data)


def plot_spectrum(data):
    try:
        from plot_style import (apply_style, label_panels, axis_label,
                                DOUBLE_COL, BLACK)
        apply_style()
    except ImportError:
        DOUBLE_COL = 7.0
        BLACK = '#000'

    cx = float(data['CX'])
    dt = float(data['DT_REC'])
    f_K = float(data['f_K'])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.4))
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.95,
                        top=0.92, bottom=0.18)

    # Magnon dispersion (Kittel + exchange) for overlay
    GAMMA_HZ = 1.76e11 / (2 * np.pi)
    MU0_MS = 4 * np.pi * 1e-7 * 140e3
    B0_FIELD = 0.05
    AEX = 3.65e-12
    MS = 140e3
    k_disp = np.linspace(-30, 30, 400) * 1e6
    om_K = 2 * np.pi * GAMMA_HZ * np.sqrt(
        (B0_FIELD + 2 * AEX / MS * k_disp ** 2) *
        (B0_FIELD + 2 * AEX / MS * k_disp ** 2 + MU0_MS))
    f_disp = om_K / (2 * np.pi) * 1e-9

    panel_specs = [
        (axes[0], 'mr_fk', '(a) MR @ $f_K$', float(data['mr_fk_kSAW']),
         False),
        (axes[1], 'mel_2fk', '(b) MEL @ $2f_K$',
         float(data['mel_2fk_kSAW']), True),
    ]

    for ax, key, label, k_pump, is_param in panel_specs:
        my_xt = data[f'{key}_my_xt']
        n_half = my_xt.shape[0] // 2
        signal = my_xt[n_half:]
        signal = signal - signal.mean()

        spec = np.fft.fftshift(np.fft.fft2(signal), axes=(0, 1))
        freqs_t = np.fft.fftshift(
            np.fft.fftfreq(signal.shape[0], d=dt)) * 1e-9
        freqs_k = np.fft.fftshift(
            np.fft.fftfreq(signal.shape[1], d=cx)) * 2 * np.pi * 1e-6
        power = np.abs(spec) ** 2
        if power.max() > 0:
            power = power / power.max()

        pos = freqs_t >= 0
        im = ax.pcolormesh(freqs_k, freqs_t[pos], power[pos],
                           shading='auto', cmap='magma',
                           vmin=0, vmax=0.3, rasterized=True)

        # Magnon dispersion overlay
        ax.plot(k_disp * 1e-6, f_disp, '-', color='white', lw=0.6,
                alpha=0.55)
        ax.plot(-k_disp * 1e-6, f_disp, '-', color='white', lw=0.6,
                alpha=0.55)

        # Reference lines
        ax.axhline(f_K * 1e-9, color='cyan', ls='--', lw=0.7, alpha=0.85)
        k_pump_um = k_pump * 1e-6
        ax.axvline(k_pump_um, color='lime', ls='--', lw=0.7, alpha=0.85)
        ax.axvline(-k_pump_um, color='lime', ls='--', lw=0.7, alpha=0.4)

        if is_param:
            ax.axhline(2 * f_K * 1e-9, color='gold', ls=':', lw=0.7,
                       alpha=0.7)
            ax.text(0.02, 0.94, r'pair band $k_1+k_2=k_\mathrm{SAW}$',
                    transform=ax.transAxes, fontsize=7, color='white',
                    va='top', ha='left')
        else:
            ax.text(0.97, 0.36,
                    r'$(k_\mathrm{SAW},\,f_K)$',
                    transform=ax.transAxes, fontsize=7, color='white',
                    va='top', ha='right')

        ax.text(0.02, 1.04, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom')
        ax.text(0.99, 1.04,
                r'$f_K$' if not is_param else r'$f_K$, $2f_K$',
                transform=ax.transAxes, fontsize=7, color='gray',
                va='bottom', ha='right', style='italic')

        ax.set_xlabel(r'$k_x$ ($\mu$m$^{-1}$)')
        if ax is axes[0]:
            ax.set_ylabel(r'$f$ (GHz)')
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 10)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, aspect=22, pad=0.02)
    cbar.set_label(r'$|M_y(k,\omega)|^2$ (norm.)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim25_kspectrum.{ext}'),
                    dpi=300)
    plt.close(fig)
    print("  Saved: fig_sim25_kspectrum.pdf/png")


if __name__ == "__main__":
    main()
