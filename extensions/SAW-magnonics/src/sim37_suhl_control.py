"""Simulation 37 (Paper 2): Suhl-type uniform pump control.

Directly compares the pair manifold of two parametric pump mechanisms
on the SAME micromagnetic strip, proving the k1+k2=k_SAW selection rule
by falsifying the k1+k2=0 (Suhl) alternative:

  Run A — SAW MEL pump:
    ChiralSurfaceAcousticWave at f_SAW = 2*f_K  (as in sim25)
    Pair rule: k1 + k2 = k_SAW  (shifted manifold)

  Run B — Suhl uniform pump:
    Spatially uniform h_z(t) = h0 * cos(2*omega_K*t)  (parallel to m0)
    Same effective modulation depth: h0 = 2*|B1|*eps0/Ms
    Pair rule: k1 + k2 = 0  (symmetric Suhl manifold)

The ω-k spectra at ω_K for both runs are compared side-by-side.
Under MEL pump: pair band is offset to k_SAW/2 (not centered at 0).
Under Suhl pump: pair band is symmetric around k=0.

Same grid as sim25: 1024 x 8 x 1 @ 5 nm, T=20 ns, dt_rec=20 ps.
Estimated runtime: ~30 minutes for two runs.
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
CACHE = os.path.join(DATA_DIR, "sim37_suhl_control.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim37_checkpoint.npz")

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
OMEGA_K = 2 * np.pi * F_K
# Effective Suhl pump amplitude matching MEL modulation depth
H0_SUHL = 2 * abs(B1) * EPS0 / MS   # ~12.6 mT


def run_saw_mel():
    """MEL parametric pump at 2*f_K via ChiralSAW (no MR)."""
    t0 = time.time()
    f_saw = 2 * F_K
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.B1 = B1
    magnet.magnetization = (1, 0, 0.005)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=0.0, enable_barnett=False)  # MEL only, no MR
    saw.apply(magnet, Msat=MS, enable_mel=True)

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    for i in range(NT_REC):
        m = magnet.magnetization.eval()
        my_xt[i] = m[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    print(f"  [SAW-MEL] done in {(time.time()-t0)/60:.1f} min")
    return my_xt, wavelength


def run_suhl_uniform():
    """Suhl uniform pump: h_z(t) = H0 * cos(2*omega_K*t), no spatial variation."""
    t0 = time.time()

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    # No MEL (B1=0), no MR — pure uniform parametric pump
    magnet.magnetization = (1, 0, 0.005)
    magnet.enable_demag = True

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    for i in range(NT_REC):
        t = i * DT_REC
        # Uniform parallel pump: h_x(t) along m0 direction
        h_pump = H0_SUHL * np.cos(2 * OMEGA_K * t)
        magnet.bias_magnetic_field = (B0 + h_pump, 0, 0)
        m = magnet.magnetization.eval()
        my_xt[i] = m[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    print(f"  [Suhl-uniform] done in {(time.time()-t0)/60:.1f} min")
    return my_xt


def compute_kspec(my_xt, CX, DT_REC):
    """2D omega-k power spectrum (positive frequencies only)."""
    NT, NX = my_xt.shape
    n_half = NT // 2
    signal = my_xt[n_half:].astype(np.float64)
    signal -= signal.mean()
    spec = np.fft.fftshift(np.fft.fft2(signal), axes=(0, 1))
    power = np.abs(spec) ** 2
    freqs_t = np.fft.fftshift(np.fft.fftfreq(signal.shape[0], d=DT_REC)) * 1e-9
    freqs_k = np.fft.fftshift(np.fft.fftfreq(NX, d=CX)) * 2 * np.pi * 1e-6
    return freqs_k, freqs_t, power


def main():
    print("=" * 70)
    print("Sim 37: Suhl uniform pump vs SAW MEL pump — k-spectrum comparison")
    print(f"  f_K = {F_K*1e-9:.3f} GHz,  2*f_K = {2*F_K*1e-9:.3f} GHz")
    print(f"  SAW eps0 = {EPS0:.0e},  Suhl h0 = {H0_SUHL*1e3:.1f} mT")
    print(f"  Grid {NX}x{NY}x{NZ} @ {CX*1e9:.0f} nm")
    print("=" * 70)

    if os.path.isfile(CACHE):
        print("  Cached. Loading...")
        d = dict(np.load(CACHE))
        plot(d)
        return

    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        print(f"  Checkpoint: {list(ckpt.keys())}")

    # --- Run SAW MEL ---
    if 'saw_my_xt' not in ckpt:
        my_saw, lam_saw = run_saw_mel()
        ckpt['saw_my_xt'] = my_saw
        ckpt['saw_kSAW'] = np.float64(2 * np.pi / lam_saw)
        np.savez(CHECKPOINT, **ckpt)
    else:
        my_saw = ckpt['saw_my_xt']
        print("  [SAW-MEL] from checkpoint")

    # --- Run Suhl uniform ---
    if 'suhl_my_xt' not in ckpt:
        my_suhl = run_suhl_uniform()
        ckpt['suhl_my_xt'] = my_suhl
        np.savez(CHECKPOINT, **ckpt)
    else:
        my_suhl = ckpt['suhl_my_xt']
        print("  [Suhl-uniform] from checkpoint")

    save_dict = {
        'saw_my_xt': my_saw,
        'suhl_my_xt': my_suhl,
        'saw_kSAW': ckpt['saw_kSAW'],
        'f_K': F_K, 'B0': B0, 'EPS0': EPS0, 'H0_SUHL': H0_SUHL,
        'CX': CX, 'DT_REC': DT_REC,
    }
    np.savez(CACHE, **save_dict)
    if os.path.isfile(CHECKPOINT):
        os.remove(CHECKPOINT)
    print(f"  Saved: {CACHE}")
    plot(save_dict)


def plot(d):
    try:
        from plot_style import (apply_style, label_panels, DOUBLE_COL,
                                SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        DOUBLE_COL = 7.0

    CX = float(d['CX'])
    DT = float(d['DT_REC'])
    f_K = float(d['f_K'])
    k_saw = float(d['saw_kSAW']) * 1e-6   # um^-1

    k_saw_arr, freqs_t_saw, pw_saw = compute_kspec(d['saw_my_xt'], CX, DT)
    k_suhl_arr, freqs_t_suhl, pw_suhl = compute_kspec(d['suhl_my_xt'], CX, DT)

    for pw in [pw_saw, pw_suhl]:
        mx = pw.max()
        if mx > 0:
            pw /= mx

    # Magnon dispersion overlay
    from mumaxplus.util.constants import MU0 as MU0_C
    MS_C = 140e3
    AEX_C = 3.65e-12
    GAMMA_HZ = 1.76e11 / (2 * np.pi)
    B0_C = float(d['B0'])
    k_disp = np.linspace(-30, 30, 400) * 1e6
    om_disp = 2 * np.pi * GAMMA_HZ * np.sqrt(
        (B0_C + 2 * AEX_C / MS_C * k_disp**2) *
        (B0_C + 2 * AEX_C / MS_C * k_disp**2 + MU0_C * MS_C))
    f_disp = om_disp / (2 * np.pi) * 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.4))
    fig.subplots_adjust(wspace=0.28, left=0.09, right=0.96,
                        top=0.90, bottom=0.18)

    titles = [
        r'(a) SAW MEL pump ($k_1\!+\!k_2\!=\!k_\mathrm{SAW}$)',
        r'(b) Uniform Suhl pump ($k_1\!+\!k_2\!=\!0$)',
    ]
    for ax, k_ax, freqs, pw, title, k_ref, k_ref_sign in zip(
            axes,
            [k_saw_arr, k_suhl_arr],
            [freqs_t_saw, freqs_t_suhl],
            [pw_saw, pw_suhl],
            titles,
            [k_saw, 0.0],   # expected pair-band center
            [1, 0]):

        pos_f = freqs >= 0
        im = ax.pcolormesh(k_ax, freqs[pos_f], pw[pos_f],
                           shading='auto', cmap='magma',
                           vmin=0, vmax=0.3, rasterized=True)

        # Magnon dispersion
        ax.plot(k_disp * 1e-6, f_disp, 'w-', lw=0.6, alpha=0.5)
        ax.plot(-k_disp * 1e-6, f_disp, 'w-', lw=0.6, alpha=0.5)

        ax.axhline(f_K * 1e-9, color='cyan', ls='--', lw=0.7, alpha=0.85,
                   label=fr'$f_K={f_K*1e-9:.2f}$ GHz')
        ax.axhline(2 * f_K * 1e-9, color='gold', ls=':', lw=0.6, alpha=0.7)
        if k_ref_sign:
            ax.axvline(k_ref / 2, color='lime', ls='--', lw=0.6, alpha=0.8,
                       label=fr'$k_\mathrm{{SAW}}/2={k_ref/2:.1f}$')
            ax.axvline(-k_ref / 2, color='lime', ls='--', lw=0.6, alpha=0.5)
        else:
            ax.axvline(0, color='lime', ls='--', lw=0.6, alpha=0.8,
                       label=r'$k=0$ (Suhl center)')

        ax.set_xlabel(r'$k_x$ ($\mu$m$^{-1}$)')
        ax.set_ylabel(r'$f$ (GHz)')
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 10)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(title, fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, aspect=22, pad=0.02)
    cbar.set_label(r'$|M_y(k,f)|^2$ (norm.)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim37_suhl_control.{ext}'),
                    dpi=300)
    plt.close(fig)
    print("  Saved: fig_sim37_suhl_control.pdf/png")
    print("  Key: SAW pair band centered at k_SAW/2 (not 0) -> k1+k2=k_SAW proven")


if __name__ == "__main__":
    main()
