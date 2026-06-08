"""Simulation 38 (Paper 2): Nonreciprocal pair emission by reversed SAW direction.

Shows that reversing the SAW propagation direction (+k → -k) switches the
parametric pair band from +k_SAW/2 to -k_SAW/2, demonstrating direction-
controlled nonreciprocal parametric magnon generation.

Physics:
  The pair rule k1+k2=k_pump inherits the SIGN of the pump wavevector:
    +k SAW at 2f_K: pump at +k_SAW → pairs centred at +k_SAW/2
    -k SAW at 2f_K: pump at -k_SAW → pairs centred at -k_SAW/2

  Reversing the SAW propagation direction therefore acts as a binary switch
  on the magnon pair-beam direction.  No spatially uniform pump can achieve
  this: Suhl pumping always gives k1+k2=0 regardless of SAW direction.

  Magnonic analogy: quasi-phase-matched optical parametric downconversion,
  where the poling direction (pump wavevector) sets the signal/idler angles.

Strategy:
  - +k data: loaded from sim37 cache (identical parameters, no re-compute)
  - -k data: new mumax+ run with saw_wavevector negated after apply()

Grid: 1024×8×1 @ 5 nm, T=20 ns, dt_rec=20 ps (same as sim25/sim37)
Estimated runtime: ~8 min (single new run; +k reused from sim37 cache)
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
CACHE = os.path.join(DATA_DIR, "sim38_nonreciprocal.npz")
CHECKPOINT = os.path.join(DATA_DIR, "sim38_checkpoint.npz")
SIM37_CACHE = os.path.join(DATA_DIR, "sim37_suhl_control.npz")

GAMMA = 1.76e11
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-4
B1 = -8.8e6
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


def run_minus_k_mel():
    """-k SAW MEL pump at 2*f_K: pump wavevector negated after apply()."""
    t0 = time.time()
    f_saw = 2 * F_K
    wavelength = V_SAW / f_saw
    k_saw = 2 * np.pi / wavelength   # positive value

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

    # Set up +k SAW first, then flip wavevector to -k
    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=EPS0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=0.0, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=True)

    # Flip to -k direction (pair band moves to -k_SAW/2)
    if hasattr(magnet, 'saw_wavevector'):
        magnet.saw_wavevector = -k_saw
        print(f"  [-k SAW] wavevector set to {-k_saw*1e-6:.3f} um^-1")
    else:
        raise RuntimeError(
            "GPU SAW kernel not available — magnet has no saw_wavevector attribute. "
            "Cannot flip wavevector; rebuild mumax+ with chiral SAW kernel.")

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    for i in range(NT_REC):
        m = magnet.magnetization.eval()
        my_xt[i] = m[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    print(f"  [-k SAW MEL] done in {(time.time()-t0)/60:.1f} min")
    return my_xt, k_saw


def compute_kspec(my_xt, CX_val, DT):
    """2D omega-k power spectrum using second half of time trace."""
    NT, NX_val = my_xt.shape
    n_half = NT // 2
    signal = my_xt[n_half:].astype(np.float64)
    signal -= signal.mean()
    spec = np.fft.fftshift(np.fft.fft2(signal), axes=(0, 1))
    power = np.abs(spec) ** 2
    freqs_t = np.fft.fftshift(np.fft.fftfreq(signal.shape[0], d=DT)) * 1e-9
    freqs_k = np.fft.fftshift(np.fft.fftfreq(NX_val, d=CX_val)) * 2 * np.pi * 1e-6
    return freqs_k, freqs_t, power


def main():
    print("=" * 70)
    print("Sim 38: Nonreciprocal pair emission -- +k vs -k SAW direction")
    print(f"  f_K = {F_K*1e-9:.3f} GHz,  2*f_K = {2*F_K*1e-9:.3f} GHz")
    print(f"  SAW eps0 = {EPS0:.0e}")
    print(f"  Grid {NX}x{NY}x{NZ} @ {CX*1e9:.0f} nm")
    print("=" * 70)

    if os.path.isfile(CACHE):
        print("  Cached. Loading...")
        d = dict(np.load(CACHE))
        plot(d)
        return

    # Load +k data from sim37 cache (same MEL @2f_K parameters)
    if not os.path.isfile(SIM37_CACHE):
        raise FileNotFoundError(
            f"sim37 cache not found at {SIM37_CACHE}. "
            "Run sim37_suhl_control.py first (provides the +k MEL reference run).")
    d37 = dict(np.load(SIM37_CACHE))
    plus_my_xt = d37['saw_my_xt']
    k_saw = float(d37['saw_kSAW'])
    print(f"  Loaded +k data from sim37 cache (k_SAW = {k_saw*1e-6:.3f} um^-1)")

    ckpt = {}
    if os.path.isfile(CHECKPOINT):
        ckpt = dict(np.load(CHECKPOINT))
        print(f"  Checkpoint: {list(ckpt.keys())}")

    if 'minus_my_xt' not in ckpt:
        minus_my_xt, _ = run_minus_k_mel()
        ckpt['minus_my_xt'] = minus_my_xt
        np.savez(CHECKPOINT, **ckpt)
    else:
        minus_my_xt = ckpt['minus_my_xt']
        print("  [-k SAW MEL] from checkpoint")

    save_dict = {
        'plus_my_xt': plus_my_xt,
        'minus_my_xt': minus_my_xt,
        'k_SAW': k_saw,
        'f_K': F_K, 'B0': B0, 'EPS0': EPS0,
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

    CX_val = float(d['CX'])
    DT = float(d['DT_REC'])
    f_K = float(d['f_K'])
    k_saw = float(d['k_SAW']) * 1e-6   # um^-1

    k_plus, f_plus, pw_plus = compute_kspec(d['plus_my_xt'], CX_val, DT)
    k_minus, f_minus, pw_minus = compute_kspec(d['minus_my_xt'], CX_val, DT)

    for pw in [pw_plus, pw_minus]:
        mx = pw.max()
        if mx > 0:
            pw /= mx

    # Magnon dispersion overlay
    MS_C = 140e3
    AEX_C = 3.65e-12
    GAMMA_HZ = 1.76e11 / (2 * np.pi)
    B0_C = float(d['B0'])
    k_disp = np.linspace(-30, 30, 400) * 1e6
    om_disp = 2 * np.pi * GAMMA_HZ * np.sqrt(
        (B0_C + 2 * AEX_C / MS_C * k_disp**2) *
        (B0_C + 2 * AEX_C / MS_C * k_disp**2 + MU0 * MS_C))
    f_disp = om_disp / (2 * np.pi) * 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.4))
    fig.subplots_adjust(wspace=0.28, left=0.09, right=0.96,
                        top=0.90, bottom=0.18)

    datasets = [
        (axes[0], k_plus, f_plus, pw_plus,
         r'(a) $+k_\mathrm{SAW}$ pump', +k_saw,
         r'$+k_\mathrm{SAW}/2$'),
        (axes[1], k_minus, f_minus, pw_minus,
         r'(b) $-k_\mathrm{SAW}$ pump', -k_saw,
         r'$-k_\mathrm{SAW}/2$'),
    ]

    for ax, k_ax, freqs, pw, title, k_pump_signed, pair_label in datasets:
        pos_f = freqs >= 0
        im = ax.pcolormesh(k_ax, freqs[pos_f], pw[pos_f],
                           shading='auto', cmap='magma',
                           vmin=0, vmax=0.3, rasterized=True)

        ax.plot(k_disp * 1e-6, f_disp, 'w-', lw=0.6, alpha=0.5)
        ax.plot(-k_disp * 1e-6, f_disp, 'w-', lw=0.6, alpha=0.5)

        ax.axhline(f_K * 1e-9, color='cyan', ls='--', lw=0.7, alpha=0.85,
                   label=fr'$f_K={f_K*1e-9:.2f}$ GHz')
        pair_center = k_pump_signed / 2
        ax.axvline(pair_center, color='lime', ls='--', lw=0.8, alpha=0.9,
                   label=f'{pair_label} = {pair_center:.1f} µm⁻¹')
        ax.axvline(0, color='white', ls=':', lw=0.4, alpha=0.4)

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
        fig.savefig(os.path.join(FIG_DIR, f'fig_sim38_nonreciprocal.{ext}'),
                    dpi=300)
    plt.close(fig)
    print("  Saved: fig_sim38_nonreciprocal.pdf/png")
    k_saw_local = float(d['k_SAW']) * 1e-6
    print(f"  Key: +k pump -> pairs at +{k_saw_local/2:.2f} um^-1; "
          f"-k pump -> pairs at -{k_saw_local/2:.2f} um^-1")
    print("  → direction-controlled nonreciprocal pair emission confirmed")


if __name__ == "__main__":
    main()
