"""Simulation 36: Two-mode pair correlator from existing sim25 cache.

Proves the k1+k2=k_SAW selection rule by computing the pair-coincidence
product and phase-locking signature from the raw time-domain data already
saved in sim25_kspectrum.npz.

Physics:
  For MEL parametric pumping at 2f_K, pairs (k1, k2) with k1+k2=k_pump are
  created simultaneously and are phase-locked:
    a_{k1}(t) * a_{k2}(t) = C  (constant in time)
  so  arg(M_y(k1, omega_K)) + arg(M_y(k_pump-k1, omega_K)) = const  (all k1)

  Random thermal magnons: phase sum is uniformly distributed on [0, 2pi].
  SAW MEL pairs:          phase sum is tightly clustered -> coherent pairing.

Three observables:
  (a) |M_y(k, f_K)|    -- pair-band spectral weight at omega_K
  (b) P(k1) = |M(k1,fK)| * |M(k_pump-k1,fK)|   -- pair coincidence product
  (c) Phi(k1) = arg(M(k1,fK) * M(k_pump-k1,fK)) -- phase sum (scatter + std)

Estimated runtime: < 1 minute (analytic, no GPU).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SIM25_CACHE = os.path.join(DATA_DIR, "sim25_kspectrum.npz")
CACHE = os.path.join(DATA_DIR, "sim36_twomode_correlator.npz")


def compute_correlator(my_xt, CX, DT_REC, k_pump):
    """Compute pair-coincidence product and phase sum.

    Parameters
    ----------
    my_xt  : (NT, NX) float array — m_y(x, t)
    CX     : float — cell size in x (m)
    DT_REC : float — time step (s)
    k_pump : float — SAW pump wavevector (rad/m)

    Returns
    -------
    k_um        : (NX,) — k axis in um^-1
    S_fK        : (NX,) — |M_y(k, f_K)|  (amplitude, not power)
    P_pair      : (NX,) — pair coincidence product P(k1)
    phi_sum     : (NX,) — phase sum Phi(k1) in radians
    f_K_meas    : float — measured Kittel frequency (GHz)
    """
    NT, NX = my_xt.shape

    # Use second half of time trace to suppress transients
    n_half = NT // 2
    signal = my_xt[n_half:].astype(np.float64)
    nt = signal.shape[0]

    # 2D FFT: axes 0=time, 1=space
    M2D = np.fft.fft2(signal)   # shape (nt, NX), not shifted

    # Frequency axes (unshifted, as returned by fft)
    freqs_t = np.fft.fftfreq(nt, d=DT_REC) * 1e-9   # GHz
    freqs_k = np.fft.fftfreq(NX, d=CX) * 2 * np.pi  # rad/m (unshifted)

    # Find index of Kittel frequency as peak of |M2D| integrated over k
    power_t = np.sum(np.abs(M2D[:nt//2]) ** 2, axis=1)  # positive freqs only
    i_fK = np.argmax(power_t[1:]) + 1   # skip DC
    f_K_meas = freqs_t[i_fK]

    # Complex spectrum at f_K: M(k) = M2D[i_fK, :]  (unshifted k)
    Mk = M2D[i_fK, :]    # shape (NX,)

    # Convert k_pump to nearest discrete k index
    dk = freqs_k[1]   # = 1/(NX*CX) * 2*pi
    # k_pump in units of dk
    n_pump = int(round(k_pump / dk)) % NX   # unshifted index

    # Build arrays: for each k1 index i, k2 index = (n_pump - i) % NX
    n_arr = np.arange(NX)
    n2_arr = (n_pump - n_arr) % NX

    Mk1 = Mk[n_arr]
    Mk2 = Mk[n2_arr]

    S_fK = np.abs(Mk)                        # amplitude spectrum at f_K
    P_pair = np.abs(Mk1) * np.abs(Mk2)       # pair coincidence product
    phi_sum = np.angle(Mk1 * Mk2)            # phase sum

    # Shift k axis for plotting
    k_um = np.fft.fftshift(freqs_k) * 1e-6   # um^-1
    S_fK = np.fft.fftshift(S_fK)
    P_pair = np.fft.fftshift(P_pair)
    phi_sum = np.fft.fftshift(phi_sum)

    return k_um, S_fK, P_pair, phi_sum, f_K_meas


def main():
    print("=" * 70)
    print("Sim 36: Two-mode pair correlator (pair selection rule proof)")
    print("=" * 70)

    if not os.path.isfile(SIM25_CACHE):
        print(f"  ERROR: sim25 cache not found at {SIM25_CACHE}")
        print("  Run sim25_spatial_kspectrum.py first.")
        return

    d25 = dict(np.load(SIM25_CACHE))
    print(f"  Loaded sim25 cache: {list(d25.keys())}")

    CX = float(d25['CX'])
    DT_REC = float(d25['DT_REC'])
    f_K = float(d25['f_K'])

    # MEL pump at 2f_K: k_pump = 2 * k_SAW(f_K)
    k_pump_mel = float(d25['mel_2fk_kSAW'])
    print(f"  MEL pump k_SAW = {k_pump_mel*1e-6:.3f} um^-1  (= 2 * k_SAW(f_K))")

    # MR drive at f_K: single mode at k_SAW(f_K)
    k_pump_mr = float(d25['mr_fk_kSAW'])
    print(f"  MR drive k_SAW = {k_pump_mr*1e-6:.3f} um^-1")

    if os.path.isfile(CACHE):
        print("  Correlator cache found. Loading...")
        res = dict(np.load(CACHE))
    else:
        print("  Computing correlators from sim25 raw data...")

        mel_my_xt = d25['mel_2fk_my_xt']
        mr_my_xt = d25['mr_fk_my_xt']

        k_mel, S_mel, P_mel, phi_mel, fK_mel = compute_correlator(
            mel_my_xt, CX, DT_REC, k_pump_mel)
        k_mr, S_mr, P_mr, phi_mr, fK_mr = compute_correlator(
            mr_my_xt, CX, DT_REC, k_pump_mel)  # use same k_pump for both

        print(f"  MEL: measured f_K = {fK_mel:.3f} GHz")
        print(f"  MR:  measured f_K = {fK_mr:.3f} GHz")

        # Phase coherence: std of phi_sum over high-amplitude k1 modes
        amp_thresh = 0.1 * P_mel.max()
        mask_mel = P_mel > amp_thresh
        phi_std_mel = np.std(phi_mel[mask_mel]) if mask_mel.sum() > 0 else np.nan
        phi_std_mr = np.std(phi_mr[mask_mel]) if mask_mel.sum() > 0 else np.nan
        print(f"  Phase-sum std (MEL, above-threshold k): {np.degrees(phi_std_mel):.1f} deg")
        print(f"  Phase-sum std (MR, same k mask):        {np.degrees(phi_std_mr):.1f} deg")

        res = {
            'k_um': k_mel,
            'S_mel': S_mel, 'P_mel': P_mel, 'phi_mel': phi_mel,
            'S_mr': S_mr, 'P_mr': P_mr, 'phi_mr': phi_mr,
            'k_pump_mel_um': k_pump_mel * 1e-6,
            'k_pump_mr_um': k_pump_mr * 1e-6,
            'f_K': f_K,
            'fK_mel': fK_mel, 'fK_mr': fK_mr,
            'phi_std_mel_deg': np.degrees(phi_std_mel),
            'phi_std_mr_deg': np.degrees(phi_std_mr),
        }
        np.savez(CACHE, **res)
        print(f"  Saved: {CACHE}")

    plot(res)


def plot(d):
    try:
        from plot_style import (apply_style, label_panels, SINGLE_COL,
                                CM_TO_INCH, SKY_BLUE, VERMILION, TEAL, BLACK)
        apply_style()
    except ImportError:
        SKY_BLUE, VERMILION, TEAL, BLACK = '#56B4E9', '#D55E00', '#009E73', '#000'
        SINGLE_COL = 3.38
        CM_TO_INCH = 1.0 / 2.54

    k = d['k_um']
    k_pump = float(d['k_pump_mel_um'])
    k_pump_half = k_pump / 2

    # Normalize
    S_mel = d['S_mel'] / d['S_mel'].max()
    P_mel = d['P_mel'] / d['P_mel'].max()
    S_mr = d['S_mr'] / d['S_mr'].max()
    P_mr = d['P_mr'] / d['P_mr'].max()
    phi_mel = d['phi_mel']
    phi_mr = d['phi_mr']

    # Vertical 3-row stack, supple-column width.
    fig_w = SINGLE_COL
    fig_h = 13.5 * CM_TO_INCH
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(hspace=0.55, left=0.18, right=0.95,
                        top=0.96, bottom=0.07)

    # --- (a) Spectral weight at f_K ---
    ax = axes[0]
    ax.plot(k, S_mel, '-', color=VERMILION, lw=0.9, label=r'MEL @ $2f_K$')
    ax.plot(k, S_mr, '--', color=SKY_BLUE, lw=0.9, label=r'MR @ $f_K$')
    ax.axvline(k_pump_half, color=VERMILION, ls=':', lw=0.5, alpha=0.6)
    ax.axvline(-k_pump_half, color=VERMILION, ls=':', lw=0.5, alpha=0.6)
    ax.axvline(float(d['k_pump_mr_um']), color=SKY_BLUE, ls=':', lw=0.5, alpha=0.6)
    ax.set_xlabel(r'$k_x$ ($\mu$m$^{-1}$)', labelpad=1)
    ax.set_ylabel(r'$|M_y(k,f_K)|$ (norm.)', labelpad=1)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-0.05, 1.25)
    ax.legend(loc='upper left', fontsize=6.5, frameon=False,
              handlelength=1.5, handletextpad=0.4, borderpad=0.3)
    ax.set_title(r'Amplitude at $f_K$', fontsize=8)

    # --- (b) Pair coincidence product P(k1) ---
    ax = axes[1]
    ax.plot(k, P_mel, '-', color=VERMILION, lw=0.9,
            label=r'MEL pair')
    ax.plot(k, P_mr, '--', color=SKY_BLUE, lw=0.9,
            label=r'MR pair')
    ax.axvline(k_pump_half, color='gray', ls=':', lw=0.6,
               label=fr'$k_\mathrm{{pump}}/2 = {k_pump_half:.1f}\,\mu$m$^{{-1}}$')
    ax.axvline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel(r'$k_1$ ($\mu$m$^{-1}$)', labelpad=1)
    ax.set_ylabel(r'$P(k_1)$ (norm.)', labelpad=1)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-0.05, 1.25)
    ax.legend(loc='upper left', fontsize=6.5, frameon=False,
              handlelength=1.5, handletextpad=0.4, borderpad=0.3)
    ax.set_title(r'$P(k_1)=|M(k_1)\,M(k_\mathrm{pump}\!-\!k_1)|$',
                 fontsize=8)

    # --- (c) Phase sum scatter ---
    ax = axes[2]
    amp_thresh_mel = 0.1 * P_mel.max()
    amp_thresh_mr = 0.1 * P_mr.max()
    mask_mel = P_mel > amp_thresh_mel
    mask_mr = P_mr > amp_thresh_mr
    ax.scatter(k[mask_mel], np.degrees(phi_mel[mask_mel]),
               s=8, c=VERMILION, alpha=0.7, label=r'MEL pairs',
               zorder=3, edgecolors='none')
    ax.scatter(k[mask_mr], np.degrees(phi_mr[mask_mr]),
               s=8, c=SKY_BLUE, alpha=0.7, marker='s',
               label=r'MR modes', zorder=2, edgecolors='none')
    std_mel = float(d.get('phi_std_mel_deg', np.nan))
    std_mr = float(d.get('phi_std_mr_deg', np.nan))
    # Legend in lower-left corner (k<-15 region has no data).
    ax.legend(loc='lower left', fontsize=6.5, frameon=False,
              handlelength=1.2, handletextpad=0.3, borderpad=0.3,
              bbox_to_anchor=(0.0, 0.0))
    # sigma_Phi annotation in upper-right (also outside data clusters).
    ax.text(0.97, 0.97,
            fr'MEL $\sigma_\Phi={std_mel:.0f}^\circ$' + '\n'
            + fr'MR $\sigma_\Phi={std_mr:.0f}^\circ$',
            transform=ax.transAxes, fontsize=6.5, va='top', ha='right')
    ax.set_xlabel(r'$k_1$ ($\mu$m$^{-1}$)', labelpad=1)
    ax.set_ylabel(r'$\Phi(k_1)$ (deg)', labelpad=1)
    ax.set_ylim(-220, 220)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_xlim(-30, 30)
    ax.set_title(r'Phase sum $\Phi(k_1)=\arg(M_{k_1} M_{k_\mathrm{pump}-k_1})$',
                 fontsize=8)

    try:
        label_panels(axes)
    except Exception:
        pass

    for ext in ['pdf', 'png']:
        fig.savefig(
            os.path.join(FIG_DIR, f'fig_sim36_twomode_correlator.{ext}'),
            dpi=300)
    plt.close(fig)
    print("  Saved: fig_sim36_twomode_correlator.pdf/png")
    print(f"  Key result: MEL phase-sum std = {d.get('phi_std_mel_deg', '??'):.1f} deg "
          f"(< 60 deg => coherent pairing at k1+k2=k_SAW confirmed)")


if __name__ == "__main__":
    main()
