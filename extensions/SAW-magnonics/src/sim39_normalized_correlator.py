"""sim39: normalized pair-correlator (heterodyne-style coherence) C(k).

Builds on the cached m_y(x,t) data of sim25 (MEL@2f_K and MR@f_K) and
sim37 (SAW MEL @ 2f_K and matched uniform Suhl pump).  Uses Welch-style
segmentation of the second half of each run to construct an ensemble of
independent M_y(k, f_K) realizations, then computes the normalized
pair coherence:

    C(k) = | <M(k, f_K) M(k_pump - k, f_K) exp(+i omega_pump t_seg)> |
           / sqrt( <|M(k, f_K)|^2> <|M(k_pump-k, f_K)|^2> )

For phase-locked pair generation, C -> 1; for independent thermal
populations, C -> 1/sqrt(M).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")

DT_REC = 20e-12  # s
CX = 5e-9        # m
M_SEG = 6        # Welch segments


def coherence(my_xt, k_pump_per_m, f_target_Hz, omega_pump_rad,
              t_start_idx=None, t_end_idx=None):
    """Normalized pair coherence C(k) using Welch segmentation.

    By default uses the second half of the run; pass t_start_idx /
    t_end_idx to restrict to a saturation-free window."""
    T, NX = my_xt.shape
    if t_start_idx is None:
        t_start_idx = T // 2
    if t_end_idx is None:
        t_end_idx = T
    my_half = my_xt[t_start_idx:t_end_idx, :]
    Th = my_half.shape[0]
    win_len = Th // M_SEG

    dx = CX
    k_axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=dx))  # /m

    # Index of pump complement: for each k1 index j, find j2 with k2 = k_pump - k1
    j_pump = np.argmin(np.abs(k_axis - k_pump_per_m))
    # complement index relative to centre
    centre = NX // 2
    j2_arr = np.array([
        np.argmin(np.abs(k_axis - (k_pump_per_m - k_axis[j]))) for j in range(NX)
    ])

    Mk_segments = []
    f_axis = None
    for i in range(M_SEG):
        seg = my_half[i * win_len:(i + 1) * win_len, :]
        # Hanning in time, rectangle in space (periodic)
        win_t = np.hanning(win_len)[:, None]
        seg_w = seg * win_t
        F = np.fft.fftshift(np.fft.fft2(seg_w), axes=(0, 1))
        f_axis = np.fft.fftshift(np.fft.fftfreq(win_len, d=DT_REC))
        i_f = int(np.argmin(np.abs(f_axis - f_target_Hz)))
        Mk = F[i_f, :].astype(np.complex128)
        Mk_segments.append(Mk)

    Mk_arr = np.array(Mk_segments)  # (M, NX)
    # Pump-phase correction for each segment
    t_seg = np.arange(M_SEG) * win_len * DT_REC
    pump_phase = np.exp(1j * omega_pump_rad * t_seg)[:, None]
    Mk_corr = Mk_arr * np.sqrt(pump_phase)  # split phase between two factors

    # Cross-correlator: <M(k) * M(k_pump-k) * exp(+i omega_pump t)>
    # Already pre-multiplied each Mk by sqrt(pump_phase) so product carries pump_phase
    pair = np.zeros(NX, dtype=complex)
    norm1 = np.zeros(NX)
    norm2 = np.zeros(NX)
    for i in range(M_SEG):
        pair += Mk_corr[i, :] * Mk_corr[i, j2_arr]
        norm1 += np.abs(Mk_arr[i, :])**2
        norm2 += np.abs(Mk_arr[i, j2_arr])**2
    pair /= M_SEG
    norm1 /= M_SEG
    norm2 /= M_SEG

    C = np.abs(pair) / (np.sqrt(norm1 * norm2) + 1e-30)
    floor = 1.0 / np.sqrt(M_SEG)  # incoherent baseline
    return k_axis, C, floor


def main():
    d25 = np.load(os.path.join(DATA, "sim25_kspectrum.npz"))
    d37 = np.load(os.path.join(DATA, "sim37_suhl_control.npz"))

    f_K = float(d25["f_K"])
    k_SAW = float(d37["saw_kSAW"])
    omega_pump = 2 * np.pi * (2 * f_K)  # rad/s

    print(f"f_K = {f_K/1e9:.3f} GHz")
    print(f"k_SAW = {k_SAW/1e6:.3f} /um  ->  k_SAW/2 = {k_SAW/2e6:.3f} /um")
    print(f"Segments: M = {M_SEG}, incoherent baseline = {1/np.sqrt(M_SEG):.3f}")

    # Empirically, saw_my_xt saturates to |m|~1 by ~t=12 ns and phase
    # coherence is lost beyond that.  Use t=4-12 ns growth window for
    # SAW MEL; full second half for MR and Suhl (linear regime).
    T_grow_start = int(4e-9 / DT_REC)    # 4 ns
    T_grow_end = int(12e-9 / DT_REC)     # 12 ns

    # FFT-sign convention: Python's fft2 of cos(omega_p t - k_p x) puts
    # the physical traveling-wave peak at (-omega_p, -k_p) in the
    # +omega_p Fourier bin we slice at f_K, so the effective pump
    # momentum in M_y(k, +f_K) space is -k_SAW.
    k_pump_mel = -k_SAW

    # (a) SAW MEL pump at 2f_K; growth window
    k_axis, C_mel, floor = coherence(
        d37["saw_my_xt"], k_pump_mel, f_K, omega_pump,
        t_start_idx=T_grow_start, t_end_idx=T_grow_end,
    )

    # (b) MR direct drive at f_K (single mode at k_SAW, no pair)
    _, C_mr, _ = coherence(
        d25["mr_fk_my_xt"], k_pump_mel, f_K, omega_pump
    )

    # (c) Uniform Suhl pump at 2omega_K: pair pump momentum = 0
    _, C_suhl, _ = coherence(
        d37["suhl_my_xt"], 0.0, f_K, omega_pump
    )

    k_um = k_axis / 1e6
    k_half_um = k_SAW / 2e6

    out = os.path.join(DATA, "sim39_normalized_correlator.npz")
    np.savez(
        out, k_um=k_um, C_mel_saw=C_mel, C_mr=C_mr, C_suhl=C_suhl,
        k_SAW_um=k_SAW/1e6, k_half_um=k_half_um,
        f_K=f_K, M_seg=M_SEG, floor=floor,
    )
    print(f"saved {out}")

    # Quick peak values.  With FFT sign convention k_pump_eff = -k_SAW,
    # the SAW MEL pair-band centroid sits at -k_SAW/2 in the plotted k.
    j_pmhalf = np.argmin(np.abs(k_um + k_half_um))  # -k_SAW/2
    j_phalf = np.argmin(np.abs(k_um - k_half_um))   # +k_SAW/2
    j_zero = np.argmin(np.abs(k_um))
    j_psaw = np.argmin(np.abs(k_um - k_SAW/1e6))
    j_msaw = np.argmin(np.abs(k_um + k_SAW/1e6))
    print("=== SAW MEL pump (growth window) ===")
    print(f"  C at -k_SAW/2 (pair centroid) = {C_mel[j_pmhalf]:.3f}")
    print(f"  C at -k_SAW                   = {C_mel[j_msaw]:.3f}")
    print(f"  C at  0                       = {C_mel[j_zero]:.3f}")
    print(f"  C at +k_SAW/2                 = {C_mel[j_phalf]:.3f}")
    print(f"  C at +k_SAW                   = {C_mel[j_psaw]:.3f}")
    # Average C over a window around the pair manifold
    win_mask = (k_um < 0) & (k_um > -k_SAW/1e6)
    print(f"  mean C in (-k_SAW, 0) window  = {C_mel[win_mask].mean():.3f}")
    print()
    print("=== MR direct drive ===")
    print(f"  C at -k_SAW/2                 = {C_mr[j_pmhalf]:.3f}")
    print(f"  C at  0                       = {C_mr[j_zero]:.3f}")
    print(f"  mean C in (-k_SAW, 0) window  = {C_mr[win_mask].mean():.3f}")
    print()
    print("=== Uniform Suhl pump (k_pump=0) ===")
    print(f"  C at -k_SAW/2                 = {C_suhl[j_pmhalf]:.3f}")
    print(f"  C at  0  (pair centroid)      = {C_suhl[j_zero]:.3f}")
    print(f"  C at +k_SAW/2                 = {C_suhl[j_phalf]:.3f}")

    # ----- Figure -----
    try:
        from plot_style import set_prb_style
        set_prb_style()
    except Exception:
        pass
    # Plot in physical (paper) k convention: k_plot = -k_fft so the SAW
    # MEL pair centroid appears at +k_SAW/2 in agreement with Figs. 2-4.
    k_plot = -k_um
    # Sort for monotonic x-axis
    order = np.argsort(k_plot)
    k_plot_s = k_plot[order]
    C_mel_s = C_mel[order]
    C_suhl_s = C_suhl[order]
    C_mr_s = C_mr[order]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    mask = (k_plot_s > -15) & (k_plot_s < 15)
    ax.plot(k_plot_s[mask], C_mel_s[mask], color="#D55E00", lw=1.5,
            label=r"SAW MEL @ $2f_K$")
    ax.plot(k_plot_s[mask], C_suhl_s[mask], color="#666666", lw=1.3,
            label=r"Uniform Suhl pump")
    ax.plot(k_plot_s[mask], C_mr_s[mask], color="#0072B2", lw=1.1,
            alpha=0.75, label=r"MR direct drive (control)")
    ax.axhline(floor, color="k", ls=":", lw=0.8,
               label=r"incoherent baseline $1/\sqrt{M}$")
    ax.axvline(k_half_um, color="#D55E00", ls="--", lw=0.7, alpha=0.7)
    ax.axvline(-k_half_um, color="#D55E00", ls="--", lw=0.4, alpha=0.4)
    ax.axvline(0, color="#666666", ls="--", lw=0.7, alpha=0.7)
    ax.text(k_half_um, 1.02, r"$+k_\mathrm{SAW}/2$",
            color="#D55E00", ha="center", va="bottom", fontsize=7)
    ax.text(0, 1.02, r"$k=0$", color="#666666",
            ha="center", va="bottom", fontsize=7)
    ax.set_xlabel(r"$k_1\ (\mu\mathrm{m}^{-1})$")
    ax.set_ylabel(r"normalized pair coherence $C(k_1)$")
    ax.set_xlim(-15, 15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc="lower center", framealpha=0.9, ncol=2)
    fig.tight_layout()
    out_pdf = os.path.join(FIGS, "fig_sim39_correlator.pdf")
    fig.savefig(out_pdf)
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
