"""Analyze sim40 cache: extract three observables vs eps_0.

Inputs
------
Per-run checkpoints in data/sim40_checkpoints/, each containing
  my_xt : (NT_REC, NX) float32  -- m_y(x, t) for one (eps0, config) combo
  eps0, f_saw, f_K, B0, enable_mel, K_mr, CX, DT_REC, NX, NY
(The final pack data/sim40_eps_kresolved.npz contains all of them under
my_xt__{label}__eps{e:.0e} keys, but we also tolerate partial runs by
reading checkpoints directly.)

Outputs (data/sim40_observables.npz)
------------------------------------
eps_values        : (n_eps,)   the strain sweep
labels            : (n_conf,)  config labels
vol_peak          : (n_eps, n_conf)  volume-averaged peak |m_y(t)| in
                                      the second half of each run
I_pair            : (n_eps, n_conf)  pair-band integrated power at f_K
                                      (k-resolved, finite-k observable)
C_K_peak          : (n_eps, n_conf)  pair-correlator value at K = k_SAW
                                      of the pump (only meaningful for
                                      full_2fK; computed for all for
                                      consistency)
v_SAW, k_SAW_2fK, k_SAW_fK  : metadata

For each run the analysis FFTs m_y(x, t) on the second half of the run,
masks to the +f_K slice, and computes:
  - vol_peak   = max_{t>=T/2} |<m_y>_x(t)|        (volume-averaged)
  - I_pair     = sum_{|k - k_pair|<Δk_pair} |M_y(k, f_K)|^2   on f_K slice
  - C_K_peak   = sum_k |M_y(k, f_K) M_y(K_pump - k, f_K)|

Definitions
-----------
* For full_2fK (parametric, f_SAW = 2 f_K): pair band centred at
  k_pair = k_SAW(2 f_K)/2 = k_SAW(f_K).  K_pump = k_SAW(2 f_K).
* For mr_fK and full_fK (direct, f_SAW = f_K): the response is a single
  forced mode at k = k_SAW(f_K), so the "pair-band" integral uses the
  same k_pair window for consistency; K_pump is undefined and we use
  K = 2 k_SAW(f_K) merely as a bookkeeping value (the correlator will
  be small for these channels).
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CKPT_DIR = os.path.join(DATA_DIR, "sim40_checkpoints")
PACK = os.path.join(DATA_DIR, "sim40_eps_kresolved.npz")
OUT = os.path.join(DATA_DIR, "sim40_observables.npz")

EPS_VALUES = np.array([1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 1e-3, 3e-3])
LABELS = ['full_2fK', 'full_fK', 'mr_fK']

V_SAW = 3500.0
F_K = 3.0e9


def fft_kf(my_xt, CX, DT_REC):
    """2D FFT on the second half of m_y(x, t). Returns sorted axes."""
    NT, NX = my_xt.shape
    sub = my_xt[NT // 2:].astype(np.float64)
    sub -= sub.mean()
    win = np.hanning(sub.shape[0])[:, None]
    F = np.fft.fftshift(np.fft.fft2(sub * win), axes=(0, 1))
    f_axis = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=DT_REC))
    k_axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=CX))
    # Display convention: flip k sign so that a forward-propagating SAW
    # at +k_SAW shows up at positive k_axis in the plotted spectrum.
    k_disp = -k_axis
    order = np.argsort(k_disp)
    return k_disp[order], f_axis, F[:, order]


def vol_peak(my_xt):
    """Volume average |<m_y>_x(t)| peak in the second half."""
    NT = my_xt.shape[0]
    half = my_xt[NT // 2:].mean(axis=1)
    return float(np.max(np.abs(half)))


def I_pair(F_kf, k_axis, f_axis, k_pair, dk_window):
    """Integrate |M_y(k, f_K)|^2 over a window centred on k_pair at f=f_K.

    Includes both +k_pair and -k_pair windows (parametric pair sits at
    both signs because pairs are co-generated)."""
    i_fK = int(np.argmin(np.abs(f_axis - F_K)))
    slice_fK = np.abs(F_kf[i_fK, :]) ** 2
    band_p = (k_axis > k_pair - dk_window) & (k_axis < k_pair + dk_window)
    band_m = (k_axis > -k_pair - dk_window) & (k_axis < -k_pair + dk_window)
    return float(slice_fK[band_p].sum() + slice_fK[band_m].sum())


def C_K(F_kf, k_axis, f_axis, K_pump):
    """C(K=K_pump) at f=f_K from the pair-product correlator.

    C(K) = sum_k |M_y(k, f_K) M_y(K - k, f_K)| evaluated at K = K_pump.
    """
    i_fK = int(np.argmin(np.abs(f_axis - F_K)))
    M = F_kf[i_fK, :]
    abs_M = np.abs(M)
    dk = k_axis[1] - k_axis[0]
    k0 = k_axis[0]
    i2 = np.rint((K_pump - k_axis - k0) / dk).astype(int)
    mask = (i2 >= 0) & (i2 < len(k_axis))
    i1 = np.where(mask)[0]
    return float(np.sum(abs_M[i1] * abs_M[i2[mask]]))


def load_run(eps0, label):
    """Try the per-run checkpoint, then fall back to the packed file."""
    cp = os.path.join(CKPT_DIR, f"sim40_{label}_eps{eps0:.0e}.npz")
    if os.path.isfile(cp):
        d = np.load(cp)
        return dict(my_xt=d['my_xt'], CX=float(d['CX']),
                    DT_REC=float(d['DT_REC']), f_saw=float(d['f_saw']))
    if os.path.isfile(PACK):
        p = np.load(PACK)
        key = f'my_xt__{label}__eps{eps0:.0e}'
        if key in p:
            return dict(my_xt=p[key], CX=float(p['CX']),
                        DT_REC=float(p['DT_REC']),
                        f_saw=float({'full_2fK': 2, 'full_fK': 1,
                                      'mr_fK': 1}[label]) * F_K)
    return None


def main():
    n_eps = len(EPS_VALUES)
    n_conf = len(LABELS)
    vol = np.full((n_eps, n_conf), np.nan)
    Ip = np.full((n_eps, n_conf), np.nan)
    CK = np.full((n_eps, n_conf), np.nan)

    # k_SAW values
    k_SAW_fK = 2 * np.pi * F_K / V_SAW
    k_SAW_2fK = 2 * np.pi * (2 * F_K) / V_SAW
    # Pair band centre for the parametric pump is k_SAW(2fK)/2 = k_SAW(fK)
    k_pair = k_SAW_fK

    n_done = 0
    for ei, eps0 in enumerate(EPS_VALUES):
        for ci, label in enumerate(LABELS):
            r = load_run(eps0, label)
            if r is None:
                print(f"  [skip] eps={eps0:.0e}  {label:10s}  (no data)")
                continue
            my_xt = r['my_xt']
            CX = r['CX']; DT_REC = r['DT_REC']

            v = vol_peak(my_xt)
            k_disp, f_axis, F_kf = fft_kf(my_xt, CX, DT_REC)

            # Δk window for pair-band integration: 1/L_x = 2π/(N_x CX),
            # multiplied by ~16 bins for a finite resolution band.
            dk_bin = abs(k_disp[1] - k_disp[0])
            dk_window = 16 * dk_bin

            Ip_val = I_pair(F_kf, k_disp, f_axis, k_pair, dk_window)
            K_pump = k_SAW_2fK if label == 'full_2fK' else 0.0
            CK_val = C_K(F_kf, k_disp, f_axis, K_pump)

            vol[ei, ci] = v
            Ip[ei, ci] = Ip_val
            CK[ei, ci] = CK_val
            n_done += 1
            print(f"  eps={eps0:.0e} {label:10s}  "
                  f"vol={v:.2e}  I_pair={Ip_val:.2e}  "
                  f"C(K)={CK_val:.2e}")
            sys.stdout.flush()

    np.savez(OUT,
             eps_values=EPS_VALUES,
             labels=np.array(LABELS),
             vol_peak=vol,
             I_pair=Ip,
             C_K_peak=CK,
             v_SAW=V_SAW,
             k_SAW_fK=k_SAW_fK,
             k_SAW_2fK=k_SAW_2fK,
             F_K=F_K)
    print(f"\nProcessed {n_done}/{n_eps * n_conf} runs.")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
