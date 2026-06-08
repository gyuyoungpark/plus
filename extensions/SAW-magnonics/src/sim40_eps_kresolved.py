"""Simulation 40: Strain-amplitude sweep with k-resolved m_y(x, t) storage.

Addresses two reviewer criticisms of the original sim21 (volume-averaged):
  - B1: 5e-5 and 7e-5 data points added between the original 3e-5 and 1e-4.
  - B2: store full m_y(x, t) so we can compute, alongside the volume average,
        the pair-band integrated power
            I_pair(eps_0) = integral_{pair band} |M_y(k, f_K)|^2 dk
        and the K-resolved correlator peak
            C_pair(eps_0) = C(K = k_SAW)
        at f_SAW = 2 f_K.  These are the directly finite-k observables.

Configurations per eps_0:
  (1) SAW Full pump at f_SAW = 2 f_K    (parametric, MEL+MR enabled)
  (2) SAW Full pump at f_SAW = f_K      (direct residual)
  (3) SAW MR-only pump at f_SAW = f_K   (linear MR baseline)

Resumable: each (eps, config) writes to a per-run npz, then a final pack.

Grid: 1024 x 8 x 1 at 5 nm  (same as sim37 spatial setup but half NX)
T_RUN = 15 ns, DT_REC = 20 ps, DT_STEP = 2e-13 s

Estimated runtime: 8 eps x 3 configs = 24 runs, ~5-10 min each on a single
GPU -> 2-4 hours total. Designed to be backgrounded.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0
from saw_chiral import ChiralSurfaceAcousticWave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CKPT_DIR = os.path.join(DATA_DIR, "sim40_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CACHE = os.path.join(DATA_DIR, "sim40_eps_kresolved.npz")

# ---------------- Material (YIG benchmark, identical to sim21/37) ----------
GAMMA = 1.76e11
MS    = 140e3
AEX   = 3.65e-12
ALPHA = 5e-4
B1    = -8.8e6
KMR   = 1.0e6
XI    = 0.68
V_SAW = 3500.0

# ---------------- Geometry ----------------
NX, NY, NZ = 1024, 8, 1
CX, CY, CZ = 5e-9, 10e-9, 20e-9

# ---------------- Run parameters ----------------
F_REF = 3.0e9   # reference Kittel target

T_RUN  = 15e-9
DT_REC = 20e-12
NT_REC = int(T_RUN / DT_REC) + 1
DT_STEP = 2e-13

# New eps grid: original sim21 + dense 5e-5, 7e-5
EPS_VALUES = np.array([1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4, 1e-3, 3e-3])

# Configurations: (label, f_saw_multiplier_of_fK, enable_mel, Kmr)
CONFIGS = [
    ('full_2fK', 2.0, True,  KMR),   # parametric MEL+MR @ 2 f_K
    ('full_fK',  1.0, True,  KMR),   # direct full @ f_K
    ('mr_fK',    1.0, False, KMR),   # MR linear baseline @ f_K
]


def find_resonance_field(f_target):
    """Solve Kittel for B0 given target f_K."""
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * MS, -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4 * a * c)) / 2


B0_RES = find_resonance_field(F_REF)


def run_single(eps0, f_saw, enable_mel, K_mr):
    """One micromagnetic run -> m_y(x, t) array (NT_REC, NX)."""
    wavelength = V_SAW / f_saw

    world = World((CX, CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=(2, 2, 0))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat = MS
    magnet.aex = AEX
    magnet.alpha = ALPHA
    if enable_mel:
        magnet.B1 = B1
    magnet.magnetization = (1, 0, 0.005)
    magnet.bias_magnetic_field = (B0_RES, 0, 0)
    magnet.enable_demag = True

    saw = ChiralSurfaceAcousticWave(
        frequency=f_saw, wavelength=wavelength, amplitude=eps0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr, enable_barnett=False)
    saw.apply(magnet, Msat=MS, enable_mel=enable_mel)

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False

    my_xt = np.zeros((NT_REC, NX), dtype=np.float32)
    for i in range(NT_REC):
        m = magnet.magnetization.eval()
        # m has shape (3, NZ, NY, NX); average over y and z (only NZ=1)
        my_xt[i] = m[1, 0].mean(axis=0)
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)
    return my_xt


def ckpt_path(eps0, label):
    return os.path.join(CKPT_DIR, f"sim40_{label}_eps{eps0:.0e}.npz")


def main():
    total_start = time.time()
    n_eps = len(EPS_VALUES)
    n_conf = len(CONFIGS)

    print("=" * 78)
    print("Sim 40: Strain sweep with k-resolved m_y(x,t) storage")
    print(f"  Grid {NX}x{NY}x{NZ} @ CX={CX*1e9:.1f} nm  (L_x={NX*CX*1e6:.2f} um)")
    print(f"  B0_res = {B0_RES*1e3:.3f} mT   (f_K = {F_REF*1e-9:.2f} GHz)")
    print(f"  T_RUN = {T_RUN*1e9:.1f} ns, DT_REC = {DT_REC*1e12:.0f} ps")
    print(f"  eps grid: {EPS_VALUES}")
    print(f"  configs:  {[c[0] for c in CONFIGS]}")
    print(f"  Total runs: {n_eps * n_conf}")
    print("=" * 78)

    # ---- Execute runs (skip those with a checkpoint) ----
    done = 0
    skipped = 0
    for ei, eps0 in enumerate(EPS_VALUES):
        for ci, (label, fmult, en_mel, kmr) in enumerate(CONFIGS):
            cp = ckpt_path(eps0, label)
            if os.path.isfile(cp):
                print(f"  [skip] eps={eps0:.0e}  {label:10s}  (cached)")
                skipped += 1
                continue
            f_saw = fmult * F_REF
            t0 = time.time()
            print(f"  ---- eps={eps0:.0e}  {label:10s}  f_SAW={f_saw*1e-9:.2f} GHz")
            sys.stdout.flush()
            my_xt = run_single(eps0, f_saw, en_mel, kmr)
            np.savez(cp,
                     my_xt=my_xt,
                     eps0=eps0, f_saw=f_saw, f_K=F_REF, B0=B0_RES,
                     enable_mel=en_mel, K_mr=kmr,
                     CX=CX, DT_REC=DT_REC, NX=NX, NY=NY)
            done += 1
            elapsed = time.time() - t0
            total_elapsed = time.time() - total_start
            remaining = n_eps * n_conf - done - skipped
            est = (total_elapsed / max(done, 1)) * remaining
            print(f"       done in {elapsed/60:.1f} min   "
                  f"(total elapsed {total_elapsed/60:.1f} min, "
                  f"~{est/60:.1f} min remaining)")
            sys.stdout.flush()

    # ---- Pack everything ----
    pack = {
        'eps_values': EPS_VALUES,
        'configs':    np.array([c[0] for c in CONFIGS]),
        'f_K': F_REF, 'B0': B0_RES,
        'CX': CX, 'DT_REC': DT_REC, 'NX': NX, 'NY': NY,
        'T_RUN': T_RUN, 'V_SAW': V_SAW,
    }
    for eps0 in EPS_VALUES:
        for (label, fmult, _, _) in CONFIGS:
            cp = ckpt_path(eps0, label)
            d = np.load(cp)
            pack[f'my_xt__{label}__eps{eps0:.0e}'] = d['my_xt']
    np.savez(CACHE, **pack)
    print(f"\n  PACKED -> {CACHE}")
    print(f"  Total wall time: {(time.time()-total_start)/60:.1f} min")


if __name__ == "__main__":
    main()
