# Data archive

The simulation scripts in `src/` write per-run NumPy archives
(`.npz`) into a `data/` directory at the SAW-magnonics root. Total
size of the cached datasets used for the published main-text and
supplemental figures is approximately 300 MB, dominated by the
$k$-resolved $m_y(x,t)$ snapshots of `sim37_suhl_control.npz`
(16 MB), `sim38_nonreciprocal.npz` (16 MB), and
`sim40_eps_kresolved.npz` (71 MB). These exceed GitHub's
recommended per-file size limit and are therefore not hosted in
this repository.

## Obtaining the cached data

The full data archive (including per-sweep checkpoints) is
available from the corresponding authors of the manuscript on
reasonable request, and is provided to referees during peer
review.

## Regenerating the data from scratch

All `sim*.py` scripts under `src/` are checkpoint-resumable and
self-contained. Running them on a single CUDA GPU regenerates
every cached dataset. Approximate wall-clock costs (NVIDIA RTX
4090, single precision):

| Script | Cost |
|---|---|
| `sim23_fft_proof.py` | ~10 min |
| `sim24_grid_convergence.py` | ~30 min |
| `sim21_eps_threshold.py` | 6 strain × 40 frequency × 2 channel ≈ 8–10 h |
| `sim37_suhl_control.py` | 2 runs of 20 ns @ 2048×8 ≈ 30 min |
| `sim38_nonreciprocal.py` | 2 runs of 15 ns @ 1024×8 ≈ 15 min |
| `sim40_eps_kresolved.py` | 8 strain × 3 channel ≈ 3–4 h |

Other simulation scripts complete in a few minutes each.
