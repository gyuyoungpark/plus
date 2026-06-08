# SAW-magnonics extension for mumax⁺

Source and simulation code for the manuscript

> *Finite-Momentum Parametric Magnon Pairing by Traveling Surface
> Acoustic Waves*, Physical Review Applied (submitted).

This extension provides the Python drivers, simulation scripts, and
post-processing code used to generate every micromagnetic dataset
underlying the manuscript. The GPU kernel
(`src/physics/chiralsawfield.cu`), the new `Ferromagnet` parameters
(`Kmr`, `enable_barnett`, `saw_enable_mel`, `saw_enable_mr`,
`saw_enable_barnett`), and the associated field quantities
(`magneto_rotation_field`, `spin_rotation_field`, `rotation_vector`,
`angular_velocity`) live in the mumax⁺ core of this repository.

## Layout

```
extensions/SAW-magnonics/
├── README.md           This file.
├── DATA_ARCHIVE.md     How to obtain the cached .npz simulation outputs.
└── src/
    ├── saw.py             Base SAW driver (magnetoelastic-only channel).
    ├── saw_chiral.py      ChiralSurfaceAcousticWave wrapper that prescribes
    │                      a traveling Rayleigh SAW (epsilon_xx, Omega_y,
    │                      d(Omega_y)/dt) and exposes the MEL/MR/Barnett
    │                      channel switches of the mumax+ chiral-SAW kernel.
    ├── analyze_sim40.py   Post-processing for the pair-band integrated
    │                      observable I_pair(eps_0) from k-resolved
    │                      m_y(x, t) snapshots.
    └── sim01_…_sim40_*.py Simulation scripts, one per cached dataset.
```

## Requirements

- mumax⁺ built from this repository (see top-level `README.md` for
  install instructions; CUDA toolkit, C++17 compiler, Python ≥ 3.11).
- Python packages: `numpy`, `scipy`.

## Running the simulations

Each `sim*.py` script is self-contained, checkpoint-resumable, and
writes its own NumPy archive into a `data/` directory at the
SAW-magnonics root. From `extensions/SAW-magnonics/`:

```bash
python src/sim37_suhl_control.py    # SAW MEL vs uniform Suhl pump
python src/sim38_nonreciprocal.py   # +k vs -k SAW direction reversal
python src/sim40_eps_kresolved.py   # strain sweep with k-resolved storage
# ... etc.
```

The cached `.npz` archives are too large to host in the repository
directly; see `DATA_ARCHIVE.md` for how to obtain them or to
regenerate them from scratch.

## Post-processing

`analyze_sim40.py` reads the cached `sim40_eps_kresolved.npz` and
computes three observables per strain amplitude:

1. Volume-averaged peak |m_perp|;
2. Pair-band integrated power
   `I_pair(eps_0) = integral_{|k +/- k_SAW/2|<dk} |M_y(k, f_K)|^2 dk`;
3. Pair-momentum correlator `C(K = k_SAW)`.

The output is written to `data/sim40_observables.npz`.

## License

Licensed under the GNU General Public License v3.0, consistent with
the parent mumax⁺ repository. See the top-level `LICENSE` file.
