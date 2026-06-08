"""Simulation 8: Standing SAW cavity - spatially resolved EP search.

A standing SAW creates a position-dependent competition between coupling
channels:
  - Strain antinodes (kx = n*pi): magnetoelastic coupling is maximum,
    magneto-rotation coupling is zero.
  - Rotation antinodes (kx = pi/2 + n*pi): magneto-rotation is maximum,
    magnetoelastic coupling is zero.

The total effective coupling g_eff(x) varies continuously with position,
potentially crossing the exceptional point (EP) threshold where the
magnon-phonon coupling transitions from strong (coherent) to weak
(dissipative) regime.

Protocol:
  1. Set up a 1D strip long enough to contain several SAW wavelengths.
  2. Apply standing SAW with MEL + MR + Barnett channels.
  3. Drive near the Kittel resonance for a duration, then let ring down.
  4. Compute local spectra vs position using windowed FFT.
  5. Extract local frequency splitting and linewidth.
  6. Map coupling regime (strong/EP/weak) vs position.
  7. Compare with MEL-only standing SAW.

Material: CoFeB in-plane film
    Msat  = 1.0 MA/m
    Aex   = 15 pJ/m
    alpha = 0.01
    B1    = -8.8 MJ/m^3
    K_mr  = 8 MJ/m^3 (enhanced for visibility)
"""

import json
import os
import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from saw import MU0, GAMMA
from saw_chiral import StandingSAW
from analysis import (windowed_local_spectrum, extract_peaks,
                      classify_coupling_regime,
                      compute_spatial_frequency_map)

from mumaxplus import Ferromagnet, Grid, World

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "raw", "sim08_standing_ep"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed", "maps"), exist_ok=True)

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
MSAT = 1.0e6
AEX = 1.5e-11
ALPHA = 0.01          # moderate damping
B1 = -8.8e6
K_MR = 8e6            # enhanced MR coupling for EP visibility
XI = 0.68
EPS0 = 3e-3           # moderate strain
V_SAW = 3500.0
B0 = 50e-3

# Grid: 1D strip, several SAW wavelengths
F_SAW = 2.0e9         # SAW frequency (Hz)
LAMBDA_SAW = V_SAW / F_SAW

# Choose grid to fit ~3 SAW wavelengths
L_TOTAL = 3 * LAMBDA_SAW    # total length
CX = 10e-9                  # cell size along x
NX = int(L_TOTAL / CX)
NY, NZ = 1, 1
CY, CZ = 30e-9, 1e-9

# Simulation timing
T_DRIVE = 10e-9       # driving time
T_RINGDOWN = 10e-9    # ringdown time
T_TOTAL = T_DRIVE + T_RINGDOWN
DT_SAVE = 0.05e-9     # time between saved snapshots
N_SAVE = int(T_TOTAL / DT_SAVE) + 1


def setup_standing_saw(K_mr_val=K_MR, enable_barnett=True, enable_mel=True):
    """Create a 1D strip with standing SAW driving.

    Returns (world, magnet, saw).
    """
    cellsize = (CX, CY, CZ)
    world = World(cellsize)
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))

    magnet.msat = MSAT
    magnet.aex = AEX
    magnet.alpha = ALPHA
    magnet.magnetization = (1, 0, 0)
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = False
    magnet.B1 = B1
    magnet.B2 = 0.0

    saw = StandingSAW(
        F_SAW, LAMBDA_SAW, EPS0,
        direction='x', ellipticity=XI,
        K_mr=K_mr_val, enable_barnett=enable_barnett)
    saw.apply(magnet, Msat=MSAT, enable_mel=enable_mel)

    return world, magnet, saw


def run_standing_saw(label, K_mr_val=K_MR, enable_barnett=True):
    """Run the standing SAW simulation and record spatiotemporal data.

    Returns time array and m_y(t, x) 2D array.
    """
    print(f"  Running: {label}")
    world, magnet, saw = setup_standing_saw(
        K_mr_val=K_mr_val, enable_barnett=enable_barnett)

    time_arr = np.linspace(0, T_TOTAL, N_SAVE)

    # Record full m_y(x) at each time step
    m_y_tx = np.zeros((N_SAVE, NX))
    m_z_tx = np.zeros((N_SAVE, NX))

    # Initial state
    m_init = magnet.magnetization.eval()
    m_y_tx[0, :] = m_init[1, 0, 0, :]
    m_z_tx[0, :] = m_init[2, 0, 0, :]

    for i in range(1, N_SAVE):
        world.timesolver.run(DT_SAVE)
        m = magnet.magnetization.eval()
        m_y_tx[i, :] = m[1, 0, 0, :]
        m_z_tx[i, :] = m[2, 0, 0, :]

        if (i + 1) % (N_SAVE // 5) == 0:
            print(f"    [{100*(i+1)/N_SAVE:5.1f}%] "
                  f"t = {time_arr[i]*1e9:.1f} ns")

    return time_arr, m_y_tx, m_z_tx


def analyze_spatial_spectra(time_arr, m_y_tx, x_positions,
                            window_cells=32, f_range=None):
    """Compute local spectra and extract frequencies/linewidths vs position."""
    dt = time_arr[1] - time_arr[0]

    x_centers, freqs, spectra = windowed_local_spectrum(
        m_y_tx, dt, x_positions, CX,
        window_width_cells=window_cells, overlap=0.5)

    # Extract peaks at each position
    freq_map = np.full((len(x_centers), 2), np.nan)
    lw_map = np.full((len(x_centers), 2), np.nan)
    regime_list = []

    for i in range(len(x_centers)):
        peaks = extract_peaks(freqs, spectra[i], n_peaks=2,
                              prominence=0.05)
        for j, p in enumerate(peaks[:2]):
            if p['success']:
                freq_map[i, j] = p['f0']
                lw_map[i, j] = p['fwhm']

        if len(peaks) >= 2 and peaks[0]['success'] and peaks[1]['success']:
            splitting = abs(peaks[1]['f0'] - peaks[0]['f0'])
            avg_lw = (peaks[0]['fwhm'] + peaks[1]['fwhm']) / 2
            regime, C = classify_coupling_regime(splitting, avg_lw)
        else:
            regime, C = 'single', 0.0
        regime_list.append((regime, C))

    return {
        'x_centers': x_centers,
        'freqs': freqs,
        'spectra': spectra,
        'freq_map': freq_map,
        'linewidth_map': lw_map,
        'regimes': regime_list,
    }


def plot_standing_saw_results(data_full, data_mel, x_positions,
                               filename="fig_standing_saw_ep.pdf"):
    """Create the standing SAW EP figure (4 panels)."""
    fig, axes = ps.quad_panel(height_cm=13.0, wspace=0.40, hspace=0.45)

    k = 2 * np.pi / LAMBDA_SAW

    # (a) Local spectrum map (full coupling)
    if data_full['spectra'].size > 0:
        x_um = data_full['x_centers'] * 1e6
        f_ghz = data_full['freqs'] * 1e-9

        # Only positive frequencies up to 2x Kittel
        f_kittel = GAMMA * np.sqrt(B0 * (B0 + MU0 * MSAT)) / (2 * np.pi)
        f_max = min(2 * f_kittel * 1e-9, f_ghz[-1])
        f_mask = (f_ghz > 0) & (f_ghz < f_max)

        S = data_full['spectra'][:, f_mask].T
        if np.max(S) > 0:
            S = S / np.max(S)
        extent = [x_um[0], x_um[-1], f_ghz[f_mask][0], f_ghz[f_mask][-1]]

        axes[0, 0].imshow(S, aspect='auto', origin='lower', extent=extent,
                          cmap='inferno', vmin=0, vmax=0.5)
        axes[0, 0].set_xlabel(r'$x$ ($\mu$m)')
        axes[0, 0].set_ylabel(r'$f$ (GHz)')
        axes[0, 0].set_title('MEL + MR + Barnett', fontsize=8)
    ps.add_panel_label(axes[0, 0], '(a)')

    # (b) Local spectrum map (MEL only)
    if data_mel['spectra'].size > 0:
        x_um_m = data_mel['x_centers'] * 1e6
        S_m = data_mel['spectra'][:, f_mask].T
        if np.max(S_m) > 0:
            S_m = S_m / np.max(S_m)
        extent_m = [x_um_m[0], x_um_m[-1], f_ghz[f_mask][0], f_ghz[f_mask][-1]]

        axes[0, 1].imshow(S_m, aspect='auto', origin='lower', extent=extent_m,
                          cmap='inferno', vmin=0, vmax=0.5)
        axes[0, 1].set_xlabel(r'$x$ ($\mu$m)')
        axes[0, 1].set_ylabel(r'$f$ (GHz)')
        axes[0, 1].set_title('MEL only', fontsize=8)
    ps.add_panel_label(axes[0, 1], '(b)')

    # (c) Frequency splitting vs position
    x_um = data_full['x_centers'] * 1e6
    freq_map = data_full['freq_map']

    for j in range(2):
        valid = ~np.isnan(freq_map[:, j])
        if np.any(valid):
            axes[1, 0].plot(x_um[valid], freq_map[valid, j] * 1e-9,
                            'o', markersize=3,
                            color=[ps.BLUE, ps.VERMILION][j],
                            label=f'Mode {j+1}')

    # Overlay coupling strength prediction
    x_pred = np.linspace(x_positions[0], x_positions[-1], 200)
    g_mel = np.abs(np.cos(k * x_pred))
    g_mr = np.abs(np.sin(k * x_pred))
    g_total = np.sqrt(g_mel**2 + g_mr**2)

    ax_twin = axes[1, 0].twinx()
    ax_twin.plot(x_pred * 1e6, g_total / np.max(g_total),
                 '-', color='gray', linewidth=0.8, alpha=0.5)
    ax_twin.set_ylabel(r'$g_\mathrm{eff}$ (norm.)', color='gray', fontsize=7)
    ax_twin.set_ylim(0, 1.3)
    ax_twin.tick_params(axis='y', labelcolor='gray', labelsize=6)

    axes[1, 0].set_xlabel(r'$x$ ($\mu$m)')
    axes[1, 0].set_ylabel(r'$f$ (GHz)')
    axes[1, 0].legend(fontsize=6, loc='upper right')
    ps.add_panel_label(axes[1, 0], '(c)')

    # (d) Linewidth and coupling regime vs position
    lw_map = data_full['linewidth_map']

    for j in range(2):
        valid = ~np.isnan(lw_map[:, j])
        if np.any(valid):
            axes[1, 1].plot(x_um[valid], lw_map[valid, j] * 1e-9,
                            'o', markersize=3,
                            color=[ps.BLUE, ps.VERMILION][j],
                            label=f'Mode {j+1}')

    # Color-code background by coupling regime
    regimes = data_full['regimes']
    for i in range(len(x_um) - 1):
        regime = regimes[i][0]
        color = {'strong': ps.TEAL, 'EP-proximal': ps.YELLOW,
                 'weak': ps.PINK, 'single': 'lightgray'}.get(regime, 'white')
        axes[1, 1].axvspan(x_um[i], x_um[min(i+1, len(x_um)-1)],
                           alpha=0.15, color=color, linewidth=0)

    axes[1, 1].set_xlabel(r'$x$ ($\mu$m)')
    axes[1, 1].set_ylabel(r'Linewidth (GHz)')
    axes[1, 1].legend(fontsize=6, loc='upper right')
    ps.add_panel_label(axes[1, 1], '(d)')

    fig.savefig(os.path.join(FIG_DIR, filename.replace(".pdf", ".png")), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


def save_all_data(time_arr, m_y_full, m_z_full, m_y_mel, m_z_mel,
                  data_full, data_mel):
    """Save all raw and processed data."""
    raw_dir = os.path.join(DATA_DIR, "raw", "sim08_standing_ep")
    proc_dir = os.path.join(DATA_DIR, "processed", "maps")

    np.savez(os.path.join(raw_dir, "raw_full_coupling.npz"),
             time=time_arr, m_y=m_y_full, m_z=m_z_full)
    np.savez(os.path.join(raw_dir, "raw_mel_only.npz"),
             time=time_arr, m_y=m_y_mel, m_z=m_z_mel)

    np.savez(os.path.join(proc_dir, "spatial_map_full.npz"),
             x_centers=data_full['x_centers'],
             freqs=data_full['freqs'],
             spectra=data_full['spectra'],
             freq_map=data_full['freq_map'],
             linewidth_map=data_full['linewidth_map'])
    np.savez(os.path.join(proc_dir, "spatial_map_mel.npz"),
             x_centers=data_mel['x_centers'],
             freqs=data_mel['freqs'],
             spectra=data_mel['spectra'],
             freq_map=data_mel['freq_map'],
             linewidth_map=data_mel['linewidth_map'])

    manifest = {
        "simulation": "sim08_standing_saw_ep_search",
        "parameters": {
            "Msat": MSAT, "Aex": AEX, "alpha": ALPHA,
            "B1": B1, "K_mr": K_MR, "eps0": EPS0, "xi": XI,
            "B0": B0, "f_SAW": F_SAW, "lambda_SAW": LAMBDA_SAW,
            "NX": NX, "CX": CX,
            "T_drive": T_DRIVE, "T_ringdown": T_RINGDOWN,
        },
        "raw_files": ["raw_full_coupling.npz", "raw_mel_only.npz"],
        "processed_files": ["spatial_map_full.npz", "spatial_map_mel.npz"],
    }
    with open(os.path.join(raw_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  Raw data saved to {raw_dir}")
    print(f"  Processed data saved to {proc_dir}")


def main():
    print("=== Sim 08: Standing SAW - Spatially resolved EP search ===")
    print(f"  Grid: {NX} x {NY} x {NZ}, CX = {CX*1e9:.1f} nm")
    print(f"  L_total = {NX*CX*1e6:.1f} um = {NX*CX/LAMBDA_SAW:.1f} lambda_SAW")
    print(f"  SAW: f = {F_SAW*1e-9:.1f} GHz, "
          f"lambda = {LAMBDA_SAW*1e6:.1f} um")
    print(f"  K_mr = {K_MR*1e-6:.1f} MJ/m^3")

    # Analytical coupling map
    x_pos = np.arange(NX) * CX
    saw = StandingSAW(F_SAW, LAMBDA_SAW, EPS0, ellipticity=XI,
                      K_mr=K_MR, enable_barnett=True)
    coupling = saw.local_coupling_map(x_pos, B1, MSAT)
    print(f"  g_mel range: {np.min(coupling['g_mel'])*1e-9:.3f} - "
          f"{np.max(coupling['g_mel'])*1e-9:.3f} GHz")
    print(f"  g_mr range:  {np.min(coupling['g_mr'])*1e-9:.3f} - "
          f"{np.max(coupling['g_mr'])*1e-9:.3f} GHz")

    # Run full coupling simulation
    print("\n--- Full coupling (MEL + MR + Barnett) ---")
    time_arr, m_y_full, m_z_full = run_standing_saw(
        "Full coupling", K_mr_val=K_MR, enable_barnett=True)

    # Run MEL-only simulation for comparison
    print("\n--- MEL only ---")
    time_arr_m, m_y_mel, m_z_mel = run_standing_saw(
        "MEL only", K_mr_val=0.0, enable_barnett=False)

    # Analyze local spectra
    print("\n--- Analyzing local spectra ---")
    data_full = analyze_spatial_spectra(time_arr, m_y_full, x_pos,
                                        window_cells=min(64, NX//4))
    data_mel = analyze_spatial_spectra(time_arr_m, m_y_mel, x_pos,
                                       window_cells=min(64, NX//4))

    # Report EP search results
    print("\n--- EP search results ---")
    regimes = data_full['regimes']
    regime_counts = {}
    for r, C in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"  Regime distribution: {regime_counts}")

    ep_found = any(r == 'EP-proximal' for r, _ in regimes)
    if ep_found:
        ep_positions = [data_full['x_centers'][i] for i, (r, _)
                        in enumerate(regimes) if r == 'EP-proximal']
        print(f"  EP-proximal regions found at x = "
              f"{[f'{x*1e6:.1f} um' for x in ep_positions]}")
    else:
        print("  No EP-proximal regime found in this parameter set.")
        print("  This may require different K_mr, eps0, or alpha tuning.")

    # Save all data
    save_all_data(time_arr, m_y_full, m_z_full, m_y_mel, m_z_mel,
                  data_full, data_mel)

    # Plot
    plot_standing_saw_results(data_full, data_mel, x_pos)


if __name__ == "__main__":
    main()
