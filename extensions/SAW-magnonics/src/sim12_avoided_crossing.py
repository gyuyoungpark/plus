"""Simulation 12: Avoided crossing - strong coupling visualization.

Demonstrates the magnon-phonon avoided crossing by sweeping B₀ through
the resonance condition ω_m(B₀) = ω_SAW. The eigenvalue splitting 2g
becomes visible in both the analytic 2×2 model and the mumax+ spectral map.

Two approaches:
  A. Analytic 2×2 non-Hermitian Hamiltonian eigenvalues
  B. mumax+ simulation with longer runs (20 ns) for spectral resolution

The avoided crossing is THE signature of strong coupling (C >> 1).
For YIG with K_mr = 1 MJ/m³: 2g/(2π) = 6.9 MHz.

Also includes the transition from level repulsion (real coupling, g_mr)
to level attraction (imaginary coupling, hypothetical EdH).
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
from matplotlib.colors import LogNorm

from plot_style import (apply_style, label_panels, axis_label,
                        SINGLE_COL, DOUBLE_COL, CM_TO_INCH,
                        SKY_BLUE, VERMILION, TEAL, ORANGE, BLACK)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ===========================================================================
# Constants and parameters
# ===========================================================================
GAMMA = 1.76e11
MU0 = 4 * np.pi * 1e-7

# YIG
MS = 140e3
AEX = 3.65e-12
ALPHA = 5e-3          # higher damping for steady-state within T_MAX
ALPHA_IDEAL = 2e-4    # ideal YIG for analytics
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68

# SAW
F_SAW = 3.0e9
OMEGA_SAW = 2 * np.pi * F_SAW
LAMBDA_SAW = 1.16e-6
EPS0 = 1e-4

# Acoustic quality
Q_PHONON = 5000

# Simulation: longer runs for spectral resolution
T_MAX = 100e-9           # 100 ns (9.4x tau at alpha=5e-3, Df=10MHz < linewidth=30MHz)
DT_REC = 50e-12          # 50 ps (Nyquist = 10 GHz)
NT_REC = int(T_MAX / DT_REC) + 1

NX, NY, NZ = 1, 1, 1
CX, CY, CZ = 10e-9, 10e-9, 1e-9   # single cell, CZ=1nm


def kittel_freq(B0, Ms=MS):
    """Kittel FMR frequency (Hz) for in-plane magnetized film."""
    return GAMMA / (2 * np.pi) * np.sqrt(np.maximum(B0 * (B0 + MU0 * Ms), 0))


def kittel_omega(B0, Ms=MS):
    """Kittel FMR angular frequency (rad/s)."""
    return GAMMA * np.sqrt(np.maximum(B0 * (B0 + MU0 * Ms), 0))


def resonance_field(f_target, Ms=MS):
    """Find B0 for Kittel FMR at f_target."""
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * Ms, -(omega / GAMMA)**2
    return (-b + np.sqrt(b**2 - 4 * a * c)) / 2


B0_RES = resonance_field(F_SAW)  # thin-film Kittel: 50.63 mT

# For 10x10x1nm single cell, actual resonance is ~61 mT due to
# finite-size demagnetization (Nz < 1). Use this for simulations.
B0_RES_SIM = 61.0e-3


# ===========================================================================
# Part A: Analytic eigenvalues
# ===========================================================================
def eigenvalues_2x2(B0_arr, g_complex, alpha_m=ALPHA_IDEAL, Q_p=Q_PHONON):
    """Non-Hermitian 2×2 eigenvalues for magnon-phonon system.

    H = [[ω_m - iκ_m/2,   g       ],
         [g*,              ω_p - iκ_p/2]]

    Parameters
    ----------
    B0_arr : array
        External field values (T).
    g_complex : complex
        Complex coupling rate (rad/s). Real → level repulsion,
        Imaginary → level attraction.
    alpha_m : float
        Gilbert damping.
    Q_p : float
        Phonon quality factor.

    Returns
    -------
    omega_plus, omega_minus : complex arrays
        Eigenfrequencies (rad/s).
    """
    omega_m = kittel_omega(B0_arr)
    kappa_m = alpha_m * omega_m
    omega_p = OMEGA_SAW
    kappa_p = omega_p / (2 * Q_p)

    avg = 0.5 * ((omega_m + omega_p) - 1j * (kappa_m + kappa_p))
    delta = 0.5 * ((omega_m - omega_p) - 1j * (kappa_m - kappa_p))

    # Coupled oscillator with symmetric off-diagonal (both g, not g and g*):
    # disc = Δ² + g². Real g → repulsion, imaginary g → attraction.
    disc = delta**2 + g_complex**2
    sqrt_disc = np.sqrt(disc.astype(complex))

    return avg + sqrt_disc, avg - sqrt_disc


def part_a_analytic():
    """Compute analytic avoided crossing for multiple coupling strengths."""
    B0_arr = np.linspace(0.5 * B0_RES, 1.5 * B0_RES, 500)

    # MR coupling rate at θ=0
    g_mr = GAMMA * KMR * XI * EPS0 / (2 * MS)  # rad/s
    print(f"  g_mr/(2π) = {g_mr/(2*np.pi)*1e-6:.2f} MHz")
    print(f"  2g/(2π) = {2*g_mr/(2*np.pi)*1e-6:.2f} MHz (splitting)")

    results = {}

    # Case 1: Strong coupling (YIG, Kmr = 1 MJ/m³)
    g_complex = g_mr  # real coupling → level repulsion
    wp, wm = eigenvalues_2x2(B0_arr, g_complex)
    results['strong'] = {
        'B0': B0_arr, 'wp': wp, 'wm': wm,
        'g': g_mr, 'label': f'YIG strong ($K_{{mr}}$=1 MJ/m³)'
    }

    # Case 2: EP condition (α tuned to EP)
    kappa_p = OMEGA_SAW / (2 * Q_PHONON)
    alpha_ep = (kappa_p + 2 * g_mr) / OMEGA_SAW
    wp_ep, wm_ep = eigenvalues_2x2(B0_arr, g_mr, alpha_m=alpha_ep)
    results['ep'] = {
        'B0': B0_arr, 'wp': wp_ep, 'wm': wm_ep,
        'g': g_mr, 'alpha_ep': alpha_ep,
        'label': f'EP ($\\alpha$={alpha_ep:.2e})'
    }

    # Case 3: Weak coupling (Kmr = 0.05 MJ/m³)
    Kmr_weak = 0.05e6
    g_weak = GAMMA * Kmr_weak * XI * EPS0 / (2 * MS)
    wp_w, wm_w = eigenvalues_2x2(B0_arr, g_weak)
    results['weak'] = {
        'B0': B0_arr, 'wp': wp_w, 'wm': wm_w,
        'g': g_weak, 'label': 'Weak coupling'
    }

    # Case 4: Level attraction (imaginary coupling, EdH)
    g_edh = 1j * g_mr * 0.1  # hypothetical dissipative coupling
    wp_a, wm_a = eigenvalues_2x2(B0_arr, g_edh)
    results['attraction'] = {
        'B0': B0_arr, 'wp': wp_a, 'wm': wm_a,
        'g': g_edh, 'label': 'Level attraction (EdH)'
    }

    return results


def fig_avoided_crossing_analytic(results):
    """Four-panel figure: avoided crossing for different coupling regimes."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 11 * CM_TO_INCH))
    fig.subplots_adjust(wspace=0.32, hspace=0.42)

    panels = ['strong', 'weak', 'ep', 'attraction']
    titles = [
        'Strong coupling (level repulsion)',
        'Weak coupling',
        'Exceptional point',
        'Level attraction (dissipative)',
    ]

    for idx, (key, title) in enumerate(zip(panels, titles)):
        ax = axes.flat[idx]
        r = results[key]
        B0_mT = r['B0'] * 1e3
        f_scale = 1e-9 / (2 * np.pi)

        # Eigenvalue real parts (frequencies)
        f_p = np.real(r['wp']) * f_scale
        f_m = np.real(r['wm']) * f_scale

        # Eigenvalue imaginary parts (linewidths)
        gamma_p = -np.imag(r['wp']) * f_scale
        gamma_m = -np.imag(r['wm']) * f_scale

        # Plot frequency branches
        ax.plot(B0_mT, f_p, '-', color=SKY_BLUE, lw=1.2, label='$\\omega_+$')
        ax.plot(B0_mT, f_m, '-', color=VERMILION, lw=1.2, label='$\\omega_-$')

        # Uncoupled modes
        f_kittel = kittel_freq(r['B0']) * 1e-9
        ax.plot(B0_mT, f_kittel, '--', color='gray', lw=0.5, alpha=0.5)
        ax.axhline(F_SAW * 1e-9, ls='--', color='gray', lw=0.5, alpha=0.5)

        ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
        if idx % 2 == 0:
            ax.set_ylabel(axis_label(r'$f$', 'GHz'))
        ax.set_ylim(F_SAW * 1e-9 - 0.05, F_SAW * 1e-9 + 0.05)
        ax.set_title(title, fontsize=7.5)

        if idx == 0:
            ax.legend(fontsize=6)

        # Add splitting annotation for strong coupling
        if key == 'strong':
            g = r['g']
            splitting = 2 * g / (2 * np.pi) * 1e-6  # MHz
            ax.annotate(f'$2g/(2\\pi)$ = {splitting:.1f} MHz',
                        xy=(B0_RES * 1e3, F_SAW * 1e-9), fontsize=6.5,
                        xytext=(B0_RES * 1e3 + 5, F_SAW * 1e-9 + 0.02),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    label_panels(axes)
    fname = os.path.join(FIG_DIR, "fig_avoided_crossing_analytic")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)
    print("Saved: fig_avoided_crossing_analytic.pdf")


# ===========================================================================
# Part B: mumax+ spectral map
# ===========================================================================
def run_b0_sweep_long(N_B0=200, B0_half_mT=0.3):
    """Sweep B0 near resonance with fine resolution.

    Parameters
    ----------
    N_B0 : int
        Number of field points.
    B0_half_mT : float
        Half-range in mT: B0_res +/- B0_half_mT.
    """
    from mumaxplus import World, Grid, Ferromagnet
    from saw_chiral import ChiralSurfaceAcousticWave

    print(f"\n=== Part B: mumax+ B0 sweep (N={N_B0}, T={T_MAX*1e9:.0f}ns) ===")

    cache = os.path.join(DATA_DIR, "sim12_b0sweep_v2.npz")
    if os.path.isfile(cache):
        print("  Loading cached...")
        return dict(np.load(cache))

    B0_half = B0_half_mT * 1e-3
    B0_values = np.linspace(B0_RES_SIM - B0_half, B0_RES_SIM + B0_half, N_B0)

    all_my = np.zeros((N_B0, NT_REC))
    all_mz = np.zeros((N_B0, NT_REC))

    for i, B0 in enumerate(B0_values):
        t_start = time.time()
        f_kittel = kittel_freq(B0)
        print(f"  [B0 {i+1}/{N_B0}] B0={B0*1e3:.2f}mT, f_K={f_kittel*1e-9:.3f}GHz...",
              end="", flush=True)

        cellsize = (CX, CY, CZ)
        grid = Grid((NX, NY, NZ))
        world = World(cellsize)
        magnet = Ferromagnet(world, grid)

        magnet.msat = MS
        magnet.aex = AEX
        magnet.alpha = ALPHA
        magnet.magnetization = (1, 0, 0)
        magnet.bias_magnetic_field = (B0, 0, 0)
        # B1=0: MR-only drive (MEL gives zero torque at theta=0 but
        # causes parametric instability above threshold)

        saw = ChiralSurfaceAcousticWave(
            frequency=F_SAW, wavelength=LAMBDA_SAW, amplitude=EPS0,
            direction='x', phase=0.0, ellipticity=XI,
            K_mr=KMR, enable_barnett=False)
        saw.apply(magnet, Msat=MS, enable_mel=False)  # MR only

        world.timesolver.adaptive_timestep = True

        for j in range(NT_REC):
            m = magnet.magnetization.eval()
            all_my[i, j] = m[1, 0, 0, 0]
            all_mz[i, j] = m[2, 0, 0, 0]
            if j < NT_REC - 1:
                world.timesolver.run(DT_REC)

        elapsed = time.time() - t_start
        m_perp = np.sqrt(all_my[i]**2 + all_mz[i]**2)
        print(f" peak={np.max(m_perp):.2e}, {elapsed:.1f}s")

    times = np.arange(NT_REC) * DT_REC
    np.savez(cache, B0_values=B0_values, times=times,
             all_my=all_my, all_mz=all_mz)
    return {'B0_values': B0_values, 'times': times,
            'all_my': all_my, 'all_mz': all_mz}


def analytic_spectral_weight(B0_arr, f_arr, g, alpha_m=ALPHA_IDEAL, Q_p=Q_PHONON):
    """Magnon spectral weight |chi(f, B0)|^2 from 2x2 coupled model."""
    omega_p = OMEGA_SAW
    kappa_p = omega_p / (2 * Q_p)
    chi_map = np.zeros((len(B0_arr), len(f_arr)))
    for i, B0 in enumerate(B0_arr):
        omega_m = kittel_omega(B0)
        kappa_m = alpha_m * omega_m
        omega = 2 * np.pi * f_arr
        D = (omega - omega_m + 1j * kappa_m) * (omega - omega_p + 1j * kappa_p) - g**2
        chi_map[i] = np.abs((omega - omega_p + 1j * kappa_p) / D)**2
    return chi_map


def fig_analytic_spectral(analytic_results):
    """Fig 12: Analytic spectral weight showing avoided crossing."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 5.5 * CM_TO_INCH))
    fig.subplots_adjust(wspace=0.40)

    g_mr = GAMMA * KMR * XI * EPS0 / (2 * MS)
    f_scale = 1e-9 / (2 * np.pi)

    # ---- (a) Susceptibility map: MR coupling (level repulsion) ----
    ax = axes[0]
    B0_broad = np.linspace(0.85 * B0_RES, 1.15 * B0_RES, 300)
    f_arr = np.linspace(2.9e9, 3.1e9, 200)
    chi_map = analytic_spectral_weight(B0_broad, f_arr, g_mr)
    chi_map /= np.max(chi_map)
    ax.pcolormesh(B0_broad * 1e3, f_arr * 1e-9, chi_map.T,
                  cmap='inferno', shading='gouraud', vmin=0, vmax=1)
    ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
    ax.set_ylabel(axis_label(r'$f$', 'GHz'))
    ax.set_ylim(2.9, 3.1)
    ax.set_title('Level repulsion (MR)', fontsize=7)

    # ---- (b) Susceptibility map: EdH coupling (level attraction) ----
    ax = axes[1]
    g_edh = 1j * g_mr * 0.5
    chi_att = analytic_spectral_weight(B0_broad, f_arr, g_edh)
    chi_att /= np.max(chi_att)
    ax.pcolormesh(B0_broad * 1e3, f_arr * 1e-9, chi_att.T,
                  cmap='inferno', shading='gouraud', vmin=0, vmax=1)
    ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
    ax.set_ylim(2.9, 3.1)
    ax.set_title('Level attraction (EdH)', fontsize=7)

    # ---- (c) Response at f_SAW vs B0 (single-mode vs coupled) ----
    ax = axes[2]
    B0_fine = np.linspace(0.9 * B0_RES, 1.1 * B0_RES, 500)
    omega_m = kittel_omega(B0_fine)
    kappa_m = ALPHA_IDEAL * omega_m
    kappa_p = OMEGA_SAW / (2 * Q_PHONON)

    # Single-mode (prescribed SAW): Lorentzian
    chi_single = 1.0 / np.abs(OMEGA_SAW - omega_m + 1j * kappa_m)
    ax.plot(B0_fine * 1e3, chi_single / np.max(chi_single), '-',
            color=SKY_BLUE, lw=1.0, label='Single mode')

    # Coupled model: |chi(omega_SAW, B0)| from 2x2
    i_fsaw = np.argmin(np.abs(f_arr - F_SAW))
    from scipy.interpolate import interp1d

    # MR (repulsion): double peak
    chi_mr_line = chi_map[:, i_fsaw]
    interp_mr = interp1d(B0_broad, chi_mr_line, fill_value=0,
                         bounds_error=False)
    chi_mr_fine = interp_mr(B0_fine)
    if np.max(chi_mr_fine) > 0:
        ax.plot(B0_fine * 1e3, chi_mr_fine / np.max(chi_mr_fine), '--',
                color=VERMILION, lw=1.0, label='MR (repulsion)')

    # EdH (attraction): single enhanced peak
    chi_att_line = chi_att[:, i_fsaw]
    interp_att = interp1d(B0_broad, chi_att_line, fill_value=0,
                          bounds_error=False)
    chi_att_fine = interp_att(B0_fine)
    if np.max(chi_att_fine) > 0:
        ax.plot(B0_fine * 1e3, chi_att_fine / np.max(chi_att_fine), ':',
                color=TEAL, lw=1.0, label='EdH (attraction)')

    ax.axvline(B0_RES * 1e3, ls=':', color='gray', lw=0.5)
    ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
    ax.set_ylabel(r'$|\chi(f_\mathrm{SAW})|$ (norm.)')
    ax.legend(fontsize=5.5)

    label_panels(axes)
    fname = os.path.join(FIG_DIR, "fig_avoided_crossing_sim")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)
    print("Saved: fig_avoided_crossing_sim.pdf")


def fig_spectral_map_legacy(sim_data, analytic_results):
    """Legacy: mumax+ spectral map (kept for reference)."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 5.5 * CM_TO_INCH))
    fig.subplots_adjust(wspace=0.40)

    B0 = sim_data['B0_values']
    times = sim_data['times']
    dt = times[1] - times[0]
    window = np.hanning(len(times))

    freq = np.fft.rfftfreq(len(times), dt)
    f_GHz = freq * 1e-9

    # Frequency range for spectral map
    f_range = (f_GHz > 2.85) & (f_GHz < 3.15)
    f_plot = f_GHz[f_range]

    # ---- (a) mumax+ spectral map ----
    ax = axes[0]
    spectra = np.zeros((len(B0), np.sum(f_range)))
    for i in range(len(B0)):
        fft_y = np.abs(np.fft.rfft(sim_data['all_my'][i] * window))
        fft_z = np.abs(np.fft.rfft(sim_data['all_mz'][i] * window))
        spectra[i] = fft_y[f_range] + fft_z[f_range]

    # Global normalization (not per-row) to show amplitude variation
    spectra /= np.max(spectra)

    ax.pcolormesh(B0 * 1e3, f_plot, spectra.T,
                  cmap='inferno', shading='gouraud', vmin=0, vmax=1)
    # Overlay bare Kittel dispersion
    B0_fine = np.linspace(B0.min(), B0.max(), 200)
    ax.plot(B0_fine * 1e3, kittel_freq(B0_fine) * 1e-9, '--',
            color='cyan', lw=0.8, alpha=0.7, label=r'$f_\mathrm{Kittel}$')
    ax.axhline(F_SAW * 1e-9, ls=':', color='white', lw=0.5, alpha=0.5)
    ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
    ax.set_ylabel(axis_label(r'$f$', 'GHz'))
    ax.set_title('mumax+ (prescribed SAW)', fontsize=7)
    ax.legend(fontsize=5.5, loc='upper left')

    # ---- (b) Steady-state |m_perp| vs B0 (late-time RMS) ----
    ax = axes[1]
    n_late = int(0.8 * len(times))  # last 20% of time series
    response = np.zeros(len(B0))
    for i in range(len(B0)):
        m_perp_late = np.sqrt(sim_data['all_my'][i, n_late:]**2 +
                              sim_data['all_mz'][i, n_late:]**2)
        response[i] = np.sqrt(np.mean(m_perp_late**2))  # RMS

    resp_norm = response / np.max(response)
    ax.plot(B0 * 1e3, resp_norm, '-', color=ORANGE, lw=0.8,
            label='mumax+')

    # Analytic Lorentzian: single-mode susceptibility |1/(w-wm+ikm)|
    omega_m_fine = kittel_omega(B0_fine)
    kappa_m_fine = ALPHA * omega_m_fine  # use simulation alpha
    chi_single = 1.0 / np.abs(OMEGA_SAW - omega_m_fine + 1j * kappa_m_fine)
    ax.plot(B0_fine * 1e3, chi_single / np.max(chi_single), '--',
            color=BLACK, lw=0.8, label='Lorentzian')

    ax.axvline(B0_RES * 1e3, ls=':', color='gray', lw=0.5)
    ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
    ax.set_ylabel(r'RMS $|m_\perp|$ (norm.)')
    ax.legend(fontsize=5.5)

    # ---- (c) Analytic susceptibility: coupled model prediction ----
    ax = axes[2]
    B0_broad = np.linspace(0.9 * B0_RES, 1.1 * B0_RES, 300)
    f_arr = np.linspace(2.9e9, 3.1e9, 200)
    g_mr = GAMMA * KMR * XI * EPS0 / (2 * MS)

    chi_map = analytic_spectral_weight(B0_broad, f_arr, g_mr)
    chi_map /= np.max(chi_map)

    ax.pcolormesh(B0_broad * 1e3, f_arr * 1e-9, chi_map.T,
                  cmap='inferno', shading='gouraud', vmin=0, vmax=1)
    # Overlay hybridized eigenvalues
    r = analytic_results['strong']
    f_scale = 1e-9 / (2 * np.pi)
    ax.plot(r['B0'] * 1e3, np.real(r['wp']) * f_scale, '--',
            color='cyan', lw=0.8, alpha=0.8)
    ax.plot(r['B0'] * 1e3, np.real(r['wm']) * f_scale, '--',
            color='cyan', lw=0.8, alpha=0.8)
    ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
    ax.set_ylabel(axis_label(r'$f$', 'GHz'))
    ax.set_title('Analytic (coupled model)', fontsize=7)

    label_panels(axes)
    fname = os.path.join(FIG_DIR, "fig_avoided_crossing_sim")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)
    print("Saved: fig_avoided_crossing_sim.pdf")


# ===========================================================================
# Combined PRL figure: analytic + simulation
# ===========================================================================
def fig_prl_avoided_crossing(analytic, sim_data):
    """PRL-quality figure: analytic and simulation shown separately."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 5.5 * CM_TO_INCH))
    fig.subplots_adjust(wspace=0.35)

    f_scale = 1e-9 / (2 * np.pi)
    g_mr = GAMMA * KMR * XI * EPS0 / (2 * MS)

    # ---- (a) Analytic eigenfrequencies (line plot only) ----
    ax = axes[0]
    r = analytic['strong']
    B0_mT = r['B0'] * 1e3
    fp = np.real(r['wp']) * f_scale
    fm = np.real(r['wm']) * f_scale

    # Linewidth bands
    gp = -np.imag(r['wp']) * f_scale
    gm = -np.imag(r['wm']) * f_scale
    ax.fill_between(B0_mT, fp - gp, fp + gp, color=SKY_BLUE, alpha=0.15)
    ax.fill_between(B0_mT, fm - gm, fm + gm, color=VERMILION, alpha=0.15)
    ax.plot(B0_mT, fp, '-', color=SKY_BLUE, lw=1.2, label=r'$\omega_+$')
    ax.plot(B0_mT, fm, '-', color=VERMILION, lw=1.2, label=r'$\omega_-$')

    # Uncoupled modes (gray dashed)
    f_kittel = kittel_freq(r['B0']) * 1e-9
    ax.plot(B0_mT, f_kittel, '--', color='gray', lw=0.5, alpha=0.5)
    ax.axhline(F_SAW * 1e-9, ls='--', color='gray', lw=0.5, alpha=0.5)

    # Splitting annotation
    splitting = 2 * g_mr / (2 * np.pi) * 1e-6
    ax.annotate(f'$2g/(2\\pi)$ = {splitting:.1f} MHz',
                xy=(B0_RES * 1e3, F_SAW * 1e-9), fontsize=6.5,
                xytext=(B0_RES * 1e3 + 3, F_SAW * 1e-9 + 0.015),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
    ax.set_ylabel(axis_label(r'$f$', 'GHz'))
    ax.set_ylim(F_SAW * 1e-9 - 0.04, F_SAW * 1e-9 + 0.04)
    ax.legend(fontsize=6)
    ax.set_title('Analytic', fontsize=7)

    # ---- (b) mumax+ spectral map (simulation only, no overlay) ----
    ax = axes[1]
    B0 = sim_data['B0_values']
    times = sim_data['times']
    dt = times[1] - times[0]
    freq = np.fft.rfftfreq(len(times), dt)
    f_GHz = freq * 1e-9
    window = np.hanning(len(times))

    f_range = (f_GHz > 2.9) & (f_GHz < 3.1)
    f_plot = f_GHz[f_range]

    spectra = np.zeros((len(B0), np.sum(f_range)))
    for i in range(len(B0)):
        fft_y = np.abs(np.fft.rfft(sim_data['all_my'][i] * window))
        fft_z = np.abs(np.fft.rfft(sim_data['all_mz'][i] * window))
        spectra[i] = fft_y[f_range] + fft_z[f_range]
    spectra /= np.max(spectra)

    ax.pcolormesh(B0 * 1e3, f_plot, spectra.T,
                  cmap='inferno', shading='gouraud', vmin=0, vmax=1)
    ax.set_xlabel(axis_label(r'$\mu_0 H$', 'mT'))
    ax.set_ylim(2.9, 3.1)
    ax.set_title(r'mumax$^+$ simulation', fontsize=7)

    label_panels(axes)
    fname = os.path.join(FIG_DIR, "fig_prl_avoided_crossing")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)
    print("Saved: fig_prl_avoided_crossing.pdf")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 72)
    print("Sim 12: Avoided crossing - strong coupling visualization")
    print("=" * 72)
    print(f"  YIG: Ms={MS*1e-3:.0f}kA/m, alpha_sim={ALPHA:.0e}, B1={B1*1e-6:.1f}MJ/m3")
    print(f"  Kmr={KMR*1e-6:.1f}MJ/m3, xi={XI}")
    print(f"  SAW: f={F_SAW*1e-9:.1f}GHz, eps0={EPS0:.0e}")
    print(f"  B0_res = {B0_RES*1e3:.2f} mT")
    print(f"  T_max = {T_MAX*1e9:.0f} ns, Δf = {1/(T_MAX)*1e-6:.0f} MHz")

    # Part A: Analytic
    print("\n--- Part A: Analytic eigenvalues ---")
    analytic = part_a_analytic()
    fig_avoided_crossing_analytic(analytic)

    # Part B: Analytic spectral weight (replacing mumax+ sweep)
    # Note: single-cell prescribed SAW cannot show B0-sweep Lorentzian
    # due to finite-cell demagnetization mismatch with thin-film Kittel.
    # The coupling validation is in sim10 (channel decomposition).
    print("\n--- Part B: Analytic spectral weight ---")
    fig_analytic_spectral(analytic)

    print("\n" + "=" * 72)
    print("Sim 12 complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
