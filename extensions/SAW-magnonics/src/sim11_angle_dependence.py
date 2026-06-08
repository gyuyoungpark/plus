"""Simulation 11: Field-angle dependence of SAW-magnon coupling.

Key discovery: For m₀ ∥ k_SAW (θ=0°), the magnetoelastic coupling gives ZERO
torque — only magneto-rotation (MR) drives FMR. As θ increases, MEL rapidly
dominates because |B₁| >> K_mr. The crossover angle θ_c is remarkably small:

    sin(θ_c) = K_mr ξ / (4|B₁|)

For YIG: θ_c ≈ 1.1° — even 1° misalignment activates MEL!

Three coupling channels with angle dependence:
    g_mel(θ) = γ |B₁| ε₀ |sin(2θ)| / Ms     [peaks at 45°]
    g_mr(θ)  = γ K_mr ξ ε₀ |cos θ| / (2 Ms)  [peaks at 0°]
    g_B(θ)   = ξ ε₀ ω |cos θ| / 2             [peaks at 0°, always tiny]

Parts:
  A. Analytic coupling rates, cooperativity, crossover analysis
  B. Material comparison (YIG, CoFeB, Ni)
  C. mumax+ simulation validation at selected angles
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

# PRB style
from plot_style import (apply_style, label_panels, axis_label,
                        SINGLE_COL, DOUBLE_COL, CM_TO_INCH,
                        SKY_BLUE, VERMILION, TEAL, YELLOW, PINK, ORANGE, BLUE, BLACK,
                        COLORS_3, COLORS_6, MARKERS)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ===========================================================================
# Physical constants
# ===========================================================================
GAMMA = 1.76e11     # rad/(s·T)
MU0 = 4 * np.pi * 1e-7

# ===========================================================================
# Material parameters: YIG (baseline)
# ===========================================================================
MS_YIG = 140e3       # A/m
ALPHA_YIG = 2e-4     # Gilbert damping (ideal YIG)
B1_YIG = -8.8e6      # J/m³ (magnetoelastic)
KMR_YIG = 1.0e6      # J/m³ (magneto-rotation)

# Material parameters: CoFeB (matched to main-text values)
MS_COFEB = 1.0e6     # A/m
ALPHA_COFEB = 5e-3
B1_COFEB = -8.0e6    # J/m³
KMR_COFEB = 1.1e6    # J/m³  (consistent with main-text Table II)

# Material parameters: Ni
MS_NI = 490e3        # A/m
ALPHA_NI = 0.064     # high damping
B1_NI = -10.0e6      # J/m³
KMR_NI = 0.6e6       # J/m³

# SAW parameters
F_SAW = 3.0e9        # Hz
OMEGA_SAW = 2 * np.pi * F_SAW
LAMBDA_SAW = 1.16e-6 # m
EPS0 = 1e-4          # peak strain
XI = 0.68            # Rayleigh ellipticity

# Acoustic loss
Q_PHONON = 5000      # SAW quality factor

# Simulation
ALPHA_SIM = 5e-4     # slightly higher for numerical stability
T_MAX = 5e-9
DT_REC = 10e-12
NT_REC = int(T_MAX / DT_REC) + 1

# Geometry (single cell)
NX, NY, NZ = 1, 1, 1
CX, CY, CZ = 10e-9, 10e-9, 20e-9


# ===========================================================================
# Part A: Analytic coupling rates
# ===========================================================================

def coupling_rates(theta, B1, Kmr, Ms, eps0=EPS0, xi=XI, omega=OMEGA_SAW):
    """Compute coupling rates for all channels at angle theta.

    Parameters
    ----------
    theta : float or array
        Angle between m₀ and k_SAW (rad).
    B1 : float
        Magnetoelastic constant (J/m³).
    Kmr : float
        Magneto-rotation constant (J/m³).
    Ms : float
        Saturation magnetization (A/m).
    eps0 : float
        Peak SAW strain.
    xi : float
        Rayleigh ellipticity.
    omega : float
        SAW angular frequency (rad/s).

    Returns
    -------
    g_mel, g_mr, g_B : array
        Coupling rates (rad/s).
    """
    g_mel = GAMMA * abs(B1) * eps0 * np.abs(np.sin(2 * theta)) / Ms
    g_mr = GAMMA * abs(Kmr) * xi * eps0 * np.abs(np.cos(theta)) / (2 * Ms)
    g_B = xi * eps0 * omega * np.abs(np.cos(theta)) / 2
    return g_mel, g_mr, g_B


def crossover_angle(B1, Kmr, xi=XI):
    """Crossover angle where g_mel = g_mr.

    sin(θ_c) = K_mr ξ / (4|B₁|)
    """
    ratio = abs(Kmr) * xi / (4 * abs(B1))
    if ratio >= 1:
        return np.pi / 4  # MEL never dominates
    return np.arcsin(ratio)


def cooperativity(g_total, alpha, omega_m, Q_p=Q_PHONON):
    """Cooperativity C = g²/(κ_m κ_p).

    Parameters
    ----------
    g_total : float or array
        Total coupling rate (rad/s).
    alpha : float
        Gilbert damping.
    omega_m : float
        Magnon frequency (rad/s).
    Q_p : float
        Phonon quality factor.
    """
    kappa_m = alpha * omega_m
    kappa_p = omega_m / (2 * Q_p)
    return g_total**2 / (kappa_m * kappa_p)


def ep_damping(g, omega_m, Q_p=Q_PHONON):
    """Critical damping for exceptional point: α_EP where 2g = |κ_m - κ_p|.

    At EP: κ_m = κ_p ± 2g  →  α_EP = (κ_p ± 2g) / ω_m
    """
    kappa_p = omega_m / (2 * Q_p)
    alpha_ep = (kappa_p + 2 * np.abs(g)) / omega_m
    return alpha_ep


def find_resonance_field(f_target, Ms):
    """Kittel resonance field for in-plane magnetized thin film."""
    omega = 2 * np.pi * f_target
    a, b, c = 1.0, MU0 * Ms, -(omega / GAMMA)**2
    disc = b**2 - 4 * a * c
    if np.isscalar(disc):
        if disc < 0:
            return np.nan
    return (-b + np.sqrt(np.maximum(disc, 0))) / 2


# ===========================================================================
# Part A: Generate analytic results
# ===========================================================================
def part_a_analytic():
    """Compute analytic coupling rates for all materials."""
    theta = np.linspace(0, np.pi / 2, 500)
    theta_deg = np.degrees(theta)

    materials = {
        'Benchmark': {'B1': B1_YIG, 'Kmr': KMR_YIG, 'Ms': MS_YIG,
                'alpha': ALPHA_YIG, 'color': BLUE, 'marker': 'o'},
        'CoFeB': {'B1': B1_COFEB, 'Kmr': KMR_COFEB, 'Ms': MS_COFEB,
                  'alpha': ALPHA_COFEB, 'color': ORANGE, 'marker': '^'},
        'Ni': {'B1': B1_NI, 'Kmr': KMR_NI, 'Ms': MS_NI,
               'alpha': ALPHA_NI, 'color': PINK, 'marker': 's'},
    }

    results = {}
    for name, mat in materials.items():
        g_mel, g_mr, g_B = coupling_rates(
            theta, mat['B1'], mat['Kmr'], mat['Ms'])
        g_total = np.sqrt(g_mel**2 + g_mr**2 + g_B**2)
        theta_c = crossover_angle(mat['B1'], mat['Kmr'])
        C = cooperativity(g_total, mat['alpha'], OMEGA_SAW)
        alpha_ep = ep_damping(g_total, OMEGA_SAW)

        results[name] = {
            'theta': theta, 'theta_deg': theta_deg,
            'g_mel': g_mel, 'g_mr': g_mr, 'g_B': g_B,
            'g_total': g_total, 'theta_c': theta_c,
            'C': C, 'alpha_ep': alpha_ep,
            **mat
        }

        print(f"\n{name}:")
        print(f"  |B₁| = {abs(mat['B1'])*1e-6:.1f} MJ/m³, Kmr = {mat['Kmr']*1e-6:.1f} MJ/m³")
        print(f"  g_mr(0°)/(2π) = {g_mr[0]/(2*np.pi)*1e-6:.2f} MHz")
        print(f"  g_mel(45°)/(2π) = {g_mel[len(theta)//2]/(2*np.pi)*1e-6:.1f} MHz")
        print(f"  θ_c = {np.degrees(theta_c):.2f}°")
        print(f"  C(0°) = {cooperativity(g_total[0], mat['alpha'], OMEGA_SAW):.1f}")
        print(f"  C(45°) = {cooperativity(g_total[len(theta)//2], mat['alpha'], OMEGA_SAW):.0f}")

    return results


# ===========================================================================
# Part A: Figures
# ===========================================================================
def fig_coupling_vs_angle(results):
    """Main PRL figure: coupling channels vs angle for YIG."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 11 * CM_TO_INCH))
    fig.subplots_adjust(wspace=0.38, hspace=0.42)

    yig = results['Benchmark']
    theta_deg = yig['theta_deg']

    # ---- (a) Coupling rates vs angle — benchmark ----
    ax = axes[0, 0]
    g_scale = 1e-6 / (2 * np.pi)  # rad/s -> MHz
    ax.semilogy(theta_deg, yig['g_mel'] * g_scale, '-', color=VERMILION,
                lw=1.2, label=r'$g_\mathrm{mel}$')
    ax.semilogy(theta_deg, yig['g_mr'] * g_scale, '-', color=SKY_BLUE,
                lw=1.2, label=r'$g_\mathrm{mr}$')
    ax.semilogy(theta_deg, yig['g_B'] * g_scale, '--', color=TEAL,
                lw=0.8, label=r'$g_\mathrm{B}$')
    ax.semilogy(theta_deg, yig['g_total'] * g_scale, '-', color=BLACK,
                lw=1.5, label=r'$g_\mathrm{total}$')
    # Crossover marker
    tc_deg = np.degrees(yig['theta_c'])
    ax.axvline(tc_deg, ls=':', color='gray', lw=0.6)
    ax.annotate(f'$\\theta_c$={tc_deg:.1f}°', xy=(tc_deg, 200),
                fontsize=7, color='gray', ha='left',
                xytext=(tc_deg + 2, 200))
    ax.set_xlabel(axis_label(r'$\theta$', 'deg'))
    ax.set_ylabel(axis_label(r'$g/(2\pi)$', 'MHz'))
    ax.set_xlim(0, 90)
    ax.set_ylim(1e-2, 1e3)
    ax.legend(fontsize=6.5, ncol=2, loc='center right',
              bbox_to_anchor=(1.0, 0.25))

    # ---- (b) Zoom near θ=0° ----
    ax = axes[0, 1]
    theta_zoom = yig['theta_deg']
    mask = theta_deg <= 10
    ax.plot(theta_deg[mask], yig['g_mel'][mask] * g_scale, '-',
            color=VERMILION, lw=1.2, label=r'$g_\mathrm{mel}$')
    ax.plot(theta_deg[mask], yig['g_mr'][mask] * g_scale, '-',
            color=SKY_BLUE, lw=1.2, label=r'$g_\mathrm{mr}$')
    ax.plot(theta_deg[mask], yig['g_total'][mask] * g_scale, '-',
            color=BLACK, lw=1.5, label=r'$g_\mathrm{total}$')
    ax.axvline(tc_deg, ls=':', color='gray', lw=0.6)
    # Shade MR-dominated region
    ax.axvspan(0, tc_deg, alpha=0.08, color=SKY_BLUE)
    ax.text(tc_deg / 2, yig['g_mr'][0] * g_scale * 1.5, 'MR\ndom.',
            fontsize=6, color=SKY_BLUE, ha='center', va='bottom')
    ax.text(tc_deg + 1.5, yig['g_mr'][0] * g_scale * 0.5, 'MEL dom.',
            fontsize=6, color=VERMILION, ha='left', va='top')
    ax.set_xlabel(axis_label(r'$\theta$', 'deg'))
    ax.set_ylabel(axis_label(r'$g/(2\pi)$', 'MHz'))
    ax.set_xlim(0, 10)
    ax.legend(fontsize=6.5)

    # ---- (c) Cooperativity vs angle — YIG ----
    ax = axes[1, 0]
    C_yig = cooperativity(yig['g_total'], ALPHA_YIG, OMEGA_SAW)
    # Clip to avoid log(0) near θ=90°
    C_yig_clip = np.maximum(C_yig, 1e-2)
    ax.semilogy(theta_deg, C_yig_clip, '-', color=BLACK, lw=1.2)
    ax.axhline(1, ls='--', color='gray', lw=0.6, label='$C = 1$')
    ax.axvline(tc_deg, ls=':', color='gray', lw=0.6)
    # Mark C at key angles — position labels to avoid overlap with curve
    c_label_offsets = {0: (6, -14), 10: (-8, 8), 45: (6, -14)}
    for th_mark in [0, 10, 45]:
        idx = np.argmin(np.abs(theta_deg - th_mark))
        C_val = C_yig[idx]
        ax.plot(th_mark, C_val, 'o', color=ORANGE, ms=4, zorder=5)
        ox, oy = c_label_offsets[th_mark]
        ax.annotate(f'$C$={C_val:.0f}', xy=(th_mark, C_val), fontsize=6,
                    xytext=(ox, oy), textcoords='offset points')
    ax.set_xlabel(axis_label(r'$\theta$', 'deg'))
    ax.set_ylabel(r'Cooperativity $C$')
    ax.set_xlim(0, 85)
    ax.set_ylim(1e-1, 1e6)
    # Place C=1 label below the dashed line, left-aligned
    ax.text(60, 0.5, '$C = 1$', fontsize=6, color='gray', ha='left', va='top')

    # ---- (d) Material comparison: crossover angle vs Kmr/|B1| ----
    ax = axes[1, 1]
    ratios = np.linspace(0.001, 0.65, 300)
    # theta_c = arcsin(ratio * xi / 4)
    tc_vals = np.degrees(np.arcsin(np.minimum(ratios * XI / 4, 1)))
    ax.plot(ratios, tc_vals, '-', color=BLACK, lw=1.2)
    # Mark materials with labels near the points
    for name, mat in results.items():
        r = abs(mat['Kmr']) / abs(mat['B1'])
        tc = np.degrees(crossover_angle(mat['B1'], mat['Kmr']))
        ax.plot(r, tc, mat['marker'], color=mat['color'], ms=7, zorder=5)
    # Manually position material labels near points
    mat_label_pos = {
        'Benchmark': (0.14, 1.1, (8, -8)),
        'CoFeB': (0.568, 5.5, (-45, -5)),
        'Ni': (0.06, 0.6, (8, -8)),
    }
    for name, mat in results.items():
        r = abs(mat['Kmr']) / abs(mat['B1'])
        tc = np.degrees(crossover_angle(mat['B1'], mat['Kmr']))
        _, _, (ox, oy) = mat_label_pos[name]
        ax.annotate(f'{name} ({tc:.1f}°)', xy=(r, tc), fontsize=6,
                    color=mat['color'], xytext=(ox, oy),
                    textcoords='offset points', ha='left', va='bottom')
    ax.set_xlabel(axis_label(r'$K_\mathrm{mr}/|B_1|$'))
    ax.set_ylabel(axis_label(r'$\theta_c$', 'deg'))
    ax.set_xlim(0, 0.65)
    label_panels(axes)
    fname = os.path.join(FIG_DIR, "fig_angle_coupling")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)
    print(f"\nSaved: fig_angle_coupling.pdf")


def fig_material_comparison(results):
    """Coupling rates vs angle for all three materials (vertical stack)."""
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COL, 14 * CM_TO_INCH))
    fig.subplots_adjust(hspace=0.40)

    g_scale = 1e-6 / (2 * np.pi)

    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        theta_deg = r['theta_deg']
        ax.semilogy(theta_deg, r['g_mel'] * g_scale, '-',
                    color=VERMILION, lw=1.0, label=r'$g_\mathrm{mel}$')
        ax.semilogy(theta_deg, r['g_mr'] * g_scale, '-',
                    color=SKY_BLUE, lw=1.0, label=r'$g_\mathrm{mr}$')
        ax.semilogy(theta_deg, r['g_total'] * g_scale, '-',
                    color=BLACK, lw=1.2, label=r'$g_\mathrm{total}$')
        tc = np.degrees(r['theta_c'])
        ax.axvline(tc, ls=':', color='gray', lw=0.6)
        if idx == 2:
            ax.set_xlabel(axis_label(r'$\theta$', 'deg'))
        ax.set_ylabel(axis_label(r'$g/(2\pi)$', 'MHz'))
        ax.set_xlim(0, 90)
        ax.set_ylim(1e-3, 2e3)
        ax.set_title(f'{name} ($\\theta_c$={tc:.1f}°)', fontsize=8)
        if idx == 0:
            ax.legend(fontsize=6, ncol=1, loc='lower right')

    label_panels(axes)
    fname = os.path.join(FIG_DIR, "fig_angle_materials")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)
    print(f"Saved: fig_angle_materials.pdf")


# ===========================================================================
# Part C: mumax+ simulation validation
# ===========================================================================
def run_oblique_saw(theta_deg, B0, direction=+1, enable_mel=True,
                    K_mr=KMR_YIG, eps0=EPS0, alpha=ALPHA_SIM, label=""):
    """Run prescribed SAW at oblique angle θ.

    The magnetization equilibrium is m₀ = (cos θ, sin θ, 0).
    MEL uses the full CUDA kernel (correct for any m).
    MR field is manually scaled by cos θ.
    """
    from mumaxplus import World, Grid, Ferromagnet
    from saw_chiral import ChiralSurfaceAcousticWave

    t_start = time.time()
    theta = np.radians(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    dir_str = "+" if direction > 0 else "-"
    print(f"  [{label}] θ={theta_deg:.1f}°, {dir_str}k, B0={B0*1e3:.1f}mT...",
          end="", flush=True)

    cellsize = (CX, CY, CZ)
    grid = Grid((NX, NY, NZ))
    world = World(cellsize)
    magnet = Ferromagnet(world, grid)

    magnet.msat = MS_YIG
    magnet.aex = 3.65e-12
    magnet.alpha = alpha
    magnet.magnetization = (ct, st, 0)
    magnet.bias_magnetic_field = (B0 * ct, B0 * st, 0)

    if enable_mel:
        magnet.B1 = B1_YIG

    # Apply prescribed SAW - CUDA kernel handles angle dependence automatically
    # Sign of K_mr encodes propagation direction (+k vs -k)
    K_mr_dir = K_mr if direction > 0 else -K_mr
    saw = ChiralSurfaceAcousticWave(
        frequency=F_SAW, wavelength=LAMBDA_SAW, amplitude=eps0,
        direction='x', phase=0.0, ellipticity=XI,
        K_mr=K_mr_dir, enable_barnett=False)
    saw.apply(magnet, Msat=MS_YIG, enable_mel=enable_mel)

    # Time evolution
    world.timesolver.adaptive_timestep = True

    times = np.zeros(NT_REC)
    m_y = np.zeros(NT_REC)
    m_z = np.zeros(NT_REC)

    for i in range(NT_REC):
        m = magnet.magnetization.eval()
        m_y[i] = m[1, 0, 0, 0]
        m_z[i] = m[2, 0, 0, 0]
        times[i] = world.timesolver.time
        if i < NT_REC - 1:
            world.timesolver.run(DT_REC)

    elapsed = time.time() - t_start

    # Transverse magnetization: perpendicular to m₀
    # m_perp_inplane = -sin θ × m_x + cos θ × m_y (in-plane transverse)
    # m_perp_oop = m_z (out-of-plane transverse)
    # For simplicity, use m_z and the transverse in-plane from m_y
    # Note: for θ=0, m_perp = sqrt(m_y² + m_z²)
    # For general θ: m_perp ~ sqrt((m_y - sinθ)² + m_z²)
    #              ≈ sqrt(δm_y² + m_z²) where δm_y = m_y - sinθ
    dm_y = m_y - st
    m_perp = np.sqrt(dm_y**2 + m_z**2)
    peak = np.max(m_perp)

    print(f" peak|m_perp|={peak:.2e}, {elapsed:.1f}s")
    return {
        'times': times, 'm_y': m_y, 'm_z': m_z,
        'm_perp': m_perp, 'peak': peak,
        'theta_deg': theta_deg, 'direction': direction,
    }


def part_c_simulation():
    """Run mumax+ simulations at selected angles."""
    print("\n=== Part C: mumax+ validation ===")

    B0_res = find_resonance_field(F_SAW, MS_YIG)
    print(f"  B0_res = {B0_res*1e3:.2f} mT")

    cache = os.path.join(DATA_DIR, "sim11_angle_sweep.npz")
    if os.path.isfile(cache):
        print("  Loading cached...")
        d = dict(np.load(cache))
        return d

    angles = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0])
    peaks = np.zeros(len(angles))

    for i, theta_deg in enumerate(angles):
        r = run_oblique_saw(theta_deg, B0_res, direction=+1,
                            label=f"θ={theta_deg:.1f}")
        peaks[i] = r['peak']

    np.savez(cache, angles=angles, peaks=peaks)
    return {'angles': angles, 'peaks': peaks}


def fig_simulation_validation(sim_data, analytic_results):
    """Compare simulation with analytic predictions (vertical stack)."""
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL, 10 * CM_TO_INCH))
    fig.subplots_adjust(hspace=0.40)

    angles = sim_data['angles']
    peaks_sim = sim_data['peaks']

    # Analytic prediction: peak |m_perp| ∝ g_total
    yig = analytic_results['Benchmark']
    g_mel, g_mr, g_B = coupling_rates(
        np.radians(angles), B1_YIG, KMR_YIG, MS_YIG)
    g_total_points = np.sqrt(g_mel**2 + g_mr**2 + g_B**2)

    # Normalize: at θ=0, peak_sim = peaks_sim[0], g_total = g_mr(0)
    # The proportionality constant: peak = k × g_total
    # Use θ=0 as reference
    if peaks_sim[0] > 0 and g_total_points[0] > 0:
        k_norm = peaks_sim[0] / g_total_points[0]
    else:
        k_norm = 1.0

    # Continuous analytic curve
    theta_fine = np.linspace(0, 50, 300)
    g_mel_f, g_mr_f, g_B_f = coupling_rates(
        np.radians(theta_fine), B1_YIG, KMR_YIG, MS_YIG)
    g_total_fine = np.sqrt(g_mel_f**2 + g_mr_f**2 + g_B_f**2)

    # ---- (a) Peak |m_perp| vs angle ----
    ax = axes[0]
    ax.plot(theta_fine, g_total_fine * k_norm, '-', color=BLACK, lw=1.0,
            label='Analytic $g_\\mathrm{total}$', zorder=1)
    ax.plot(theta_fine, g_mr_f * k_norm, '--', color=SKY_BLUE, lw=0.8,
            label='MR only', zorder=1)
    ax.plot(theta_fine, g_mel_f * k_norm, '--', color=VERMILION, lw=0.8,
            label='MEL only', zorder=1)
    ax.plot(angles, peaks_sim, 'o', color=ORANGE, ms=5, zorder=5,
            label='mumax+', markeredgecolor='k', markeredgewidth=0.3)
    tc = np.degrees(crossover_angle(B1_YIG, KMR_YIG))
    ax.axvline(tc, ls=':', color='gray', lw=0.6)
    ax.set_xlabel(axis_label(r'$\theta$', 'deg'))
    ax.set_ylabel(axis_label(r'Peak $|m_\perp|$'))
    ax.set_xlim(0, 50)
    ax.legend(fontsize=6, loc='upper left')

    # ---- (b) Ratio sim/theory ----
    ax = axes[1]
    predicted = g_total_points * k_norm
    mask = predicted > 0
    ratio = np.ones_like(peaks_sim)
    ratio[mask] = peaks_sim[mask] / predicted[mask]
    ax.plot(angles[mask], ratio[mask], 'o-', color=ORANGE, ms=5, lw=1.0,
            markeredgecolor='k', markeredgewidth=0.3)
    ax.axhline(1, ls='--', color='gray', lw=0.6)
    ax.set_xlabel(axis_label(r'$\theta$', 'deg'))
    ax.set_ylabel(r'Simulation / Theory')
    ax.set_xlim(0, 50)
    ax.set_ylim(0.5, 1.5)

    label_panels(axes)
    fname = os.path.join(FIG_DIR, "fig_angle_validation")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)
    print(f"Saved: fig_angle_validation.pdf")


# ===========================================================================
# Summary table
# ===========================================================================
def print_summary_table(results):
    """Print a summary table of key results."""
    print("\n" + "=" * 72)
    print("Summary: Angle-dependent coupling rates")
    print("=" * 72)
    print(f"{'Material':<8} {'|B₁|':>8} {'Kmr':>8} {'g_mr(0°)':>10} "
          f"{'g_mel(45°)':>12} {'θ_c':>6} {'C(0°)':>8} {'C(45°)':>10}")
    print(f"{'':8} {'MJ/m³':>8} {'MJ/m³':>8} {'MHz':>10} "
          f"{'MHz':>12} {'deg':>6} {'':>8} {'':>10}")
    print("-" * 72)

    g_scale = 1e-6 / (2 * np.pi)
    for name, r in results.items():
        g_mel_45 = r['g_mel'][len(r['theta']) // 2]
        g_mr_0 = r['g_mr'][0]
        tc = np.degrees(r['theta_c'])
        C0 = cooperativity(r['g_total'][0], r['alpha'], OMEGA_SAW)
        C45 = cooperativity(r['g_total'][len(r['theta']) // 2],
                            r['alpha'], OMEGA_SAW)
        print(f"{name:<8} {abs(r['B1'])*1e-6:>8.1f} {r['Kmr']*1e-6:>8.1f} "
              f"{g_mr_0*g_scale:>10.2f} {g_mel_45*g_scale:>12.1f} "
              f"{tc:>6.2f} {C0:>8.1f} {C45:>10.0f}")
    print("=" * 72)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 72)
    print("Sim 11: Field-angle dependence of SAW-magnon coupling")
    print("=" * 72)

    # Part A: Analytic
    print("\n--- Part A: Analytic coupling rates ---")
    results = part_a_analytic()
    print_summary_table(results)

    # Figures
    fig_coupling_vs_angle(results)
    fig_material_comparison(results)

    # Part C: mumax+ validation
    print("\n--- Part C: mumax+ simulation validation ---")
    sim_data = part_c_simulation()

    # Validation figure
    fig_simulation_validation(sim_data, results)

    print("\n" + "=" * 72)
    print("Sim 11 complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
