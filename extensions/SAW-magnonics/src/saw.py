"""Surface Acoustic Wave (SAW) coupled to magnons via magnetoelastic interaction.

This module provides a Python-level SAW driver for mumax+ that uses the built-in
rigid strain and magnetoelastic field infrastructure.  No C++ or CUDA
modifications are required.

A Rayleigh SAW propagating along x (or y) in a piezoelectric substrate creates
a time- and space-dependent normal strain in the ferromagnetic thin film:

    epsilon_xx(x, t) = epsilon_0 * sin(k x - omega t + phi)

This strain couples to the magnetization through the magnetoelastic energy:

    E_mel = B1 (eps_xx m_x^2 + eps_yy m_y^2 + eps_zz m_z^2)
          + B2 (eps_xy m_x m_y + eps_xz m_x m_z + eps_yz m_y m_z)

producing an effective field

    H_mel,i = -(2/mu0 M_s) [B1 eps_ii m_i + B2 sum_{j!=i} eps_ij m_j]

The SAW strain profile is set analytically via the mumax+ ``add_time_term``
mechanism on ``rigid_norm_strain``, using the product-to-sum identity

    sin(kx - wt) = sin(kx) cos(wt) - cos(kx) sin(wt)

so that two ``add_time_term`` calls (separable time x space) suffice.

Typical usage::

    from saw import SurfaceAcousticWave

    saw = SurfaceAcousticWave(frequency=200e6, wavelength=20e-6,
                              amplitude=6e-3)
    saw.apply(magnet)          # register strain on the magnet
    magnet.B1 = -8.8e6        # magnetoelastic coupling constant
    world.timesolver.solve(time_array, quantity_dict)

Material reference: CoFeB thin film
    Msat  = 0.6 -- 1.2 MA/m
    Aex   = 10 -- 20 pJ/m
    B1    = -8.8 MJ/m^3
    alpha = 0.005 -- 0.05
"""

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GAMMA = 1.76e11            # gyromagnetic ratio  (rad / s / T)
MU0 = 4 * np.pi * 1e-7    # vacuum permeability  (T m / A)


class SurfaceAcousticWave:
    """Rayleigh SAW coupled to a ferromagnetic thin film via magnetoelastic
    interaction.

    This class sets up an analytical, time- and space-dependent strain
    profile on a mumax+ ``Ferromagnet`` using the existing
    ``rigid_norm_strain.add_time_term()`` API.  The magnetoelastic effective
    field is then computed by the built-in CUDA kernel
    ``k_rigidMagnetoelasticField`` at every LLG time step on the GPU.

    Parameters
    ----------
    frequency : float
        SAW frequency (Hz).
    wavelength : float
        SAW wavelength (m).
    amplitude : float
        Peak strain epsilon_0 (dimensionless, typically 1e-4 -- 1e-2).
    direction : str, optional
        Propagation direction: ``'x'`` (default) or ``'y'``.
    phase : float, optional
        Initial phase offset (rad).  Default 0.
    envelope : callable or None, optional
        Time envelope function ``f(t) -> float``.  If None (default), the
        SAW is continuous wave (CW).  For a burst, use e.g.
        ``lambda t: np.exp(-0.5*((t - t0)/sigma)**2)``.
    """

    def __init__(self, frequency, wavelength, amplitude,
                 direction='x', phase=0.0, envelope=None):
        self.frequency = frequency
        self.wavelength = wavelength
        self.amplitude = amplitude          # epsilon_0
        self.direction = direction
        self.phase = phase
        self.envelope = envelope

        # Derived quantities
        self.omega = 2 * np.pi * frequency
        self.k = 2 * np.pi / wavelength
        self.v_saw = frequency * wavelength  # phase velocity (m/s)

    # ------------------------------------------------------------------
    # Core method: apply SAW strain to magnet
    # ------------------------------------------------------------------

    def apply(self, magnet):
        """Register the SAW strain profile on the magnet.

        Sets time-dependent ``rigid_norm_strain`` using the identity

            sin(kx - wt + phi) = sin(kx + phi) cos(wt)
                                - cos(kx + phi) sin(wt)

        so that two separable ``add_time_term(f(t), g(x,y,z))`` calls are
        made.  If an envelope function is provided, the time parts become
        ``env(t) * cos(wt)`` and ``-env(t) * sin(wt)``.

        Parameters
        ----------
        magnet : mumaxplus.Ferromagnet
            The ferromagnetic sample.  Must have ``B1`` (and optionally
            ``B2``) already set.

        Notes
        -----
        - Only normal strain is set (``rigid_norm_strain``).  Shear
          components are zero in the thin-film Rayleigh SAW limit.
        - For propagation along x: ``eps_xx(x,t) = eps_0 sin(kx - wt + phi)``
        - For propagation along y: ``eps_yy(y,t) = eps_0 sin(ky - wt + phi)``
        """
        # Bind to local variables for correct closure capture
        _k = self.k
        _omega = self.omega
        _eps0 = self.amplitude
        _phi = self.phase
        _env = self.envelope

        if self.direction == 'x':
            if _env is None:
                # Term 1: eps0 * sin(kx + phi) * cos(omega*t)
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (np.cos(_omega * t), 0., 0.),
                    lambda x, y, z: (_eps0 * np.sin(_k * x + _phi), 0., 0.))
                # Term 2: -eps0 * cos(kx + phi) * sin(omega*t)
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (-np.sin(_omega * t), 0., 0.),
                    lambda x, y, z: (_eps0 * np.cos(_k * x + _phi), 0., 0.))
            else:
                # With envelope
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (_env(t) * np.cos(_omega * t), 0., 0.),
                    lambda x, y, z: (_eps0 * np.sin(_k * x + _phi), 0., 0.))
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (-_env(t) * np.sin(_omega * t), 0., 0.),
                    lambda x, y, z: (_eps0 * np.cos(_k * x + _phi), 0., 0.))

        elif self.direction == 'y':
            if _env is None:
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (0., np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., _eps0 * np.sin(_k * y + _phi), 0.))
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (0., -np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., _eps0 * np.cos(_k * y + _phi), 0.))
            else:
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (0., _env(t) * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., _eps0 * np.sin(_k * y + _phi), 0.))
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (0., -_env(t) * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., _eps0 * np.cos(_k * y + _phi), 0.))
        else:
            raise ValueError(f"direction must be 'x' or 'y', got '{self.direction}'")

    # ------------------------------------------------------------------
    # Analytical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def kittel_frequency(B_ext, Msat, gamma=GAMMA):
        """Kittel FMR frequency for a thin film with in-plane field.

        omega_K = gamma * sqrt(B_0 * (B_0 + mu_0 M_s))

        Parameters
        ----------
        B_ext : float or array
            External magnetic field magnitude (T).
        Msat : float
            Saturation magnetization (A/m).
        gamma : float
            Gyromagnetic ratio (rad/s/T).

        Returns
        -------
        float or array
            Kittel frequency (rad/s).
        """
        B = np.abs(B_ext)
        return gamma * np.sqrt(B * (B + MU0 * Msat))

    @staticmethod
    def resonance_field(frequency, Msat, gamma=GAMMA):
        """External field B_0 for Kittel resonance at given frequency.

        Solves  omega = gamma * sqrt(B * (B + mu0 * Ms))  for B.

        Parameters
        ----------
        frequency : float
            Target frequency (Hz).
        Msat : float
            Saturation magnetization (A/m).
        gamma : float
            Gyromagnetic ratio (rad/s/T).

        Returns
        -------
        float
            Resonance field (T).
        """
        omega = 2 * np.pi * frequency
        # B^2 + mu0*Ms*B - (omega/gamma)^2 = 0
        a = 1.0
        b = MU0 * Msat
        c = -(omega / gamma) ** 2
        return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    @staticmethod
    def coupling_field(B1, Msat, epsilon_0):
        """Effective magnon-phonon coupling field.

        h_mel = 2 |B1| epsilon_0 / (mu_0 M_s)

        This is the amplitude of the oscillating magnetoelastic field
        produced by the SAW strain.

        Parameters
        ----------
        B1 : float
            First magnetoelastic coupling constant (J/m^3).
        Msat : float
            Saturation magnetization (A/m).
        epsilon_0 : float
            Peak strain amplitude.

        Returns
        -------
        float
            Effective coupling field (T).
        """
        return 2 * abs(B1) * abs(epsilon_0) / (MU0 * Msat)

    @staticmethod
    def dw_velocity_theory(epsilon_0, B1, Msat, alpha, delta_w, omega, k):
        """Analytical DW velocity from SAW driving (leading-order estimate).

        The SAW-driven domain wall velocity scales quadratically with
        strain amplitude:

            v_DW ~ (B1^2 epsilon_0^2) / (alpha mu_0^2 M_s^2 delta_w omega)
                   * geometric_factor(k delta_w)

        This is a simplified estimate based on PRB 108, 104420 (2023).
        The exact prefactor depends on the DW profile and SAW
        wavelength relative to the DW width.

        Parameters
        ----------
        epsilon_0 : float or array
            Peak strain amplitude.
        B1 : float
            First magnetoelastic coupling constant (J/m^3).
        Msat : float
            Saturation magnetization (A/m).
        alpha : float
            Gilbert damping constant.
        delta_w : float
            Domain wall width (m), delta_w = sqrt(A / K_eff).
        omega : float
            SAW angular frequency (rad/s).
        k : float
            SAW wave vector (1/m).

        Returns
        -------
        float or array
            Domain wall velocity (m/s).
        """
        # Leading-order scaling (dimensionally consistent estimate)
        # v ~ (gamma * B1^2 * eps0^2 * delta_w) / (alpha * mu0 * Ms^2 * omega)
        # The k*delta_w factor enters through the overlap integral
        kd = k * delta_w
        geometric = kd / (1 + kd**2)
        v = (GAMMA * B1**2 * epsilon_0**2 * delta_w
             / (alpha * MU0 * Msat**2)) * geometric / omega
        return np.abs(v)

    @staticmethod
    def magnon_phonon_coupling(B1, Msat, epsilon_0, omega_m):
        """Effective magnon-phonon coupling rate.

        g_mp = gamma * B1 * epsilon_0 / (mu_0 * M_s)

        This gives the coupling strength in units of rad/s, analogous
        to the vacuum Rabi rate in cavity magnonics.

        Parameters
        ----------
        B1 : float
            First magnetoelastic coupling constant (J/m^3).
        Msat : float
            Saturation magnetization (A/m).
        epsilon_0 : float
            Peak strain amplitude.
        omega_m : float
            Magnon frequency (rad/s).  Not used in leading order but
            reserved for frequency-dependent corrections.

        Returns
        -------
        float
            Coupling rate (rad/s).
        """
        return GAMMA * abs(B1) * abs(epsilon_0) / (MU0 * Msat)

    def __repr__(self):
        return (f"SurfaceAcousticWave(f={self.frequency*1e-6:.1f} MHz, "
                f"lambda={self.wavelength*1e6:.1f} um, "
                f"eps0={self.amplitude:.1e}, "
                f"dir='{self.direction}', "
                f"v_SAW={self.v_saw:.0f} m/s)")
