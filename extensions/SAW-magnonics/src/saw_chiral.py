"""Chiral Surface Acoustic Wave with magneto-rotation and Barnett couplings.

Extends the base SurfaceAcousticWave class to include the full Rayleigh wave
chirality: the elliptical particle motion in the sagittal plane (x-z) creates
a rotation pseudovector Omega_y that is pi/2 out of phase with the strain
epsilon_xx, plus a Barnett effective field from the angular velocity.

Three coupling channels to the magnetization:

1. Magnetoelastic (inherited):
     H_mel ~ B1 * eps_xx * m_i
     eps_xx(x,t) = eps_0 * sin(kx - wt + phi)

2. Magneto-rotation (via CUDA kernel):
     Omega_y(x,t) = (xi * eps_0 / 2) * cos(kx - wt + phi)
     Omega prescribed via rigid_rotation -> CUDA kernel computes H_mr(Omega, m, Kmr)
     Nonlinear: uses current m at every timestep

3. Barnett (via CUDA kernel):
     omega_y(x,t) = d(Omega_y)/dt = (xi * eps_0 * w / 2) * sin(kx - wt + phi)
     omega prescribed via rigid_angular_velocity -> CUDA kernel computes H_B = omega/gamma

Phase relationships (key to chirality physics):
  - H_mel  ~ sin(kx - wt)   [in-phase with strain]
  - H_mr   ~ cos(kx - wt)   [pi/2 out of phase, from rotation]
  - H_B    ~ sin(kx - wt)   [in-phase with strain, from angular velocity]

For reversed propagation (-k), the rotation chirality flips:
  - H_mel  ~ sin(-kx - wt)
  - H_mr   ~ -cos(-kx - wt)  [sign flip! nonreciprocity source]
  - H_B    ~ -sin(-kx - wt)  [sign flip]

This direction-dependent sign creates nonreciprocal magnon-phonon coupling.
For a standing SAW (superposition of +k and -k), the interference pattern
varies with position x, potentially producing local exceptional points.

Typical parameters:
    xi   ~ 0.68   (Rayleigh wave ellipticity for nu ~ 0.3)

    YIG:   B1 ~ -8.8 MJ/m^3,  K_mr ~ 1 MJ/m^3,  M_s = 140 kA/m
    CoFeB: B1 ~ -8.8 MJ/m^3,  K_mr ~ 5 MJ/m^3,  M_s = 1200 kA/m
"""

import numpy as np
from saw import SurfaceAcousticWave, GAMMA, MU0


def rayleigh_ellipticity(poisson_ratio=0.3):
    """Compute Rayleigh wave ellipticity ratio xi.

    The ellipticity ratio relates the vertical displacement amplitude
    to the horizontal displacement amplitude at the surface:
        |u_z| / |u_x| = xi

    For a Poisson ratio nu ~ 0.3, xi ~ 0.68.

    Parameters
    ----------
    poisson_ratio : float
        Poisson ratio of the substrate material.

    Returns
    -------
    float
        Ellipticity ratio (dimensionless, positive).
    """
    nu = poisson_ratio
    # Approximate formula for Rayleigh wave ellipticity
    # (exact solution requires solving the Rayleigh secular equation)
    eta = (0.87 + 1.12 * nu) / (1 + nu)  # Rayleigh velocity ratio v_R/v_s
    xi = np.sqrt(1 - eta**2 * (1 - 2*nu) / (2*(1 - nu)))
    return abs(xi)


class ChiralSurfaceAcousticWave(SurfaceAcousticWave):
    """Rayleigh SAW with full chirality: magnetoelastic + magneto-rotation
    + Barnett couplings.

    The Rayleigh wave has elliptical particle motion in the sagittal plane,
    characterized by the ellipticity ratio xi.  This creates a lattice
    rotation Omega_y and angular velocity omega_y that couple to the
    magnetization through additional effective fields.

    Parameters
    ----------
    frequency : float
        SAW frequency (Hz).
    wavelength : float
        SAW wavelength (m).
    amplitude : float
        Peak strain epsilon_0 (dimensionless).
    direction : str
        Propagation direction: 'x' or 'y'.
    phase : float
        Initial phase (rad).
    envelope : callable or None
        Time envelope function.
    ellipticity : float
        Rayleigh wave ellipticity ratio xi (default: 0.68).
    K_mr : float
        Magneto-rotation coupling constant (J/m^3). Default: 0 (disabled).
    enable_barnett : bool
        Enable Barnett effective field. Default: False.
    """

    def __init__(self, frequency, wavelength, amplitude,
                 direction='x', phase=0.0, envelope=None,
                 ellipticity=0.68, K_mr=0.0, enable_barnett=False):
        super().__init__(frequency, wavelength, amplitude,
                         direction, phase, envelope)
        self.ellipticity = ellipticity
        self.K_mr = K_mr
        self.enable_barnett = enable_barnett

    def apply(self, magnet, Msat=None, enable_mel=True, use_gpu_kernel=True):
        """Register all coupling channels on the magnet.

        Parameters
        ----------
        magnet : mumaxplus.Ferromagnet
            The ferromagnetic sample.
        Msat : float or None
            Saturation magnetization (A/m). Required for magneto-rotation
            and Barnett fields. If None, these are disabled.
        enable_mel : bool
            Enable magnetoelastic coupling (default: True).
        use_gpu_kernel : bool
            If True (default), use the dedicated chiral SAW CUDA kernel
            that computes all three channels in a single GPU pass with
            no Python callbacks. If False, fall back to the legacy
            add_time_term approach.
        """
        if use_gpu_kernel and hasattr(magnet, 'enable_saw'):
            self._apply_gpu_kernel(magnet, enable_mel)
        else:
            self._apply_legacy(magnet, Msat, enable_mel)

    def _apply_gpu_kernel(self, magnet, enable_mel=True):
        """Set SAW parameters for the dedicated CUDA kernel.

        All three coupling channels (MEL, MR, Barnett) are computed in
        a single GPU kernel (k_chiralSAWField) at every LLG timestep
        with no Python callback overhead.
        """
        magnet.enable_saw = True
        magnet.saw_frequency = self.omega
        magnet.saw_wavevector = self.k if self.K_mr >= 0 else -self.k
        magnet.saw_amplitude = self.amplitude
        magnet.saw_ellipticity = self.ellipticity
        magnet.saw_phase = self.phase
        magnet.saw_direction = 0 if self.direction == 'x' else 1
        magnet.saw_gamma = GAMMA
        magnet.saw_enable_mel = enable_mel
        magnet.saw_enable_barnett = self.enable_barnett

        # B1 and Kmr are set on the magnet by the simulation script
        # (magnet.B1 = ..., magnet.Kmr = ...)
        # The kernel reads them directly.
        if enable_mel:
            # Ensure B1 is set (the kernel needs it)
            pass  # User must set magnet.B1 before calling apply()

        if abs(self.K_mr) > 0:
            magnet.Kmr = abs(self.K_mr)
            magnet.anisU = (0, 0, 1)

    def _apply_legacy(self, magnet, Msat=None, enable_mel=True):
        """Legacy approach using add_time_term Python callbacks.

        Falls back to the original implementation for compatibility
        with older mumax+ builds that lack the chiral SAW kernel.
        """
        # 1. Magnetoelastic coupling (parent class)
        if enable_mel:
            super().apply(magnet)

        # 2. Magneto-rotation coupling via CUDA kernel
        if abs(self.K_mr) > 0:
            self._apply_magneto_rotation(magnet)

        # 3. Barnett coupling via CUDA kernel
        if self.enable_barnett:
            self._apply_barnett(magnet)

    def _apply_magneto_rotation(self, magnet):
        """Prescribe rotation pseudovector Omega(x,t) via rigid_rotation.

        The CUDA kernel k_magnetoRotationField then computes the full
        nonlinear MR effective field H_mr = f(Omega, m, Kmr) at every
        cell using the current magnetization.

        Omega_y(x,t) = (xi * eps_0 / 2) * cos(kx - wt + phi)

        Decomposed: cos(kx - wt + phi) =
            cos(kx + phi) cos(wt) + sin(kx + phi) sin(wt)
        """
        _k = self.k
        _omega = self.omega
        _eps0 = self.amplitude
        _phi = self.phase
        _env = self.envelope
        _xi = self.ellipticity

        # Set |Kmr| and anisU on the magnet for the CUDA kernel
        # Sign of K_mr encodes propagation direction (+k or -k)
        magnet.Kmr = abs(self.K_mr)
        # anisU = surface normal (z) for thin-film MR coupling convention
        magnet.anisU = (0, 0, 1)

        # Rotation amplitude (rad), sign encodes chirality (+k or -k)
        sign = 1.0 if self.K_mr > 0 else -1.0
        Omega_amp = sign * _xi * _eps0 / 2.0

        if self.direction == 'x':
            if _env is None:
                # Term 1: Omega_amp * cos(kx+phi) * cos(wt) along y
                magnet.rigid_rotation.add_time_term(
                    lambda t: (0., Omega_amp * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.cos(_k * x + _phi), 0.))
                # Term 2: Omega_amp * sin(kx+phi) * sin(wt) along y
                magnet.rigid_rotation.add_time_term(
                    lambda t: (0., Omega_amp * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * x + _phi), 0.))
            else:
                magnet.rigid_rotation.add_time_term(
                    lambda t: (0., Omega_amp * _env(t) * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.cos(_k * x + _phi), 0.))
                magnet.rigid_rotation.add_time_term(
                    lambda t: (0., Omega_amp * _env(t) * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * x + _phi), 0.))

        elif self.direction == 'y':
            if _env is None:
                magnet.rigid_rotation.add_time_term(
                    lambda t: (0., Omega_amp * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.cos(_k * y + _phi), 0.))
                magnet.rigid_rotation.add_time_term(
                    lambda t: (0., Omega_amp * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * y + _phi), 0.))
            else:
                magnet.rigid_rotation.add_time_term(
                    lambda t: (0., Omega_amp * _env(t) * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.cos(_k * y + _phi), 0.))
                magnet.rigid_rotation.add_time_term(
                    lambda t: (0., Omega_amp * _env(t) * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * y + _phi), 0.))

    def _apply_barnett(self, magnet):
        """Prescribe angular velocity omega(x,t) via rigid_angular_velocity.

        The CUDA kernel k_spinRotationField then computes
        H_Barnett = omega / gamma at every cell.

        omega_y(x,t) = (xi * eps_0 * w / 2) * sin(kx - wt + phi)

        Decompose sin(kx - wt + phi) = sin(kx+phi)cos(wt) - cos(kx+phi)sin(wt).
        """
        _k = self.k
        _omega = self.omega
        _eps0 = self.amplitude
        _phi = self.phase
        _env = self.envelope
        _xi = self.ellipticity

        # Enable Barnett on the magnet for the CUDA kernel
        magnet.enable_barnett = True

        # Angular velocity amplitude (rad/s)
        omega_amp = _xi * _eps0 * _omega / 2.0

        if self.direction == 'x':
            if _env is None:
                # Term 1: omega_amp * sin(kx+phi) * cos(wt) along y
                magnet.rigid_angular_velocity.add_time_term(
                    lambda t: (0., omega_amp * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * x + _phi), 0.))
                # Term 2: -omega_amp * cos(kx+phi) * sin(wt) along y
                magnet.rigid_angular_velocity.add_time_term(
                    lambda t: (0., -omega_amp * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., np.cos(_k * x + _phi), 0.))
            else:
                magnet.rigid_angular_velocity.add_time_term(
                    lambda t: (0., omega_amp * _env(t) * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * x + _phi), 0.))
                magnet.rigid_angular_velocity.add_time_term(
                    lambda t: (0., -omega_amp * _env(t) * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., np.cos(_k * x + _phi), 0.))

        elif self.direction == 'y':
            if _env is None:
                magnet.rigid_angular_velocity.add_time_term(
                    lambda t: (0., omega_amp * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * y + _phi), 0.))
                magnet.rigid_angular_velocity.add_time_term(
                    lambda t: (0., -omega_amp * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., np.cos(_k * y + _phi), 0.))
            else:
                magnet.rigid_angular_velocity.add_time_term(
                    lambda t: (0., omega_amp * _env(t) * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * y + _phi), 0.))
                magnet.rigid_angular_velocity.add_time_term(
                    lambda t: (0., -omega_amp * _env(t) * np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., np.cos(_k * y + _phi), 0.))

    # ------------------------------------------------------------------
    # Analytical helpers for the extended coupling channels
    # ------------------------------------------------------------------

    def rotation_amplitude(self):
        """Peak rotation pseudovector amplitude Omega_y (rad).

        Omega_y_peak = xi * eps_0 / 2
        """
        return self.ellipticity * self.amplitude / 2.0

    def angular_velocity_amplitude(self):
        """Peak angular velocity amplitude omega_y (rad/s).

        omega_y_peak = xi * eps_0 * omega_SAW / 2
        """
        return self.ellipticity * self.amplitude * self.omega / 2.0

    def magneto_rotation_field(self, Msat):
        """Peak magneto-rotation effective field (T).

        H_mr = K_mr * xi * eps_0 / (2 * mu_0 * M_s)

        Parameters
        ----------
        Msat : float
            Saturation magnetization (A/m).

        Returns
        -------
        float
            Peak MR field (T).
        """
        return abs(self.K_mr) * self.ellipticity * self.amplitude / (2.0 * Msat)

    def barnett_field(self):
        """Peak Barnett effective field (T).

        H_B = xi * eps_0 * omega / (2 * gamma)

        Returns
        -------
        float
            Peak Barnett field (T).
        """
        return self.ellipticity * self.amplitude * self.omega / (2.0 * GAMMA)

    def coupling_hierarchy(self, B1, Msat):
        """Print the coupling field hierarchy for diagnostics.

        All fields converted to Tesla for comparison.

        Parameters
        ----------
        B1 : float
            Magnetoelastic coupling constant (J/m^3).
        Msat : float
            Saturation magnetization (A/m).
        """
        # coupling_field returns A/m; convert to Tesla
        h_mel_Am = self.coupling_field(B1, Msat, self.amplitude)
        h_mel = MU0 * h_mel_Am  # Tesla
        h_mr = self.magneto_rotation_field(Msat)  # already Tesla
        h_b = self.barnett_field()  # already Tesla

        print(f"  Coupling field hierarchy (all in Tesla):")
        print(f"    H_mel     = {h_mel*1e3:.3f} mT  (magnetoelastic)")
        print(f"    H_mr      = {h_mr*1e3:.3f} mT  (magneto-rotation)")
        print(f"    H_Barnett = {h_b*1e6:.3f} uT  (Barnett)")
        if h_mel > 0:
            print(f"    H_mr/H_mel = {h_mr/h_mel:.4f}")
            print(f"    H_B/H_mel  = {h_b/h_mel:.2e}")

        return h_mel, h_mr, h_b

    @staticmethod
    def effective_coupling_rate(B1, K_mr, Msat, eps0, xi, omega_saw,
                                direction=+1, gamma=GAMMA):
        """Complex effective coupling rate including all channels.

        The effective coupling strength depends on propagation direction
        due to the chirality-induced phase relationship:

            g_eff(+k) = g_mel + i * g_mr + g_B     (constructive MR)
            g_eff(-k) = g_mel - i * g_mr - g_B     (destructive MR)

        Parameters
        ----------
        B1 : float
            Magnetoelastic coupling constant (J/m^3).
        K_mr : float
            Magneto-rotation coupling constant (J/m^3).
        Msat : float
            Saturation magnetization (A/m).
        eps0 : float
            Peak strain amplitude.
        xi : float
            Ellipticity ratio.
        omega_saw : float
            SAW angular frequency (rad/s).
        direction : int
            +1 for +k propagation, -1 for -k propagation.
        gamma : float
            Gyromagnetic ratio (rad/s/T).

        Returns
        -------
        complex
            Effective coupling rate (rad/s).
        """
        g_mel = gamma * abs(B1) * abs(eps0) / (MU0 * Msat)
        g_mr = gamma * abs(K_mr) * xi * abs(eps0) / (2 * MU0 * Msat)
        g_B = xi * abs(eps0) * omega_saw / 2.0

        # Direction-dependent combination
        # MR is pi/2 out of phase → imaginary part
        # Barnett is in-phase → adds to real part
        # Both flip sign with propagation direction
        g_eff = g_mel + direction * 1j * g_mr + direction * g_B

        return g_eff

    def __repr__(self):
        base = super().__repr__()
        return (base[:-1] +
                f", xi={self.ellipticity:.2f}, "
                f"K_mr={self.K_mr*1e-6:.1f} MJ/m^3, "
                f"Barnett={'ON' if self.enable_barnett else 'OFF'})")


class StandingSAW:
    """Standing SAW formed by two counter-propagating Rayleigh waves.

    A standing wave creates spatially varying coupling:
    - At strain antinodes: magnetoelastic coupling is maximum
    - At rotation antinodes (shifted by lambda/4): MR coupling is maximum
    - The competition between channels varies with position

    The standing SAW is a superposition:
        epsilon_xx(x,t) = eps_0 * [sin(kx - wt) + sin(-kx - wt)]
                        = -2 * eps_0 * cos(kx) * sin(wt)

    The rotation from counter-propagating waves:
        Omega_y = Omega_y(+k) + Omega_y(-k)
        For a standing wave, the rotation has DIFFERENT spatial nodes
        than the strain, shifted by lambda/4.

    Parameters
    ----------
    frequency : float
        SAW frequency (Hz).
    wavelength : float
        SAW wavelength (m).
    amplitude : float
        Peak strain per traveling wave (total strain = 2*amplitude).
    direction : str
        Standing wave direction: 'x' or 'y'.
    ellipticity : float
        Rayleigh wave ellipticity ratio.
    K_mr : float
        Magneto-rotation coupling constant (J/m^3).
    enable_barnett : bool
        Enable Barnett effective field.
    """

    def __init__(self, frequency, wavelength, amplitude,
                 direction='x', ellipticity=0.68,
                 K_mr=0.0, enable_barnett=False):
        self.frequency = frequency
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.direction = direction
        self.ellipticity = ellipticity
        self.K_mr = K_mr
        self.enable_barnett = enable_barnett

        self.omega = 2 * np.pi * frequency
        self.k = 2 * np.pi / wavelength

    def apply(self, magnet, Msat=None, enable_mel=True):
        """Register standing SAW strain and fields on the magnet.

        Standing wave strain:
            eps_xx(x,t) = 2 * eps_0 * sin(kx) * cos(wt)
                        (using sin(kx-wt) + sin(-kx-wt) = -2cos(kx)sin(wt))

        Actually: sin(kx - wt) + sin(-kx - wt) = sin(kx-wt) - sin(kx+wt)
                = -2 cos(kx) sin(wt)

        For the standing wave, we use:
            eps_xx(x,t) = 2 * eps_0 * cos(kx) * sin(wt)  [choosing sign]

        This gives strain nodes at kx = pi/2 + n*pi and
        strain antinodes at kx = n*pi.
        """
        _k = self.k
        _omega = self.omega
        _eps0 = self.amplitude
        _xi = self.ellipticity
        _K = self.K_mr

        if self.direction == 'x':
            # Magnetoelastic: eps_xx(x,t) = 2*eps0 * cos(kx) * sin(wt)
            if enable_mel:
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (np.sin(_omega * t), 0., 0.),
                    lambda x, y, z: (2 * _eps0 * np.cos(_k * x), 0., 0.))

            # Magneto-rotation for standing wave:
            # Omega_y(+k) = (xi*eps0/2) cos(kx - wt)
            # Omega_y(-k) = -(xi*eps0/2) cos(-kx - wt) = -(xi*eps0/2) cos(kx + wt)
            # Sum: (xi*eps0/2)[cos(kx-wt) - cos(kx+wt)]
            #     = (xi*eps0/2) * 2*sin(kx)*sin(wt)
            #     = xi*eps0 * sin(kx) * sin(wt)
            #
            # Note: Omega nodes at kx = n*pi (strain antinodes!)
            #        Omega antinodes at kx = pi/2 + n*pi (strain nodes!)
            if abs(_K) > 0 and Msat is not None:
                H_amp = _K * _xi * _eps0 / Msat  # factor of 2 from standing wave
                magnet.bias_magnetic_field.add_time_term(
                    lambda t: (0., 0., -H_amp * np.sin(_omega * t)),
                    lambda x, y, z: (0., 0., np.sin(_k * x)))

            # Barnett for standing wave:
            # omega_y = d(Omega_y)/dt
            # = xi*eps0*w * sin(kx) * cos(wt)
            if self.enable_barnett:
                H_B = _xi * _eps0 * _omega / GAMMA
                magnet.bias_magnetic_field.add_time_term(
                    lambda t: (0., H_B * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * x), 0.))

        elif self.direction == 'y':
            if enable_mel:
                magnet.rigid_norm_strain.add_time_term(
                    lambda t: (0., np.sin(_omega * t), 0.),
                    lambda x, y, z: (0., 2 * _eps0 * np.cos(_k * y), 0.))

            if abs(_K) > 0 and Msat is not None:
                H_amp = _K * _xi * _eps0 / Msat
                magnet.bias_magnetic_field.add_time_term(
                    lambda t: (0., 0., -H_amp * np.sin(_omega * t)),
                    lambda x, y, z: (0., 0., np.sin(_k * y)))

            if self.enable_barnett:
                H_B = _xi * _eps0 * _omega / GAMMA
                magnet.bias_magnetic_field.add_time_term(
                    lambda t: (0., H_B * np.cos(_omega * t), 0.),
                    lambda x, y, z: (0., np.sin(_k * y), 0.))

    def local_coupling_map(self, x_positions, B1, Msat, gamma=GAMMA):
        """Compute position-dependent effective coupling for the standing wave.

        At each position x, the effective coupling is a superposition of
        MEL, MR, and Barnett channels with position-dependent amplitudes.

        Parameters
        ----------
        x_positions : array
            Positions along the SAW direction (m).
        B1 : float
            Magnetoelastic coupling constant (J/m^3).
        Msat : float
            Saturation magnetization (A/m).

        Returns
        -------
        dict with keys:
            'g_mel': magnetoelastic coupling amplitude vs x
            'g_mr': magneto-rotation coupling amplitude vs x
            'g_B': Barnett coupling amplitude vs x
            'g_total': total effective coupling vs x
        """
        k = self.k
        eps0 = self.amplitude
        xi = self.ellipticity

        # Position-dependent amplitudes for standing wave
        # MEL: eps_xx ~ cos(kx) → coupling g_mel ~ |cos(kx)|
        g_mel = gamma * abs(B1) * 2 * eps0 * np.abs(np.cos(k * x_positions)) / (MU0 * Msat)

        # MR: Omega_y ~ sin(kx) → coupling g_mr ~ |sin(kx)|
        g_mr = gamma * abs(self.K_mr) * xi * eps0 * np.abs(np.sin(k * x_positions)) / (MU0 * Msat)

        # Barnett: omega_y ~ sin(kx) → coupling g_B ~ |sin(kx)|
        g_B = xi * eps0 * self.omega * np.abs(np.sin(k * x_positions)) / 2.0

        g_total = np.sqrt(g_mel**2 + g_mr**2 + g_B**2)

        return {
            'g_mel': g_mel,
            'g_mr': g_mr,
            'g_B': g_B,
            'g_total': g_total,
            'x': x_positions,
        }

    def __repr__(self):
        return (f"StandingSAW(f={self.frequency*1e-6:.1f} MHz, "
                f"lambda={self.wavelength*1e6:.1f} um, "
                f"eps0={self.amplitude:.1e}, "
                f"xi={self.ellipticity:.2f}, "
                f"K_mr={self.K_mr*1e-6:.1f} MJ/m^3)")
