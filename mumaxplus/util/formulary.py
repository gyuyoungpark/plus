"""Functions to calculate common micromagnetic quantities."""

import numpy as _np
from .constants import MU0


def magnetostatic_energy_density(msat):
    """Magnetostatic energy density (J/m³).
    1/2 μ0 msat²

    Parameters
    ----------
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Magnetostatic energy density (J/m³).
    """
    return 0.5 * MU0 * msat**2

def Km(msat):
    """Magnetostatic energy density (J/m³).
    1/2 μ0 msat²
    This is a short alias for `magnetostatic_energy_density(msat)`.

    Parameters
    ----------
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Magnetostatic energy density (J/m³).
    """
    return magnetostatic_energy_density(msat)


def exchange_length(aex, msat):
    """Ferromagnetic exchange length (m). Beware that different definitions exist
    without the √2 prefactor, but this one seems to be used most often.
    l_ex = √(aex/Km) = √(2*aex/(μ0 msat²))
    
    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Exchange length (m).
    """
    return _np.sqrt(aex/Km(msat))

def l_ex(aex, msat):
    """Ferromagnetic exchange length (m). Beware that different definitions exist
    without the √2 prefactor, but this one seems to be used most often.
    l_ex = √(aex/Km) = √(2*aex/(μ0 msat²))
    This is a short alias for `exchange_length(aex, msat)`.
    
    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Exchange length (m).
    """
    return exchange_length(aex, msat)


def wall_width(aex, K_eff):
    """Bloch wall width (m). Mind the lack of any prefactor! Different
    definitions exist, for example with a prefactor π.
    √(aex/K_eff)

    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    K_eff : float
        Effective anisotropy constant (J/m³), for example Ku1.

    Returns
    -------
    float
        Wall width (m).
    """
    return _np.sqrt(aex/K_eff)

def wall_energy(aex, K_eff):
    """Bloch wall energy (J/m²): the energy per unit of Bloch domain wall area.
    Mind the lack of any prefactor, different definitions exist!
    √(aex*K_eff)

    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    K_eff : float
        Effective anisotropy constant (J/m³), for example Ku1.

    Returns
    -------
    float
        Wall energy (J/m²).
    """
    return _np.sqrt(aex*K_eff)


def helical_length(aex, D):
    """Characteristic length scale of a DMI dominated system, such as with
    skyrmions. Mind the lack of any prefactor! Different definitions exist,
    with prefactors like 1, 2 or 4π.
    aex/D

    Parameters
    ----------
    aex : float
        Exchange stiffness (J/m).
    D : float
        Strength of DMI (J/m²), for example interfacial DMI strength.

    Returns
    -------
    float
        Helical length (m).
    """
    return aex/D


def magnetic_hardness(K1, msat):
    """Magnetic hardness parameter κ (dimensionless). It should be greater than
    1 for a permanent magnet and much less than 1 for a good temporary magnet.
    √(|K1|/(μ0 msat²))
    https://doi.org/10.1016/j.scriptamat.2015.09.021

    Parameters
    ----------
    K1 : float
        Anisotropy energy density (J/m³), for example ku1.
    msat : float
        Saturation magnetization (A/m).

    Returns
    -------
    float
        Magnetic hardness parameter (dimensionless).
    """
    return _np.sqrt(abs(K1)/(MU0 * msat**2))


def bulk_modulus(C11, C44):
    """For isotropic materials, the bulk modulus is given by
    K = C11 - 4/3 C44 = C12 + 2/3 C44.
    https://en.wikipedia.org/wiki/Hooke%27s_law
    
    Parameters
    ----------
    C11 : float
        Stiffness constant C11 (Pa).
    C44 : float
        Stiffness constant C44 (Pa).

    Returns
    -------
    float
        Bulk modulus (Pa).
    """
    return C11 - 4/3 * C44

def shear_modulus(C44):
    """For isotropic materials, the shear modulus is equal to the stiffness
    constant C44 = c1212.
    
    Parameters
    ----------
    C44 : float
        Stiffness constant C44 (Pa).
    
    Returns
    -------
    float
        Shear modulus (Pa).
    """
    return C44


def Rayleigh_damping_mass_coefficient(frequency_1, damping_ratio_1, frequency_2, damping_ratio_2):
    """Rayleigh damping mass coefficient calculated by providing damping ratios
    at specified frequencies.

    4π f₁ f₂ * (ζ₁f₂ - ζ₂f₁) / (f₂² - f₁²)
    with f₁/f₂ < ζ₂/ζ₁ < f₂/f₁

    Based on https://www.comsol.com/blogs/how-to-model-different-types-of-damping-in-comsol-multiphysics.
    
    Parameters
    ----------
    frequency_1 : float
        Low frequency at which damping_ratio_1 is expected.
        This should be smaller than frequency_2.
    damping_ratio_1 : float
        Positive damping ratio expected at frequency_1.
    frequency_2 : float
        High frequency at which damping_ratio_2 is expected.
        This should be larger than frequency_1.
    damping_ratio_2 : float
        Positive damping ratio expected at frequency_2.

    Returns
    -------
    float
        Rayleigh damping mass coefficient (1/s).
    """
    assert damping_ratio_1 >= 0 and damping_ratio_2 >= 0
    assert frequency_1 >= 0 and frequency_2 >= 0 and frequency_2 > frequency_1
    assert frequency_1 / frequency_2 < damping_ratio_2 / damping_ratio_1 and \
           damping_ratio_2 / damping_ratio_1 < frequency_2 / frequency_1

    return 4 * _np.pi * frequency_1 * frequency_2 \
        * (damping_ratio_1 * frequency_2 - damping_ratio_2 * frequency_1) \
        / (frequency_2*frequency_2 - frequency_1*frequency_1)

def Rayleigh_damping_stiffness_coefficient(frequency_1, damping_ratio_1, frequency_2, damping_ratio_2):
    """Rayleigh damping stiffness coefficient calculated by providing damping
    ratios at specified frequencies.

    1/π * (ζ₂f₂ - ζ₁f₁) / (f₂² - f₁²)
    with f₁/f₂ < ζ₂/ζ₁ < f₂/f₁

    Based on https://www.comsol.com/blogs/how-to-model-different-types-of-damping-in-comsol-multiphysics.

    Parameters
    ----------
    frequency_1 : float
        Low frequency at which damping_ratio_1 is expected.
        This should be smaller than frequency_2.
    damping_ratio_1 : float
        Positive damping ratio expected at frequency_1.
    frequency_2 : float
        High frequency at which damping_ratio_2 is expected.
        This should be larger than frequency_1.
    damping_ratio_2 : float
        Positive damping ratio expected at frequency_2.

    Returns
    -------
    float
        Rayleigh damping stiffness coefficient (s).
    """
    assert damping_ratio_1 >= 0 and damping_ratio_2 >= 0
    assert frequency_1 >= 0 and frequency_2 >= 0 and frequency_2 > frequency_1
    assert frequency_1 / frequency_2 < damping_ratio_2 / damping_ratio_1 and \
           damping_ratio_2 / damping_ratio_1 < frequency_2 / frequency_1

    return 1 / _np.pi \
        * (damping_ratio_2 * frequency_2 - damping_ratio_1 * frequency_1) \
        / (frequency_2*frequency_2 - frequency_1*frequency_1)

def Rayleigh_mass_coef_to_damping_const(mass_coef, rho):
    """Returns the elastic damping constant, given the rayleigh damping mass
    coefficient and the mass density of the material.

    η = mass_coef * ρ

    Parameters
    ----------
    mass_coef : float
        Rayleigh damping mass coefficient (1/s).
    rho : float
        Mass density (kg/m³).
    
    Returns
    -------
    float
        Phenomenological elastic damping constant (kg/m³s).
    """
    return mass_coef * rho

def Rayleigh_stiffness_coef_to_viscosity(stiffness_coef, C11, C44):
    """Returns both the bulk and shear viscosity, given the rayleigh damping
    stiffness coefficient and the stiffness constants C11 and C44 of the material.
    Isotropy is assumed.

    η_b = stiffness_coef * 3K = stiffness_coef * 3 * (C11 - 4/3 C44)

    η_s = stiffness_coef * 2G = stiffness_coef * 2 * C44

    Parameters
    ----------
    stiffness_coef : float
        Rayleigh damping stiffness coefficient (s).
    C11 : float
        Stiffness constant C11 (Pa).
    C44 : float
        Stiffness constant C44 (Pa).
    
    Returns
    -------
    tuple[float] of size 2
        Bulk and shear viscosity (Pa s).
    """
    return (stiffness_coef * 3 * bulk_modulus(C11, C44),
            stiffness_coef * 2 * shear_modulus(C44))
