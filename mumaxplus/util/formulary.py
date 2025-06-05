"""Functions to calculate common micromagnetic quantities."""

import numpy as _np
from .constants import MU0


def magnetostatic_energy_density(msat) -> float:
    r"""Magnetostatic energy density (J/m³).
    
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

def Km(msat) -> float:
    r"""Magnetostatic energy density (J/m³).
    
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


def exchange_length(aex, msat) -> float:
    r"""Ferromagnetic exchange length (m). Beware that different definitions exist
    without the √2 prefactor, but this one seems to be used most often.
    
    .. math:: l_{ex} = \sqrt{a_\text{ex}/K_\text{m}} = \sqrt{2a_\text{ex}/(\mu_0 m_\text{sat}^2)}
    
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

def l_ex(aex, msat) -> float:
    r"""Ferromagnetic exchange length (m). Beware that different definitions exist
    without the √2 prefactor, but this one seems to be used most often.

    .. math:: l_{ex} = \sqrt{a_\text{ex}/K_\text{m}} = \sqrt{2a_\text{ex}/(\mu_0 m_\text{sat}^2)}

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


def wall_width(aex, K_eff) -> float:
    r"""Bloch wall width (m). Mind the lack of any prefactor! Different
    definitions exist, for example with a prefactor π.
    
    .. math:: \sqrt{a_\text{ex}/K_{\text{eff}}}

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

def wall_energy(aex, K_eff) -> float:
    r"""Bloch wall energy (J/m²): the energy per unit of Bloch domain wall area.
    Mind the lack of any prefactor, different definitions exist!

    .. math:: \sqrt{a_\text{ex} K_{\text{eff}}}

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


def helical_length(aex, D) -> float:
    r"""Characteristic length scale of a DMI dominated system, such as with
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


def magnetic_hardness(K1, msat) -> float:
    r"""Magnetic hardness parameter κ (dimensionless). It should be greater than
    1 for a permanent magnet and much less than 1 for a good temporary magnet.
    
    .. math:: \sqrt{\frac{|K_1|}{\mu_0 m_\text{sat}^2}}
    
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
