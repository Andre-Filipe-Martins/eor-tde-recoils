"""
physics_relations.py
--------------------
Physical relations and empirical scaling laws used throughout the
Monte Carlo simulation pipeline. This module is not intended to be
run directly; it is imported by the simulation scripts.

Contents (in order):
  1.  Jiménez-Forteza+2017 remnant mass (non-spinning)
  2.  Reines & Volonteri (2015) M_BH–M_* relation
  3.  Morishita+2024 galaxy size–mass–redshift relation
  4.  TDE rate building blocks (Merritt+2009 formalism)
  5.  Ballistic central escape speed (NFW + Hernquist bulge)
  6.  FIRE-2 stellar–halo mass relation (z = 5–12)
  7.  FIRE-2 galaxy stellar mass function (Schechter fits from gsmf_fitting.py)
  8.  Cosmology helpers (cosmic age, time to z = 6)
  9.  Duan+2025 major-merger rate

Units convention
----------------
- Masses   : solar masses (Msun) unless a function docstring says otherwise.
- Distances: kpc for galaxy sizes; metres (SI) inside TDE helpers.
- Velocities: km/s for kicks and escape speeds; m/s inside TDE helpers.
- Redshift z is dimensionless.
- All logarithms are base-10.

Each function docstring states input and output units explicitly.
Most relations are deterministic unless a scatter argument is explicitly provided.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.interpolate import PchipInterpolator


# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # 1. Remnant mass (JF2017)
    "eta", "mass_ratio", "symmetric_mass_ratio_from_masses",
    "erad_fraction_ns_jf2017", "final_mass_and_fraction_ns_jf2017",
    # 2. M_BH–M_* (Reines & Volonteri 2015)
    "RV15AGNParams", "log10_mstar_from_mbh", "mstar_from_mbh",
    # 3. Galaxy size–mass–redshift (Morishita+2024)
    "M24SizeParams", "log10_re_kpc_m24", "re_kpc_m24",
    # 4. TDE building blocks
    "G", "M_sun", "pc", "km",
    "r_k", "sigma_from_mbh", "r_eff_from_rk_gamma1",
    "r_t", "m_b_collisional", "f_b_from_mb", "tde_rate_resonant",
    # 5. NFW+Hernquist escape speed
    "hubble_z", "r_vir_mpc", "nfw_concentration", "vesc0_nfw_hernquist",
    # 6. FIRE-2 SHMR
    "FIRE2_SHMR_Params", "fire2_log10_mstar_from_mh", "fire2_mstar_from_mh",
    "fire2_log10_mh_from_mstar", "fire2_mh_from_mstar",
    # 7. FIRE-2 GSMF (Schechter)
    "SchechterParams", "schechter_phi_perdex", "schechter_log10phi_perdex",
    "schechter_params_by_z",
    "schechter_z6", "schechter_z7", "schechter_z8", "schechter_z9",
    "schechter_z10", "schechter_z11", "schechter_z12",
    "schechter_z_6", "schechter_z_7", "schechter_z_8", "schechter_z_9",
    "schechter_z_10", "schechter_z_11", "schechter_z_12",
    "get_schechter_params_at_z",
    "schechter_phi_perdex_at_z", "schechter_log10phi_perdex_at_z",
    "gsmf_number_density_in_bin", "number_of_galaxies_in_bin_at_z",
    # 8. Cosmology helpers
    "age_of_universe_at_z", "time_until_z6",
    # 9. Merger rate (Duan+2025)
    "MergerRateFit", "merger_rate_z",
]


# ============================================================================
# 1. REMNANT MASS — JIMÉNEZ-FORTEZA ET AL. (2017), NON-SPINNING FIT
#    Radiated energy fraction E_rad(η) from Table VII, Eq. (21).
# ============================================================================

def eta(q: float) -> float:
    """Symmetric mass ratio η = q / (1+q)², with q = m1/m2 ≤ 1."""
    if not (0 < q <= 1):
        raise ValueError("q must be in (0,1]. Define q = m1/m2 with m1 <= m2.")
    return q / (1.0 + q)**2


def mass_ratio(m1: float, m2: float) -> float:
    """Return q = m1/m2 ≤ 1, enforcing m1 ≤ m2 internally."""
    if m1 <= 0 or m2 <= 0:
        raise ValueError("Component masses must be positive.")
    q = min(m1, m2) / max(m1, m2)
    return min(q, 1.0)


def symmetric_mass_ratio_from_masses(m1: float, m2: float) -> float:
    """Compute the symmetric mass ratio η directly from (m1, m2)."""
    return eta(mass_ratio(m1, m2))


def erad_fraction_ns_jf2017(eta_value: float) -> float:
    """Dimensionless radiated-energy fraction E_rad(η) for non-spinning binaries.

    Uses the JF2017 polynomial fit (Eq. 21, Table VII):
        E_rad = a4·η⁴ + a3·η³ + a2·η² + (1 − 2√2/3)·η
    Coefficients: a2 = 0.5610, a3 = −0.847, a4 = 3.145. Valid for η ∈ (0, 0.25].
    """
    eta_value = float(eta_value)
    if eta_value <= 0.0:
        raise ValueError("η must be in (0, 0.25].")
    eta_value = min(eta_value, 0.25)
    a2, a3, a4 = 0.5610, -0.847, 3.145
    linear = 1.0 - 2.0 * math.sqrt(2.0) / 3.0
    return a4*eta_value**4 + a3*eta_value**3 + a2*eta_value**2 + linear*eta_value


def final_mass_and_fraction_ns_jf2017(m1: float, m2: float) -> tuple[float, float]:
    """Remnant mass and radiated fraction using JF2017 for non-spinning binaries.

    Returns (M_final, f_rad) where M_final = (m1 + m2) * (1 - E_rad(η)).
    Applicability note: this non-spinning limit is appropriate when aligned
    spin contributions are negligible, such as idealised superkick setups.
    """
    M_total = m1 + m2
    eta_val = symmetric_mass_ratio_from_masses(m1, m2)
    f_rad = erad_fraction_ns_jf2017(eta_val)
    return M_total * (1.0 - f_rad), f_rad


# ============================================================================
# 2. M_BH — M_* RELATION — REINES & VOLONTERI (2015), AGN HOSTS
#    Parameterisation: log10(M_BH) = alpha + beta * log10(M_*/M_pivot)
# ============================================================================

@dataclass(frozen=True)
class RV15AGNParams:
    """Parameters for the Reines & Volonteri (2015) M_BH–M_* relation.

    alpha, beta : regression intercept and slope in log-space.
    M_pivot     : pivot stellar mass in Msun.
    sigma_intrinsic_bh : intrinsic scatter in log10(M_BH) at fixed M_* [dex].
    sigma_rms_bh       : total rms scatter in log10(M_BH) at fixed M_* [dex].
    """
    alpha: float = 7.45
    beta: float = 1.05
    M_pivot: float = 1e11
    sigma_intrinsic_bh: float = 0.24
    sigma_rms_bh: float = 0.55


def log10_mstar_from_mbh(m_bh: float,
                         params: RV15AGNParams = RV15AGNParams(),
                         scatter_in_bh_dex: float | None = None,
                         scatter_in_mstar_dex: float | None = None,
                         rng: random.Random | None = None) -> float:
    """Invert the RV15 relation to get log10(M_*/Msun) from M_BH [Msun].

    Base relation (no scatter):
        log10 M_* = log10(M_pivot) + (log10 M_BH − alpha) / beta

    Optional scatter is available through the scatter arguments:
      - scatter_in_bh_dex   : adds scatter in M_BH at fixed M_*, propagated to M_*.
      - scatter_in_mstar_dex: adds scatter directly in M_* (dex).
    If neither is given, returns the deterministic inversion.
    """
    if m_bh <= 0:
        raise ValueError("M_BH must be positive [Msun].")
    log10_mbh = math.log10(m_bh)
    base = math.log10(params.M_pivot) + (log10_mbh - params.alpha) / params.beta

    if scatter_in_mstar_dex is not None or scatter_in_bh_dex is not None:
        if rng is None:
            rng = random
        if scatter_in_mstar_dex is not None:
            delta = rng.gauss(0.0, scatter_in_mstar_dex)
        else:
            delta = rng.gauss(0.0, scatter_in_bh_dex / params.beta)
        return base + delta

    return base


def mstar_from_mbh(m_bh: float,
                   params: RV15AGNParams = RV15AGNParams(),
                   **scatter_kwargs) -> float:
    """Return M_* [Msun] (not log10) from M_BH [Msun] using the RV15 relation."""
    return 10.0 ** log10_mstar_from_mbh(m_bh, params=params, **scatter_kwargs)


# ============================================================================
# 3. GALAXY SIZE–MASS–REDSHIFT — MORISHITA ET AL. (2024)
#    log10(Re/kpc) = alpha * log10(M*/M0) + beta_z * log10(1+z) + alpha_z
# ============================================================================

@dataclass(frozen=True)
class M24SizeParams:
    """Parameters for the Morishita+2024 size–mass–redshift relation (Eqs. 6–7).

    alpha         : stellar-mass slope.
    beta_z        : redshift slope.
    alpha_z       : zero-point.
    M0            : pivot stellar mass [Msun].
    sigma_log10_Re: intrinsic scatter in log10(Re) [dex]; stored but not applied
                    by the deterministic helpers.
    """
    alpha: float = 0.19
    beta_z: float = -0.21
    alpha_z: float = -0.44
    M0: float = 1e8
    sigma_log10_Re: float = 0.30


def log10_re_kpc_m24(mstar_msun: float, z: float,
                     params: M24SizeParams = M24SizeParams()) -> float:
    """Deterministic effective radius log10(Re/kpc) from Morishita+2024.

    Parameters
    ----------
    mstar_msun : float  -- Stellar mass [Msun].
    z          : float  -- Redshift.
    """
    if mstar_msun <= 0.0:
        raise ValueError("M* must be positive [Msun].")
    if z < 0.0:
        raise ValueError("Redshift z must be >= 0.")
    return (params.alpha * math.log10(mstar_msun / params.M0)
            + params.beta_z * math.log10(1.0 + z)
            + params.alpha_z)


def re_kpc_m24(mstar_msun: float, z: float,
               params: M24SizeParams = M24SizeParams()) -> float:
    """Deterministic effective radius Re [kpc] from Morishita+2024 (no scatter)."""
    return 10.0 ** log10_re_kpc_m24(mstar_msun, z, params=params)


# ============================================================================
# 4. TDE RATE BUILDING BLOCKS — MERRITT ET AL. (2009) FORMALISM
#    Physical constants in SI; function docstrings specify units explicitly.
# ============================================================================

# Physical constants (SI)
G     = 6.67430e-11   # m^3 kg^-1 s^-2
M_sun = 1.989e30      # kg
pc    = 3.085677581e16 # m
km    = 1_000.0        # m

# Prefactor for r_eff at fixed γ = 1 (Dehnen profile half-mass radius)
_F2_GAMMA1 = 1.5 * (np.sqrt(2.0) + 1.0)  # ≈ 3.621


def r_k(M_BH: float, v_k: float) -> float:
    """Kick (bound-cluster) scale r_k = G M_BH / v_k² (Merritt+2009 Eq. 1a).

    Parameters: M_BH [kg], v_k [m/s]. Returns r_k [m].
    """
    return G * M_BH / v_k**2


def sigma_from_mbh(M_BH, fit: str = "full"):
    """Stellar velocity dispersion from the McConnell & Ma (2013) M–σ relation.

    Parameters
    ----------
    M_BH : float or array-like  -- Black-hole mass [kg].
    fit  : {"full","early","late"} -- Which sub-sample calibration to use.

    Returns
    -------
    sigma_km_s : ndarray  -- Velocity dispersion [km/s].
    sigma_m_s  : ndarray  -- Velocity dispersion [m/s].

    Relation: log10(M_BH/Msun) = a + b * log10(σ / 200 km/s).
    """
    coeffs = {"full": (8.32, 5.64), "early": (8.39, 5.20), "late": (8.07, 5.06)}
    a, b = coeffs[fit]
    logM = np.log10(np.asarray(M_BH, float) / M_sun)
    sigma_km = 200.0 * 10.0 ** ((logM - a) / b)
    return sigma_km, sigma_km * km


def r_eff_from_rk_gamma1(rk) -> np.ndarray:
    """Effective (half-mass) radius for γ = 1: r_eff = F2 * r_k. Same units as rk."""
    return _F2_GAMMA1 * np.asarray(rk, float)


def r_t(M_BH: float, M_star: float, R_star: float) -> float:
    """Tidal radius r_t = R_star * (M_BH/M_star)^(1/3). Units follow R_star."""
    return R_star * (M_BH / M_star)**(1.0 / 3.0)


def m_b_collisional(M_BH_solar: float, sigma_km_s: float, r_eff_pc: float) -> float:
    """Collisional bound mass from Merritt+2009 [returns kg].

    Parameters: M_BH_solar [Msun], sigma_km_s [km/s], r_eff_pc [pc].
    """
    Mb_Msun = (4e4
               * (M_BH_solar / 1e7)**(-0.25)
               * (sigma_km_s / 100.0)**2.5
               * (r_eff_pc / 0.1)**1.25)
    return Mb_Msun * M_sun


def f_b_from_mb(M_b, M_BH) -> np.ndarray:
    """Bound fraction f_b = M_b / M_BH (dimensionless)."""
    return np.asarray(M_b, float) / np.asarray(M_BH, float)


def tde_rate_resonant(M_BH, M_star, r_k_val, r_t_val, v_k, f_b,
                      lnLambda=None):
    r"""Simplified resonant-relaxation TDE rate estimate (Merritt+2009).

    Scaling: Γ ≈ (lnΛ / ln(r_k/r_t)) * (v_k / r_k) * f_b [s^-1]

    Parameters
    ----------
    M_BH    : float        -- BH mass [kg].
    M_star  : float        -- Stellar mass [kg] (used for lnΛ if not provided).
    r_k_val : float/array  -- Kick radius [m].
    r_t_val : float/array  -- Tidal radius [m].
    v_k     : float/array  -- Kick speed [m/s].
    f_b     : float/array  -- Bound fraction (dimensionless).
    lnLambda: float/array  -- Coulomb logarithm; defaults to ln(M_BH/M_star).

    Returns: rate [s^-1].
    """
    rk = np.asarray(r_k_val, float)
    rt = np.asarray(r_t_val, float)
    ln_ratio = np.where(rk > rt, np.log(rk / rt), np.inf)
    if lnLambda is None:
        lnLambda = np.log(np.asarray(M_BH, float) / np.asarray(M_star, float))
    return (lnLambda / ln_ratio) * (v_k / rk) * f_b


# ============================================================================
# 5. BALLISTIC CENTRAL ESCAPE SPEED — NFW HALO + HERNQUIST BULGE
#    Cosmology defaults: flat ΛCDM, H0 = 67.74 km/s/Mpc, Ωm = 0.31.
# ============================================================================

_H0_FID = 67.74
_OM_FID = 0.31
_OL_FID = 1.0 - _OM_FID


def hubble_z(z: float, H0: float = _H0_FID, Om: float = _OM_FID,
        Ol: float | None = None) -> float:
    """Hubble parameter H(z) [km/s/Mpc] in flat ΛCDM."""
    if Ol is None:
        Ol = 1.0 - Om
    return H0 * math.sqrt(Om * (1.0 + z)**3 + Ol)


def r_vir_mpc(Mh: float, z: float, Delta: float = 200.0,
              H0: float = _H0_FID, Om: float = _OM_FID,
              Ol: float | None = None) -> float:
    """Virial radius R_vir [Mpc] for halo mass M_h [Msun] at redshift z.

    Uses the critical density ρ_c(z) = 3 H(z)² / (8πG) and overdensity Delta.
    """
    Hz = hubble_z(z, H0=H0, Om=Om, Ol=Ol)
    G_Mpc = 4.30091e-6 * 1e-3  # (km/s)^2 Mpc / Msun
    rho_c = 3.0 * Hz**2 / (8.0 * math.pi * G_Mpc)
    return (3.0 * Mh / (4.0 * math.pi * Delta * rho_c))**(1.0 / 3.0)


def nfw_concentration(Mh: float, z: float, h: float = 0.6774) -> float:
    """NFW concentration c200 from Dutton & Macciò (2014), Eqs. (10)–(11).

    b(z) = −0.101 + 0.026z
    a(z) = 0.520 + (0.905 − 0.520) * exp(−0.617 * z^1.21)
    log10 c200 = a(z) + b(z) * log10(M200 / (10^12 h^-1 Msun))
    """
    if Mh <= 0 or z < 0:
        raise ValueError("Require Mh > 0 and z >= 0.")
    b_z   = -0.101 + 0.026 * z
    a_z   = 0.520 + (0.905 - 0.520) * math.exp(-0.617 * z**1.21)
    log10c = a_z + b_z * math.log10(Mh / (1.0e12 / h))
    return 10.0**log10c


def vesc0_nfw_hernquist(Mh: float, Mb: float, Re_kpc: float, z: float,
                        c: float | None = None) -> float:
    """Central escape speed v_esc(0) [km/s] for an NFW halo + Hernquist bulge.

    Parameters
    ----------
    Mh     : float  -- Halo mass M_200 [Msun].
    Mb     : float  -- Bulge/stellar mass [Msun].
    Re_kpc : float  -- Effective radius [kpc].
    z      : float  -- Redshift.
    c      : float  -- Concentration (uses Dutton & Macciò 2014 if None).
    """
    if not (Mh > 0 and Mb >= 0 and Re_kpc > 0 and z >= 0):
        raise ValueError("Require Mh > 0, Mb >= 0, Re_kpc > 0, z >= 0.")
    G_kpc   = 4.30091e-6  # (km/s)^2 kpc / Msun
    Rvir_kpc = r_vir_mpc(Mh, z) * 1.0e3
    c_val   = nfw_concentration(Mh, z) if c is None else float(c)
    f_c     = math.log(1.0 + c_val) - c_val / (1.0 + c_val)
    Phi_nfw   = G_kpc * Mh / Rvir_kpc * (c_val / f_c)
    a_kpc     = max(1.0e-4, Re_kpc / 1.8153)
    Phi_bulge = G_kpc * Mb / a_kpc
    return math.sqrt(2.0 * (Phi_nfw + Phi_bulge))


# ============================================================================
# 6. FIRE-2 STELLAR–HALO MASS RELATION — MEDIAN (z ≈ 5–12)
#    log10 M_* = alpha * [log10 M_halo − log10 M_pivot] + beta
#    Best-fit (FIRE-2): alpha = 1.58, beta = 7.10
# ============================================================================

@dataclass(frozen=True)
class FIRE2_SHMR_Params:
    """Parameters for the FIRE-2 median stellar–halo mass relation.

    Relation (redshift-independent for z ∈ [5, 12]):
        log10(M_*) = alpha * [log10(M_halo) − log10(M_pivot)] + beta
    """
    alpha:   float = 1.58
    beta:    float = 7.10
    M_pivot: float = 1.0e10  # Msun


def fire2_log10_mstar_from_mh(Mh_msun: float,
                               params: FIRE2_SHMR_Params = FIRE2_SHMR_Params()) -> float:
    """Median FIRE-2 SHMR: log10(M_* / Msun) given M_halo [Msun]."""
    if Mh_msun <= 0.0:
        raise ValueError("Mh_msun must be positive [Msun].")
    return params.alpha * math.log10(Mh_msun / params.M_pivot) + params.beta


def fire2_mstar_from_mh(Mh_msun: float,
                         params: FIRE2_SHMR_Params = FIRE2_SHMR_Params()) -> float:
    """Median FIRE-2 SHMR: M_* [Msun] given M_halo [Msun]."""
    return 10.0 ** fire2_log10_mstar_from_mh(Mh_msun, params=params)


def fire2_log10_mh_from_mstar(Mstar_msun: float,
                               params: FIRE2_SHMR_Params = FIRE2_SHMR_Params()) -> float:
    """Analytic inverse of FIRE-2 SHMR (median): log10(M_halo / Msun) given M_* [Msun]."""
    if Mstar_msun <= 0.0:
        raise ValueError("Mstar_msun must be positive [Msun].")
    return math.log10(params.M_pivot) + (math.log10(Mstar_msun) - params.beta) / params.alpha


def fire2_mh_from_mstar(Mstar_msun: float,
                         params: FIRE2_SHMR_Params = FIRE2_SHMR_Params()) -> float:
    """Analytic inverse of FIRE-2 SHMR (median): M_halo [Msun] given M_* [Msun]."""
    return 10.0 ** fire2_log10_mh_from_mstar(Mstar_msun, params=params)


# ============================================================================
# 7. FIRE-2 GALAXY STELLAR MASS FUNCTION — SCHECHTER FITS (z = 6–12)
#     Fitted to Ma et al. (2018) Appendix C completeness-limited bins via
#     gsmf_fitting.py. At z >= 11, M_char and alpha are shape-locked to z = 10.
#
#     φ(M*) [Mpc^-3 dex^-1] — per-dex single-Schechter form:
#       φ = ln(10) * φ* * (M*/M_char)^(α+1) * exp(−M*/M_char)
# ============================================================================

@dataclass(frozen=True)
class SchechterParams:
    """Single-Schechter GSMF parameters (per-dex form).

    log10_phi_star : log10(φ*) [Mpc^-3 dex^-1].
    log10_M_char   : log10(M_char) [Msun].
    alpha          : faint-end slope.
    """
    log10_phi_star: float
    log10_M_char:  float
    alpha:         float


def schechter_phi_perdex(Mstar_Msun, p: SchechterParams) -> np.ndarray:
    r"""Per-dex Schechter GSMF: φ(M*) [Mpc^-3 dex^-1].

    φ = ln(10) * φ* * (M*/M_char)^(α+1) * exp(−M*/M_char)

    Parameters: Mstar_Msun [Msun] (float or array), p : SchechterParams.
    """
    M        = np.asarray(Mstar_Msun, dtype=float)
    phi_star = 10.0 ** p.log10_phi_star
    M_char   = 10.0 ** p.log10_M_char
    x = M / M_char
    return np.log(10.0) * phi_star * np.power(x, p.alpha + 1.0) * np.exp(-x)


def schechter_log10phi_perdex(log10_Mstar, p: SchechterParams) -> np.ndarray:
    """log10(φ) given log10(M_*/Msun) and Schechter parameters."""
    return np.log10(schechter_phi_perdex(10.0**np.asarray(log10_Mstar, float), p))


# --- Build schechter_params_by_z from fitted parameters loaded from gsmf_fitting.py ---
try:
    from gsmf_fitting import get_gsmf_params as _get_gsmf_params
    _raw_fits = _get_gsmf_params()
    _SCHECHTER_PARAMS_BY_Z: Dict[int, tuple] = {
        z: (d["log10_phi_star"], d["log10_M_char"], d["alpha"])
        for z, d in _raw_fits.items()
    }
except ImportError:
    # Fallback values used when `gsmf_fitting.py` is unavailable.
    _SCHECHTER_PARAMS_BY_Z = {
        6:  (-3.283, 9.398, -1.757),
        7:  (-3.326, 9.047, -1.805),
        8:  (-3.186, 8.578, -1.839),
        9:  (-3.488, 8.457, -1.905),
        10: (-3.494, 8.111, -1.948),
        11: (-3.750, 8.111, -1.948),
        12: (-4.016, 8.111, -1.948),
    }

# Public dict: integer redshift → SchechterParams
schechter_params_by_z: Dict[int, SchechterParams] = {
    z: SchechterParams(*params) for z, params in _SCHECHTER_PARAMS_BY_Z.items()
}

# Arrays used by the PCHIP interpolators
_Z_KNOTS              = np.array(sorted(_SCHECHTER_PARAMS_BY_Z), dtype=float)
_LOG10_PHISTAR_KNOTS  = np.array([_SCHECHTER_PARAMS_BY_Z[int(z)][0] for z in _Z_KNOTS])
_LOG10_MCHAR_KNOTS    = np.array([_SCHECHTER_PARAMS_BY_Z[int(z)][1] for z in _Z_KNOTS])
_ALPHA_KNOTS          = np.array([_SCHECHTER_PARAMS_BY_Z[int(z)][2] for z in _Z_KNOTS])

# Convenience wrappers per integer redshift
def schechter_z6(Mstar_Msun):  return schechter_phi_perdex(Mstar_Msun, schechter_params_by_z[6])
def schechter_z7(Mstar_Msun):  return schechter_phi_perdex(Mstar_Msun, schechter_params_by_z[7])
def schechter_z8(Mstar_Msun):  return schechter_phi_perdex(Mstar_Msun, schechter_params_by_z[8])
def schechter_z9(Mstar_Msun):  return schechter_phi_perdex(Mstar_Msun, schechter_params_by_z[9])
def schechter_z10(Mstar_Msun): return schechter_phi_perdex(Mstar_Msun, schechter_params_by_z[10])
def schechter_z11(Mstar_Msun): return schechter_phi_perdex(Mstar_Msun, schechter_params_by_z[11])
def schechter_z12(Mstar_Msun): return schechter_phi_perdex(Mstar_Msun, schechter_params_by_z[12])

# Underscore-style aliases
schechter_z_6  = schechter_z6
schechter_z_7  = schechter_z7
schechter_z_8  = schechter_z8
schechter_z_9  = schechter_z9
schechter_z_10 = schechter_z10
schechter_z_11 = schechter_z11
schechter_z_12 = schechter_z12


# --- PCHIP interpolators (initialised lazily on first use) ---
_pchip_phi: PchipInterpolator | None = None
_pchip_mchar: PchipInterpolator | None = None
_pchip_alpha: PchipInterpolator | None = None


def _ensure_interpolators(extrapolate: bool = False) -> None:
    """Initialise the global PCHIP interpolators if not yet built."""
    global _pchip_phi, _pchip_mchar, _pchip_alpha
    if _pchip_phi is None:
        _pchip_phi   = PchipInterpolator(_Z_KNOTS, _LOG10_PHISTAR_KNOTS, extrapolate=extrapolate)
        _pchip_mchar = PchipInterpolator(_Z_KNOTS, _LOG10_MCHAR_KNOTS,   extrapolate=extrapolate)
        _pchip_alpha = PchipInterpolator(_Z_KNOTS, _ALPHA_KNOTS,         extrapolate=extrapolate)


def get_schechter_params_at_z(z: float, *, method: str = "pchip",
                               allow_extrapolation: bool = False,
                               atol_exact: float = 1e-9) -> SchechterParams:
    """Interpolate Schechter parameters to an arbitrary redshift z.

    Policy
    ------
    - Exact integer nodes: return the stored best-fit parameters verbatim.
    - z < 10 : PCHIP over all integer knots for all three parameters.
    - z ≥ 10 : shape-lock (M_char, alpha fixed to z = 10); PCHIP only on
               log10(φ*) over the high-z knots {10, 11, 12}.

    Parameters
    ----------
    z                   : float -- Redshift.
    method              : {"pchip","linear"} -- Interpolation type.
    allow_extrapolation : bool  -- If False (default), clamp z to knot range.
    atol_exact          : float -- Tolerance for snapping to an integer knot.
    """
    z = float(z)

    # Return stored values verbatim at integer knots
    z_round = int(round(z))
    if abs(z - z_round) <= atol_exact and z_round in schechter_params_by_z:
        return schechter_params_by_z[z_round]

    _ensure_interpolators(extrapolate=allow_extrapolation)

    if z < 10.0:
        log10_phi = float(_pchip_phi(z))
        log10_Mch = float(_pchip_mchar(z))
        alpha     = float(_pchip_alpha(z))
        return SchechterParams(log10_phi, log10_Mch, alpha)

    # z >= 10: interpolate only log10(phi*) over {10, 11, 12}
    z_hi   = np.array([10.0, 11.0, 12.0])
    phi_hi = np.array([_SCHECHTER_PARAMS_BY_Z[zz][0] for zz in z_hi])
    if method == "pchip":
        log10_phi = float(PchipInterpolator(z_hi, phi_hi, extrapolate=allow_extrapolation)(z))
    else:
        log10_phi = float(np.interp(z, z_hi, phi_hi))
    log10_Mch = schechter_params_by_z[10].log10_M_char
    alpha     = schechter_params_by_z[10].alpha
    return SchechterParams(log10_phi, log10_Mch, alpha)


def schechter_phi_perdex_at_z(Mstar_Msun, z: float, *,
                               method: str = "pchip",
                               allow_extrapolation: bool = False):
    """Evaluate φ(M*, z) [Mpc^-3 dex^-1] using parameter interpolation in z."""
    p = get_schechter_params_at_z(z, method=method, allow_extrapolation=allow_extrapolation)
    return schechter_phi_perdex(Mstar_Msun, p)


def schechter_log10phi_perdex_at_z(log10_Mstar, z: float, *,
                                    method: str = "pchip",
                                    allow_extrapolation: bool = False):
    """Evaluate log10(φ)(log10 M*, z) using parameter interpolation in z."""
    p = get_schechter_params_at_z(z, method=method, allow_extrapolation=allow_extrapolation)
    return schechter_log10phi_perdex(log10_Mstar, p)


def gsmf_number_density_in_bin(z: float, log10M_min: float, log10M_max: float,
                                n_steps: int = 2048, method: str = "pchip",
                                allow_extrapolation: bool = False) -> float:
    """Comoving number density ∫φ d(log10 M*) [Mpc^-3] over a stellar-mass bin at z.

    Integrates the per-dex Schechter GSMF using the trapezoid rule.
    Recommended redshift range: 6 ≤ z ≤ 12.
    """
    lo, hi = float(log10M_min), float(log10M_max)
    if hi <= lo:
        return 0.0
    grid    = np.linspace(lo, hi, int(n_steps))
    log10_phi = schechter_log10phi_perdex_at_z(grid, z, method=method,
                                               allow_extrapolation=allow_extrapolation)
    return np.trapezoid(10.0**log10_phi, grid)


def number_of_galaxies_in_bin_at_z(
    z0: float,
    log10M_min: float,
    log10M_max: float,
    dz: float = 0.1,
    *,
    method: str = "pchip",
    allow_extrapolation: bool = False,
    solid_angle_sr: float = 4.0 * np.pi,
    area_deg2: float = None,
    cosmo=None,
) -> tuple[float, float, float]:
    """Total galaxy count in a stellar-mass bin within a thin redshift shell.

    Steps:
      1. Integrate the GSMF over [log10M_min, log10M_max] → n_bin [Mpc^-3].
      2. Multiply by the comoving shell volume dV_c/dz|z0 * dz * Ω.

    Parameters
    ----------
    z0               : float  -- Central redshift of the shell.
    log10M_min/max   : float  -- Bin edges in log10(M_*/Msun).
    dz               : float  -- Shell thickness in redshift.
    solid_angle_sr   : float  -- Sky coverage [sr]; default full sky (4π).
    area_deg2        : float  -- Overrides solid_angle_sr if provided [deg^2].
    cosmo            : astropy cosmology instance or None (defaults to FIRE-2).

    Returns: (N_galaxies, n_bin [Mpc^-3], V_shell [Mpc^3]).
    """
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM

    n_bin = gsmf_number_density_in_bin(z0, log10M_min, log10M_max,
                                        method=method,
                                        allow_extrapolation=allow_extrapolation)
    if cosmo is None:
        cosmo = _default_cosmo()

    if area_deg2 is not None:
        solid_angle_sr = float(area_deg2) * (np.pi / 180.0)**2

    dV_dz = cosmo.differential_comoving_volume(z0)  # Mpc³ / sr
    V_shell = float((dV_dz * dz * (solid_angle_sr * u.sr)).to(u.Mpc**3).value)
    return float(n_bin * V_shell), float(n_bin), V_shell


# ============================================================================
# 8. COSMOLOGY HELPERS — FLAT ΛCDM (FIRE-2 / PLANCK-LIKE DEFAULTS)
#     H0 = 67.74 km/s/Mpc, Ωm = 0.31, Ωb = 0.048
# ============================================================================

def _default_cosmo():
    """Return a flat ΛCDM cosmology matching the FIRE-2 parameter choices."""
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    return FlatLambdaCDM(H0=67.74, Om0=0.31, Ob0=0.048, Tcmb0=2.725 * u.K)


def age_of_universe_at_z(z, cosmo=None) -> np.ndarray:
    """Cosmic age at redshift z [years]. Accepts scalar or array input."""
    from astropy import units as u
    if cosmo is None:
        cosmo = _default_cosmo()
    return cosmo.age(np.asarray(z, dtype=float)).to(u.yr).value


def time_until_z6(z, cosmo=None) -> np.ndarray:
    """Time interval [years] from redshift z down to z = 6: Δt = t(z=6) − t(z).

    Clamped to zero for z < 6 (no negative time intervals).
    """
    from astropy import units as u
    if cosmo is None:
        cosmo = _default_cosmo()
    z = np.asarray(z, dtype=float)
    t_z = cosmo.age(z).to(u.yr).value
    t_6 = cosmo.age(6.0).to(u.yr).value
    return np.maximum(t_6 - t_z, 0.0)


# ============================================================================
# 9. HIGH-z MAJOR-MERGER RATE — DUAN ET AL. (2025)
#     R_M(z) = f0 * (1+z)^m * exp[tau * (1+z)]  [Gyr^-1]
#     Parameters (Table 4, JWST + Casteels zero-point):
#       f0 = 0.013,  m = 3.359,  tau = -0.138
#     Calibrated for major mergers with log10(M_*/Msun) ∈ [8, 10].
# ============================================================================

@dataclass(frozen=True)
class MergerRateFit:
    """Parameters for the Duan+2025 merger rate fit.

    R_M(z) = f0 * (1+z)^m * exp[tau * (1+z)]  [Gyr^-1]
    """
    f0:  float
    m:   float
    tau: float


_MERGER_RATE = MergerRateFit(f0=0.013, m=3.359, tau=-0.138)


def merger_rate_z(z) -> np.ndarray:
    """Per-galaxy major-merger rate R_M(z) [Gyr^-1] from Duan+2025 (Eq. 20).

    Parameters: z (float or array-like). Returns R_M [Gyr^-1].
    """
    z = np.asarray(z, dtype=float)
    p = _MERGER_RATE
    return p.f0 * np.power(1.0 + z, p.m) * np.exp(p.tau * (1.0 + z))
