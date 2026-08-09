"""
merger_pair_sampling.py
-----------------------
Shared population-building helpers for the ratio-scan catalogue and the final
simulation.  The module constructs explicit pairs of merging galaxies, maps
both progenitors to central black holes, and returns the merged galaxy and BH
properties used by the recoil/TDE pipeline.

The event-rate normalisation is deliberately split into two parts:

- Duan et al. (2025) supplies the high-redshift major-merger rate.
- Rodriguez-Gomez et al. (2015) supplies only the relative stellar-mass-ratio
  dependence, extrapolated to the redshifts and masses used here.

Galaxy mergers and BH coalescences are treated as occurring at the same
snapshot. The GSMF, merger rate, descendant properties, recoil, and TDE
calculations therefore use one common redshift convention throughout.

Every physical context is cap-sampled once.  A saved row therefore carries the
single physical weight n_phys / n_samp used by all downstream weighted sums.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import physics_relations as pr

_TRAPZ = getattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# Canonical population limits and bin definitions
# ---------------------------------------------------------------------------

MBH_MIN = 1.0e3
MBH_MAX = 1.0e6
MBH_POST_MAX = 2.0e6

RV15_PARAMS = pr.RV15AGNParams()

LOG10_MSTAR_MIN = float(pr.log10_mstar_from_mbh(MBH_MIN, params=RV15_PARAMS))
LOG10_MSTAR_MAX = float(pr.log10_mstar_from_mbh(MBH_MAX, params=RV15_PARAMS))
MSTAR_MIN = 10.0 ** LOG10_MSTAR_MIN
MSTAR_MAX = 10.0 ** LOG10_MSTAR_MAX

# A single physical stellar-mass grid is used throughout the pipeline.  The
# progenitor masses remain continuous integration/sampling variables, whereas
# all physical targets and downstream grouping are indexed by descendant mass.
LOG10_MSTAR_DESC_MIN = float(np.log10(2.0 * MSTAR_MIN))
LOG10_MSTAR_DESC_MAX = float(np.log10(2.0 * MSTAR_MAX))
DESCENDANT_LOGM_EDGES = np.linspace(
    LOG10_MSTAR_DESC_MIN,
    LOG10_MSTAR_DESC_MAX,
    10,
    dtype=float,
)

MERGER_MAJOR = "major"
MERGER_MINOR = "minor"
MERGER_CLASSES = (MERGER_MAJOR, MERGER_MINOR)
MU_MAJOR_MIN = 0.25

# Written into every catalogue so downstream scripts cannot silently mix an
# older population model with the revised major-plus-minor calculation.
MODEL_VERSION = "descendant_target_duan_major_rg_minor_no_delay_cap100k_v5"

F_BULGE = 0.1548


@dataclass(frozen=True)
class SamplingControls:
    """Numerical controls shared by both catalogue generators."""

    max_events_per_context: int = 100_000
    integration_steps: int = 2048
    max_tries_multiplier: int = 200
    max_tries_per_context: int = 10_000_000


DEFAULT_CONTROLS = SamplingControls()


# ---------------------------------------------------------------------------
# Bin helpers
# ---------------------------------------------------------------------------

def bin_pairs(edges: np.ndarray) -> list[tuple[float, float]]:
    """Return consecutive (lower, upper) bin-edge pairs."""
    arr = np.asarray(edges, dtype=float)
    return [(float(lo), float(hi)) for lo, hi in zip(arr[:-1], arr[1:])]



def descendant_bin_pairs() -> list[tuple[float, float]]:
    """Return the merged-galaxy stellar-mass bins."""
    return bin_pairs(DESCENDANT_LOGM_EDGES)


def bin_label(lo: float, hi: float) -> str:
    """Human-readable mass-bin label used by the JSON and figure outputs."""
    return f"[{float(lo):.2f}, {float(hi):.2f}]"


def descendant_bin_index(log10_mstar: float) -> int | None:
    """Return the descendant-bin index, including the exact final upper edge."""
    value = float(log10_mstar)
    if not np.isfinite(value):
        return None

    if np.isclose(value, DESCENDANT_LOGM_EDGES[-1], rtol=0.0, atol=1e-10):
        return len(DESCENDANT_LOGM_EDGES) - 2

    idx = int(np.searchsorted(DESCENDANT_LOGM_EDGES, value, side="right") - 1)
    if 0 <= idx < len(DESCENDANT_LOGM_EDGES) - 1:
        return idx
    return None


# ---------------------------------------------------------------------------
# Merger-rate and sampling distributions
# ---------------------------------------------------------------------------

def mu_bounds_for_descendant_bin(
    mstar_primary_msun,
    descendant_logm_lo: float,
    descendant_logm_hi: float,
    merger_class: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorised mu bounds for one descendant bin and merger class.

    Bounds simultaneously enforce the component stellar-mass floor, the
    primary >= secondary convention, the major/minor definition, and
    Mstar_desc = Mstar_primary * (1 + mu) inside the requested descendant bin.
    Empty intervals are marked by the returned boolean mask.
    """
    m1 = np.asarray(mstar_primary_msun, dtype=float)
    desc_lo = 10.0 ** float(descendant_logm_lo)
    desc_hi = 10.0 ** float(descendant_logm_hi)

    mu_floor = np.divide(
        MSTAR_MIN,
        m1,
        out=np.full_like(m1, np.inf, dtype=float),
        where=np.isfinite(m1) & (m1 > 0.0),
    )
    mu_desc_lo = desc_lo / m1 - 1.0
    mu_desc_hi = desc_hi / m1 - 1.0

    if merger_class == MERGER_MAJOR:
        lo = np.maximum.reduce([
            mu_floor,
            np.full_like(m1, MU_MAJOR_MIN),
            mu_desc_lo,
            np.zeros_like(m1),
        ])
        hi = np.minimum.reduce([
            np.ones_like(m1),
            mu_desc_hi,
        ])
    elif merger_class == MERGER_MINOR:
        lo = np.maximum.reduce([
            mu_floor,
            mu_desc_lo,
            np.zeros_like(m1),
        ])
        hi = np.minimum.reduce([
            np.full_like(m1, MU_MAJOR_MIN),
            mu_desc_hi,
            np.ones_like(m1),
        ])
    else:
        raise ValueError(f"Unknown merger class: {merger_class!r}")

    valid = (
        np.isfinite(m1) & (m1 >= MSTAR_MIN) & (m1 <= MSTAR_MAX)
        & np.isfinite(lo) & np.isfinite(hi) & (lo > 0.0) & (hi > lo)
    )
    return lo, hi, valid


def _mu_kernel(mstar_primary_msun, mu, z_rate: float) -> np.ndarray:
    """Rodriguez-Gomez relative dR/dmu kernel using descendant stellar mass."""
    m1, ratio = np.broadcast_arrays(
        np.asarray(mstar_primary_msun, dtype=float),
        np.asarray(mu, dtype=float),
    )
    mdesc = m1 * (1.0 + ratio)
    exponent = np.asarray(
        pr.rodriguez_gomez_mu_exponent(mdesc, float(z_rate)),
        dtype=float,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        kernel = np.power(ratio, exponent)
    return np.where(
        np.isfinite(kernel) & np.isfinite(ratio) & (ratio > 0.0),
        kernel,
        0.0,
    )


def _mu_quadrature_grid(n_steps: int) -> np.ndarray:
    """Shared mu grid resolving both the minor and major intervals."""
    n = max(256, int(n_steps))
    mu_global_min = MSTAR_MIN / MSTAR_MAX
    n_minor = max(128, n // 2)
    n_major = max(128, n - n_minor + 1)
    minor = np.geomspace(mu_global_min, MU_MAJOR_MIN, n_minor, dtype=float)
    major = np.linspace(MU_MAJOR_MIN, 1.0, n_major, dtype=float)
    return np.unique(np.concatenate([minor, major]))


def _cumulative_trapezoid_axis1(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral along axis 1 with a leading zero."""
    values = np.asarray(y, dtype=float)
    grid = np.asarray(x, dtype=float)
    out = np.zeros_like(values, dtype=float)
    if grid.size > 1:
        out[:, 1:] = np.cumsum(
            0.5 * (values[:, :-1] + values[:, 1:]) * np.diff(grid)[None, :],
            axis=1,
        )
    return out


def _rowwise_cumulative_at(
    cumulative: np.ndarray,
    grid: np.ndarray,
    x_values: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate row-wise cumulative integrals on a common grid."""
    cdf = np.asarray(cumulative, dtype=float)
    x = np.asarray(grid, dtype=float)
    values = np.asarray(x_values, dtype=float)
    clipped = np.clip(values, x[0], x[-1])
    right = np.searchsorted(x, clipped, side="right")
    right = np.clip(right, 1, len(x) - 1)
    left = right - 1
    rows = np.arange(cdf.shape[0])
    x0 = x[left]
    x1 = x[right]
    frac = np.divide(
        clipped - x0,
        x1 - x0,
        out=np.zeros_like(clipped),
        where=(x1 > x0),
    )
    return cdf[rows, left] + frac * (cdf[rows, right] - cdf[rows, left])


def _integral_between_bounds(
    cumulative: np.ndarray,
    grid: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Evaluate row-wise kernel integrals between arbitrary bounds."""
    lower = _rowwise_cumulative_at(cumulative, grid, lo)
    upper = _rowwise_cumulative_at(cumulative, grid, hi)
    result = np.where(valid, np.maximum(upper - lower, 0.0), 0.0)
    return np.where(np.isfinite(result), result, 0.0)

def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral with a leading zero."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    if len(x) > 1:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out


def _normalised_cdf(x: np.ndarray, density: np.ndarray) -> np.ndarray | None:
    """Return a monotonic [0,1] CDF for a non-negative tabulated density."""
    x = np.asarray(x, dtype=float)
    density = np.asarray(density, dtype=float)
    density = np.where(np.isfinite(density) & (density > 0.0), density, 0.0)
    cdf = _cumulative_trapezoid(density, x)
    total = float(cdf[-1]) if len(cdf) else 0.0
    if not np.isfinite(total) or total <= 0.0:
        return None
    cdf /= total
    cdf[-1] = 1.0
    return cdf


def sample_from_tabulated_cdf(grid: np.ndarray, cdf: np.ndarray, rng) -> float:
    """Inverse-CDF draw from a tabulated one-dimensional distribution."""
    u = float(rng.random())
    return float(np.interp(u, cdf, grid))


def sample_truncated_powerlaw(mu_lo: float, mu_hi: float, exponent: float, rng) -> float:
    """Draw mu from p(mu) proportional to mu**exponent on [mu_lo, mu_hi]."""
    lo = float(mu_lo)
    hi = float(mu_hi)
    s = float(exponent)
    if not (0.0 < lo < hi <= 1.0):
        raise ValueError(f"Invalid mu interval [{lo}, {hi}]")

    u = float(rng.random())
    p = s + 1.0
    if abs(p) < 1e-10:
        return float(lo * np.exp(u * np.log(hi / lo)))

    lo_p = lo ** p
    hi_p = hi ** p
    value = (lo_p + u * (hi_p - lo_p)) ** (1.0 / p)
    return float(np.clip(value, lo, np.nextafter(hi, lo)))


# ---------------------------------------------------------------------------
# Vectorised event drawing
# ---------------------------------------------------------------------------

def _sample_from_tabulated_cdf_batch(
    grid: np.ndarray,
    cdf: np.ndarray,
    rng,
    size: int,
) -> np.ndarray:
    """Draw several values from a tabulated CDF in one NumPy operation."""
    return np.interp(rng.random(int(size)), cdf, grid)


def _sample_truncated_powerlaw_batch(
    mu_lo: np.ndarray,
    mu_hi: np.ndarray,
    exponent: np.ndarray,
    rng,
) -> np.ndarray:
    """Vectorised inverse-CDF draw from p(mu) proportional to mu**s."""
    lo, hi, s = np.broadcast_arrays(
        np.asarray(mu_lo, dtype=float),
        np.asarray(mu_hi, dtype=float),
        np.asarray(exponent, dtype=float),
    )
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(~np.isfinite(s)):
        raise ValueError("Non-finite values in truncated power-law inputs.")
    if np.any((lo <= 0.0) | (hi <= lo) | (hi > 1.0)):
        raise ValueError("Invalid interval in vectorised mu draw.")

    u = rng.random(lo.size).reshape(lo.shape)
    p = s + 1.0
    log_case = np.abs(p) < 1.0e-10
    values = np.empty_like(lo, dtype=float)

    values[log_case] = lo[log_case] * np.exp(
        u[log_case] * np.log(hi[log_case] / lo[log_case])
    )
    regular = ~log_case
    lo_p = np.power(lo[regular], p[regular])
    hi_p = np.power(hi[regular], p[regular])
    values[regular] = np.power(
        lo_p + u[regular] * (hi_p - lo_p),
        1.0 / p[regular],
    )

    # Keep a nominally open upper boundary inside the requested class.  This
    # matters only for the minor interval, where mu=0.25 belongs to the major
    # class; it also protects against round-off at any other upper edge.
    upper_inside = np.nextafter(hi, lo)
    return np.clip(values, lo, upper_inside)


def _sample_mu_kernel_batch(
    mstar_primary: np.ndarray,
    mu_lo: np.ndarray,
    mu_hi: np.ndarray,
    z_rate: float,
    rng,
) -> np.ndarray:
    """Draw mu exactly from the descendant-mass Rodriguez-Gomez kernel.

    The target kernel can be factored as

        mu**s(M1 * (1 + mu), z)
        = mu**s(M1, z) * exp[c ln(1 + mu) ln(mu)],

    where c = gamma / ln(10).  We draw from the first (truncated power-law)
    factor and accept with the second factor.  The rejection envelope uses
    ln(1+mu) <= mu and max[-mu ln(mu)] = 1/e, so it is a strict global bound
    over 0 < mu <= 1.  Acceptance is close to unity and no per-event CDF grid
    is required.
    """
    m1, lo, hi = np.broadcast_arrays(
        np.asarray(mstar_primary, dtype=float),
        np.asarray(mu_lo, dtype=float),
        np.asarray(mu_hi, dtype=float),
    )
    if np.any(~np.isfinite(m1)) or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError("Non-finite inputs in conditional mu sampling.")
    if np.any((lo <= 0.0) | (hi <= lo) | (hi > 1.0)):
        raise ValueError("Invalid bounds in conditional mu sampling.")

    proposal_exponent = np.asarray(
        pr.rodriguez_gomez_mu_exponent(m1, float(z_rate)),
        dtype=float,
    )
    gamma = float(pr.RodriguezGomezMuParams().gamma)
    c_corr = gamma / np.log(10.0)
    log_envelope = abs(c_corr) / np.e

    out = np.empty_like(m1, dtype=float)
    accepted = np.zeros_like(m1, dtype=bool)
    while not np.all(accepted):
        idx = np.flatnonzero(~accepted)
        proposal = _sample_truncated_powerlaw_batch(
            lo[idx], hi[idx], proposal_exponent[idx], rng
        )
        log_correction = c_corr * np.log1p(proposal) * np.log(proposal)
        accept_probability = np.exp(log_correction - log_envelope)
        take = rng.random(len(idx)) < accept_probability
        if np.any(take):
            chosen = idx[take]
            out[chosen] = proposal[take]
            accepted[chosen] = True

    return out


def _descendant_bin_indices(log10_mstar: np.ndarray) -> np.ndarray:
    """Vectorised descendant-bin lookup, including the exact final edge."""
    values = np.asarray(log10_mstar, dtype=float)
    idx = np.searchsorted(DESCENDANT_LOGM_EDGES, values, side="right") - 1
    on_last = np.isclose(
        values,
        DESCENDANT_LOGM_EDGES[-1],
        rtol=0.0,
        atol=1.0e-10,
    )
    idx[on_last] = len(DESCENDANT_LOGM_EDGES) - 2
    return idx.astype(np.int16, copy=False)


def _vectorised_descendant_properties(
    mstar_rem: np.ndarray,
    z_snapshot: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Re, log10(Mhalo), and central escape speed for many descendants.

    These expressions are algebraically identical to the scalar helpers in
    physics_relations.py.  Keeping them in array form removes the expensive
    Python function call for every event without changing the adopted model.
    """
    mstar = np.asarray(mstar_rem, dtype=float)
    z = float(z_snapshot)

    size_params = pr.M24SizeParams()
    log10_re = (
        size_params.alpha * np.log10(mstar / size_params.M0)
        + size_params.beta_z * np.log10(1.0 + z)
        + size_params.alpha_z
    )
    re_kpc = np.power(10.0, log10_re)

    shmr = pr.FIRE2_SHMR_Params()
    log10_mh = (
        np.log10(shmr.M_pivot)
        + (np.log10(mstar) - shmr.beta) / shmr.alpha
    )
    mh_msun = np.power(10.0, log10_mh)
    mb_msun = F_BULGE * mstar

    # Vectorised copy of r_vir_mpc(), nfw_concentration(), and
    # vesc0_nfw_hernquist() using the same constants and formulae.
    H0 = 67.74
    omega_m = 0.31
    omega_l = 1.0 - omega_m
    hz = H0 * np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l)
    g_mpc = 4.30091e-6 * 1.0e-3
    rho_c = 3.0 * hz**2 / (8.0 * np.pi * g_mpc)
    rvir_kpc = np.power(
        3.0 * mh_msun / (4.0 * np.pi * 200.0 * rho_c),
        1.0 / 3.0,
    ) * 1.0e3

    h = 0.6774
    b_z = -0.101 + 0.026 * z
    a_z = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    log10_c = a_z + b_z * np.log10(mh_msun / (1.0e12 / h))
    concentration = np.power(10.0, log10_c)
    f_c = np.log1p(concentration) - concentration / (1.0 + concentration)

    g_kpc = 4.30091e-6
    phi_nfw = g_kpc * mh_msun / rvir_kpc * (concentration / f_c)
    a_kpc = np.maximum(1.0e-4, re_kpc / 1.8153)
    phi_bulge = g_kpc * mb_msun / a_kpc
    vesc0_kms = np.sqrt(2.0 * (phi_nfw + phi_bulge))

    return re_kpc, log10_mh, vesc0_kms


def _draw_galaxy_pair_batch(
    context: dict,
    z_snapshot: float,
    rng,
    size: int,
) -> pd.DataFrame:
    """Draw a validated batch for one descendant-bin/class context."""
    n_draw = int(size)
    if n_draw <= 0:
        return pd.DataFrame()

    merger_class = str(context["merger_class"])
    log_mstar_primary = _sample_from_tabulated_cdf_batch(
        context["primary_logm_grid"],
        context["primary_cdf"],
        rng,
        n_draw,
    )
    mstar_primary = np.power(10.0, log_mstar_primary)
    mu_lo, mu_hi, interval_ok = mu_bounds_for_descendant_bin(
        mstar_primary,
        context["descendant_bin_lo_log10M"],
        context["descendant_bin_hi_log10M"],
        merger_class,
    )
    if not np.any(interval_ok):
        return pd.DataFrame()

    keep0 = np.flatnonzero(interval_ok)
    mstar_primary = mstar_primary[keep0]
    mu_lo = mu_lo[keep0]
    mu_hi = mu_hi[keep0]
    mu_star = _sample_mu_kernel_batch(
        mstar_primary,
        mu_lo,
        mu_hi,
        context["z_rate"],
        rng,
    )
    mstar_secondary = mu_star * mstar_primary

    mbh_primary = np.asarray(pr.mbh_from_mstar(mstar_primary, params=RV15_PARAMS), dtype=float)
    mbh_secondary = np.asarray(pr.mbh_from_mstar(mstar_secondary, params=RV15_PARAMS), dtype=float)
    m1_bh = np.minimum(mbh_primary, mbh_secondary)
    m2_bh = np.maximum(mbh_primary, mbh_secondary)
    q = m1_bh / m2_bh

    eta_value = q / np.square(1.0 + q)
    a2, a3, a4 = 0.5610, -0.847, 3.145
    linear = 1.0 - 2.0 * np.sqrt(2.0) / 3.0
    f_rad = (
        a4 * eta_value**4
        + a3 * eta_value**3
        + a2 * eta_value**2
        + linear * eta_value
    )
    mrem_bh = np.minimum((m1_bh + m2_bh) * (1.0 - f_rad), MBH_POST_MAX)

    mstar_rem = mstar_primary + mstar_secondary
    log_mstar_rem = np.log10(mstar_rem)
    desc_idx = _descendant_bin_indices(log_mstar_rem)
    re_kpc, log10_mh, vesc0_kms = _vectorised_descendant_properties(
        mstar_rem,
        z_snapshot,
    )

    finite_columns = np.column_stack([
        mstar_primary, mstar_secondary, mu_star, m1_bh, m2_bh, q,
        mrem_bh, mstar_rem, re_kpc, log10_mh, vesc0_kms,
    ])
    valid = np.all(np.isfinite(finite_columns), axis=1)
    valid &= (mstar_primary >= MSTAR_MIN * (1.0 - 1.0e-8))
    valid &= (mstar_primary <= MSTAR_MAX * (1.0 + 1.0e-8))
    valid &= (mstar_secondary >= MSTAR_MIN * (1.0 - 1.0e-8))
    valid &= (mstar_secondary <= MSTAR_MAX * (1.0 + 1.0e-8))
    valid &= (mstar_primary >= mstar_secondary * (1.0 - 1.0e-12))
    valid &= (mu_star > 0.0) & (mu_star <= 1.0)
    if merger_class == MERGER_MAJOR:
        valid &= mu_star >= MU_MAJOR_MIN - 1.0e-12
    else:
        valid &= mu_star < MU_MAJOR_MIN
    valid &= (m1_bh >= MBH_MIN * (1.0 - 1.0e-8))
    valid &= (m2_bh <= MBH_MAX * (1.0 + 1.0e-8))
    valid &= (m1_bh <= m2_bh) & (q > 0.0) & (q <= 1.0)
    valid &= (mrem_bh > 0.0) & (re_kpc > 0.0) & (vesc0_kms > 0.0)
    valid &= desc_idx == int(context["descendant_bin_index"])
    valid &= np.isclose(
        mstar_rem, mstar_primary + mstar_secondary, rtol=1.0e-12, atol=0.0
    )

    if not np.any(valid):
        return pd.DataFrame()

    mstar_primary = mstar_primary[valid]
    mstar_secondary = mstar_secondary[valid]
    mu_star = mu_star[valid]
    m1_bh = m1_bh[valid]
    m2_bh = m2_bh[valid]
    q = q[valid]
    mrem_bh = mrem_bh[valid]
    f_rad = f_rad[valid]
    mstar_rem = mstar_rem[valid]
    re_kpc = re_kpc[valid]
    log10_mh = log10_mh[valid]
    vesc0_kms = vesc0_kms[valid]
    n_valid = len(mstar_primary)
    desc_lo = float(context["descendant_bin_lo_log10M"])
    desc_hi = float(context["descendant_bin_hi_log10M"])

    return pd.DataFrame({
        "z": np.full(n_valid, float(z_snapshot)),
        "z_rate": np.full(n_valid, float(context["z_rate"])),
        "population_model_version": np.full(n_valid, MODEL_VERSION, dtype=object),
        "merger_class": np.full(n_valid, merger_class, dtype=object),
        "descendant_bin_lo_log10M": np.full(n_valid, desc_lo),
        "descendant_bin_hi_log10M": np.full(n_valid, desc_hi),
        "desc_bin_lo_log10M": np.full(n_valid, desc_lo),
        "desc_bin_hi_log10M": np.full(n_valid, desc_hi),
        "bin_lo_log10M": np.full(n_valid, desc_lo),
        "bin_hi_log10M": np.full(n_valid, desc_hi),
        "Mstar_primary_Msun": mstar_primary,
        "Mstar_secondary_Msun": mstar_secondary,
        "mu_star": mu_star,
        "Mstar_rem_Msun": mstar_rem,
        "m1_BH_Msun": m1_bh,
        "m2_BH_Msun": m2_bh,
        "q": q,
        "Mrem_BH_Msun": mrem_bh,
        "f_rad_bh_merger": f_rad,
        "Re_kpc": re_kpc,
        "log10_Mh_fire2": log10_mh,
        "Vesc0_kms": vesc0_kms,
        "weight": np.full(n_valid, float(context["weight"])),
    })


# ---------------------------------------------------------------------------
# Deterministic target construction
# ---------------------------------------------------------------------------

def build_snapshot_contexts(
    z_snapshot: float,
    z_rate: float,
    dt_gyr: float,
    shell_volume_mpc3: float,
    *,
    min_primary_count: float = 1.0,
    controls: SamplingControls = DEFAULT_CONTROLS,
) -> tuple[list[dict], list[dict]]:
    """Build descendant-bin physical targets and matching sampling CDFs.

    The deterministic integral is two-dimensional in primary progenitor mass
    and stellar-mass ratio.  Duan supplies the absolute major rate.  At each
    primary mass, the descendant-mass-dependent Rodriguez-Gomez kernel is
    normalised over the *valid* major interval so that it integrates to the
    Duan rate; the same normalisation extends the kernel into the minor regime.
    """
    contexts: list[dict] = []
    target_rows: list[dict] = []

    z_snapshot = float(z_snapshot)
    z_rate = float(z_rate)
    dt_gyr = float(dt_gyr)
    shell_volume_mpc3 = float(shell_volume_mpc3)

    n_primary_steps = int(controls.integration_steps)
    log_grid = np.linspace(LOG10_MSTAR_MIN, LOG10_MSTAR_MAX, n_primary_steps)
    mstar_grid = np.power(10.0, log_grid)
    phi = np.asarray(
        pr.schechter_phi_perdex_at_z(
            mstar_grid,
            z_snapshot,
            method="pchip",
            allow_extrapolation=True,
        ),
        dtype=float,
    )
    phi = np.where(np.isfinite(phi) & (phi > 0.0), phi, 0.0)
    n_primary_density = float(_TRAPZ(phi, log_grid))
    n_primary_total = n_primary_density * shell_volume_mpc3
    if not np.isfinite(n_primary_total) or n_primary_total < float(min_primary_count):
        return contexts, target_rows

    mu_grid = _mu_quadrature_grid(controls.integration_steps)
    kernel = _mu_kernel(mstar_grid[:, None], mu_grid[None, :], z_rate)
    cumulative = _cumulative_trapezoid_axis1(kernel, mu_grid)

    mu_floor = MSTAR_MIN / mstar_grid
    major_lo = np.maximum(mu_floor, MU_MAJOR_MIN)
    major_hi = np.ones_like(major_lo)
    major_valid = major_hi > major_lo
    major_norm = _integral_between_bounds(
        cumulative, mu_grid, major_lo, major_hi, major_valid
    )
    major_rate = float(np.asarray(pr.merger_rate_z(z_rate)))

    expected_by_class: dict[str, list[float]] = {
        cls: [] for cls in MERGER_CLASSES
    }
    rate_density_by_context: dict[tuple[int, str], np.ndarray] = {}

    for desc_idx, (desc_lo, desc_hi) in enumerate(descendant_bin_pairs()):
        for merger_class in MERGER_CLASSES:
            mu_lo, mu_hi, interval_valid = mu_bounds_for_descendant_bin(
                mstar_grid, desc_lo, desc_hi, merger_class
            )
            numerator = _integral_between_bounds(
                cumulative, mu_grid, mu_lo, mu_hi, interval_valid
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                class_rate = np.divide(
                    major_rate * numerator,
                    major_norm,
                    out=np.zeros_like(numerator),
                    where=major_valid & np.isfinite(major_norm) & (major_norm > 0.0),
                )
            event_density = phi * np.where(
                np.isfinite(class_rate) & (class_rate > 0.0), class_rate, 0.0
            )
            expected = shell_volume_mpc3 * dt_gyr * float(_TRAPZ(event_density, log_grid))
            expected = float(expected) if np.isfinite(expected) and expected > 0.0 else 0.0
            expected_by_class[merger_class].append(expected)
            rate_density_by_context[(desc_idx, merger_class)] = event_density

    # Independent full-range totals validate that descendant bins partition the
    # complete accessible pair domain without double-counting.
    full_major_rate = np.where(major_valid & (major_norm > 0.0), major_rate, 0.0)
    minor_lo = mu_floor
    minor_hi = np.full_like(mu_floor, MU_MAJOR_MIN)
    minor_valid = minor_hi > minor_lo
    minor_integral = _integral_between_bounds(
        cumulative, mu_grid, minor_lo, minor_hi, minor_valid
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        full_minor_rate = np.divide(
            major_rate * minor_integral,
            major_norm,
            out=np.zeros_like(minor_integral),
            where=major_valid & np.isfinite(major_norm) & (major_norm > 0.0),
        )

    independent_totals = {
        MERGER_MAJOR: shell_volume_mpc3 * dt_gyr * float(_TRAPZ(phi * full_major_rate, log_grid)),
        MERGER_MINOR: shell_volume_mpc3 * dt_gyr * float(_TRAPZ(phi * full_minor_rate, log_grid)),
    }
    for merger_class in MERGER_CLASSES:
        binned_total = float(np.sum(expected_by_class[merger_class]))
        independent = float(independent_totals[merger_class])
        if not np.isclose(binned_total, independent, rtol=2.0e-5, atol=1.0e-6):
            raise RuntimeError(
                "Descendant-bin target partition failed: "
                f"z={z_snapshot:.2f}, class={merger_class}, "
                f"sum_bins={binned_total:.12e}, independent={independent:.12e}."
            )

    # Convert floating expectations to integer physical targets while preserving
    # each independently integrated class total exactly.  Largest-remainder
    # allocation avoids a small cumulative drift from rounding nine bins
    # separately.
    rounded_by_class: dict[str, np.ndarray] = {}
    for merger_class in MERGER_CLASSES:
        raw = np.asarray(expected_by_class[merger_class], dtype=float)
        floors = np.floor(raw).astype(np.int64)
        total_target = int(np.round(independent_totals[merger_class]))
        remainder = total_target - int(np.sum(floors))
        if remainder < 0 or remainder > len(raw):
            raise RuntimeError(
                f"Invalid target-rounding remainder for {merger_class}: {remainder}"
            )
        rounded = floors.copy()
        if remainder:
            order = np.argsort(-(raw - floors), kind="stable")
            rounded[order[:remainder]] += 1
        if int(np.sum(rounded)) != total_target:
            raise RuntimeError(f"Rounded target audit failed for {merger_class}.")
        rounded_by_class[merger_class] = rounded

    for desc_idx, (desc_lo, desc_hi) in enumerate(descendant_bin_pairs()):
        for merger_class in MERGER_CLASSES:
            expected = float(expected_by_class[merger_class][desc_idx])
            n_phys = int(rounded_by_class[merger_class][desc_idx])
            n_samp = min(n_phys, int(controls.max_events_per_context)) if n_phys > 0 else 0
            weight = float(n_phys) / float(n_samp) if n_samp > 0 else 0.0

            # Keep an explicit zero row for unavailable/empty contexts so the
            # physical target table has all nine bins for both merger classes.
            target_rows.append({
                "redshift": z_snapshot,
                "z": z_snapshot,
                "z_rate": z_rate,
                "population_model_version": MODEL_VERSION,
                "descendant_bin_lo": float(desc_lo),
                "descendant_bin_hi": float(desc_hi),
                "descendant_bin_lo_log10M": float(desc_lo),
                "descendant_bin_hi_log10M": float(desc_hi),
                "bin_lo_log10M": float(desc_lo),
                "bin_hi_log10M": float(desc_hi),
                "merger_class": merger_class,
                "expected_mergers_float": expected,
                "physical_target": int(n_phys),
                "simulated_events": int(n_samp),
                "n_events_phys": int(n_phys),
                "n_events_samp": int(n_samp),
                "weight": weight,
                "rounding_method": "largest_remainder_by_redshift_and_class",
                "independent_class_total_float": float(independent_totals[merger_class]),
                "independent_class_total_rounded": int(np.round(independent_totals[merger_class])),
            })

            if n_samp <= 0:
                continue
            event_density = rate_density_by_context[(desc_idx, merger_class)]
            cdf = _normalised_cdf(log_grid, event_density)
            if cdf is None:
                raise RuntimeError(
                    f"Missing sampling CDF for non-zero target: bin={desc_idx}, "
                    f"class={merger_class}, target={n_phys}."
                )

            context = {
                "z": z_snapshot,
                "z_rate": z_rate,
                "population_model_version": MODEL_VERSION,
                "descendant_bin_index": int(desc_idx),
                "descendant_bin_lo_log10M": float(desc_lo),
                "descendant_bin_hi_log10M": float(desc_hi),
                "merger_class": merger_class,
                "expected_mergers_float": expected,
                "n_events_phys": int(n_phys),
                "n_events_samp": int(n_samp),
                "weight": weight,
                "primary_logm_grid": log_grid,
                "primary_cdf": cdf,
            }
            contexts.append(context)

    return contexts, target_rows


# ---------------------------------------------------------------------------
# Pair construction and validation
# ---------------------------------------------------------------------------

def _is_within(value: float, lo: float, hi: float, *, atol: float = 1e-8) -> bool:
    return (value >= lo * (1.0 - atol)) and (value <= hi * (1.0 + atol))


def draw_galaxy_pair_event(context: dict, z_snapshot: float, rng) -> dict | None:
    """Draw and validate one merger event for a descendant-bin context."""
    batch = _draw_galaxy_pair_batch(context, z_snapshot, rng, 1)
    if batch.empty:
        return None
    return batch.iloc[0].to_dict()


def context_key(context: dict) -> tuple[float, float, str]:
    """Stable identifier for a descendant-bin/merger-class context."""
    return (
        float(context["descendant_bin_lo_log10M"]),
        float(context["descendant_bin_hi_log10M"]),
        str(context["merger_class"]),
    )


def fill_snapshot_contexts(
    contexts: Iterable[dict],
    z_snapshot: float,
    rng,
    *,
    controls: SamplingControls = DEFAULT_CONTROLS,
) -> tuple[pd.DataFrame, dict]:
    """Fill every capped descendant-bin/class context exactly.

    Physically unavailable pair regions were removed from the target integral.
    A numerical or Monte Carlo failure therefore triggers another batch and
    never changes the requested physical target.
    """
    frames: list[pd.DataFrame] = []
    summary: dict = {}

    for context in contexts:
        n_goal = int(context["n_events_samp"])
        if n_goal <= 0:
            continue

        key = context_key(context)
        filled = 0
        attempts = 0
        max_attempts = min(
            int(controls.max_tries_per_context),
            max(10_000, int(controls.max_tries_multiplier) * n_goal),
        )

        while filled < n_goal and attempts < max_attempts:
            remaining = n_goal - filled
            # Direct inverse-CDF draws are normally all valid.  A small margin
            # avoids an extra loop when a handful of boundary rows are rejected.
            batch_size = min(
                max(1_024, int(np.ceil(1.02 * remaining))),
                200_000,
                max_attempts - attempts,
            )
            if batch_size <= 0:
                break

            attempts += batch_size
            batch = _draw_galaxy_pair_batch(
                context,
                z_snapshot,
                rng,
                batch_size,
            )
            if batch.empty:
                continue

            take = min(remaining, len(batch))
            frames.append(batch.iloc[:take].copy())
            filled += take

        summary[key] = {
            "target_phys": int(context["n_events_phys"]),
            "target_samp": n_goal,
            "filled": filled,
            "tries": attempts,
            "weight": float(context["weight"]),
        }

        if filled != n_goal:
            lo, hi, merger_class = key
            raise RuntimeError(
                "Could not fill galaxy-pair sampling context: "
                f"z={z_snapshot:.2f}, z_rate={context['z_rate']:.3f}, "
                f"descendant_bin=[{lo:.6f}, {hi:.6f}], class={merger_class}, "
                f"n_phys={context['n_events_phys']}, n_samp={n_goal}, "
                f"filled={filled}, deficit={n_goal - filled}, attempts={attempts}."
            )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # The physical weight must recover the rounded target exactly in each
    # context; downstream scripts must not apply another target multiplication.
    if not df.empty:
        grouped = df.groupby(
            ["descendant_bin_lo_log10M", "descendant_bin_hi_log10M", "merger_class"],
            dropna=False,
        )["weight"].sum()
        for context in contexts:
            key = context_key(context)
            total_weight = float(grouped.get(key, 0.0))
            if not np.isclose(
                total_weight,
                context["n_events_phys"],
                rtol=1.0e-10,
                atol=1.0e-6,
            ):
                raise RuntimeError(
                    f"Weight audit failed for context {key}: "
                    f"sum(weight)={total_weight}, expected={context['n_events_phys']}"
                )

    return df, summary

