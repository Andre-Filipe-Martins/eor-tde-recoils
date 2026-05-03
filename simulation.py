#!/usr/bin/env python3
"""
simulation.py
-------------
Full Monte Carlo pipeline for BH merger remnants across the EoR (z = 11.8 -> 6.2).

This is the final downstream Monte Carlo catalogue used for external-TDE and
ionising-energy post-processing.

Inputs
------
  - ratio_scan_vcent_rpeak__<fcent_tag>.json

For each redshift snapshot the pipeline:
  1. Computes expected galaxy counts and merger targets per stellar-mass bin
     using the FIRE-2 GSMF and the Duan+2025 merger rate.
  2. Draws BH–BH merger events by rejection sampling and computes remnant
     properties (Mrem, M*, Re, Vesc0, q).
  3. Assigns a GW recoil kick to each remnant using the per-bin peak ratio
     R_peak = V_kick / V_cent, loaded from the JSON output of ratio_scan_vcent.py.
     V_cent is the speed needed to reach R_cent = F_NUC * Re from the centre.
  4. Attaches TDE fields: rate, orbit travel times, and expected TDE counts
     split by phase (central phase before boundary crossing and external phase
     after boundary crossing).
  5. Saves lean per-snapshot Parquet files to simulation_results/<run_tag>/.

Dependencies
------------
  physics_relations.py    — all physical scaling laws
  ratio_scan_vcent.py     — must be run first to produce the peak-ratio JSON

Outputs
-------
  - simulation_results/runXX/data_z_*.parquet
"""

import os

# Limit BLAS thread counts before importing NumPy for reproducible, stable runtime.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")

import gc
import json
import math
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

import physics_relations as pr
from physics_relations import (
    # GSMF and cosmology
    gsmf_number_density_in_bin, merger_rate_z, age_of_universe_at_z, time_until_z6,
    # BH-galaxy mapping and galaxy sizes
    RV15AGNParams, mstar_from_mbh, re_kpc_m24, FIRE2_SHMR_Params,
    # Non-spinning remnant mass
    final_mass_and_fraction_ns_jf2017,
    # Host potential and halo helpers
    fire2_log10_mh_from_mstar, vesc0_nfw_hernquist,
    # Orbit helpers
    r_vir_mpc, nfw_concentration,
    # TDE helpers and SI constants
    r_k, sigma_from_mbh, r_eff_from_rk_gamma1, r_t,
    m_b_collisional, f_b_from_mb, tde_rate_resonant,
    G, M_sun, pc, km,
)

_TRAPZ = getattr(np, "trapezoid", np.trapz)


# ===========================================================================
# Paths
# ===========================================================================

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

RESULTS_DIR = os.path.join(BASE_DIR, "simulation_results")


# ===========================================================================
# Parquet engine detection
# ===========================================================================

_PARQUET_ENGINE = None
_PARQUET_CODEC  = None

try:
    import pyarrow as pa
    _PARQUET_ENGINE = "pyarrow"
    try:
        _zstd_ok = pa.Codec.is_available("zstd")
    except Exception:
        _zstd_ok = False
    _PARQUET_CODEC = "zstd" if _zstd_ok else "snappy"
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _PARQUET_ENGINE = "fastparquet"
        try:
            from fastparquet.compress import compr
            _PARQUET_CODEC = "ZSTD" if "ZSTD" in compr else "SNAPPY"
        except Exception:
            _PARQUET_CODEC = "SNAPPY"
    except ImportError:
        raise ImportError(
            "A Parquet engine is required. Install pyarrow: pip install pyarrow"
        )


# ===========================================================================
# Cosmology
# ===========================================================================

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.31, Ob0=0.048, Tcmb0=2.725 * u.K)


# ===========================================================================
# EoR redshift window
# ===========================================================================

Z_EOR_INITIAL = 12.0   # upper EoR boundary (cosmic-age reference)
Z_START       = 11.8   # first snapshot
Z_END         = 6.2    # last snapshot (inclusive)
DZ_SNAP       = -0.2   # step between snapshots


# ===========================================================================
# Mass binning  (log10 M_sun)
# ===========================================================================

LOGM_MIN  = 6.75
LOGM_MAX  = 11.65
BIN_WIDTH = 0.35

# Stellar-mass bins retained in the ratio scan and final output.
KEEP_LOGM_MIN = 6.75
KEEP_LOGM_MAX = 9.90


# ===========================================================================
# BH mass bounds
# ===========================================================================

MBH_MIN      = 1.0e3   # lower component mass [M_sun]; log10 = 3
MBH_MAX      = 1.0e6   # upper component mass [M_sun]; log10 = 6
MBH_POST_MAX = 2.0e6   # remnant cap [M_sun] — equal-mass merger at MBH_MAX


# ===========================================================================
# Physical / model parameters
# ===========================================================================

DZ_SLICE   = 0.01            # comoving shell thickness for galaxy counts
OMEGA_FULL = 4.0 * np.pi     # full sky [sr]
MIN_COUNT  = 1.0             # minimum galaxy count to include a bin

F_BULGE           = 0.1548   # bulge mass fraction (= Omega_b / Omega_m)
F_NUC             = 0.05     # central-boundary radius as a fraction of Re
HERNQUIST_RE_OVER_A = 1.8153 # Hernquist scale length from effective radius

SEC_PER_YEAR = 365.25 * 24.0 * 3600.0
KPC_TO_KM    = (1000.0 * pc) / km


# ===========================================================================
# Monte Carlo controls
# ===========================================================================

N_RUNS           = 10
BASE_SEED        = 12345
SEED_STRIDE      = 10_000

MAX_EVENTS_PER_BIN   = 50_000   # per-(snapshot, bin) sampling cap
MAX_TRIES_MULTIPLIER = 200
MAX_TRIES_PER_BIN    = 200_000


# ===========================================================================
# Cached GSMF integrals
# ===========================================================================

_gsmf_cache = {}

rv15         = RV15AGNParams()
fire2_params = FIRE2_SHMR_Params()

_LOG10_MBH_MIN = float(np.log10(MBH_MIN))


# ===========================================================================
# Peak-ratio table — loaded from ratio_scan_vcent.py JSON output
# ===========================================================================

def load_peak_ratios(f_nuc: float = F_NUC) -> dict:
    """
    Load per-bin peak kick ratios R_peak = V_kick / V_cent from the JSON file
    produced by the ratio scan for the same central-boundary choice.

    The JSON filename follows the convention:
        ratio_scan_vcent_rpeak__fcent{F_NUC_tag}.json
    where the tag converts dots to 'p', e.g. 0.05 → 'fcent0p05'.
    """
    tag      = f"fcent{f_nuc:.2f}".replace(".", "p")
    filename = f"ratio_scan_vcent_rpeak__{tag}.json"
    path     = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Peak-ratio JSON not found: {path}\n"
            "Run ratio_scan_vcent.py first to generate it."
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rpeak = data.get("R_peak_by_bin", {})
    if not rpeak:
        raise ValueError(f"'R_peak_by_bin' is empty in {path}")

    print(f"[peak ratios] Loaded {len(rpeak)} bins from {filename}")
    return rpeak


def _bin_label(lo: float, hi: float) -> str:
    """Bin key matching the format used by ratio_scan_vcent.py."""
    return f"[{float(lo):.2f}, {float(hi):.2f}]"


def peak_ratio_for_bin(rpeak_table: dict, log_lo: float, log_hi: float,
                       default: float = 3.0) -> float:
    """Return the peak kick ratio for a mass bin; `default` is used only if absent."""
    return float(rpeak_table.get(_bin_label(log_lo, log_hi), default))


# ===========================================================================
# Mass-bin helpers
# ===========================================================================

def make_mass_bin_edges(logm_min=LOGM_MIN, logm_max=LOGM_MAX, width=BIN_WIDTH):
    """Return an array of bin edges spanning [logm_min, logm_max] in steps of `width`."""
    n_steps = int(np.floor((logm_max - logm_min) / width + 0.5))
    edges   = logm_min + width * np.arange(n_steps + 1)
    if edges[-1] < logm_max - 1e-9:
        edges = np.append(edges, logm_max)
    else:
        edges[-1] = logm_max
    return edges


def snapshot_file_tag(z: float) -> str:
    """Filename-safe redshift string for a snapshot, e.g. z = 11.8 → 'z_11_8'."""
    z_tag = f"{z:.2f}".replace(".", "_")
    return "z_" + z_tag.rstrip("0").rstrip("_")


def _format_mb(path: str) -> str:
    """Return a file size string in MB, or '?' if the size cannot be read."""
    try:
        return f"{os.path.getsize(path) / (1024**2):.2f} MB"
    except Exception:
        return "?"


# ===========================================================================
# GSMF galaxy count (cached)
# ===========================================================================

def cached_galaxy_count(z: float, log10M_lo: float, log10M_hi: float,
                         dz: float = DZ_SLICE):
    """
    Expected galaxy count in a stellar-mass bin within a full-sky shell of
    thickness dz at redshift z. Results are cached to avoid redundant integrals.

    Returns (N_galaxies, n_bin [Mpc^-3], V_shell [Mpc^3]).
    """
    z_key     = float(f"{z:.2f}")
    cache_key = (z_key, float(log10M_lo), float(log10M_hi), float(dz))
    if cache_key in _gsmf_cache:
        return _gsmf_cache[cache_key]

    n_bin = gsmf_number_density_in_bin(
        z_key, log10M_lo, log10M_hi,
        n_steps=2048, method="pchip", allow_extrapolation=False,
    )
    dVc_dz_dOmega = cosmo.differential_comoving_volume(z_key).to(u.Mpc**3 / u.sr)
    V_shell       = float(dVc_dz_dOmega.value) * dz * OMEGA_FULL
    result        = (float(n_bin * V_shell), float(n_bin), float(V_shell))

    _gsmf_cache[cache_key] = result
    return result


# ===========================================================================
# Bin metadata (BH mass windows and peak ratios)
# ===========================================================================

def build_bin_metadata(active_edges: list, rpeak_table: dict) -> dict:
    """
    Precompute per-bin BH mass windows and peak kick ratios.

    For each (log_lo, log_hi) bin, RV15 defines the proposal BH-mass window
    implied by the stellar-mass bin edges. The peak ratio R_peak is looked up
    from the ratio-scan JSON.

    Returns a dict keyed by (log_lo, log_hi).
    """
    meta  = {}
    alpha = float(rv15.alpha)
    beta  = float(rv15.beta)
    Mpiv  = float(rv15.M_pivot)

    for (log_lo, log_hi) in active_edges:
        Mstar_lo = 10.0 ** float(log_lo)
        Mstar_hi = 10.0 ** float(log_hi)

        mbh_lo = 10.0 ** (alpha + beta * np.log10(Mstar_lo / Mpiv))
        mbh_hi = 10.0 ** (alpha + beta * np.log10(Mstar_hi / Mpiv))

        bh_lo = max(MBH_MIN, min(mbh_lo, mbh_hi))
        bh_hi = min(MBH_MAX, max(mbh_lo, mbh_hi))

        meta[(float(log_lo), float(log_hi))] = {
            "bh_lo":   float(bh_lo),
            "bh_hi":   float(bh_hi),
            "R_peak":  peak_ratio_for_bin(rpeak_table, log_lo, log_hi),
        }

    return meta


# ===========================================================================
# Snapshot plan (precomputed once, reused across all runs)
# ===========================================================================

def precompute_snapshot_plan(z_values: list, active_edges: list) -> list:
    """
    For each snapshot, compute expected galaxy counts, physical merger targets,
    capped sampled targets, and time remaining until z = 6. These depend only
    on the GSMF and merger rate, so they are computed once and reused across
    all Monte Carlo runs.

    Returns a list of snapshot plan dicts.
    """
    plan = []

    for idx, z_cur in enumerate(z_values):
        if idx == 0:
            z_mid  = float(z_cur)
            t_prev = float(age_of_universe_at_z(Z_EOR_INITIAL))
            t_curr = float(age_of_universe_at_z(z_cur))
        else:
            z_prev = z_values[idx - 1]
            z_mid  = 0.5 * (z_prev + z_cur)
            t_prev = float(age_of_universe_at_z(z_prev))
            t_curr = float(age_of_universe_at_z(z_cur))

        R_Gyr  = float(merger_rate_z(z_mid))
        dt_Gyr = (t_curr - t_prev) / 1e9

        counts_rows   = []
        targets_phys  = {}
        targets_samp  = {}
        total_N       = 0.0
        total_mergers = 0.0

        for (lo, hi) in active_edges:
            N, n_bin, _ = cached_galaxy_count(z_cur, lo, hi, dz=DZ_SLICE)
            if N < MIN_COUNT:
                continue
            mergers_bin = N * R_Gyr * dt_Gyr
            counts_rows.append((float(lo), float(hi), float(N), float(n_bin),
                                 float(mergers_bin)))
            total_N       += float(N)
            total_mergers += float(mergers_bin)

            n_phys = int(np.round(mergers_bin))
            if n_phys > 0:
                n_samp = int(min(n_phys, MAX_EVENTS_PER_BIN))
                targets_phys[(float(lo), float(hi))] = n_phys
                targets_samp[(float(lo), float(hi))] = n_samp

        plan.append({
            "idx":          idx,
            "z":            float(z_cur),
            "z_mid":        float(z_mid),
            "R_Gyr":        R_Gyr,
            "dt_Gyr":       dt_Gyr,
            "counts_rows":  counts_rows,
            "totals":       (total_N, total_mergers),
            "targets_phys": targets_phys,
            "targets_samp": targets_samp,
            "dt_to_z6_yr":  float(time_until_z6(z_cur)),
        })

    return plan


# ===========================================================================
# Gravitational potential (NFW + Hernquist), dimensionless form
# ===========================================================================

def _psi_shape(x, Mh_Msun: float, Mb_Msun: float, Re_kpc: float, z: float):
    """
    Dimensionless positive potential depth at x = r/Re:
        psi(x) = Psi(r) / Psi(0),   Psi(r) = -Phi(r),   Psi(0) = 0.5 * Vesc0^2.
    NFW halo + Hernquist bulge model.
    """
    x    = np.asarray(x, dtype=float)
    Re_m = (Re_kpc * 1000.0) * pc
    r_m  = x * Re_m

    c        = float(nfw_concentration(Mh_Msun, z))
    rvir_m   = float(r_vir_mpc(Mh_Msun, z)) * 1.0e6 * pc
    rs_m     = rvir_m / c
    f_c      = np.log(1.0 + c) - c / (1.0 + c)
    Mh_kg    = Mh_Msun * M_sun
    Mb_kg    = Mb_Msun * M_sun
    a_m      = Re_m / HERNQUIST_RE_OVER_A

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        psi_nfw = G * Mh_kg * np.log(1.0 + r_m / rs_m) / (r_m * f_c)
    psi_nfw_0 = G * Mh_kg / (rs_m * f_c)
    psi_nfw   = np.where(r_m > 0.0, psi_nfw, psi_nfw_0)

    psi_h   = G * Mb_kg / (r_m + a_m)
    psi_h_0 = G * Mb_kg / a_m

    psi0 = psi_nfw_0 + psi_h_0
    return (psi_nfw + psi_h) / psi0


def _find_rmax_bisect(target, Mh_Msun, Mb_Msun, Re_kpc, z, x_lo, x_hi_init):
    """Find the apocentric x = r/Re where psi(x) = target, by bisection."""
    x_hi = float(x_hi_init)
    for _ in range(80):
        if float(_psi_shape(x_hi, Mh_Msun, Mb_Msun, Re_kpc, z)) < target:
            break
        x_hi *= 2.0
    else:
        return x_hi

    x_lo = float(x_lo)
    for _ in range(80):
        x_mid = 0.5 * (x_lo + x_hi)
        if float(_psi_shape(x_mid, Mh_Msun, Mb_Msun, Re_kpc, z)) > target:
            x_lo = x_mid
        else:
            x_hi = x_mid
    return x_hi


def _integrate_travel_time(R, Mh_Msun, Mb_Msun, Re_kpc, z,
                             x0, x1, n=4096, cluster_near_x1=False):
    """
    Integrate dx / sqrt(R^2 - 1 + psi(x)) from x0 to x1.
    The returned value is dimensionless before conversion by the caller.
    """
    x0, x1 = float(x0), float(x1)
    if not (np.isfinite(x0) and np.isfinite(x1) and x1 > x0):
        return np.nan

    if cluster_near_x1:
        u  = np.linspace(0.0, 1.0, n, endpoint=False)
        xs = x0 + (x1 - x0) * (1.0 - (1.0 - u) ** 2)
    else:
        xs = np.linspace(x0, x1, n, endpoint=True)

    psi       = _psi_shape(xs, Mh_Msun, Mb_Msun, Re_kpc, z)
    arg       = np.maximum(R * R - 1.0 + psi, 0.0)
    denom     = np.sqrt(arg)
    denom     = np.where(denom > 0.0, denom, np.nan)
    integrand = 1.0 / denom

    m = np.isfinite(integrand)
    if m.sum() < 2:
        return np.nan
    return float(_TRAPZ(integrand[m], xs[m]))


def _representative_orbit_times(rep_Mh, rep_Mb, rep_Re, rep_Vesc0, z, R):
    """
    Compute representative orbit travel times for the effective ratio
    R = V_kick / Vesc0 using the model potential for a given mass bin.

    Returns (t_leave_yr, t_ext_max_yr, t_return_yr, is_bound).
    """
    if not all(np.isfinite(v) for v in [rep_Mh, rep_Mb, rep_Re, rep_Vesc0]):
        return np.nan, 0.0, np.nan, False

    x_nuc = float(F_NUC)

    if R >= 1.0:
        # Escape-like or unbound trajectory.
        I_leave = _integrate_travel_time(R, rep_Mh, rep_Mb, rep_Re, z,
                                          0.0, x_nuc, cluster_near_x1=False)
        if not np.isfinite(I_leave):
            return np.nan, 0.0, np.inf, False
        scale_yr = (rep_Re * KPC_TO_KM / rep_Vesc0) / SEC_PER_YEAR
        return scale_yr * I_leave, np.inf, np.inf, True

    psi_xnuc = float(_psi_shape(x_nuc, rep_Mh, rep_Mb, rep_Re, z))
    target   = 1.0 - R * R
    if psi_xnuc <= target:
        return np.nan, 0.0, np.nan, False

    try:
        rvir_kpc = float(r_vir_mpc(rep_Mh, z)) * 1.0e3
        x_hi0    = max(10.0, rvir_kpc / rep_Re)
    except Exception:
        x_hi0 = 1.0e3

    x_max = _find_rmax_bisect(target, rep_Mh, rep_Mb, rep_Re, z,
                               x_lo=x_nuc, x_hi_init=x_hi0)

    I_leave = _integrate_travel_time(R, rep_Mh, rep_Mb, rep_Re, z,
                                      0.0, x_nuc, cluster_near_x1=False)
    I_out   = _integrate_travel_time(R, rep_Mh, rep_Mb, rep_Re, z,
                                      x_nuc, x_max, cluster_near_x1=True)
    if not (np.isfinite(I_leave) and np.isfinite(I_out)):
        return np.nan, 0.0, np.nan, False

    scale_yr  = (rep_Re * KPC_TO_KM / rep_Vesc0) / SEC_PER_YEAR
    t_leave   = scale_yr * I_leave
    t_ext_max = scale_yr * (2.0 * I_out)
    return t_leave, t_ext_max, t_leave + t_ext_max, True


def _integral_exp(rate0, t_decay, t0, t1):
    """
    Integral of rate0 * exp(-t / t_decay) from t0 to t1 (vectorised).
    Returns zero where inputs are non-positive or non-finite.
    """
    rate0, t_decay, t0, t1 = (np.asarray(x, dtype=float)
                                for x in (rate0, t_decay, t0, t1))
    if t0.shape == ():
        t0 = np.broadcast_to(t0, rate0.shape)
    if t1.shape == ():
        t1 = np.broadcast_to(t1, rate0.shape)
    if t_decay.shape == ():
        t_decay = np.broadcast_to(t_decay, rate0.shape)

    out = np.zeros_like(rate0, dtype=float)
    m = (
        np.isfinite(rate0)  & (rate0  > 0.0) &
        np.isfinite(t_decay) & (t_decay > 0.0) &
        np.isfinite(t0) & np.isfinite(t1) & (t1 > t0)
    )
    if np.any(m):
        out[m] = rate0[m] * t_decay[m] * (
            np.exp(-t0[m] / t_decay[m]) - np.exp(-t1[m] / t_decay[m])
        )
    return out


# ===========================================================================
# Kick assignment from V_cent peak ratios
# ===========================================================================

def assign_kicks(df: pd.DataFrame, z: float, rpeak_table: dict) -> pd.DataFrame:
    """
    Assign a GW recoil kick to each remnant using the per-bin peak ratio
        R_peak = V_kick / V_cent,
    where V_cent is estimated from a representative potential for each
    (z, mass-bin) group and individual values are scaled using each remnant's
    Vesc0:
        V_cent(BH) ~= (V_cent / Vesc0)_rep * Vesc0(BH).

    Adds / overwrites columns: v_cent_kms, Vkick_kms, Vkick_over_Vesc0,
    Vkick_over_Vcent, escaped.
    """
    if df is None or len(df) == 0:
        return df

    bin_lo = np.round(df["bin_lo_log10M"].to_numpy(float), 2)
    bin_hi = np.round(df["bin_hi_log10M"].to_numpy(float), 2)
    uniq_pairs, inv = np.unique(np.stack([bin_lo, bin_hi], axis=1),
                                 axis=0, return_inverse=True)
    n_bins = uniq_pairs.shape[0]

    Re_kpc   = df["Re_kpc"].to_numpy(float)
    Vesc0    = df["Vesc0_kms"].to_numpy(float)
    log10_Mh = df["log10_Mh_fire2"].to_numpy(float)
    Mstar    = df["Mstar_rem_Msun"].to_numpy(float)
    Rpeak_v  = df["Rpeak_vcent"].to_numpy(float)

    vcent_over_vesc0 = np.ones(n_bins, dtype=float)

    for j in range(n_bins):
        bin_mask = (inv == j)
        if not np.any(bin_mask):
            continue

        rep_Re   = float(np.nanmedian(Re_kpc[bin_mask]))
        rep_Vesc = float(np.nanmedian(Vesc0[bin_mask]))
        rep_Mh   = float(np.nanmedian(10.0 ** log10_Mh[bin_mask]))
        rep_Mb   = float(F_BULGE * np.nanmedian(Mstar[bin_mask]))

        # Estimate V_cent/Vesc0 from the median host potential in each active bin.
        psi_xnuc = float(_psi_shape(F_NUC, rep_Mh, rep_Mb, rep_Re, z))
        if not np.isfinite(psi_xnuc):
            vcent_over_vesc0[j] = 1.0
        else:
            psi_xnuc = float(np.clip(psi_xnuc, 0.0, 1.0))
            vcent_over_vesc0[j] = math.sqrt(max(0.0, 1.0 - psi_xnuc))

    vcent_row = vcent_over_vesc0[inv] * Vesc0
    Vkick     = Rpeak_v * vcent_row

    with np.errstate(divide="ignore", invalid="ignore"):
        Vk_over_vesc0 = Vkick / Vesc0
        Vk_over_vcent = Vkick / vcent_row

    df["v_cent_kms"]       = vcent_row
    df["Vkick_kms"]        = Vkick
    df["Vkick_over_Vesc0"] = Vk_over_vesc0
    df["Vkick_over_Vcent"] = Vk_over_vcent
    df["escaped"]          = (Vkick >= Vesc0)
    return df


# ===========================================================================
# Event drawing (rejection sampling)
# ===========================================================================

def draw_merger_event(log_lo: float, log_hi: float, z: float, rng,
                       bin_meta: dict, rpeak_table: dict):
    """
    Draw one BH–BH merger remnant landing in the host bin [log_lo, log_hi).

    Procedure:
      1. Draw the heavier component m2 log-uniformly in the bin's BH window.
      2. Draw the lighter component m1 log-uniformly in [MBH_MIN, m2].
      3. Compute remnant mass with the JF2017 non-spinning fit.
      4. Invert RV15 to get host-galaxy stellar mass; reject if outside the bin.
      5. Compute Re, M_h, and Vesc0.

    Returns a tuple on success, None if any step fails or the event falls
    outside the target stellar-mass bin.
    """
    info = bin_meta.get((log_lo, log_hi))
    if info is None:
        return None

    bh_lo = info["bh_lo"]
    bh_hi = info["bh_hi"]
    if not (np.isfinite(bh_lo) and np.isfinite(bh_hi) and bh_hi > bh_lo):
        return None

    m2 = 10.0 ** rng.uniform(np.log10(bh_lo), np.log10(bh_hi))
    if not (np.isfinite(m2) and m2 > 0.0):
        return None

    log10_m2 = float(np.log10(m2))
    if log10_m2 <= _LOG10_MBH_MIN:
        return None
    m1 = 10.0 ** rng.uniform(_LOG10_MBH_MIN, log10_m2)
    if not (np.isfinite(m1) and m1 > 0.0):
        return None

    if m1 > m2:
        m1, m2 = m2, m1

    Mrem, _ = final_mass_and_fraction_ns_jf2017(float(m1), float(m2))
    if not (np.isfinite(Mrem) and Mrem > 0.0):
        return None
    Mrem = min(Mrem, MBH_POST_MAX)

    Mstar = mstar_from_mbh(float(Mrem))
    if not (np.isfinite(Mstar) and Mstar > 0.0):
        return None
    if not (log_lo <= np.log10(Mstar) < log_hi):
        return None

    Re = re_kpc_m24(float(Mstar), float(z))
    if not (np.isfinite(Re) and Re > 0.0):
        return None

    log10_Mh = fire2_log10_mh_from_mstar(float(Mstar), fire2_params)
    if not np.isfinite(log10_Mh):
        return None
    Mh    = 10.0 ** float(log10_Mh)
    Mb    = F_BULGE * float(Mstar)
    Vesc0 = vesc0_nfw_hernquist(float(Mh), float(Mb), float(Re), float(z))
    if not (np.isfinite(Vesc0) and Vesc0 > 0.0):
        return None

    q       = float(m1 / m2)
    R_peak  = float(info["R_peak"])

    return (float(log_lo), float(log_hi), float(z),
            float(m1), float(m2), float(q),
            float(Mrem), float(Mstar),
            float(Re), float(log10_Mh),
            float(Vesc0), float(R_peak))


def sample_snapshot_events(targets_phys: dict, targets_samp: dict,
                             z: float, rng, bin_meta: dict,
                             rpeak_table: dict) -> tuple[pd.DataFrame, dict]:
    """
    Fill all bins to their capped sampled targets by rejection sampling.

    Events in bins where the physical count exceeds the sampling cap are
    assigned a per-row weight (n_phys / n_samp) so that the scaled totals
    still represent the physical population.

    Physical and sampled counts are tracked in the summary. Placeholder kick
    columns are filled by assign_kicks().

    Returns (df_events, summary_dict).
    """
    cols = {k: [] for k in [
        "bin_lo_log10M", "bin_hi_log10M", "z",
        "m1_BH_Msun", "m2_BH_Msun", "q",
        "Mrem_BH_Msun", "Mstar_rem_Msun",
        "Re_kpc", "log10_Mh_fire2",
        "Vesc0_kms", "Rpeak_vcent",
    ]}

    summary = {}

    for (lo, hi) in sorted(targets_samp.keys()):
        n_phys = int(targets_phys.get((lo, hi), 0))
        n_goal = int(targets_samp[(lo, hi)])
        if n_goal <= 0:
            continue

        filled    = 0
        tries     = 0
        max_tries = min(MAX_TRIES_PER_BIN,
                        max(10_000, MAX_TRIES_MULTIPLIER * n_goal))

        while filled < n_goal and tries < max_tries:
            tries += 1
            out = draw_merger_event(lo, hi, z, rng, bin_meta, rpeak_table)
            if out is None:
                continue

            (b_lo, b_hi, zc, m1, m2, q, Mrem, Mstar,
             Re, log10_Mh, Vesc0, R_peak) = out

            cols["bin_lo_log10M"].append(b_lo)
            cols["bin_hi_log10M"].append(b_hi)
            cols["z"].append(zc)
            cols["m1_BH_Msun"].append(m1)
            cols["m2_BH_Msun"].append(m2)
            cols["q"].append(q)
            cols["Mrem_BH_Msun"].append(Mrem)
            cols["Mstar_rem_Msun"].append(Mstar)
            cols["Re_kpc"].append(Re)
            cols["log10_Mh_fire2"].append(log10_Mh)
            cols["Vesc0_kms"].append(Vesc0)
            cols["Rpeak_vcent"].append(R_peak)
            filled += 1

        summary[(lo, hi)] = {
            "target_phys": n_phys,
            "target_samp": n_goal,
            "filled":      filled,
            "tries":       tries,
            "cap_applied": (n_phys > n_goal),
        }

    df = pd.DataFrame(cols)

    # Placeholder kick columns — filled by assign_kicks()
    df["v_cent_kms"]       = np.nan
    df["Vkick_kms"]        = np.nan
    df["Vkick_over_Vesc0"] = np.nan
    df["Vkick_over_Vcent"] = np.nan
    df["escaped"]          = False

    return df, summary


# ===========================================================================
# TDE field attachment
# ===========================================================================

def attach_tde_fields(df: pd.DataFrame, z: float, dt_to_z6_yr: float) -> pd.DataFrame:
    """
    Attach TDE rates and integrated TDE counts to every remnant.

    For each remnant the procedure is:
      1. Compute the resonant-relaxation TDE rate (Merritt+2009).
      2. Determine the orbit travel time to/from the central-region boundary using
         representative (median) galaxy properties per mass bin.
      3. Integrate an exponentially decaying rate over three time segments:
           a. Central phase before boundary crossing.
           b. External phase outside the central region.
           c. Central phase after the BH returns (bound orbits only).

    Only TDEs accumulated during the external phase (b) contribute to
    IGM ionising budget.

    Adds columns: t_esc_yr, t_external_yr, tde_rate_per_yr,
                  tde_nuclear_preescape, tde_external_post, tde_count_per_bh.
    """
    if len(df) == 0:
        return df

    M_star_SI = 1.0 * M_sun
    R_sun_m   = 6.957e8

    Mbh_Msun   = df["Mrem_BH_Msun"].to_numpy(float)
    Re_kpc_arr = df["Re_kpc"].to_numpy(float)
    Vesc0_kms  = df["Vesc0_kms"].to_numpy(float)
    Vkick_kms  = df["Vkick_kms"].to_numpy(float)
    Mstar_Msun = df["Mstar_rem_Msun"].to_numpy(float)
    log10_Mh   = df["log10_Mh_fire2"].to_numpy(float)

    Mbh_SI  = Mbh_Msun * M_sun
    v_k_ms  = Vkick_kms * km

    # TDE rate (Merritt+2009)
    rk_val       = r_k(Mbh_SI, v_k_ms)
    sigma_km_s, _ = sigma_from_mbh(Mbh_SI)
    r_eff_pc     = pr.r_eff_from_rk_gamma1(rk_val) / pc
    Mb_coll_kg   = m_b_collisional(Mbh_Msun, sigma_km_s, r_eff_pc)
    cap_stars    = np.where(np.isfinite(Mb_coll_kg) & (Mb_coll_kg > 0.0),
                            Mb_coll_kg / M_sun, 0.0)

    f_b     = np.clip(f_b_from_mb(Mb_coll_kg, Mbh_SI), 0.0, 1.0)
    rt_val  = r_t(Mbh_SI, M_star_SI, R_sun_m)
    rate_s  = tde_rate_resonant(Mbh_SI, M_star_SI, rk_val, rt_val, v_k_ms, f_b)
    rate_yr = np.where(np.isfinite(rate_s) & (rate_s > 0.0),
                       rate_s * SEC_PER_YEAR, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t_decay_yr = 1.5e9 * (Mbh_Msun / 1.0e7) ** 2 * (Vkick_kms / 1.0e3) ** (-3.0)
    t_decay_yr = np.where(np.isfinite(t_decay_yr) & (t_decay_yr > 0.0),
                          t_decay_yr, np.nan)

    t_cap = float(dt_to_z6_yr) if (np.isfinite(dt_to_z6_yr) and dt_to_z6_yr > 0.0) else 0.0

    # Representative orbit times per mass bin (Re/Vesc0 scaling per galaxy)
    bin_lo = np.round(df["bin_lo_log10M"].to_numpy(float), 2)
    bin_hi = np.round(df["bin_hi_log10M"].to_numpy(float), 2)
    uniq_pairs, inv = np.unique(np.stack([bin_lo, bin_hi], axis=1),
                                 axis=0, return_inverse=True)
    n_bins = uniq_pairs.shape[0]

    rep_scale_base = np.full(n_bins, np.nan, float)
    rep_t_leave    = np.full(n_bins, np.nan, float)
    rep_t_ext_max  = np.zeros(n_bins, float)
    rep_leave_ok   = np.zeros(n_bins, bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        Vkick_over_Vesc0 = Vkick_kms / Vesc0_kms

    for j in range(n_bins):
        lo, hi = float(uniq_pairs[j, 0]), float(uniq_pairs[j, 1])
        bin_mask = (inv == j)
        if not np.any(bin_mask):
            continue

        rep_Re   = float(np.nanmedian(Re_kpc_arr[bin_mask]))
        rep_Vesc = float(np.nanmedian(Vesc0_kms[bin_mask]))
        rep_Mh   = float(np.nanmedian(10.0 ** log10_Mh[bin_mask]))
        rep_Mb   = float(F_BULGE * np.nanmedian(Mstar_Msun[bin_mask]))

        # Convert the V_cent peak ratio to an effective Vkick/Vesc0 ratio for
        # the orbit-time calculation (R_eff = R_peak * (Vcent/Vesc0)_rep).
        R_peak   = peak_ratio_for_bin({}, lo, hi)  # not needed here; use Vkick_over_Vesc0
        psi_xnuc = float(_psi_shape(F_NUC, rep_Mh, rep_Mb, rep_Re, z))
        psi_xnuc = float(np.clip(psi_xnuc, 0.0, 1.0)) if np.isfinite(psi_xnuc) else 0.0
        vcent_over_vesc0_rep = math.sqrt(max(0.0, 1.0 - psi_xnuc))

        # Median Vkick / Vesc0 across the bin's events
        R_eff = float(np.nanmedian(Vkick_over_Vesc0[bin_mask]))

        tL, t_ext_max_rep, _tRet, ok = _representative_orbit_times(
            rep_Mh, rep_Mb, rep_Re, rep_Vesc, z, R_eff
        )

        rep_scale_base[j] = (rep_Re / rep_Vesc) if (rep_Re > 0 and rep_Vesc > 0) else np.nan
        rep_t_leave[j]    = float(tL)
        rep_t_ext_max[j]  = float(t_ext_max_rep)
        rep_leave_ok[j]   = bool(ok)

    with np.errstate(divide="ignore", invalid="ignore"):
        base_i    = Re_kpc_arr / Vesc0_kms
        scale_fac = base_i / rep_scale_base[inv]
    scale_fac = np.where(np.isfinite(scale_fac) & (scale_fac > 0.0), scale_fac, 1.0)

    leave_ok  = rep_leave_ok[inv]
    t_leave   = rep_t_leave[inv]   * scale_fac
    t_ext_max = rep_t_ext_max[inv] * scale_fac
    t_leave   = np.where(leave_ok, t_leave, np.nan)
    t_ext_max = np.where(leave_ok, t_ext_max, 0.0)

    is_bound = (Vkick_over_Vesc0 < 1.0)
    rem_after_leave = np.where(
        leave_ok & np.isfinite(t_leave),
        np.clip(t_cap - t_leave, 0.0, None), 0.0
    )
    t_ext = np.where(
        leave_ok,
        np.where(is_bound, np.minimum(rem_after_leave, t_ext_max), rem_after_leave),
        0.0,
    )

    # --- Integrate the exponentially decaying TDE rate over each time segment. ---
    t_nuc_pre = np.where(
        leave_ok & np.isfinite(t_leave),
        np.minimum(t_leave, t_cap), t_cap,
    )
    N_nuc_pre_unc = _integral_exp(rate_yr, t_decay_yr, 0.0, t_nuc_pre)

    t0_ext    = np.where(np.isfinite(t_leave), t_leave, 0.0)
    N_ext_unc = _integral_exp(rate_yr, t_decay_yr, t0_ext, t0_ext + t_ext)

    t_return     = np.where(leave_ok & np.isfinite(t_leave), t_leave + t_ext_max, np.nan)
    t_nuc_post   = np.where(
        leave_ok & is_bound & np.isfinite(t_return),
        np.clip(t_cap - t_return, 0.0, None), 0.0,
    )
    t0_nuc_post    = np.clip(t_cap - t_nuc_post, 0.0, None)
    N_nuc_post_unc = _integral_exp(rate_yr, t_decay_yr, t0_nuc_post, t_cap)

    # Apply the bound-star cap sequentially across central, external, and return segments.
    N_nuc_pre  = np.minimum(N_nuc_pre_unc, cap_stars)
    cap_rem    = np.clip(cap_stars - N_nuc_pre, 0.0, None)
    N_ext      = np.minimum(N_ext_unc, cap_rem)
    cap_rem2   = np.clip(cap_rem - N_ext, 0.0, None)
    N_nuc_post = np.minimum(N_nuc_post_unc, cap_rem2)

    df["t_esc_yr"]              = t_leave
    df["t_external_yr"]         = t_ext
    df["tde_rate_per_yr"]       = rate_yr
    df["tde_nuclear_preescape"] = N_nuc_pre + N_nuc_post
    df["tde_external_post"]     = N_ext
    df["tde_count_per_bh"]      = N_ext

    # Diagnostic medians for monitoring each snapshot.
    N_total = df["tde_nuclear_preescape"] + df["tde_external_post"]
    with np.errstate(invalid="ignore", divide="ignore"):
        print("\n--- Remnant medians (all bound + unbound) ---")
        print(f"  log10 Mrem_BH         = {np.nanmedian(np.log10(Mbh_Msun)):6.3f}")
        print(f"  log10 M*_rem          = {np.nanmedian(np.log10(Mstar_Msun)):6.3f}")
        print(f"  Re [kpc]              = {np.nanmedian(df['Re_kpc']):6.3f}")
        print(f"  q                     = {np.nanmedian(df['q']):6.3f}")
        print(f"  log10 Mh (FIRE-2)     = {np.nanmedian(df['log10_Mh_fire2']):6.3f}")
        print(f"  Vesc0 [km/s]          = {np.nanmedian(df['Vesc0_kms']):7.1f}")
        print(f"  V_cent [km/s]         = {np.nanmedian(df['v_cent_kms']):7.1f}")
        print(f"  V_kick [km/s]         = {np.nanmedian(df['Vkick_kms']):7.1f}")
        print(f"  V_kick / Vesc0        = {np.nanmedian(df['Vkick_over_Vesc0']):6.3f}")
        print(f"  V_kick / V_cent       = {np.nanmedian(df['Vkick_over_Vcent']):6.3f}")
        print(f"  TDE rate [yr^-1]      = {np.nanmedian(df['tde_rate_per_yr']):.3e}")
        print(f"  t_leave nucleus [yr]  = {np.nanmedian(df['t_esc_yr']):.3e}")
        print(f"  t_external [yr]       = {np.nanmedian(df['t_external_yr']):.3e}")
        print(f"  TDEs nuclear (median) = {np.nanmedian(df['tde_nuclear_preescape']):.3f}")
        print(f"  TDEs external (median)= {np.nanmedian(df['tde_external_post']):.3f}")
        print(f"  TDEs total (median)   = {np.nanmedian(N_total):.3f}")

    return df


# ===========================================================================
# Save to Parquet
# ===========================================================================

def save_parquet(df: pd.DataFrame, out_base: str, run_tag: str) -> None:
    """
    Save the lean set of columns needed by downstream post-processing to Parquet.

    Output: simulation_results/<run_tag>/<out_base>.parquet
    """
    out_dir  = os.path.join(RESULTS_DIR, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_base}.parquet")

    desired_cols = [
        "event_id", "run_id",
        "q", "Mrem_BH_Msun", "Mstar_rem_Msun",
        "Re_kpc", "log10_Mh_fire2",
        "Vesc0_kms", "v_cent_kms", "Vkick_kms",
        "Vkick_over_Vesc0", "Vkick_over_Vcent",
        "tde_rate_per_yr", "t_esc_yr", "t_external_yr",
        "tde_nuclear_preescape", "tde_external_post", "tde_count_per_bh",
    ]
    keep_cols = [c for c in desired_cols if c in df.columns]

    if len(df) == 0:
        df_save = pd.DataFrame(columns=keep_cols)
    else:
        df_save = df[keep_cols].copy()
        for c in df_save.select_dtypes(include=["float64"]).columns:
            df_save[c] = pd.to_numeric(df_save[c], downcast="float")
        for c in df_save.select_dtypes(include=["int64"]).columns:
            df_save[c] = pd.to_numeric(df_save[c], downcast="integer")

    codec = _PARQUET_CODEC
    if _PARQUET_ENGINE == "fastparquet":
        codec = codec.upper()
    df_save.to_parquet(out_path, engine=_PARQUET_ENGINE, compression=codec, index=False)
    print(f"  Saved: {out_path}  ({len(df_save):,} rows, {_format_mb(out_path)})")


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Run the final Monte Carlo simulation using peak ratios from the ratio scan."""
    # Load peak ratios from ratio_scan_vcent.py output (must be run first)
    rpeak_table = load_peak_ratios(f_nuc=F_NUC)

    # Build the EoR snapshot list (z = 11.8 → 6.2, step = −0.2)
    z_values = []
    z = Z_START
    while z >= Z_END - 1e-8:
        z_values.append(float(f"{z:.2f}"))
        z += DZ_SNAP

    # Stellar-mass bins in the range used by the ratio scan
    all_edges    = make_mass_bin_edges()
    active_edges = [
        (lo, hi)
        for lo, hi in zip(all_edges[:-1], all_edges[1:])
        if lo >= KEEP_LOGM_MIN and hi <= KEEP_LOGM_MAX
    ]

    # Precompute bin metadata (BH windows, peak ratios)
    bin_meta = build_bin_metadata(active_edges, rpeak_table)

    # Precompute snapshot plan (galaxy counts, merger targets) — once for all runs
    print("Precomputing snapshot plan...")
    snapshot_plan = precompute_snapshot_plan(z_values, active_edges)

    # Monte Carlo loop
    for run_idx in range(N_RUNS):
        run_tag   = f"run{run_idx:02d}"
        seed_base = BASE_SEED + run_idx * SEED_STRIDE

        print("\n" + "#" * 70)
        print(f"### RUN {run_tag}  (seed_base = {seed_base})")
        print("#" * 70 + "\n")

        verbose = (run_idx == 0)   # print the counts table only on the first run

        for snap in snapshot_plan:
            z_cur    = snap["z"]
            rng      = np.random.default_rng(seed_base + snap["idx"])
            file_tag = snapshot_file_tag(z_cur)
            out_base = f"data_{file_tag}"

            if verbose:
                print(f"\n=== [z = {z_cur:.2f}] Galaxy counts and mergers "
                      f"(Δz = {DZ_SLICE}, full sky) ===")
                print(f"  Merger rate R_M(z_mid = {snap['z_mid']:.2f}) = "
                      f"{snap['R_Gyr']:.4e} Gyr^-1   "
                      f"Δt = {snap['dt_Gyr']:.4e} Gyr")
                print(f"\n  {'Mass bin [log10 Msun]':>24}  {'N (count)':>14}  "
                      f"{'n [Mpc^-3]':>14}  {'Mergers':>12}")
                print("  " + "-" * 70)
                for lo, hi, N, n_bin, mergers_bin in snap["counts_rows"]:
                    print(f"  [{lo:5.2f}, {hi:5.2f}]".rjust(28),
                          f"{N:14.6e}  {n_bin:14.6e}  {mergers_bin:12.6e}")
                total_N, total_mergers = snap["totals"]
                print("  " + "-" * 70)
                print(f"  {'TOTAL:':>28}  {total_N:14.6e}  {'':14}  {total_mergers:12.6e}")

            # Step 1: sample merger events for this snapshot
            df_events, bin_summary = sample_snapshot_events(
                targets_phys=snap["targets_phys"],
                targets_samp=snap["targets_samp"],
                z=z_cur,
                rng=rng,
                bin_meta=bin_meta,
                rpeak_table=rpeak_table,
            )

            # Stable event IDs and run label
            if "event_id" not in df_events.columns:
                df_events = df_events.reset_index(drop=False).rename(
                    columns={"index": "event_id"}
                )
            df_events["run_id"] = run_tag

            # Print sampling diagnostics before kick and TDE fields are attached.
            print(f"\n=== [z = {z_cur:.2f}] Per-bin fill summary ===")
            print(f"  {'Bin [log10 M*]':>18}  {'Target':>10}  "
                  f"{'Sampled':>10}  {'Filled':>10}  {'Escaped':>8}")
            for (lo, hi), info in sorted(bin_summary.items()):
                bin_mask  = ((df_events["bin_lo_log10M"] == lo) &
                             (df_events["bin_hi_log10M"] == hi))
                n_escaped = int((bin_mask & df_events["escaped"]).sum())
                print(f"  [{lo:5.2f}, {hi:5.2f}]  "
                      f"{info['target_phys']:10d}  {info['target_samp']:10d}  "
                      f"{info['filled']:10d}  {n_escaped:8d}")
            print(f"\n  [{run_tag}] Total events sampled: {len(df_events):,}")

            # Step 2: assign GW recoil kicks
            if len(df_events):
                df_events = assign_kicks(df_events, z_cur, rpeak_table)

            # Step 3: attach TDE fields
            df_remnants = df_events.reset_index(drop=True)
            if len(df_remnants):
                df_remnants = attach_tde_fields(
                    df_remnants, z_cur, snap["dt_to_z6_yr"]
                )

            # Step 4: save snapshot to Parquet
            save_parquet(df_remnants, out_base, run_tag)

            del df_events, df_remnants
            gc.collect()

        print(f"\n### FINISHED {run_tag}\n")


if __name__ == "__main__":
    main()
