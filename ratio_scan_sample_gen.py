#!/usr/bin/env python3
"""
ratio_scan_sample_gen.py
----------------------------
Monte Carlo catalogue generator for BH merger remnants across the Epoch of Reionization.

For each snapshot in the EoR redshift window (z = 11.8 → 6.2 in steps of 0.2), this script
generates a population of BH merger remnants with the host-galaxy properties needed by the
downstream kick/TDE analysis:

  - Remnant BH mass         Mrem_BH_Msun      [M_sun]
  - Host stellar mass       Mstar_rem_Msun    [M_sun]  (via RV15 inversion)
  - Effective radius        Re_kpc            [kpc]    (Morishita+2024)
  - Central escape speed    Vesc0_kms         [km/s]   (NFW + Hernquist, Model B)

Kick velocities and TDE yields are not computed here. The downstream
`ratio_scan_vcent.py` script uses this catalogue to scan R = V_kick / V_cent.
The bin_targets_physical.parquet table stores uncapped physical merger targets
used later for rescaling.

Output
------
  ratio_scan_catalogue/runXX/data_z_*.parquet       one file per snapshot per run
  results_bin_targets/bin_targets_physical.parquet

Dependencies
------------
  physics_relations.py  —  all physical relations and scaling laws
"""

import os

# Limit BLAS thread counts before importing NumPy for reproducible, stable runtime.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from physics_relations import (
    gsmf_number_density_in_bin,
    merger_rate_z,
    age_of_universe_at_z,
    RV15AGNParams,
    mstar_from_mbh,
    final_mass_and_fraction_ns_jf2017,
    re_kpc_m24,
    FIRE2_SHMR_Params,
    fire2_log10_mh_from_mstar,
    vesc0_nfw_hernquist,
)


# ===========================================================================
# Output paths
# ===========================================================================

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

CATALOGUE_DIR   = os.path.join(BASE_DIR, "ratio_scan_catalogue")
BIN_TARGETS_DIR = os.path.join(BASE_DIR, "results_bin_targets")


# ===========================================================================
# Parquet engine detection (pyarrow preferred, fastparquet as fallback)
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

# Used here for comoving shell volumes; physics_relations uses the same parameters internally.
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.31, Ob0=0.048, Tcmb0=2.725 * u.K)


# ===========================================================================
# EoR redshift window
# ===========================================================================

Z_EOR_INITIAL = 12.0   # upper EoR boundary used for the first time interval
Z_START       = 11.8   # first snapshot
Z_END         = 6.2    # last snapshot
DZ_SNAP       = -0.2   # snapshot spacing


# ===========================================================================
# Comoving shell thickness for galaxy counts
# ===========================================================================

DZ_SLICE   = 0.01
OMEGA_FULL = 4.0 * np.pi  # full sky [sr]
MIN_COUNT  = 1.0           # minimum galaxy count to include a bin


# ===========================================================================
# Host stellar-mass binning  (log10 M_sun)
# ===========================================================================

LOGM_MIN  = 6.75    # lower edge of the global mass grid
LOGM_MAX  = 11.65   # upper edge of the global mass grid
BIN_WIDTH = 0.35

# Only bins within this range are kept for the downstream V_kick/V_cent analysis
KEEP_LOGM_MIN = 6.75
KEEP_LOGM_MAX = 9.90


# ===========================================================================
# BH mass bounds  (log10 M_BH / M_sun in [3, 6]; component masses)
# ===========================================================================

MBH_MIN      = 1.0e3   # lower component mass bound [M_sun]
MBH_MAX      = 1.0e6   # upper component mass bound [M_sun]   — log10 range [3, 6] per thesis
MBH_POST_MAX = 2.0e6   # post-merger remnant mass cap [M_sun]  — equal-mass merger at MBH_MAX


# ===========================================================================
# Bulge fraction  (Model B: NFW + Hernquist bulge)
# ===========================================================================

# Omega_b / Omega_m from the adopted FIRE-2 cosmology
F_BULGE = 0.1548


# ===========================================================================
# Multi-run Monte Carlo controls
# ===========================================================================

N_RUNS      = 10
BASE_SEED   = 12345
SEED_STRIDE = 10_000


# ===========================================================================
# Rejection-sampling controls
# ===========================================================================

# Exponent used when prioritising bins with larger remaining sampled deficits.
ALPHA_DEFICIT        = 1.5

# Maximum total draw attempts = multiplier × total target events for the snapshot
MAX_TRIES_MULTIPLIER = 200

# Per-bin Monte Carlo cap; per-row weights rescale capped samples to physical targets.
MAX_EVENTS_PER_BIN   = 50_000


# ===========================================================================
# Verbosity
# ===========================================================================

PRINT_COUNTS_TABLE = False
PRINT_FILL_SUMMARY = True


# ===========================================================================
# Static parameter objects  (initialised once at import time)
# ===========================================================================

rv15         = RV15AGNParams()
fire2_params = FIRE2_SHMR_Params()

# Pre-compute log10(MBH_MIN) for use inside the tight sampling loop
_LOG10_MBH_MIN = float(np.log10(MBH_MIN))


# ===========================================================================
# Caches  (avoid recomputing GSMF integrals and shell volumes across bins)
# ===========================================================================

_gsmf_cache   = {}  # (z_rounded, log10M_lo, log10M_hi) -> n_bin  [Mpc^-3]
_volume_cache = {}  # (z_rounded, dz)                   -> V_shell [Mpc^3]


# ===========================================================================
# Helpers
# ===========================================================================

def make_mass_bin_edges(logm_min=LOGM_MIN, logm_max=LOGM_MAX, width=BIN_WIDTH):
    """Return an array of bin edges spanning [logm_min, logm_max] in steps of `width`."""
    n_steps = int(np.floor((logm_max - logm_min) / width + 0.5))
    edges = logm_min + width * np.arange(n_steps + 1)
    # Snap the final edge to logm_max to avoid floating-point drift
    if edges[-1] < logm_max - 1e-9:
        edges = np.append(edges, logm_max)
    else:
        edges[-1] = logm_max
    return edges


def snapshot_file_tag(z):
    """Return a filename-safe redshift string for a snapshot, e.g. 'z_11_8'."""
    z_tag = f"{z:.2f}".replace(".", "_")
    return "z_" + z_tag.rstrip("0").rstrip("_")


def cached_volume_shell(z, dz=DZ_SLICE):
    """Full-sky comoving volume of a thin redshift shell of thickness dz at z [Mpc^3]."""
    z_key = float(f"{z:.2f}")
    key = (z_key, float(dz))
    if key in _volume_cache:
        return _volume_cache[key]
    dV_dz_dOmega = cosmo.differential_comoving_volume(z_key).to(u.Mpc**3 / u.sr).value
    V_shell = dV_dz_dOmega * dz * OMEGA_FULL
    _volume_cache[key] = V_shell
    return V_shell


def cached_gsmf_number_density(z, log10M_lo, log10M_hi):
    """Comoving number density ∫φ d(log10 M*) [Mpc^-3] in a mass bin at z (cached)."""
    z_key = float(f"{z:.2f}")
    key = (z_key, float(log10M_lo), float(log10M_hi))
    if key in _gsmf_cache:
        return _gsmf_cache[key]
    n_bin = gsmf_number_density_in_bin(
        z_key, log10M_lo, log10M_hi,
        n_steps=2048, method="pchip", allow_extrapolation=False,
    )
    _gsmf_cache[key] = n_bin
    return n_bin


def galaxy_count_in_mass_bin(z, log10M_lo, log10M_hi, dz=DZ_SLICE):
    """
    Expected galaxy count in a stellar-mass bin within a full-sky shell of thickness dz.

    Returns (N_galaxies, n_bin [Mpc^-3], V_shell [Mpc^3]).
    """
    n_bin   = cached_gsmf_number_density(z, log10M_lo, log10M_hi)
    V_shell = cached_volume_shell(z, dz=dz)
    return float(n_bin * V_shell), float(n_bin), float(V_shell)


# ===========================================================================
# Rejection-sampling core
# ===========================================================================

def build_bin_sampling_contexts(bin_targets):
    """
    Precompute per-bin BH-mass windows for rejection sampling.

    For each target bin, the RV15 forward mapping is used only to define the
    proposal interval for BH component masses implied by the stellar-mass bin
    edges. Only bins with a valid BH window (and at least one event to draw)
    are included.

    Parameters
    ----------
    bin_targets : list of (log_lo, log_hi, n_sample)

    Returns
    -------
    list of context dicts, one per valid bin.
    """
    contexts = []
    for log_lo, log_hi, n_events in bin_targets:
        if n_events <= 0:
            continue

        Mstar_lo = 10.0 ** float(log_lo)
        Mstar_hi = 10.0 ** float(log_hi)

        # RV15 forward map: BH mass window implied by the stellar-mass bin edges
        mbh_at_lo = 10.0 ** (rv15.alpha + rv15.beta * np.log10(Mstar_lo / rv15.M_pivot))
        mbh_at_hi = 10.0 ** (rv15.alpha + rv15.beta * np.log10(Mstar_hi / rv15.M_pivot))

        bh_lo = max(MBH_MIN, min(mbh_at_lo, mbh_at_hi))
        bh_hi = min(MBH_MAX, max(mbh_at_lo, mbh_at_hi))

        if not (np.isfinite(bh_lo) and np.isfinite(bh_hi) and bh_hi > bh_lo):
            continue

        contexts.append({
            "log_lo":            float(log_lo),
            "log_hi":            float(log_hi),
            "n_events":          int(n_events),
            "bh_lo":             float(bh_lo),
            "bh_hi":             float(bh_hi),
            "log10_bh_lo":       float(np.log10(bh_lo)),
            "log10_bh_hi":       float(np.log10(bh_hi)),
            # Filled later after physical and sampled targets are known.
            "n_events_physical": None,
            "weight":            None,
        })
    return contexts


def draw_merger_remnant(ctx, z, rng):
    """
    Attempt to draw one BH–BH merger remnant landing in the host bin [log_lo, log_hi).

    Procedure:
      1. Draw the heavier component m2 log-uniformly in the bin's BH window.
      2. Draw the lighter component m1 log-uniformly in [MBH_MIN, m2].
      3. Compute the remnant mass with the JF2017 non-spinning fit.
      4. Invert RV15 to obtain the host-galaxy stellar mass; reject if outside the bin.
      5. Compute effective radius and central escape speed
         (NFW + Hernquist, Model B).

    Returns a tuple on success, or None if any step fails or the event falls
    outside the target stellar-mass bin.
    """
    log_lo = ctx["log_lo"]
    log_hi = ctx["log_hi"]

    # Draw the heavier component.
    m2 = 10.0 ** rng.uniform(ctx["log10_bh_lo"], ctx["log10_bh_hi"])
    if not (np.isfinite(m2) and m2 > 0.0):
        return None

    # Draw the lighter component.
    log10_m2 = float(np.log10(m2))
    if log10_m2 <= _LOG10_MBH_MIN:
        return None
    m1 = 10.0 ** rng.uniform(_LOG10_MBH_MIN, log10_m2)
    if not (np.isfinite(m1) and m1 > 0.0):
        return None

    if m1 > m2:
        m1, m2 = m2, m1

    # Compute the remnant mass with the JF2017 non-spinning fit.
    Mrem, _ = final_mass_and_fraction_ns_jf2017(float(m1), float(m2))
    if not (np.isfinite(Mrem) and Mrem > 0.0):
        return None
    # Apply the post-merger remnant-mass cap.
    Mrem = min(Mrem, MBH_POST_MAX)

    # Invert RV15 to obtain the host-galaxy stellar mass.
    Mstar = mstar_from_mbh(float(Mrem))
    if not (np.isfinite(Mstar) and Mstar > 0.0):
        return None
    if not (log_lo <= np.log10(Mstar) < log_hi):
        return None

    # Compute effective radius.
    Re = re_kpc_m24(float(Mstar), float(z))
    if not (np.isfinite(Re) and Re > 0.0):
        return None

    # Compute halo mass and central escape speed.
    try:
        log10_Mh = fire2_log10_mh_from_mstar(float(Mstar), fire2_params)
        Mh    = 10.0 ** float(log10_Mh)
        Mb    = F_BULGE * float(Mstar)
        Vesc0 = vesc0_nfw_hernquist(float(Mh), float(Mb), float(Re), float(z))
    except Exception:
        return None

    if not (np.isfinite(Vesc0) and Vesc0 > 0.0):
        return None

    return (log_lo, log_hi, float(z), float(Mrem), float(Mstar), float(Re), float(Vesc0))


def fill_snapshot_remnants(bin_targets_print, contexts, z, rng):
    """
    Fill all bins to their sampled integer targets via rejection sampling.

    At each trial, a bin is chosen with probability proportional to its remaining
    deficit raised to ALPHA_DEFICIT. A merger remnant is then drawn and accepted
    only if it falls inside the chosen bin. This continues until all bins are
    filled or the maximum draw count is reached.

    Parameters
    ----------
    bin_targets_print : list of (lo, hi, n_phys, n_samp) — used for summary printing only.
    contexts          : prebuilt sampling contexts from build_bin_sampling_contexts.
    z                 : current snapshot redshift.
    rng               : numpy Generator instance.

    Returns
    -------
    df      : DataFrame of generated remnants for this snapshot.
    summary : dict keyed by (lo, hi) with target and fill counts.
    """
    if not contexts:
        return pd.DataFrame(), {}

    n_bins = len(contexts)

    bin_lo_arr  = np.array([c["log_lo"] for c in contexts], dtype=float)
    bin_hi_arr  = np.array([c["log_hi"] for c in contexts], dtype=float)
    bin_weights = np.array([float(c["weight"]) for c in contexts], dtype=float)

    deficits        = np.array([int(c["n_events"]) for c in contexts], dtype=np.int64)
    remaining_total = int(deficits.sum())
    max_tries       = int(MAX_TRIES_MULTIPLIER * remaining_total)

    filled = np.zeros(n_bins, dtype=np.int64)

    # Output columns stored as lists for fast appending; assembled into a DataFrame at the end
    out_bin_lo = []
    out_bin_hi = []
    out_z      = []
    out_mrem   = []
    out_mstar  = []
    out_re     = []
    out_vesc0  = []
    out_weight = []

    n_tries = 0
    while n_tries < max_tries and remaining_total > 0:
        # Select a bin weighted by (remaining deficit)^ALPHA_DEFICIT
        bin_weights_raw = deficits.astype(float) ** ALPHA_DEFICIT
        bin_weights_sum = bin_weights_raw.sum()
        if bin_weights_sum <= 0.0:
            break
        probs = bin_weights_raw / bin_weights_sum

        bin_idx = int(rng.choice(n_bins, p=probs))
        if deficits[bin_idx] <= 0:
            n_tries += 1
            continue

        row = draw_merger_remnant(contexts[bin_idx], z, rng)
        n_tries += 1

        if row is None:
            continue

        lo, hi, zc, Mrem, Mstar, Re, Vesc0 = row

        out_bin_lo.append(lo)
        out_bin_hi.append(hi)
        out_z.append(zc)
        out_mrem.append(Mrem)
        out_mstar.append(Mstar)
        out_re.append(Re)
        out_vesc0.append(Vesc0)
        out_weight.append(bin_weights[bin_idx])

        deficits[bin_idx]  -= 1
        filled[bin_idx]    += 1
        remaining_total    -= 1

    df = pd.DataFrame({
        "bin_lo_log10M":  out_bin_lo,
        "bin_hi_log10M":  out_bin_hi,
        "z":              out_z,
        "Mrem_BH_Msun":   out_mrem,
        "Mstar_rem_Msun": out_mstar,
        "Re_kpc":         out_re,
        "Vesc0_kms":      out_vesc0,
        "weight":         out_weight,
    })

    summary = {}
    if PRINT_FILL_SUMMARY:
        print("\n  === Per-bin fill summary ===")
        print(f"  {'Bin [log10 M*]':>18}  {'Target (phys)':>14}  {'Sampled':>10}  {'Filled':>8}")

        bin_index = {(float(bin_lo_arr[j]), float(bin_hi_arr[j])): j for j in range(n_bins)}

        for lo, hi, n_phys, n_samp in bin_targets_print:
            key = (float(lo), float(hi))
            j = bin_index.get(key, None)
            n_filled = int(filled[j]) if j is not None else 0
            summary[key] = {"target_phys": int(n_phys), "target_samp": int(n_samp), "filled": n_filled}
            print(f"  [{float(lo):5.2f}, {float(hi):5.2f}]  {int(n_phys):14d}  {int(n_samp):10d}  {n_filled:8d}")

        # Warn about any bins that fell short of their sampled target
        short_bins = [
            (float(bin_lo_arr[j]), float(bin_hi_arr[j]), int(deficits[j]))
            for j in range(n_bins) if deficits[j] > 0
        ]
        if short_bins:
            print("\n  [WARNING] Some bins did not reach their sampled target (max tries exceeded):")
            for lo, hi, deficit in short_bins:
                print(f"    [{lo:.2f}, {hi:.2f}]  short by {deficit:,}")

    return df, summary


# ===========================================================================
# Snapshot plan  (precomputed once, reused across all Monte Carlo runs)
# ===========================================================================

def precompute_snapshot_plan(z_values, active_bin_edges):
    """
    For each snapshot, compute the expected galaxy counts and integer merger targets per bin.

    These quantities depend only on the GSMF and merger rate — not on the random seed —
    so they are computed once and reused across all N_RUNS Monte Carlo realisations.
    The sampling contexts (BH windows, per-row weights) are also built here and stored
    in the plan for direct reuse by fill_snapshot_remnants.

    Parameters
    ----------
    z_values         : list of snapshot redshifts (descending).
    active_bin_edges : list of (lo, hi) tuples for the stellar-mass bins to process.

    Returns
    -------
    list of snapshot plan dicts.
    """
    plan = []

    for idx, z_cur in enumerate(z_values):
        # Merger rate at the midpoint of the redshift interval, dt from ages
        if idx == 0:
            z_mid  = z_cur
            t_prev = float(age_of_universe_at_z(Z_EOR_INITIAL))
            t_curr = float(age_of_universe_at_z(z_cur))
        else:
            z_prev = z_values[idx - 1]
            z_mid  = 0.5 * (z_prev + z_cur)
            t_prev = float(age_of_universe_at_z(z_prev))
            t_curr = float(age_of_universe_at_z(z_cur))

        R_Gyr  = float(merger_rate_z(z_mid))
        dt_Gyr = (t_curr - t_prev) / 1e9  # yr → Gyr

        # Expected galaxy counts and merger targets per mass bin
        counts_rows   = []
        total_N       = 0.0
        total_mergers = 0.0

        for lo, hi in active_bin_edges:
            N, n_bin, _ = galaxy_count_in_mass_bin(z_cur, lo, hi, dz=DZ_SLICE)
            if N < MIN_COUNT:
                continue
            mergers_bin = N * R_Gyr * dt_Gyr
            counts_rows.append((float(lo), float(hi), float(N), float(n_bin), float(mergers_bin)))
            total_N       += float(N)
            total_mergers += float(mergers_bin)

        # Convert to integer targets and apply the per-bin sample cap
        phys_by_bin        = {}
        bin_targets_sample = []
        bin_targets_print  = []

        for lo, hi, N, _, _ in counts_rows:
            n_phys = int(np.round(N * R_Gyr * dt_Gyr))
            if n_phys <= 0:
                continue
            n_samp = int(min(n_phys, MAX_EVENTS_PER_BIN))

            phys_by_bin[(float(lo), float(hi))] = n_phys

            if n_samp > 0:
                bin_targets_sample.append((float(lo), float(hi), int(n_samp)))
                bin_targets_print.append((float(lo), float(hi), int(n_phys), int(n_samp)))

        # Build sampling contexts (BH windows, etc.) — shared and reused across all runs
        contexts = build_bin_sampling_contexts(bin_targets_sample)

        # Attach physical targets and per-row weights to each context
        for ctx in contexts:
            key    = (float(ctx["log_lo"]), float(ctx["log_hi"]))
            n_phys = phys_by_bin.get(key)
            if n_phys is None:
                continue
            n_samp = int(ctx["n_events"])
            ctx["n_events_physical"] = int(n_phys)
            ctx["weight"] = float(n_phys) / float(n_samp) if n_samp > 0 else np.nan

        plan.append({
            "z_cur":             float(z_cur),
            "z_mid":             float(z_mid),
            "R_Gyr":             float(R_Gyr),
            "dt_Gyr":            float(dt_Gyr),
            "counts_rows":       counts_rows,
            "total_N":           float(total_N),
            "total_mergers":     float(total_mergers),
            "bin_targets_print": bin_targets_print,
            "contexts":          contexts,
        })

    return plan


# ===========================================================================
# Save routines
# ===========================================================================

def _write_parquet(df, path):
    """Write a DataFrame to Parquet using the detected engine and codec."""
    codec = _PARQUET_CODEC
    if _PARQUET_ENGINE == "fastparquet":
        codec = codec.upper()
    df.to_parquet(path, engine=_PARQUET_ENGINE, compression=codec, index=False)


def save_remnant_catalogue(df, out_base, run_tag):
    """
    Save the merger-remnant catalogue for one snapshot to Parquet.

    Float and integer columns are downcast only to reduce file size.
    Output: ratio_scan_catalogue/<run_tag>/<out_base>.parquet
    """
    out_dir  = os.path.join(CATALOGUE_DIR, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_base}.parquet")

    desired_cols = [
        "event_id", "run_id", "z",
        "bin_lo_log10M", "bin_hi_log10M",
        "Mrem_BH_Msun", "Mstar_rem_Msun",
        "Re_kpc", "Vesc0_kms", "weight",
    ]
    keep_cols = [c for c in desired_cols if c in df.columns]

    if len(df) == 0:
        df_save = pd.DataFrame(columns=keep_cols)
    else:
        df_save = df[keep_cols].copy()
        for col in df_save.select_dtypes(include=["float64"]).columns:
            df_save[col] = pd.to_numeric(df_save[col], downcast="float")
        for col in df_save.select_dtypes(include=["int64"]).columns:
            df_save[col] = pd.to_numeric(df_save[col], downcast="integer")

    _write_parquet(df_save, out_path)
    print(f"  Saved: {out_path}  ({len(df_save):,} rows)")


def save_bin_targets(snapshot_plan):
    """
    Save the physical (pre-cap) expected merger counts per (z, bin) to Parquet.

    This file records the deterministic, sampling-cap-independent merger targets
    and serves as a reference table for the downstream analysis.

    Columns: z, bin_lo_log10M, bin_hi_log10M, expected_mergers_float, n_events_phys.
    Output: results_bin_targets/bin_targets_physical.parquet
    """
    os.makedirs(BIN_TARGETS_DIR, exist_ok=True)

    rows = []
    for snap in snapshot_plan:
        z_cur = float(snap["z_cur"])
        for lo, hi, _, _, mergers_float in snap["counts_rows"]:
            exp_mergers = float(mergers_float)
            if not np.isfinite(exp_mergers) or exp_mergers <= 0.0:
                continue
            n_phys = int(np.round(exp_mergers))
            if n_phys <= 0:
                continue
            rows.append({
                "z":                      z_cur,
                "bin_lo_log10M":          float(lo),
                "bin_hi_log10M":          float(hi),
                "expected_mergers_float": exp_mergers,
                "n_events_phys":          n_phys,
            })

    df = pd.DataFrame(rows)

    if len(df):
        df["z"]                      = pd.to_numeric(df["z"],                      downcast="float")
        df["bin_lo_log10M"]          = pd.to_numeric(df["bin_lo_log10M"],          downcast="float")
        df["bin_hi_log10M"]          = pd.to_numeric(df["bin_hi_log10M"],          downcast="float")
        df["expected_mergers_float"] = pd.to_numeric(df["expected_mergers_float"], downcast="float")
        df["n_events_phys"]          = pd.to_numeric(df["n_events_phys"],          downcast="integer")

    out_path = os.path.join(BIN_TARGETS_DIR, "bin_targets_physical.parquet")
    _write_parquet(df, out_path)
    print(f"[bin targets] Saved: {out_path}  ({len(df):,} rows)")


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Generate capped Monte Carlo remnant catalogues and physical target tables for the ratio scan."""
    # Build the list of EoR snapshots (z = 11.8 -> 6.2, step = -0.2)
    z_values = []
    z = Z_START
    while z >= Z_END - 1e-8:
        z_values.append(float(f"{z:.2f}"))
        z += DZ_SNAP

    # Stellar-mass bin edges and the subset used for the V_kick/V_cent analysis
    all_edges    = make_mass_bin_edges()
    active_edges = [
        (lo, hi)
        for lo, hi in zip(all_edges[:-1], all_edges[1:])
        if lo >= KEEP_LOGM_MIN and hi <= KEEP_LOGM_MAX
    ]

    # Precompute the snapshot plan (counts, targets, sampling contexts) — done once for all runs
    print("Precomputing snapshot plan...")
    snapshot_plan = precompute_snapshot_plan(z_values, active_edges)

    # Save the deterministic bin targets (independent of run seed)
    save_bin_targets(snapshot_plan)

    # Monte Carlo loop: N_RUNS independent realisations
    for run_idx in range(N_RUNS):
        run_tag   = f"run{run_idx:02d}"
        seed_base = BASE_SEED + run_idx * SEED_STRIDE

        print("\n" + "=" * 70)
        print(f"  RUN {run_tag}   (seed_base = {seed_base})")
        print("=" * 70)

        for idx, snap in enumerate(snapshot_plan):
            z_cur    = snap["z_cur"]
            rng      = np.random.default_rng(seed_base + idx)
            file_tag = snapshot_file_tag(z_cur)
            out_base = f"data_{file_tag}"

            z_mid         = snap["z_mid"]
            R_Gyr         = snap["R_Gyr"]
            dt_Gyr        = snap["dt_Gyr"]
            counts_rows   = snap["counts_rows"]
            total_N       = snap["total_N"]
            total_mergers = snap["total_mergers"]
            contexts      = snap["contexts"]
            bin_targets_display = snap["bin_targets_print"]

            print(f"\n  [{run_tag}] z = {z_cur:.2f}  |  "
                  f"R_M(z_mid={z_mid:.2f}) = {R_Gyr:.4e} Gyr^-1  |  dt = {dt_Gyr:.4e} Gyr")
            print(f"  Sample cap: {MAX_EVENTS_PER_BIN:,} events/bin  "
                  f"(per-row weights scale to physical totals)")

            if PRINT_COUNTS_TABLE:
                print(f"\n  {'Mass bin [log10 Msun]':>24}  {'N (count)':>14}  "
                      f"{'n [Mpc^-3]':>14}  {'Mergers':>12}")
                print("  " + "-" * 68)
                for lo, hi, N, n_bin, mergers_bin in counts_rows:
                    print(f"  [{lo:5.2f}, {hi:5.2f}]".rjust(28),
                          f"{N:14.6e}  {n_bin:14.6e}  {mergers_bin:12.6e}")
                print("  " + "-" * 68)
                print(f"  {'TOTAL:':>28}  {total_N:14.6e}  {'':14}  {total_mergers:12.6e}")

            # Generate remnants for this snapshot
            df_snap, _ = fill_snapshot_remnants(bin_targets_display, contexts, z_cur, rng)

            # Add stable event IDs and run label
            if "event_id" not in df_snap.columns:
                df_snap = df_snap.reset_index(drop=False).rename(columns={"index": "event_id"})
            df_snap["run_id"] = run_tag

            print(f"  [{run_tag}] z = {z_cur:.2f}  →  {len(df_snap):,} remnants generated")

            save_remnant_catalogue(df_snap, out_base, run_tag)

        print(f"\n  Finished {run_tag}.")

    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
