#!/usr/bin/env python3
"""
simulation.py
-------------
Final downstream recoil/TDE simulation for the revised explicit galaxy-pair
population.

This version does not regenerate the Monte Carlo population.  It reuses the
catalogues already produced by ratio_scan_sample_gen.py, which guarantees that
the final simulation and the V_kick/V_cent scan use the same major+minor merger
events, descendant mass bins, and single cap-rescaling weight.

Inputs
------
  ratio_scan_catalogue/runXX/data_z_*.parquet
  ratio_scan_vcent_rpeak__fcent0p05.json

For each catalogue the script:
  1. validates the shared population-model version;
  2. loads the combined major+minor R_peak for each descendant mass bin;
  3. assigns V_kick = R_peak V_cent using the same representative-potential
     prescription as ratio_scan_vcent.py;
  4. computes the central/external orbit times and TDE counts;
  5. writes the enriched catalogue to simulation_results/runXX/.

The expensive population sampling is therefore performed only once, upstream.
All physical equations are unchanged.  Speed improvements come from catalogue
reuse, vectorised TDE calculations, one representative orbit calculation per
(snapshot, descendant bin), minimal Parquet I/O, and optional snapshot-level
parallelism.

Outputs
-------
  simulation_results/runXX/data_z_*.parquet
"""

from __future__ import annotations

import os

# Keep each worker single-threaded.  Parallelism is across independent files.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import merger_pair_sampling as mps
import physics_relations as pr


# ============================================================================
# Paths and execution controls
# ============================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

CATALOGUE_DIR = BASE_DIR / "ratio_scan_catalogue"
RESULTS_DIR = BASE_DIR / "simulation_results"

F_NUC = 0.05
F_BULGE = float(mps.F_BULGE)
HERNQUIST_RE_OVER_A = 1.8153

# Same representative-orbit integration resolution as ratio_scan_vcent.py.
N_INT = 320

# Two workers matches the conservative setting used by the updated ratio scan.
# Set SIMULATION_WORKERS=1 for serial execution or increase it when RAM allows.
DEFAULT_WORKERS = 2
N_WORKERS = max(1, int(os.environ.get("SIMULATION_WORKERS", DEFAULT_WORKERS)))

PRINT_PER_FILE_SUMMARY = True


# ============================================================================
# Shared mass bins and ratio grid
# ============================================================================

MASS_EDGES = np.asarray(mps.DESCENDANT_LOGM_EDGES, dtype=float)
MASS_LABELS = [mps.bin_label(MASS_EDGES[i], MASS_EDGES[i + 1])
               for i in range(len(MASS_EDGES) - 1)]
N_MASS = len(MASS_LABELS)

RATIO_MIN = 1.0
RATIO_MAX = 10.0
DR = 0.10
N_R = int(np.round((RATIO_MAX - RATIO_MIN) / DR))
RATIO_EDGES = RATIO_MIN + DR * np.arange(N_R + 1)
RATIO_CENTERS = RATIO_EDGES[:-1] + 0.5 * DR


# ============================================================================
# Physical constants and TDE controls
# ============================================================================

SEC_PER_YEAR = 365.25 * 24.0 * 3600.0
KPC_TO_KM = 3.085677581e16
M_STAR_SI = 1.3 * pr.M_sun
R_STAR_SI = 1.3 * 6.957e8

TDECAY_MODE = "vkick"
TDECAY_NO_DECAY_YR = 1.0e30


# ============================================================================
# Parquet support
# ============================================================================

_PARQUET_ENGINE = None
_PARQUET_CODEC = None
try:
    import pyarrow as pa

    _PARQUET_ENGINE = "pyarrow"
    try:
        _PARQUET_CODEC = "zstd" if pa.Codec.is_available("zstd") else "snappy"
    except Exception:
        _PARQUET_CODEC = "snappy"
except ImportError:
    try:
        import fastparquet  # noqa: F401

        _PARQUET_ENGINE = "fastparquet"
        try:
            from fastparquet.compress import compr

            _PARQUET_CODEC = "ZSTD" if "ZSTD" in compr else "SNAPPY"
        except Exception:
            _PARQUET_CODEC = "SNAPPY"
    except ImportError as exc:
        raise ImportError(
            "A Parquet engine is required. Install pyarrow: pip install pyarrow"
        ) from exc


SOURCE_COLUMNS = [
    "event_id", "run_id", "z", "z_rate", "population_model_version",
    "merger_class",
    "descendant_bin_lo_log10M", "descendant_bin_hi_log10M",
    "desc_bin_lo_log10M", "desc_bin_hi_log10M",
    "bin_lo_log10M", "bin_hi_log10M",
    "Mstar_primary_Msun", "Mstar_secondary_Msun", "mu_star",
    "m1_BH_Msun", "m2_BH_Msun", "q", "Mrem_BH_Msun",
    "Mstar_rem_Msun", "Re_kpc", "log10_Mh_fire2", "Vesc0_kms",
    "weight",
]

REQUIRED_SOURCE_COLUMNS = {
    "population_model_version", "merger_class",
    "Mstar_rem_Msun", "Mrem_BH_Msun", "Re_kpc",
    "log10_Mh_fire2", "Vesc0_kms", "weight",
}


# ============================================================================
# Catalogue discovery and validation
# ============================================================================


def _parse_redshift_from_filename(filename: str) -> float | None:
    """Extract z from data_z_11_8.parquet-like filenames."""
    match = re.fullmatch(r"data_z_(\d+(?:_\d+)?)\.parquet", filename)
    if match is None:
        return None
    try:
        return round(float(match.group(1).replace("_", ".")), 2)
    except ValueError:
        return None


def discover_catalogues(root: Path = CATALOGUE_DIR) -> tuple[list[float], dict[float, list[Path]]]:
    """Return descending redshifts and sorted run files for each snapshot."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing ratio-scan catalogue directory: {root}\n"
            "Run ratio_scan_sample_gen.py first."
        )

    paths_by_z: dict[float, list[Path]] = {}
    for path in root.glob("run*/data_z_*.parquet"):
        z = _parse_redshift_from_filename(path.name)
        if z is not None:
            paths_by_z.setdefault(z, []).append(path)

    # Legacy root-level support, without weakening the model-version audit.
    for path in root.glob("data_z_*.parquet"):
        z = _parse_redshift_from_filename(path.name)
        if z is not None:
            paths_by_z.setdefault(z, []).append(path)

    if not paths_by_z:
        raise FileNotFoundError(f"No data_z_*.parquet catalogues found below {root}")

    for paths in paths_by_z.values():
        paths.sort(key=lambda p: (p.parent.name, p.name))

    return sorted(paths_by_z, reverse=True), paths_by_z


def _read_catalogue(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read one revised catalogue and fail fast on incompatible inputs."""
    requested = list(columns) if columns is not None else SOURCE_COLUMNS
    try:
        df = pd.read_parquet(path, columns=requested)
    except Exception:
        # A full read gives a clearer missing-column error for unusual engines.
        df = pd.read_parquet(path)

    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(df.columns))
    if missing:
        raise KeyError(
            f"Incomplete revised catalogue {path}: missing {missing}. "
            "Rerun ratio_scan_sample_gen.py with the matching updated files."
        )

    versions = set(df["population_model_version"].dropna().astype(str).unique())
    if versions != {mps.MODEL_VERSION}:
        raise RuntimeError(
            f"Incompatible population model in {path}: found {sorted(versions)}, "
            f"expected {mps.MODEL_VERSION!r}."
        )

    classes = set(df["merger_class"].dropna().astype(str).str.lower().unique())
    if not classes.issubset({mps.MERGER_MAJOR, mps.MERGER_MINOR}):
        raise RuntimeError(f"Unexpected merger_class values in {path}: {sorted(classes)}")

    return df


# ============================================================================
# Peak-ratio loading
# ============================================================================


def _fcent_tag(value: float) -> str:
    return f"fcent{value:.2f}".replace(".", "p")


def load_peak_ratios(f_nuc: float = F_NUC) -> tuple[dict[str, float], np.ndarray]:
    """Load and validate the combined major+minor R_peak table."""
    filename = f"ratio_scan_vcent_rpeak__{_fcent_tag(f_nuc)}.json"
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Peak-ratio JSON not found: {path}\nRun ratio_scan_vcent.py first."
        )

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    meta = payload.get("meta", {})
    if "F_CENT" in meta and not np.isclose(float(meta["F_CENT"]), f_nuc):
        raise RuntimeError(
            f"Peak file {filename} has F_CENT={meta['F_CENT']}, expected {f_nuc}."
        )
    if meta.get("mass_bin_meaning") not in (None, "descendant galaxy stellar mass"):
        raise RuntimeError(f"Unexpected mass-bin meaning in {filename}: {meta.get('mass_bin_meaning')}")

    table = payload.get("R_peak_by_bin", {})
    missing = [label for label in MASS_LABELS if label not in table]
    if missing:
        raise RuntimeError(f"Peak table {filename} is missing descendant bins: {missing}")

    values = np.asarray([float(table[label]) for label in MASS_LABELS], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < RATIO_MIN) or np.any(values > RATIO_MAX):
        raise RuntimeError(f"Invalid R_peak values in {filename}: {values}")

    # Peak values must be ratio-bin centres.  Snapping only removes JSON round-off.
    nearest = np.argmin(np.abs(values[:, None] - RATIO_CENTERS[None, :]), axis=1)
    snapped = RATIO_CENTERS[nearest]
    if not np.allclose(values, snapped, rtol=0.0, atol=1.0e-8):
        raise RuntimeError(
            f"R_peak values do not match the ratio-scan bin centres in {filename}."
        )

    print(f"[peak ratios] Loaded {len(values)} descendant bins from {filename}")
    return {label: float(value) for label, value in zip(MASS_LABELS, snapped)}, snapped


# ============================================================================
# Descendant-bin helpers
# ============================================================================


def descendant_bin_indices(mstar_msun: np.ndarray) -> np.ndarray:
    """Vectorised descendant-bin assignment, including the exact final edge."""
    mstar = np.asarray(mstar_msun, dtype=float)
    logm = np.log10(mstar, where=(mstar > 0.0), out=np.full_like(mstar, np.nan))
    idx = np.searchsorted(MASS_EDGES, logm, side="right") - 1
    idx[np.isclose(logm, MASS_EDGES[-1], rtol=0.0, atol=1.0e-10)] = N_MASS - 1
    return idx.astype(np.int16, copy=False)


# ============================================================================
# Representative potential and orbit calculations
# These functions reproduce the updated ratio_scan_vcent.py implementation.
# ============================================================================

G_KPC = 4.30091e-6  # (km/s)^2 kpc / Msun


def psi_total_scaled_factory(rep_mstar: float, rep_re_kpc: float,
                             rep_vesc0_kms: float, z: float):
    """Return the positive NFW+Hernquist potential scaled to catalogue Vesc0."""
    mh = float(pr.fire2_mh_from_mstar(rep_mstar))
    mb = float(F_BULGE * rep_mstar)

    rvir_kpc = float(pr.r_vir_mpc(mh, z) * 1.0e3)
    concentration = float(pr.nfw_concentration(mh, z))
    f_c = math.log(1.0 + concentration) - concentration / (1.0 + concentration)
    rs = rvir_kpc / concentration
    a = rep_re_kpc / HERNQUIST_RE_OVER_A

    def psi_nfw(r_kpc):
        r = np.asarray(r_kpc, dtype=float)
        term = np.empty_like(r)
        positive = r > 0.0
        term[positive] = np.log1p(r[positive] / rs) / r[positive]
        term[~positive] = 1.0 / rs
        return (G_KPC * mh / f_c) * term

    def psi_hernquist(r_kpc):
        r = np.asarray(r_kpc, dtype=float)
        return (G_KPC * mb) / (r + a)

    psi0_model = float(psi_nfw(0.0) + psi_hernquist(0.0))
    psi0_target = 0.5 * rep_vesc0_kms**2
    scale = 1.0 if (not np.isfinite(psi0_model) or psi0_model <= 0.0) \
        else psi0_target / psi0_model

    def psi(r_kpc):
        return scale * (psi_nfw(r_kpc) + psi_hernquist(r_kpc))

    return psi, rvir_kpc


def travel_time_years(psi, psi_target: float, r1_kpc: float,
                      r2_kpc: float, n: int = N_INT) -> float:
    """Integrate dt=dr/sqrt(2(Psi-psi_target)) using midpoint cells."""
    if r2_kpc <= r1_kpc:
        return 0.0

    edges = np.linspace(r1_kpc, r2_kpc, int(n) + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    diff = psi(mids) - psi_target
    if not np.all(np.isfinite(diff)) or np.any(diff <= 0.0):
        return np.nan

    velocity = np.sqrt(2.0 * diff)
    dt_seconds = (dr / velocity) * KPC_TO_KM
    value = np.sum(dt_seconds) / SEC_PER_YEAR
    return float(value) if np.isfinite(value) else np.nan


def find_rmax_bisect(psi, psi_target: float, r_lo: float,
                     r_hi: float) -> float | None:
    """Find the radial turning point Psi(r)=psi_target by bisection."""
    f_lo = float(psi(r_lo) - psi_target)
    f_hi = float(psi(r_hi) - psi_target)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo <= 0.0 or f_hi >= 0.0:
        return None

    a, b = float(r_lo), float(r_hi)
    for _ in range(70):
        mid = 0.5 * (a + b)
        f_mid = float(psi(mid) - psi_target)
        if not np.isfinite(f_mid):
            return None
        if f_mid > 0.0:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def build_selected_rep_data(df: pd.DataFrame, z: float,
                            rpeak_by_bin: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build representative orbit quantities only at the selected R_peak values."""
    rep_t_leave = np.full(N_MASS, np.nan, dtype=float)
    rep_t_ext_max = np.full(N_MASS, np.nan, dtype=float)
    rep_scale_base = np.full(N_MASS, np.nan, dtype=float)
    rep_vcent_over_vesc0 = np.full(N_MASS, np.nan, dtype=float)

    mstar = pd.to_numeric(df["Mstar_rem_Msun"], errors="coerce").to_numpy(float)
    re_kpc = pd.to_numeric(df["Re_kpc"], errors="coerce").to_numpy(float)
    vesc0 = pd.to_numeric(df["Vesc0_kms"], errors="coerce").to_numpy(float)
    mass_i = descendant_bin_indices(mstar)

    for mass_bin in range(N_MASS):
        selected = (
            (mass_i == mass_bin)
            & np.isfinite(mstar) & (mstar > 0.0)
            & np.isfinite(re_kpc) & (re_kpc > 0.0)
            & np.isfinite(vesc0) & (vesc0 > 0.0)
        )
        if not np.any(selected):
            continue

        rep_mstar = float(np.nanmedian(mstar[selected]))
        rep_re = float(np.nanmedian(re_kpc[selected]))
        rep_vesc = float(np.nanmedian(vesc0[selected]))
        if not (rep_mstar > 0.0 and rep_re > 0.0 and rep_vesc > 0.0):
            continue

        rep_scale_base[mass_bin] = rep_re / rep_vesc
        psi, rvir_kpc = psi_total_scaled_factory(rep_mstar, rep_re, rep_vesc, z)
        r_cent = F_NUC * rep_re
        psi0 = 0.5 * rep_vesc**2
        psi_rcent = float(psi(r_cent))

        if np.isfinite(psi_rcent) and psi0 > 0.0 and psi_rcent < psi0:
            vcent_ratio = math.sqrt(max(0.0, 1.0 - psi_rcent / psi0))
        else:
            vcent_ratio = 1.0
        rep_vcent_over_vesc0[mass_bin] = vcent_ratio

        ratio = float(rpeak_by_bin[mass_bin])
        effective_ratio = ratio * vcent_ratio
        psi_target = psi0 * (1.0 - effective_ratio**2)
        if not np.isfinite(psi_rcent) or psi_rcent <= psi_target:
            continue

        t_leave = travel_time_years(psi, psi_target, 0.0, r_cent)
        if not (np.isfinite(t_leave) and t_leave > 0.0):
            continue
        rep_t_leave[mass_bin] = t_leave

        # psi_target <= 0 is escape-like: the external phase lasts until z=6.
        if psi_target <= 0.0:
            continue

        r_hi = min(max(2.0 * r_cent, 1.2 * r_cent), rvir_kpc)
        psi_hi = float(psi(r_hi))
        tries = 0
        while (np.isfinite(psi_hi) and psi_hi > psi_target
               and r_hi < rvir_kpc and tries < 25):
            r_hi = min(r_hi * 1.6, rvir_kpc)
            psi_hi = float(psi(r_hi))
            tries += 1

        if not (np.isfinite(psi_hi) and psi_hi < psi_target):
            continue

        rmax = find_rmax_bisect(psi, psi_target, r_cent, r_hi)
        if rmax is None or not (np.isfinite(rmax) and rmax > r_cent):
            continue

        t_half = travel_time_years(psi, psi_target, r_cent, rmax)
        if np.isfinite(t_half) and t_half > 0.0:
            rep_t_ext_max[mass_bin] = 2.0 * t_half

    return rep_t_leave, rep_t_ext_max, rep_scale_base, rep_vcent_over_vesc0


def representative_data_for_snapshot(paths: list[Path], z: float,
                                     rpeak_by_bin: np.ndarray) -> tuple[np.ndarray, ...]:
    """Reproduce the ratio scan's first-valid-run representative-bin policy."""
    combined = (
        np.full(N_MASS, np.nan, dtype=float),
        np.full(N_MASS, np.nan, dtype=float),
        np.full(N_MASS, np.nan, dtype=float),
        np.full(N_MASS, np.nan, dtype=float),
    )
    rep_t_leave, rep_t_ext_max, rep_scale, rep_vcent = combined

    minimal = [
        "population_model_version", "merger_class",
        "Mstar_rem_Msun", "Mrem_BH_Msun", "Re_kpc",
        "log10_Mh_fire2", "Vesc0_kms", "weight",
    ]

    for path in paths:
        df = _read_catalogue(path, columns=minimal)
        if df.empty:
            continue

        mstar = pd.to_numeric(df["Mstar_rem_Msun"], errors="coerce").to_numpy(float)
        present = np.unique(descendant_bin_indices(mstar))
        present = present[(present >= 0) & (present < N_MASS)]
        missing = [int(idx) for idx in present if not np.isfinite(rep_scale[int(idx)])]
        if not missing:
            continue

        cand_leave, cand_ext, cand_scale, cand_vcent = build_selected_rep_data(
            df, z, rpeak_by_bin
        )
        for idx in missing:
            if np.isfinite(cand_scale[idx]):
                rep_t_leave[idx] = cand_leave[idx]
                rep_t_ext_max[idx] = cand_ext[idx]
                rep_scale[idx] = cand_scale[idx]
                rep_vcent[idx] = cand_vcent[idx]

    return combined


# ============================================================================
# Vectorised kick and TDE calculations
# ============================================================================


def _integral_exp(rate_yr, t_decay_yr, t0, t1) -> np.ndarray:
    """Integrate rate*exp(-t/t_decay) over [t0,t1] with broadcasting."""
    rate, decay, start, end = np.broadcast_arrays(
        np.asarray(rate_yr, dtype=float),
        np.asarray(t_decay_yr, dtype=float),
        np.asarray(t0, dtype=float),
        np.asarray(t1, dtype=float),
    )
    out = np.zeros_like(rate, dtype=float)
    valid = (
        np.isfinite(rate) & (rate > 0.0)
        & np.isfinite(decay) & (decay > 0.0)
        & np.isfinite(start) & np.isfinite(end) & (end > start)
    )
    if np.any(valid):
        x0 = np.clip(start[valid] / decay[valid], 0.0, 1.0e6)
        x1 = np.clip(end[valid] / decay[valid], 0.0, 1.0e6)
        values = rate[valid] * decay[valid] * (np.exp(-x0) - np.exp(-x1))
        out[valid] = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    return out


def attach_kicks_and_tdes(df: pd.DataFrame, z: float, dt_to_z6_yr: float,
                          rpeak_by_bin: np.ndarray,
                          rep_data: tuple[np.ndarray, ...]) -> pd.DataFrame:
    """Attach kicks, orbit times, and TDE counts without changing the model."""
    if df.empty:
        return df

    rep_t_leave, rep_t_ext_max, rep_scale_base, rep_vcent_ratio = rep_data

    mstar = pd.to_numeric(df["Mstar_rem_Msun"], errors="coerce").to_numpy(float)
    mbh = pd.to_numeric(df["Mrem_BH_Msun"], errors="coerce").to_numpy(float)
    re_kpc = pd.to_numeric(df["Re_kpc"], errors="coerce").to_numpy(float)
    vesc0 = pd.to_numeric(df["Vesc0_kms"], errors="coerce").to_numpy(float)
    mass_i = descendant_bin_indices(mstar).astype(int)

    valid_bins = (mass_i >= 0) & (mass_i < N_MASS)
    if not np.all(valid_bins):
        bad = int(np.count_nonzero(~valid_bins))
        raise RuntimeError(f"{bad} catalogue rows fall outside the shared descendant bins.")

    missing_rep = np.unique(mass_i[~np.isfinite(rep_scale_base[mass_i])])
    if len(missing_rep):
        labels = [MASS_LABELS[int(i)] for i in missing_rep]
        raise RuntimeError(f"Missing representative host potentials for bins {labels} at z={z:.2f}")

    ratio = rpeak_by_bin[mass_i]
    vcent_ratio = rep_vcent_ratio[mass_i]
    vcent_ratio = np.where(np.isfinite(vcent_ratio) & (vcent_ratio > 0.0),
                            vcent_ratio, 1.0)

    vcent = vcent_ratio * vesc0
    # Preserve the multiplication order used by ratio_scan_vcent.py.
    vkick = ratio * vcent_ratio * vesc0
    vkick = np.where(np.isfinite(vkick) & (vkick > 0.0), vkick, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        vkick_over_vesc = vkick / vesc0
        vkick_over_vcent = vkick / vcent

    df["Rpeak_vcent"] = ratio
    df["v_cent_kms"] = vcent
    df["Vkick_kms"] = vkick
    df["Vkick_over_Vesc0"] = vkick_over_vesc
    df["Vkick_over_Vcent"] = vkick_over_vcent
    df["escaped"] = vkick >= vesc0

    mbh_si = mbh * pr.M_sun
    v_k_ms = vkick * pr.km

    # Ratio-invariant row factors, matching the optimised ratio scan.
    sigma_km_s, _ = pr.sigma_from_mbh(mbh_si)
    rt = pr.r_t(mbh_si, M_STAR_SI, R_STAR_SI)
    mb_coll_prefactor_msun = (
        4.0e4
        * (mbh / 1.0e7) ** (-0.25)
        * (sigma_km_s / 100.0) ** 2.5
    )
    ln_lambda = np.log(mbh_si / M_STAR_SI)

    rk = pr.r_k(mbh_si, v_k_ms)
    r_eff_pc = pr.r_eff_from_rk_gamma1(rk) / pr.pc
    mb_coll_kg = (
        mb_coll_prefactor_msun * (r_eff_pc / 0.1) ** 1.25
    ) * pr.M_sun
    cap_stars = np.where(
        np.isfinite(mb_coll_kg) & (mb_coll_kg > 0.0),
        mb_coll_kg / M_STAR_SI,
        0.0,
    )
    f_b_raw = mb_coll_kg / mbh_si
    f_b = np.clip(np.where(np.isfinite(f_b_raw), f_b_raw, 0.0), 0.0, 1.0)

    rk_arr = np.asarray(rk, dtype=float)
    ln_ratio = np.where(rk_arr > rt, np.log(rk_arr / rt), np.inf)
    rate_s = (ln_lambda / ln_ratio) * (v_k_ms / rk_arr) * f_b
    rate_yr = np.where(
        np.isfinite(rate_s) & (rate_s > 0.0), rate_s * SEC_PER_YEAR, 0.0
    )

    t_decay_mass_prefactor = 1.5e9 * (mbh / 1.0e7) ** 2
    if TDECAY_MODE == "vkick":
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            t_decay_yr = t_decay_mass_prefactor * (vkick / 1.0e3) ** (-3.0)
        t_decay_yr = np.where(
            (vkick <= 0.0) & np.isfinite(mbh), TDECAY_NO_DECAY_YR, t_decay_yr
        )
    elif TDECAY_MODE == "mass_only":
        t_decay_yr = t_decay_mass_prefactor
    else:
        raise ValueError(f"Unknown TDECAY_MODE: {TDECAY_MODE}")
    t_decay_yr = np.where(
        np.isfinite(t_decay_yr) & (t_decay_yr > 0.0), t_decay_yr, 0.0
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (re_kpc / vesc0) / rep_scale_base[mass_i]
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)

    t_leave = rep_t_leave[mass_i] * scale
    t_leave = np.where(np.isfinite(t_leave) & (t_leave > 0.0), t_leave, np.nan)
    t_ext_max = rep_t_ext_max[mass_i] * scale

    t_cap = float(dt_to_z6_yr) if np.isfinite(dt_to_z6_yr) and dt_to_z6_yr > 0.0 else 0.0
    active = np.isfinite(t_leave) & (t_leave < t_cap)
    remaining_time = np.where(active, np.maximum(0.0, t_cap - t_leave), 0.0)

    # Same convention as the ratio scan: NaN t_ext_max denotes escape-like motion.
    t_external = np.where(
        active,
        np.where(
            np.isfinite(t_ext_max) & (t_ext_max > 0.0),
            np.minimum(t_ext_max, remaining_time),
            remaining_time,
        ),
        0.0,
    )

    t_nuclear_pre = np.where(active, t_leave, t_cap)
    n_nuclear_pre_unc = _integral_exp(rate_yr, t_decay_yr, 0.0, t_nuclear_pre)
    n_external_unc = _integral_exp(
        rate_yr,
        t_decay_yr,
        np.where(active, t_leave, 0.0),
        np.where(active, t_leave + t_external, 0.0),
    )

    n_nuclear_pre = np.minimum(n_nuclear_pre_unc, cap_stars)
    cap_after_pre = np.maximum(0.0, cap_stars - n_nuclear_pre)
    n_external = np.minimum(n_external_unc, cap_after_pre)

    is_bound = vkick_over_vesc < 1.0
    t_return = t_leave + t_ext_max
    post_duration = np.where(
        active & is_bound & np.isfinite(t_ext_max) & np.isfinite(t_return),
        np.maximum(0.0, t_cap - t_return),
        0.0,
    )
    post_start = np.maximum(0.0, t_cap - post_duration)
    n_nuclear_post_unc = _integral_exp(rate_yr, t_decay_yr, post_start, t_cap)
    cap_after_external = np.maximum(0.0, cap_after_pre - n_external)
    n_nuclear_post = np.minimum(n_nuclear_post_unc, cap_after_external)

    df["tde_rate_per_yr"] = rate_yr
    df["t_decay_yr"] = t_decay_yr
    df["t_esc_yr"] = t_leave
    df["t_external_yr"] = t_external
    df["t_return_yr"] = np.where(
        active & is_bound & np.isfinite(t_return), t_return, np.nan
    )
    df["tde_nuclear_pre"] = n_nuclear_pre
    df["tde_nuclear_post"] = n_nuclear_post
    df["tde_nuclear_preescape"] = n_nuclear_pre + n_nuclear_post
    df["tde_external_post"] = n_external
    df["tde_count_per_bh"] = n_external

    return df


# ============================================================================
# Output
# ============================================================================

OUTPUT_COLUMNS = [
    # Provenance and population weighting
    "event_id", "run_id", "z", "z_rate", "population_model_version",
    "merger_class", "weight",
    "descendant_bin_lo_log10M", "descendant_bin_hi_log10M",
    "desc_bin_lo_log10M", "desc_bin_hi_log10M",
    "bin_lo_log10M", "bin_hi_log10M",
    # Explicit galaxy and BH pair
    "Mstar_primary_Msun", "Mstar_secondary_Msun", "mu_star",
    "m1_BH_Msun", "m2_BH_Msun", "q", "Mrem_BH_Msun",
    "Mstar_rem_Msun", "Re_kpc", "log10_Mh_fire2", "Vesc0_kms",
    # Kick and TDE results
    "Rpeak_vcent", "v_cent_kms", "Vkick_kms",
    "Vkick_over_Vesc0", "Vkick_over_Vcent", "escaped",
    "tde_rate_per_yr", "t_decay_yr", "t_esc_yr", "t_external_yr",
    "t_return_yr", "tde_nuclear_pre", "tde_nuclear_post",
    "tde_nuclear_preescape", "tde_external_post", "tde_count_per_bh",
]


def _downcast_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce file size while preserving exact weights and shared bin edges."""
    out = df.copy()
    preserve_float64 = {
        "weight", "descendant_bin_lo_log10M", "descendant_bin_hi_log10M",
        "desc_bin_lo_log10M", "desc_bin_hi_log10M",
        "bin_lo_log10M", "bin_hi_log10M",
    }
    for column in out.select_dtypes(include=["float64"]).columns:
        if column not in preserve_float64:
            out[column] = pd.to_numeric(out[column], downcast="float")
    for column in out.select_dtypes(include=["int64"]).columns:
        out[column] = pd.to_numeric(out[column], downcast="integer")
    return out


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = _PARQUET_CODEC.upper() if _PARQUET_ENGINE == "fastparquet" else _PARQUET_CODEC
    df.to_parquet(path, engine=_PARQUET_ENGINE, compression=codec, index=False)


def _output_path_for_source(source_path: Path) -> Path:
    run_tag = source_path.parent.name if source_path.parent != CATALOGUE_DIR else "run00"
    return RESULTS_DIR / run_tag / source_path.name


def process_one_catalogue(source_path_str: str, z: float,
                          rpeak_values: np.ndarray,
                          rep_data: tuple[np.ndarray, ...]) -> dict:
    """Worker: enrich one precursor catalogue and write its final Parquet."""
    source_path = Path(source_path_str)
    df = _read_catalogue(source_path)

    if "event_id" not in df.columns:
        df.insert(0, "event_id", np.arange(len(df), dtype=np.int64))
    if "run_id" not in df.columns:
        df.insert(1, "run_id", source_path.parent.name)

    dt_to_z6 = float(pr.time_until_z6(z))
    df = attach_kicks_and_tdes(df, z, dt_to_z6, rpeak_values, rep_data)

    keep = [column for column in OUTPUT_COLUMNS if column in df.columns]
    df_save = _downcast_for_storage(df[keep]) if not df.empty else pd.DataFrame(columns=keep)
    output_path = _output_path_for_source(source_path)
    _write_parquet(df_save, output_path)

    weight = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(float)
    n_ext = pd.to_numeric(df["tde_external_post"], errors="coerce").fillna(0.0).to_numpy(float)
    classes = df["merger_class"].astype(str).str.lower().to_numpy()
    weighted_ext = weight * np.where(np.isfinite(n_ext) & (n_ext > 0.0), n_ext, 0.0)

    return {
        "source": str(source_path),
        "output": str(output_path),
        "z": float(z),
        "run": source_path.parent.name,
        "rows": int(len(df)),
        "weight_sum": float(np.sum(weight)),
        "external_combined": float(np.sum(weighted_ext)),
        "external_major": float(np.sum(weighted_ext[classes == mps.MERGER_MAJOR])),
        "external_minor": float(np.sum(weighted_ext[classes == mps.MERGER_MINOR])),
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Enrich all revised ratio-scan catalogues with final kick/TDE fields."""
    _, rpeak_values = load_peak_ratios(F_NUC)
    z_values, paths_by_z = discover_catalogues()

    print(f"[catalogues] Found {sum(len(v) for v in paths_by_z.values())} files "
          f"across {len(z_values)} snapshots")
    print(f"[workers] SIMULATION_WORKERS={N_WORKERS}")

    print("[representative orbits] Precomputing one selected-ratio orbit per bin and snapshot...")
    rep_by_z: dict[float, tuple[np.ndarray, ...]] = {}
    for z in z_values:
        rep_by_z[z] = representative_data_for_snapshot(paths_by_z[z], z, rpeak_values)
        present = np.count_nonzero(np.isfinite(rep_by_z[z][2]))
        print(f"  z={z:.2f}: representative bins={present}/{N_MASS}")

    tasks = [
        (str(path), float(z), rpeak_values, rep_by_z[z])
        for z in z_values
        for path in paths_by_z[z]
    ]

    summaries: list[dict] = []
    if N_WORKERS == 1:
        for args in tasks:
            summary = process_one_catalogue(*args)
            summaries.append(summary)
            if PRINT_PER_FILE_SUMMARY:
                print(
                    f"  [{summary['run']} z={summary['z']:.2f}] "
                    f"rows={summary['rows']:,}, "
                    f"weighted N_ext={summary['external_combined']:.6e}"
                )
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            future_map = {
                executor.submit(process_one_catalogue, *args): (args[0], args[1])
                for args in tasks
            }
            for future in as_completed(future_map):
                source, z = future_map[future]
                try:
                    summary = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed while processing {source} at z={z:.2f}") from exc
                summaries.append(summary)
                if PRINT_PER_FILE_SUMMARY:
                    print(
                        f"  [{summary['run']} z={summary['z']:.2f}] "
                        f"rows={summary['rows']:,}, "
                        f"weighted N_ext={summary['external_combined']:.6e}"
                    )

    # A final class-sum audit catches any class-label or weighting mistakes.
    for summary in summaries:
        if not np.isclose(
            summary["external_combined"],
            summary["external_major"] + summary["external_minor"],
            rtol=1.0e-10,
            atol=1.0e-6,
        ):
            raise RuntimeError(f"Major+minor audit failed for {summary['source']}")

    print("\nSimulation catalogues complete.")
    print(f"Outputs: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
