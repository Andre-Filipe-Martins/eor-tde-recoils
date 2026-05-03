#!/usr/bin/env python3
"""
external_tde_ionizing_energy.py
--------------------------------
Compute the hydrogen-ionising energy budget (E > 13.6 eV) from external TDEs,
defined as TDEs occurring outside the adopted central-region boundary across
the EoR simulation runs, and plot per-host-mass-bin and total ionising energy
histories vs. redshift.

Because the simulation Parquets are cap-sampled per (z, mass bin), we first
compute per-BH means within each (z, bin) from the capped catalogues, average
those means over all Monte Carlo runs, and then multiply by the physical
rounded targets from bin_targets_physical.parquet to recover physical totals.
This post-processing stage reads existing simulation outputs and does not
regenerate the Monte Carlo population.

Input
-----
  simulation_results/<run_tag>/data_z_*.parquet  — simulation snapshot Parquets
  results_bin_targets/bin_targets_physical.parquet

Output
------
  external_tde_ionizing_energy.json    machine-readable summary (totals + fits)
  external_tde_ionizing_energy.xlsx    formatted Excel workbook (totals + fits)
  figures/external_tde_ionizing_energy_per_bin.png
  figures/external_tde_ionizing_energy_total.png
"""

import json
import os
import re
from copy import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend so figures can be written on headless systems.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from openpyxl.utils import get_column_letter
    _HAS_OPENPYXL = True
except Exception:
    _HAS_OPENPYXL = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

PARQUET_DIR = os.path.join(BASE_DIR, "simulation_results")
TARGETS_DIR = os.path.join(BASE_DIR, "results_bin_targets")
TARGETS_PQ  = os.path.join(TARGETS_DIR, "bin_targets_physical.parquet")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCRIPT_STEM  = os.path.splitext(os.path.basename(__file__))[0] if "__file__" in globals() \
               else "external_tde_ionizing_energy"
JSON_OUTPATH = os.path.join(BASE_DIR, f"{SCRIPT_STEM}.json")
XLSX_OUTPATH = os.path.join(BASE_DIR, f"{SCRIPT_STEM}.xlsx")


# ---------------------------------------------------------------------------
# Host stellar-mass binning — must match the binning used in simulation.py
# ---------------------------------------------------------------------------
LOGM_MIN, LOGM_MAX, BIN_WIDTH = 6.75, 9.90, 0.35

# If True, keep only rows with `t_external_yr > 0` when that column is
# available. If the column is absent, the filter is skipped.
REQUIRE_EXTERNAL_WINDOW = False

# Verbosity flags
PRINT_FILE_PATHS  = False
PRINT_PER_Z_MEANS = False  # prints per-BH mean values per z (can be verbose)


# ---------------------------------------------------------------------------
# Physics knobs for ionising energy per TDE
# ---------------------------------------------------------------------------
M_STAR_TDE_MSUN = 1.0   # disrupted stellar mass [M_sun]
F_DISK          = 0.5   # fraction of stellar debris that forms the accretion disc
F_ION_PHASE     = 1.0   # fixed-energy path fraction going into ionising radiation
ETA_FIXED       = 0.10  # fixed radiative efficiency (used when spin is unavailable)

USE_SPIN_EFFICIENCY = False  # set True only if a_spin column is present in Parquets

# BH-mass-dependent accreted mass using the Mummery analytic viscous disc model
USE_MUMMERY_MBH_DEPENDENCE = True

# Mummery model parameters
MUM_N_DECAY      = 1.2
MUM_XP           = 10.0
MUM_V_NUIS       = 1000.0
MUM_TION_NORM_YR = 200.0
MUM_MBH_REF_MSUN = 1.0e6

# Diagnostic Rydberg-equivalent photon count: N_equiv = E_ion / (13.6 eV).
PRINT_EQUIV_PHOTONS = True
E_PHOTON_EFF_EV     = 13.6


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT   = 299_792_458.0   # speed of light [m/s]
M_SUN     = 1.98847e30      # solar mass [kg]
EV_TO_ERG = 1.602176634e-12 # 1 eV in erg
G_GRAV    = 6.67430e-11     # gravitational constant [SI]
R_SUN     = 6.957e8         # solar radius [m]
YR_TO_S   = 365.25 * 24 * 3600.0


# ---------------------------------------------------------------------------
# Mass-bin helpers
# ---------------------------------------------------------------------------
def build_mass_bin_edges_and_labels(
    lo: float = LOGM_MIN,
    hi: float = LOGM_MAX,
    width: float = BIN_WIDTH,
) -> Tuple[np.ndarray, List[str]]:
    """Return bin edges array and matching human-readable labels."""
    n = int(np.floor((hi - lo) / width + 0.5))
    edges = lo + width * np.arange(n + 1)
    if edges[-1] < hi - 1e-9:
        edges = np.append(edges, hi)
    labels = [f"[{edges[i]:0.2f}, {edges[i+1]:0.2f}]" for i in range(len(edges) - 1)]
    return edges, labels


EDGES, ALL_LABELS = build_mass_bin_edges_and_labels()
NBINS = len(ALL_LABELS)

# Consistent plotting colour assigned to each host-mass bin.
_default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
COLOR_BY_LABEL  = {lbl: _default_colors[j % len(_default_colors)]
                   for j, lbl in enumerate(ALL_LABELS)}


# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------
def _parse_redshift_from_filename(fname: str) -> Optional[float]:
    """Extract the redshift encoded in a data_z_*.parquet filename."""
    match = re.match(r"data_z_(\d+(?:_\d+)?)\.parquet$", fname)
    if not match:
        return None
    z_str = match.group(1).replace("_", ".")
    try:
        return round(float(z_str), 2)
    except ValueError:
        return None


def discover_parquet_snapshots(
    parquet_dir: str,
) -> Tuple[np.ndarray, Dict[float, List[str]]]:
    """
    Walk parquet_dir for data_z_*.parquet files at root level and in run
    subdirectories. Both legacy root-level files and current runXX/
    subdirectories are supported. Returns a sorted array of redshifts and a dict mapping
    each redshift to the list of file paths for that snapshot.
    """
    if not os.path.isdir(parquet_dir):
        raise SystemExit(f"[ERROR] Missing simulation results directory: {parquet_dir}")

    paths_by_z: Dict[float, List[str]] = {}

    # Root-level files (legacy layout)
    for fname in os.listdir(parquet_dir):
        full = os.path.join(parquet_dir, fname)
        if os.path.isfile(full) and fname.startswith("data_z_") and fname.endswith(".parquet"):
            z = _parse_redshift_from_filename(fname)
            if z is not None:
                paths_by_z.setdefault(z, []).append(full)

    # Per-run subdirectories (current layout: simulation_results/<run_tag>/)
    for entry in os.scandir(parquet_dir):
        if not entry.is_dir():
            continue
        for fname in os.listdir(entry.path):
            if not (fname.startswith("data_z_") and fname.endswith(".parquet")):
                continue
            z = _parse_redshift_from_filename(fname)
            if z is not None:
                paths_by_z.setdefault(z, []).append(os.path.join(entry.path, fname))

    if not paths_by_z:
        raise SystemExit("[ERROR] No Parquet snapshot files found in simulation_results/.")

    z_array = np.array(sorted(paths_by_z.keys(), reverse=True), dtype=float)
    print("Parquet directory:", parquet_dir)
    print("Redshift snapshots found:", ", ".join(f"{z:.2f}" for z in z_array))
    return z_array, paths_by_z


# ---------------------------------------------------------------------------
# Physical bin targets loader
# ---------------------------------------------------------------------------
def _first_matching_column(df_columns, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists in df_columns."""
    cols = set(df_columns)
    for name in candidates:
        if name in cols:
            return name
    return None


def load_physical_targets(targets_pq: str) -> Dict[float, np.ndarray]:
    """
    Load bin_targets_physical.parquet and return a dict mapping each redshift
    to a length-NBINS array of physical (rounded, pre-cap) merger-event counts per bin.
    """
    if not os.path.exists(targets_pq):
        raise SystemExit(f"[ERROR] Missing targets Parquet: {targets_pq}")

    tdf = pd.read_parquet(targets_pq)
    if tdf is None or tdf.empty:
        raise SystemExit(f"[ERROR] Targets Parquet is empty: {targets_pq}")

    z_col  = _first_matching_column(tdf.columns, ["z", "z_cur", "z_current", "redshift"])
    lo_col = _first_matching_column(tdf.columns, ["bin_lo_log10M", "logM_lo", "lo", "bin_lo", "m_lo"])
    hi_col = _first_matching_column(tdf.columns, ["bin_hi_log10M", "logM_hi", "hi", "bin_hi", "m_hi"])
    n_col  = _first_matching_column(tdf.columns, [
        "n_phys", "N_phys",
        "n_events_phys", "n_events_physical",
        "n_events_rounded", "n_events_round",
        "n_events", "N_events",
        "target_phys", "target",
        "mergers_rounded", "n_mergers_rounded",
    ])

    missing = [name for name, col in [("z", z_col), ("lo", lo_col), ("hi", hi_col), ("n", n_col)]
               if col is None]
    if missing:
        raise SystemExit(
            "[ERROR] Could not identify required columns in bin_targets_physical.parquet. "
            f"Missing: {missing}. Columns present: {list(tdf.columns)}"
        )

    z_vals  = pd.to_numeric(tdf[z_col],  errors="coerce").to_numpy(float)
    lo_vals = pd.to_numeric(tdf[lo_col], errors="coerce").to_numpy(float)
    hi_vals = pd.to_numeric(tdf[hi_col], errors="coerce").to_numpy(float)
    n_vals  = pd.to_numeric(tdf[n_col],  errors="coerce").to_numpy(float)

    targets_by_z: Dict[float, np.ndarray] = {}
    for z, lo, hi, n in zip(z_vals, lo_vals, hi_vals, n_vals):
        if not all(np.isfinite(v) for v in (z, lo, hi, n)):
            continue
        z  = round(float(z),  2)
        lo = round(float(lo), 2)
        hi = round(float(hi), 2)

        # Map (lo, hi) pair to a bin index
        matches = np.where(np.isclose(EDGES[:-1], lo, atol=1e-6))[0]
        if len(matches) == 0:
            continue
        i = int(matches[0])
        if not (0 <= i < NBINS) or not np.isclose(EDGES[i + 1], hi, atol=1e-6):
            continue

        arr = targets_by_z.setdefault(z, np.zeros(NBINS, dtype=float))
        arr[i] = float(n)

    return targets_by_z


# ---------------------------------------------------------------------------
# Physics: ionising energy per TDE
# ---------------------------------------------------------------------------
def ionizing_energy_per_tde_erg(eta: float = ETA_FIXED) -> float:
    """
    Return the ionising energy per TDE for the fixed-accreted-mass path.
    """
    delta_m_acc_kg = M_STAR_TDE_MSUN * M_SUN * F_DISK * F_ION_PHASE
    return float(eta * delta_m_acc_kg * C_LIGHT**2 * 1e7)  # J -> erg


def _mummery_viscous_timescale_sec(v_nuis: float = MUM_V_NUIS) -> float:
    """Viscous timescale t_visc for the Mummery analytic disc model [seconds]."""
    m_star = 1.0 * M_SUN
    r_star = 1.0 * R_SUN
    t_star = np.sqrt(8.0 * r_star**3 / (G_GRAV * m_star))
    return float(v_nuis * t_star)


def _mummery_ionizing_timescale_sec(
    mbh_msun: np.ndarray,
    v_nuis: float = MUM_V_NUIS,
    n_decay: float = MUM_N_DECAY,
    x_peak: float = MUM_XP,
) -> np.ndarray:
    """
    BH-mass-dependent ionising timescale t_ion [seconds] from Mummery et al.,
    parameterised by the disc viscosity nuisance parameter v_nuis.
    """
    mbh = np.asarray(mbh_msun, dtype=float)
    mbh = np.where(np.isfinite(mbh) & (mbh > 0.0), mbh, np.nan)
    t_yr = (
        MUM_TION_NORM_YR
        * (v_nuis / 1000.0)
        * (mbh / MUM_MBH_REF_MSUN) ** (-1.0 / n_decay)
        * (x_peak / 10.0) ** (-2.0 / n_decay)
    )
    return t_yr * YR_TO_S


def mummery_accreted_mass_msun(
    mbh_msun: np.ndarray,
    m_disk_msun: float,
    n_decay: float = MUM_N_DECAY,
    x_peak: float = MUM_XP,
    v_nuis: float = MUM_V_NUIS,
) -> np.ndarray:
    """
    Accreted disc mass [M_sun] as a function of BH mass, using the Mummery
    analytic viscous-disc solution.  The returned fraction is clipped to [0, 1].
    """
    t_visc = _mummery_viscous_timescale_sec(v_nuis=v_nuis)
    t_ion  = _mummery_ionizing_timescale_sec(mbh_msun, v_nuis=v_nuis, n_decay=n_decay, x_peak=x_peak)

    ratio = t_ion / t_visc
    frac  = np.where(ratio > 1.0, 1.0 - ratio ** (1.0 - n_decay), 0.0)
    frac  = np.clip(frac, 0.0, 1.0)
    frac  = np.where(np.isfinite(frac), frac, 0.0)
    return float(m_disk_msun) * frac


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------
def accumulate_external_tdes_and_energy(
    z_array: np.ndarray,
    paths_by_z: Dict[float, List[str]],
    targets_by_z: Dict[float, np.ndarray],
) -> Tuple[np.ndarray, Dict, Dict, np.ndarray, np.ndarray]:
    """
    Read capped simulation catalogues for each redshift snapshot, compute
    per-BH means per (z, bin, run), average those means over runs, and multiply
    by physical targets to recover population totals.

    Returns (all arrays ordered by ascending redshift for plotting):
      z_plot                  — sorted redshift array
      series_ext_by_bin       — dict[bin_label] -> array of external TDE totals per z
      eion_series_by_bin      — dict[bin_label] -> array of external ionizing energy [erg] per z
      overall_ext             — total external TDEs summed over bins, per z
      overall_eion            — total external ionizing energy [erg] summed over bins, per z
    """
    n_z = len(z_array)
    series_ext_by_bin  = {lbl: np.zeros(n_z, float) for lbl in ALL_LABELS}
    eion_series_by_bin = {lbl: np.zeros(n_z, float) for lbl in ALL_LABELS}
    overall_ext  = np.zeros(n_z, float)
    overall_eion = np.zeros(n_z, float)

    eion_fixed             = ionizing_energy_per_tde_erg(ETA_FIXED)
    warned_missing_targets = False

    for i, z in enumerate(z_array):
        paths  = paths_by_z[z]
        n_runs = len(paths)
        print(f"\n[z = {z:.2f}] Averaging over {n_runs} file(s)")
        if PRINT_FILE_PATHS:
            for p in paths:
                print("  ", p)

        n_phys = targets_by_z.get(z)
        if n_phys is None:
            if not warned_missing_targets:
                print("  [warning] No entry for this z in bin_targets_physical.parquet; "
                      "totals for missing snapshots will be zero.")
                warned_missing_targets = True
            n_phys = np.zeros(NBINS, dtype=float)

        # Accumulate per-run BH means, then average over runs (bin-wise)
        acc_mean_ext  = np.zeros(NBINS, float)
        acc_mean_eion = np.zeros(NBINS, float)
        acc_n_runs    = np.zeros(NBINS, float)

        for path in paths:
            # Load only the columns we need; fall back to full read if columns
            # are absent in some Parquet versions.
            needed_cols = ["Mstar_rem_Msun", "Mrem_BH_Msun", "tde_external_post"]
            optional_col = "t_external_yr"
            try:
                df = pd.read_parquet(path, columns=needed_cols + [optional_col])
            except Exception:
                try:
                    df = pd.read_parquet(path, columns=needed_cols)
                except Exception:
                    try:
                        df = pd.read_parquet(path)
                    except Exception as exc:
                        print(f"  [warning] Could not read {path}: {exc}")
                        continue

            if df is None or df.empty:
                continue

            # Optional filter: only keep BHs that had time to reach the external
            # region before z = 6. The column may not exist in some Parquet versions.
            if REQUIRE_EXTERNAL_WINDOW and (optional_col in df.columns):
                t_avail = pd.to_numeric(df[optional_col], errors="coerce").fillna(0.0).to_numpy(float)
                df = df.loc[t_avail > 0.0].copy()
                if df.empty:
                    continue

            # Assign each row to a stellar-mass bin
            m_star = pd.to_numeric(df["Mstar_rem_Msun"], errors="coerce").to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_mstar = np.log10(m_star)

            bin_idx = np.searchsorted(EDGES, log_mstar, side="right") - 1
            valid = (
                np.isfinite(log_mstar)
                & (bin_idx >= 0)
                & (bin_idx < NBINS)
                & (log_mstar < EDGES[-1])
            )
            if not np.any(valid):
                continue

            bin_idx = bin_idx[valid].astype(int)

            # External TDE counts per row
            n_ext = pd.to_numeric(df["tde_external_post"], errors="coerce").fillna(0.0).to_numpy(float)
            n_ext = np.where(np.isfinite(n_ext) & (n_ext > 0.0), n_ext, 0.0)
            n_ext = n_ext[valid]

            # Radiative efficiency per row (spin-dependent or fixed)
            if USE_SPIN_EFFICIENCY and ("a_spin" in df.columns):
                a_spin = pd.to_numeric(df["a_spin"], errors="coerce").to_numpy(float)
                a_spin = np.clip(a_spin, 0.0, 0.998)
                eta = np.where(np.isfinite(a_spin), 0.057 + 0.3 * a_spin, ETA_FIXED)
                eta = eta[valid]
            else:
                eta = np.full(int(np.count_nonzero(valid)), ETA_FIXED, dtype=float)

            # Ionizing energy per TDE: BH-mass-dependent or constant
            if USE_MUMMERY_MBH_DEPENDENCE:
                mbh = pd.to_numeric(df["Mrem_BH_Msun"], errors="coerce").to_numpy(float)
                mbh = np.where(np.isfinite(mbh) & (mbh > 0.0), mbh, np.nan)
                mbh = mbh[valid]

                m_disk = M_STAR_TDE_MSUN * F_DISK
                delta_m_acc = mummery_accreted_mass_msun(mbh, m_disk)
                eion_per_tde = eta * (delta_m_acc * M_SUN) * C_LIGHT**2 * 1e7
                eion_per_tde = np.where(np.isfinite(eion_per_tde) & (eion_per_tde > 0.0),
                                        eion_per_tde, 0.0)
            else:
                eion_per_tde = np.full_like(eta, eion_fixed, dtype=float)

            eion_row = n_ext * eion_per_tde
            eion_row = np.where(np.isfinite(eion_row) & (eion_row > 0.0), eion_row, 0.0)

            # Per-bin sums and counts for this run
            cnt    = np.bincount(bin_idx, minlength=NBINS).astype(float)
            s_ext  = np.bincount(bin_idx, weights=n_ext,    minlength=NBINS)
            s_eion = np.bincount(bin_idx, weights=eion_row, minlength=NBINS)

            # Per-run per-BH means (avoids bias from the cap-sampling)
            mean_ext  = np.divide(s_ext,  cnt, out=np.zeros_like(s_ext),  where=cnt > 0)
            mean_eion = np.divide(s_eion, cnt, out=np.zeros_like(s_eion), where=cnt > 0)

            has_data = cnt > 0
            acc_mean_ext[has_data]  += mean_ext[has_data]
            acc_mean_eion[has_data] += mean_eion[has_data]
            acc_n_runs[has_data]    += 1.0

        # Average the per-run means (bin-wise)
        denom       = np.where(acc_n_runs > 0, acc_n_runs, 1.0)
        mean_ext_z  = acc_mean_ext  / denom
        mean_eion_z = acc_mean_eion / denom

        # Scale up to physical totals using the rounded targets
        tot_ext_z  = mean_ext_z  * n_phys
        tot_eion_z = mean_eion_z * n_phys

        for b, lbl in enumerate(ALL_LABELS):
            series_ext_by_bin[lbl][i]  = float(tot_ext_z[b])
            eion_series_by_bin[lbl][i] = float(tot_eion_z[b])

        overall_ext[i]  = float(np.sum(tot_ext_z))
        overall_eion[i] = float(np.sum(tot_eion_z))

        if PRINT_PER_Z_MEANS:
            print("  Per-BH means (averaged over runs):")
            for b, lbl in enumerate(ALL_LABELS):
                if n_phys[b] <= 0:
                    continue
                print(
                    f"    {lbl}: <N_ext>={mean_ext_z[b]:.3e} per BH, "
                    f"<E_ion,ext>={mean_eion_z[b]:.3e} erg per BH, "
                    f"N_phys={n_phys[b]:.3e}"
                )

        # Sanity check: populated bins with no sampled remnants
        problem_bins = np.where((n_phys > 0) & (acc_n_runs == 0))[0]
        if len(problem_bins) > 0:
            print("  [warning] Bins with N_phys > 0 but no sampled remnants in any run:")
            for b in problem_bins:
                print(f"    {ALL_LABELS[b]}: N_phys = {n_phys[b]:.3e}")

    # Re-order arrays in ascending redshift for plotting
    order = np.argsort(z_array)
    z_plot = z_array[order]
    for lbl in ALL_LABELS:
        series_ext_by_bin[lbl]  = series_ext_by_bin[lbl][order]
        eion_series_by_bin[lbl] = eion_series_by_bin[lbl][order]
    overall_ext  = overall_ext[order]
    overall_eion = overall_eion[order]

    return z_plot, series_ext_by_bin, eion_series_by_bin, overall_ext, overall_eion


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_external_energy_summary(series_ext_by_bin, eion_series_by_bin):
    """Print a formatted per-bin table of total external TDEs and ionising energy."""
    print("\n=== External TDE & ionizing-energy totals by host stellar-mass bin ===")
    header = (
        f"{'Bin [log10 M*]':>18}  "
        f"{'N_ext_TDE':>14}  "
        f"{'E_ion,ext [erg]':>20}"
    )
    print(header)
    print("-" * len(header))

    total_tde  = 0.0
    total_eion = 0.0
    for i, lbl in enumerate(ALL_LABELS):
        n_ext = float(np.nan_to_num(series_ext_by_bin[lbl], nan=0.0).sum())
        e_ext = float(np.nan_to_num(eion_series_by_bin[lbl], nan=0.0).sum())
        if n_ext <= 0.0 and e_ext <= 0.0:
            continue
        print(
            f"[{EDGES[i]:5.2f}, {EDGES[i+1]:5.2f}]  "
            f"{n_ext:14.3e}  "
            f"{e_ext:20.3e}"
        )
        total_tde  += n_ext
        total_eion += e_ext

    print("-" * len(header))
    print(f"{'TOTAL (by-bin sum)':>18}  {total_tde:14.3e}  {total_eion:20.3e}")


# ---------------------------------------------------------------------------
# Table builder for JSON / XLSX
# ---------------------------------------------------------------------------
def build_external_totals_table(series_ext_by_bin, eion_series_by_bin) -> pd.DataFrame:
    """Return a DataFrame with per-bin external TDE counts and ionising energies."""
    rows = []
    for i, lbl in enumerate(ALL_LABELS):
        n_ext = float(np.nan_to_num(series_ext_by_bin[lbl], nan=0.0).sum())
        e_ext = float(np.nan_to_num(eion_series_by_bin[lbl], nan=0.0).sum())
        if n_ext <= 0.0 and e_ext <= 0.0:
            continue
        rows.append({
            "section":        "external_totals",
            "bin_label":      f"[{EDGES[i]:0.2f}, {EDGES[i+1]:0.2f}]",
            "bin_lo_log10M":  float(EDGES[i]),
            "bin_hi_log10M":  float(EDGES[i + 1]),
            "N_ext_TDE":      n_ext,
            "E_ion_ext_erg":  e_ext,
        })

    total_tde  = sum(r["N_ext_TDE"]     for r in rows)
    total_eion = sum(r["E_ion_ext_erg"] for r in rows)
    rows.append({
        "section":        "external_totals",
        "bin_label":      "TOTAL (by-bin sum)",
        "bin_lo_log10M":  float("nan"),
        "bin_hi_log10M":  float("nan"),
        "N_ext_TDE":      total_tde,
        "E_ion_ext_erg":  total_eion,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Log-polynomial fit helper
# ---------------------------------------------------------------------------
def _fit_log_polynomial(
    z_centers: np.ndarray,
    y_values: np.ndarray,
    max_degree: int = 2,
    label: str = "",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit log10(y) as a diagnostic polynomial in z (up to max_degree), then
    return (z_line, y_line, coefficients) evaluated on a dense grid. Totals are
    computed from binned simulation outputs, not from this fit.
    Returns (None, None, None) if there are too few points to fit.
    """
    z_arr = np.asarray(z_centers, dtype=float)
    y_arr = np.asarray(y_values,  dtype=float)

    mask   = y_arr > 0.0
    n_pts  = int(mask.sum())
    if n_pts < 2:
        print(f"[fit {label}] too few positive points (N={n_pts}); skipping")
        return None, None, None

    z_fit = z_arr[mask]
    y_fit = y_arr[mask]
    degree = 1 if n_pts <= 3 else min(max_degree, n_pts - 1)

    coeffs = np.polyfit(z_fit, np.log10(y_fit), degree)
    z_line = np.linspace(z_fit.min(), z_fit.max(), 400)
    y_line = 10.0 ** np.polyval(coeffs, z_line)

    sum_data  = float(y_fit.sum())
    sum_model = float((10.0 ** np.polyval(coeffs, z_fit)).sum())
    print(f"[fit {label}] deg={degree}, N={n_pts}, sum_data={sum_data:.3e}, sum_model={sum_model:.3e}")

    return z_line, y_line, coeffs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_figure(fig, outpath: str, dpi: int = 150) -> None:
    """Save and close a Matplotlib figure, reporting whether the file was created or overwritten."""
    existed = os.path.exists(outpath)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    status = "Updated (overwrote)" if existed else "Saved (new)"
    print(f"{status}: {outpath}")


def plot_external_ionizing_energy(
    z_plot: np.ndarray,
    eion_series_by_bin: Dict,
    overall_eion: np.ndarray,
) -> pd.DataFrame:
    """
    Produce two figures:
      (1) Per-mass-bin grid: ionising energy vs redshift
      (2) Single-panel total across all bins

    Returns a fit-coefficient DataFrame for use in the JSON / XLSX output.
    """
    active_bins = [
        lbl for lbl in ALL_LABELS
        if np.any(np.nan_to_num(eion_series_by_bin[lbl], nan=0.0) > 0.0)
    ]
    if not active_bins:
        print("\n[plot] No external ionizing energy to plot.")
        return pd.DataFrame(columns=["section", "bin_label", "deg", "poly_form", "a0", "a1", "a2"])

    active_bins = sorted(active_bins, key=lambda lbl: EDGES[ALL_LABELS.index(lbl)])

    # Bar width derived from the redshift grid spacing
    bar_width = float(np.abs(np.median(np.diff(z_plot)))) if len(z_plot) > 1 else 0.2

    # Shared y-limits across both figures
    all_positive = []
    for lbl in active_bins:
        vals = np.nan_to_num(eion_series_by_bin[lbl], nan=0.0)
        all_positive.append(vals[vals > 0.0])
    total_vals = np.nan_to_num(overall_eion, nan=0.0)
    all_positive.append(total_vals[total_vals > 0.0])
    all_positive = [a for a in all_positive if len(a) > 0]

    if all_positive:
        concat = np.concatenate(all_positive)
        y_min = max(float(np.min(concat)) * 0.85, 1.0)
        y_max = float(np.max(concat)) * 1.15
    else:
        y_min, y_max = 1.0, 1e6

    fit_rows_print = []
    fit_rows_df    = []

    # --- Figure 1: per-bin grid ---
    n_panels = len(active_bins)
    ncols    = 2 if n_panels <= 6 else 3
    nrows    = int(np.ceil(n_panels / ncols))

    fig1, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 2.6 * nrows),
                              sharex=True, sharey=True)
    axes = np.ravel(axes)

    for j, lbl in enumerate(active_bins):
        ax    = axes[j]
        y     = np.nan_to_num(eion_series_by_bin[lbl], nan=0.0)
        color = COLOR_BY_LABEL.get(lbl, "C0")

        ax.bar(z_plot, y, width=bar_width, align="center",
               color=color, edgecolor="black", linewidth=0.5, alpha=0.35, zorder=1)

        z_line, y_line, coeffs = _fit_log_polynomial(z_plot, y, max_degree=2, label=lbl)
        if z_line is not None:
            ax.plot(z_line, y_line, "-", color=color, linewidth=2.0, zorder=3)
            degree      = len(coeffs) - 1
            coeffs_asc  = coeffs[::-1]   # [a0, a1, a2, ...]
            poly_form   = ("a0 + a1 z" if degree == 1
                           else "a0 + a1 z + a2 z^2" if degree == 2
                           else f"poly deg {degree}")
            coeff_str   = ", ".join(f"a{k}={c:.3e}" for k, c in enumerate(coeffs_asc))

            i_all     = ALL_LABELS.index(lbl)
            bin_label = f"[{EDGES[i_all]:4.2f}, {EDGES[i_all+1]:4.2f}]"
            fit_rows_print.append((bin_label, degree, poly_form, coeff_str))
            fit_rows_df.append({
                "section":    "logpoly_fits",
                "bin_label":  bin_label,
                "deg":        int(degree),
                "poly_form":  poly_form,
                "a0": float(coeffs_asc[0]) if len(coeffs_asc) > 0 else float("nan"),
                "a1": float(coeffs_asc[1]) if len(coeffs_asc) > 1 else float("nan"),
                "a2": float(coeffs_asc[2]) if len(coeffs_asc) > 2 else float("nan"),
            })

        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Bin {lbl}")
        ax.grid(True, linestyle=":", linewidth=0.7)

    for k in range(n_panels, len(axes)):
        axes[k].axis("off")

    for ax in axes[:n_panels]:
        ax.set_xlim(z_plot.min() - 0.5 * bar_width, z_plot.max() + 0.5 * bar_width)

    axes[0].set_ylabel(r"$E_{\rm ion,ext}$ [erg] per snapshot")
    for ax in axes[(nrows - 1) * ncols: nrows * ncols]:
        if ax.get_visible():
            ax.set_xlabel("Redshift $z$")

    selection_note = " (filter: t_external_yr > 0)" if REQUIRE_EXTERNAL_WINDOW else ""
    fig1.suptitle(
        "External TDE ionizing energy vs. redshift\n"
        f"Per host stellar-mass bin{selection_note}",
        y=0.99, fontsize=12,
    )
    fig1.tight_layout(rect=[0, 0.02, 1, 0.96])

    out1 = os.path.join(FIG_DIR, "external_tde_ionizing_energy_per_bin.png")
    save_figure(fig1, out1)

    # --- Figure 2: total across all bins ---
    fig2, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.bar(z_plot, total_vals, width=bar_width, align="center",
           color="black", edgecolor="red", linewidth=0.8, alpha=0.4, zorder=1)

    z_line_tot, y_line_tot, coeffs_tot = _fit_log_polynomial(
        z_plot, total_vals, max_degree=2, label="TOTAL"
    )
    if z_line_tot is not None:
        ax.plot(z_line_tot, y_line_tot, "-", color="red", linewidth=2.5, zorder=3)
        degree_tot     = len(coeffs_tot) - 1
        coeffs_asc_tot = coeffs_tot[::-1]
        poly_form_tot  = ("a0 + a1 z" if degree_tot == 1
                          else "a0 + a1 z + a2 z^2" if degree_tot == 2
                          else f"poly deg {degree_tot}")
        coeff_str_tot  = ", ".join(f"a{k}={c:.3e}" for k, c in enumerate(coeffs_asc_tot))
        fit_rows_print.append(("TOTAL", degree_tot, poly_form_tot, coeff_str_tot))
        fit_rows_df.append({
            "section":   "logpoly_fits",
            "bin_label": "TOTAL",
            "deg":       int(degree_tot),
            "poly_form": poly_form_tot,
            "a0": float(coeffs_asc_tot[0]) if len(coeffs_asc_tot) > 0 else float("nan"),
            "a1": float(coeffs_asc_tot[1]) if len(coeffs_asc_tot) > 1 else float("nan"),
            "a2": float(coeffs_asc_tot[2]) if len(coeffs_asc_tot) > 2 else float("nan"),
        })

    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(z_plot.min() - 0.5 * bar_width, z_plot.max() + 0.5 * bar_width)
    ax.set_title("Total external ionizing energy (all host-mass bins)")
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$E_{\rm ion,ext}$ [erg] per snapshot")
    ax.grid(True, linestyle=":", linewidth=0.7)

    fig2.tight_layout()
    out2 = os.path.join(FIG_DIR, "external_tde_ionizing_energy_total.png")
    save_figure(fig2, out2)

    # Print fit coefficient table
    if fit_rows_print:
        print("\n=== Log-polynomial fits for external ionizing-energy histories ===")
        print("log10 E_ion,ext(z) = a0 + a1*z + ... + aN*z^N\n")
        header = (
            f"{'Bin [log10 M*]':>18}  "
            f"{'deg':>3}  "
            f"{'poly form':>22}  "
            f"{'coefficients':>60}"
        )
        print(header)
        print("-" * len(header))
        for bin_label, deg, poly_form, coeff_str in fit_rows_print:
            print(f"{bin_label:>18}  {deg:3d}  {poly_form:>22}  {coeff_str:>60}")

    return pd.DataFrame(fit_rows_df)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def write_tables_json(
    outpath: str,
    totals_df: pd.DataFrame,
    fits_df: pd.DataFrame,
) -> None:
    """
    Write the external TDE totals and log-polynomial fit coefficients to a
    JSON file, matching the structure used by ratio_scan_vcent.py.
    """
    def _df_to_records(df: pd.DataFrame) -> list:
        """Convert a DataFrame to a list of dicts, replacing NaN with None."""
        records = []
        for row in df.to_dict(orient="records"):
            records.append({
                k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in row.items()
            })
        return records

    payload = {
        "meta": {
            "script":                  SCRIPT_STEM,
            "M_star_TDE_Msun":         M_STAR_TDE_MSUN,
            "f_disk":                  F_DISK,
            "f_ion_phase":             F_ION_PHASE,
            "eta_fixed":               ETA_FIXED,
            "use_mummery_mbh_dep":     USE_MUMMERY_MBH_DEPENDENCE,
            "use_spin_efficiency":     USE_SPIN_EFFICIENCY,
            "require_external_window": REQUIRE_EXTERNAL_WINDOW,
        },
        "external_totals": _df_to_records(totals_df) if totals_df is not None else [],
        "logpoly_fits":    _df_to_records(fits_df)   if fits_df   is not None else [],
    }

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON tables: {outpath}")


# ---------------------------------------------------------------------------
# XLSX output
# ---------------------------------------------------------------------------
def write_tables_xlsx(
    outpath: str,
    totals_df: pd.DataFrame,
    fits_df: pd.DataFrame,
) -> None:
    """Write external TDE totals and fit coefficients to a formatted Excel workbook."""
    if not _HAS_OPENPYXL:
        print(f"[warning] openpyxl not available; skipping XLSX output: {outpath}")
        print("          Install with: pip install openpyxl")
        return

    if (totals_df is None or totals_df.empty) and (fits_df is None or fits_df.empty):
        print(f"[warning] No tables to write to XLSX: {outpath}")
        return

    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:
        if totals_df is not None and not totals_df.empty:
            totals_df.to_excel(writer, sheet_name="external_totals", index=False)
        if fits_df is not None and not fits_df.empty:
            fits_df.to_excel(writer, sheet_name="logpoly_fits", index=False)

        wb = writer.book

        def _format_sheet(ws, numfmt_by_col: dict, freeze_at: str = "A2"):
            ws.freeze_panes = freeze_at
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                f = copy(cell.font); f.bold = True; cell.font = f
                a = copy(cell.alignment); a.horizontal = "center"; cell.alignment = a

            header     = [c.value for c in ws[1]]
            col_index  = {name: j + 1 for j, name in enumerate(header) if name is not None}

            for col_name, fmt in numfmt_by_col.items():
                if col_name not in col_index:
                    continue
                cidx = col_index[col_name]
                for row in range(2, ws.max_row + 1):
                    ws.cell(row=row, column=cidx).number_format = fmt

            for j in range(1, ws.max_column + 1):
                col_letter = get_column_letter(j)
                max_len = max(
                    (len(str(ws.cell(row=r, column=j).value or ""))
                     for r in range(1, ws.max_row + 1)),
                    default=0,
                )
                ws.column_dimensions[col_letter].width = min(max_len + 2, 55)

        if "external_totals" in wb.sheetnames:
            _format_sheet(wb["external_totals"], {
                "bin_lo_log10M": "0.00",
                "bin_hi_log10M": "0.00",
                "N_ext_TDE":     "0.00E+00",
                "E_ion_ext_erg": "0.00E+00",
            })

        if "logpoly_fits" in wb.sheetnames:
            _format_sheet(wb["logpoly_fits"], {
                "a0": "0.0E+00",
                "a1": "0.0E+00",
                "a2": "0.0E+00",
            })

    print(f"Saved XLSX tables: {outpath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Aggregate external TDE counts and ionising energy from simulation outputs, then write summary tables and figures."""
    print("\n=== External TDE ionizing energy budget ===\n")

    eion_ref = ionizing_energy_per_tde_erg(ETA_FIXED)
    print("Per-TDE ionizing energy parameters:")
    print(f"  M_star_TDE     = {M_STAR_TDE_MSUN:.3f} M_sun")
    print(f"  f_disk         = {F_DISK:.3f}")
    print(f"  f_ion_phase    = {F_ION_PHASE:.3f}  (legacy fixed-DeltaM path only)")
    print(f"  eta_fixed      = {ETA_FIXED:.3f}")
    print(f"  E_ion (fixed)  = {eion_ref:.3e} erg")

    if PRINT_EQUIV_PHOTONS:
        n_equiv = eion_ref / (E_PHOTON_EFF_EV * EV_TO_ERG)
        print(f"  Rydberg-equivalent photons = E_ion / 13.6 eV = {n_equiv:.3e}")

    print("\nModel switches:")
    print(f"  USE_MUMMERY_MBH_DEPENDENCE = {USE_MUMMERY_MBH_DEPENDENCE}")
    print(f"  USE_SPIN_EFFICIENCY        = {USE_SPIN_EFFICIENCY}")
    print()

    if USE_MUMMERY_MBH_DEPENDENCE:
        print("Mummery ΔM_acc sanity check (solar-type star, M_disk = M_star_TDE * f_disk):")
        m_disk = M_STAR_TDE_MSUN * F_DISK
        for mbh_test in [1e5, 1e6, 1e7, 1e8, 2e8]:
            d_m = float(mummery_accreted_mass_msun(np.array([mbh_test]), m_disk)[0])
            e   = ETA_FIXED * (d_m * M_SUN) * C_LIGHT**2 * 1e7
            print(f"  M_BH = {mbh_test:.1e} M_sun:  DeltaM_acc = {d_m:.3f} M_sun  ->  E_ion = {e:.3e} erg")
        print()

    # Load data
    z_array, paths_by_z = discover_parquet_snapshots(PARQUET_DIR)
    targets_by_z        = load_physical_targets(TARGETS_PQ)

    z_plot, series_ext_by_bin, eion_series_by_bin, overall_ext, overall_eion = (
        accumulate_external_tdes_and_energy(z_array, paths_by_z, targets_by_z)
    )

    print_external_energy_summary(series_ext_by_bin, eion_series_by_bin)

    total_tde  = float(np.nan_to_num(overall_ext,  nan=0.0).sum())
    total_eion = float(np.nan_to_num(overall_eion, nan=0.0).sum())

    print("\n=== Global totals (external, mean over runs; physical targets applied) ===")
    print(f"  Total external TDEs          = {total_tde:.3e}")
    print(f"  Total external E_ion [erg]   = {total_eion:.3e}")

    if PRINT_EQUIV_PHOTONS:
        n_equiv = total_eion / (E_PHOTON_EFF_EV * EV_TO_ERG)
        print(f"  Rydberg-equivalent photons   = {n_equiv:.3e}")

    print()

    # Build output tables and write files
    totals_df = build_external_totals_table(series_ext_by_bin, eion_series_by_bin)
    fits_df   = plot_external_ionizing_energy(z_plot, eion_series_by_bin, overall_eion)

    write_tables_json(JSON_OUTPATH, totals_df, fits_df)
    write_tables_xlsx(XLSX_OUTPATH, totals_df, fits_df)

    print(f"\n>>> {SCRIPT_STEM}.py finished successfully.")


if __name__ == "__main__":
    main()
