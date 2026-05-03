#!/usr/bin/env python3
"""
ratio_scan_vcent.py — Parametric scan of R = V_kick / V_cent

Scans the ratio grid R = V_kick / V_cent across all stellar-mass bins and EoR
snapshots to determine which kick velocities maximise external TDE yields.
V_cent is the minimum speed needed to reach R_cent = F_CENT * R_e:
    V_cent = sqrt( 2 * (Psi(0) - Psi(R_cent)) )
with Psi(r) = -Phi(r) the positive potential depth, normalised so Psi(0) = 0.5 * V_esc(0)^2.

The scan is run for two values of F_CENT (1.00 and 0.05) back-to-back.

Inputs
------
  - ratio_scan_catalogue/runXX/data_z_*.parquet
  - results_bin_targets/bin_targets_physical.parquet

Outputs (per F_CENT run):
  - XLSX table:  ratio_scan_vcent_tables__<fcent_tag>.xlsx
  - JSON table:  ratio_scan_vcent_rpeak__<fcent_tag>.json, including per-bin
    R_peak values used by the downstream final simulation

Combined figure:
  - external_tdes_vs_ratio_fcent_comparison.png
"""

import os
import re
import math
import json
import time
from copy import copy

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics_relations as pr


# ===================== Optional openpyxl =====================
# openpyxl is required only for formatted XLSX output.
try:
    from openpyxl.utils.cell import get_column_letter
    _HAS_OPENPYXL = True
except Exception as _e:
    _HAS_OPENPYXL = False
    _OPENPYXL_ERR = f"{type(_e).__name__}: {_e}"


# ===================== PATHS =====================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

PARQUET_ROOT = os.path.join(BASE_DIR, "ratio_scan_catalogue")

TARGETS_DIR = os.path.join(BASE_DIR, "results_bin_targets")
TARGETS_PQ  = os.path.join(TARGETS_DIR, "bin_targets_physical.parquet")

FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCRIPT_STEM = "ratio_scan_vcent"


# ===================== MASS BINS =====================
MASS_EDGES_ALL  = np.array([6.75, 7.10, 7.45, 7.80, 8.15, 8.50, 8.85, 9.20, 9.55, 9.90])
MASS_LABELS_ALL = [f"[{MASS_EDGES_ALL[i]:.2f}, {MASS_EDGES_ALL[i+1]:.2f}]"
                   for i in range(len(MASS_EDGES_ALL) - 1)]
N_MASS_ALL = len(MASS_LABELS_ALL)


# ===================== RATIO GRID =====================
# R = V_kick / V_cent
RATIO_MIN = 1.0
RATIO_MAX = 10.0
DR        = 0.10
N_R       = int(np.round((RATIO_MAX - RATIO_MIN) / DR))
RATIO_EDGES   = RATIO_MIN + DR * np.arange(N_R + 1)
RATIO_CENTERS = RATIO_EDGES[:-1] + 0.5 * DR  # ratio-bin centres used for peak selection

# Two F_CENT values scanned in sequence
F_CENT_LIST = [1.00, 0.05]

TDECAY_MODE = "vkick"   # exponential decay timescale set by V_kick

N_INT = 320             # integration resolution for representative-orbit travel times

# Physical constants / unit conversions
M_STAR_SI      = 1.0 * pr.M_sun
R_SUN_M        = 6.957e8
SEC_PER_YEAR   = 365.25 * 24 * 3600
KPC_TO_KM      = 3.085677581e16   # km per kpc

F_BULGE        = 0.1548           # bulge mass fraction (Model B)

# Placeholder decay time for zero-kick events
TDECAY_NO_DECAY_YR = 1e30


# ===================== PARQUET COLUMN DETECTION =====================
# Candidate column names for compatibility with different Parquet versions
CAND = {
    "MSTAR": ["Mstar_rem_Msun", "Mstar_Msun", "Mstar_msun", "Mstar"],
    "MBH":   ["Mrem_BH_Msun", "MBH_Msun", "Mbh_Msun", "Mbh_msun", "Mbh"],
    "RE":    ["Re_kpc", "Reff_kpc", "R_e_kpc", "R_eff_kpc", "Re"],
    "VESC":  ["Vesc0_kms", "vesc0_kms", "Vesc_kms", "vesc_kms", "Vesc0"],
    "W":     ["weight", "w", "count", "N_weight", "N", "N_galaxies", "N_targets"],
}

try:
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except Exception:
    HAVE_PYARROW = False

NEEDED_CAND_COLS = sorted({c for lst in CAND.values() for c in lst})


def first_col(df, names):
    """Return the first name from `names` that exists as a column in `df`."""
    for n in names:
        if n in df.columns:
            return n
    return None


def parse_z_from_filename(path: str):
    """Extract the redshift value encoded in a data_z_*.parquet filename."""
    base = os.path.basename(path)
    m = re.match(r"data_z_(\d+(?:_\d+)?)\.parquet$", base)
    if not m:
        return None
    return float(m.group(1).replace("_", "."))


def discover_parquet_snapshots(parquet_root: str):
    """
    Discover snapshot Parquet files under `parquet_root`.
    Searches both root-level files and one level of subdirectories (e.g. run00/).
    Returns a sorted list of redshifts and a dict mapping z -> list of file paths.
    """
    if not os.path.isdir(parquet_root):
        raise FileNotFoundError(f"Missing parquet root dir: {parquet_root}")

    paths_by_z = {}

    for fn in os.listdir(parquet_root):
        full = os.path.join(parquet_root, fn)
        if os.path.isfile(full) and fn.startswith("data_z_") and fn.endswith(".parquet"):
            z = parse_z_from_filename(full)
            if z is not None:
                paths_by_z.setdefault(z, []).append(full)

    for entry in os.scandir(parquet_root):
        if not entry.is_dir():
            continue
        for fn in os.listdir(entry.path):
            if fn.startswith("data_z_") and fn.endswith(".parquet"):
                full = os.path.join(entry.path, fn)
                z = parse_z_from_filename(full)
                if z is not None:
                    paths_by_z.setdefault(z, []).append(full)

    if not paths_by_z:
        raise FileNotFoundError(
            f"No data_z_*.parquet found in {parquet_root} or its subdirectories."
        )

    Zs = sorted(paths_by_z.keys(), reverse=True)
    return Zs, paths_by_z


# ===================== PHYSICAL BIN TARGETS =====================
def _bin_key(z, lo, hi):
    """Hashable key for a (z, bin_lo, bin_hi) entry."""
    return (f"{float(z):.2f}", f"{float(lo):.2f}", f"{float(hi):.2f}")


def load_bin_targets_physical():
    """
    Load the rounded, pre-cap physical merger-event counts per (z, bin_lo, bin_hi).
    Expected columns: z, bin_lo_log10M, bin_hi_log10M, n_events_phys.
    Written by ratio_scan_sample_gen.py.
    """
    if not os.path.exists(TARGETS_PQ):
        raise FileNotFoundError(
            f"Physical targets file not found: {TARGETS_PQ}\n"
            "Run ratio_scan_sample_gen.py first to generate it."
        )
    return pd.read_parquet(TARGETS_PQ)


def build_phys_target_map(df_targets):
    """
    Convert the targets DataFrame into a dict keyed by _bin_key(z, lo, hi)
    returning the integer physical event count for each bin.
    """
    needed = {"z", "bin_lo_log10M", "bin_hi_log10M", "n_events_phys"}
    missing = [c for c in needed if c not in df_targets.columns]
    if missing:
        raise KeyError(f"[targets] Missing columns in targets file: {missing}")

    phys_count_map = {}
    for row in df_targets.itertuples(index=False):
        z  = getattr(row, "z")
        lo = getattr(row, "bin_lo_log10M")
        hi = getattr(row, "bin_hi_log10M")
        n  = getattr(row, "n_events_phys")
        if not np.isfinite(z) or not np.isfinite(lo) or not np.isfinite(hi):
            continue
        try:
            n_int = int(n)
        except Exception:
            continue
        if n_int <= 0:
            continue
        phys_count_map[_bin_key(z, lo, hi)] = n_int
    return phys_count_map


# ===================== GRAVITATIONAL POTENTIAL =====================
G_KPC = 4.30091e-6  # (km/s)^2 kpc / Msun


def psi_total_scaled_factory(rep_Mstar, rep_Re_kpc, rep_Vesc0_kms, z):
    """
    Build the positive potential depth Psi(r) = -Phi(r), normalised so that
    Psi(0) = 0.5 * V_esc(0)^2. Uses an NFW halo + Hernquist bulge model and
    rescales the model potential to match the catalogue value of Vesc0_kms.
    """
    Mh = float(pr.fire2_mh_from_mstar(rep_Mstar))
    Mb = float(F_BULGE * rep_Mstar)

    Rvir_kpc = float(pr.r_vir_mpc(Mh, z) * 1e3)
    c = float(pr.nfw_concentration(Mh, z))

    f_c = math.log(1.0 + c) - c / (1.0 + c)
    rs  = Rvir_kpc / c
    a   = rep_Re_kpc / 1.8153   # Hernquist scale length from effective radius

    def psi_nfw(r_kpc):
        r = np.asarray(r_kpc, float)
        term = np.empty_like(r)
        mask = (r > 0)
        term[mask]  = np.log1p(r[mask] / rs) / r[mask]
        term[~mask] = 1.0 / rs
        return (G_KPC * Mh / f_c) * term

    def psi_hernquist(r_kpc):
        r = np.asarray(r_kpc, float)
        return (G_KPC * Mb) / (r + a)

    psi0_model  = float(psi_nfw(0.0) + psi_hernquist(0.0))
    psi0_target = 0.5 * (rep_Vesc0_kms ** 2)
    scale = 1.0 if (not np.isfinite(psi0_model) or psi0_model <= 0) \
            else (psi0_target / psi0_model)

    def psi(r_kpc):
        return scale * (psi_nfw(r_kpc) + psi_hernquist(r_kpc))

    return psi, Rvir_kpc


def travel_time_years(psi, psi_target, r1_kpc, r2_kpc, n=N_INT):
    """
    Integrate dt = dr / sqrt(2*(Psi(r) - psi_target)) from r1 to r2 (kpc).
    Returns NaN if the integrand becomes non-physical anywhere in the interval.
    """
    if r2_kpc <= r1_kpc:
        return 0.0

    edges = np.linspace(r1_kpc, r2_kpc, n + 1)
    mids  = 0.5 * (edges[:-1] + edges[1:])
    dr    = edges[1:] - edges[:-1]

    psi_mid = psi(mids)
    diff    = psi_mid - psi_target
    if not np.all(np.isfinite(diff)) or np.any(diff <= 0):
        return np.nan

    v          = np.sqrt(2.0 * diff)         # km/s
    dt_s       = (dr / v) * KPC_TO_KM        # seconds
    dt_yr      = np.sum(dt_s) / SEC_PER_YEAR
    return float(dt_yr) if np.isfinite(dt_yr) else np.nan


def find_rmax_bisect(psi, psi_target, r_lo, r_hi):
    """
    Find the apocentric radius where Psi(r) = psi_target by bisection.
    Assumes Psi decreases monotonically with r.
    """
    flo = float(psi(r_lo) - psi_target)
    fhi = float(psi(r_hi) - psi_target)
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return None
    if flo <= 0 or fhi >= 0:
        return None

    a, b = r_lo, r_hi
    for _ in range(70):
        mid = 0.5 * (a + b)
        fm  = float(psi(mid) - psi_target)
        if not np.isfinite(fm):
            return None
        if fm > 0:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def build_rep_times_for_snapshot(df, z, F_CENT):
    """
    Compute representative orbit travel times per (mass bin, ratio bin) for one snapshot.

    The representative galaxy in each mass bin is taken as the median of Mstar, Re, Vesc0.
    Travel times are later scaled to individual remnants by
    (Re/Vesc0)/(Re_rep/Vesc0_rep).

    Returns
    -------
    rep_t_leave          : (N_MASS_ALL, N_R) array, years to leave central region
    rep_t_ext_max        : (N_MASS_ALL, N_R) array, full external orbit time (NaN if unbound)
    rep_scale_base       : (N_MASS_ALL,) array, Re/Vesc0 for the representative galaxy
    rep_Vcent_over_Vesc0 : (N_MASS_ALL,) array, V_cent / V_esc(0) for the representative
    """
    rep_t_leave          = np.full((N_MASS_ALL, N_R), np.nan, float)
    rep_t_ext_max        = np.full((N_MASS_ALL, N_R), np.nan, float)
    rep_scale_base       = np.full(N_MASS_ALL, np.nan, float)
    rep_Vcent_over_Vesc0 = np.full(N_MASS_ALL, np.nan, float)

    Mstar = pd.to_numeric(df["_MSTAR"], errors="coerce").to_numpy(float)
    Re    = pd.to_numeric(df["_RE"],    errors="coerce").to_numpy(float)
    Vesc0 = pd.to_numeric(df["_VESC"],  errors="coerce").to_numpy(float)

    logM   = np.log10(Mstar, where=(Mstar > 0), out=np.full_like(Mstar, np.nan))
    mass_i = np.digitize(logM, MASS_EDGES_ALL) - 1

    for m in range(N_MASS_ALL):
        sel = (
            (mass_i == m) &
            np.isfinite(Mstar) & (Mstar > 0) &
            np.isfinite(Re)    & (Re    > 0) &
            np.isfinite(Vesc0) & (Vesc0 > 0)
        )
        if not np.any(sel):
            continue

        rep_Mstar = float(np.nanmedian(Mstar[sel]))
        rep_Re    = float(np.nanmedian(Re[sel]))
        rep_Vesc0 = float(np.nanmedian(Vesc0[sel]))
        if not (rep_Mstar > 0 and rep_Re > 0 and rep_Vesc0 > 0):
            continue

        rep_scale_base[m] = rep_Re / rep_Vesc0

        psi, Rvir_kpc = psi_total_scaled_factory(rep_Mstar, rep_Re, rep_Vesc0, z)
        R_cent = float(F_CENT) * rep_Re
        psi0   = 0.5 * rep_Vesc0 ** 2

        psi_Rc0 = float(psi(R_cent))
        if np.isfinite(psi_Rc0) and (psi0 > 0) and (psi_Rc0 < psi0):
            Vcent_over_Vesc0 = math.sqrt(max(0.0, 1.0 - psi_Rc0 / psi0))
            rep_Vcent_over_Vesc0[m] = Vcent_over_Vesc0
        else:
            rep_Vcent_over_Vesc0[m] = 1.0
            Vcent_over_Vesc0 = 1.0

        for rbin, rc in enumerate(RATIO_CENTERS):
            rc_eff     = rc * Vcent_over_Vesc0
            psi_target = psi0 * (1.0 - rc_eff ** 2)

            psi_Rc = float(psi(R_cent))
            if not np.isfinite(psi_Rc) or psi_Rc <= psi_target:
                continue

            t_leave = travel_time_years(psi, psi_target, 0.0, R_cent)
            if not (np.isfinite(t_leave) and t_leave > 0):
                continue
            rep_t_leave[m, rbin] = t_leave

            # Orbits with psi_target <= 0 are escape-like; no turning point
            if psi_target <= 0:
                continue

            r_hi    = min(max(2.0 * R_cent, 1.2 * R_cent), Rvir_kpc)
            psi_hi  = float(psi(r_hi))
            tries   = 0
            while np.isfinite(psi_hi) and psi_hi > psi_target and (r_hi < Rvir_kpc) and tries < 25:
                r_hi   = min(r_hi * 1.6, Rvir_kpc)
                psi_hi = float(psi(r_hi))
                tries += 1

            if not (np.isfinite(psi_hi) and psi_hi < psi_target):
                continue

            rmax = find_rmax_bisect(psi, psi_target, R_cent, r_hi)
            if rmax is None or not (np.isfinite(rmax) and rmax > R_cent):
                continue

            t_out_half = travel_time_years(psi, psi_target, R_cent, rmax)
            if not (np.isfinite(t_out_half) and t_out_half > 0):
                continue

            rep_t_ext_max[m, rbin] = 2.0 * t_out_half

    return rep_t_leave, rep_t_ext_max, rep_scale_base, rep_Vcent_over_Vesc0


def integral_exp(rate_yr, tdecay_yr, t0, t1):
    """
    Integrate an exponentially decaying rate over a time interval.
    Supports NumPy broadcasting and returns zero where inputs are non-positive
    or non-finite.
    """
    r, td, t0a, t1a = np.broadcast_arrays(
        np.asarray(rate_yr,   dtype=float),
        np.asarray(tdecay_yr, dtype=float),
        np.asarray(t0,        dtype=float),
        np.asarray(t1,        dtype=float),
    )

    out = np.zeros_like(r, dtype=float)
    ok  = (
        np.isfinite(r)   & (r   > 0) &
        np.isfinite(td)  & (td  > 0) &
        np.isfinite(t0a) & np.isfinite(t1a) &
        (t1a > t0a)
    )
    if not np.any(ok):
        return out

    x0     = np.clip(t0a[ok] / td[ok], 0.0, 1e6)
    x1     = np.clip(t1a[ok] / td[ok], 0.0, 1e6)
    out[ok] = r[ok] * td[ok] * (np.exp(-x0) - np.exp(-x1))
    return np.where(np.isfinite(out) & (out > 0), out, 0.0)


# ===================== PLOT HELPERS =====================
def save_fig(fig, path_png):
    """Save a Matplotlib figure to disk."""
    fig.savefig(path_png, dpi=200, bbox_inches="tight")


def set_logy_positive(ax, y2d, pad_top=1.10, pad_bottom=1.30):
    """Set a log y-scale using only the positive finite values in y2d."""
    y   = np.asarray(y2d, float)
    pos = y[np.isfinite(y) & (y > 0)]
    if pos.size == 0:
        return False, None, None
    ymin = float(np.min(pos))
    ymax = float(np.max(pos))
    ax.set_yscale("log")
    ax.set_ylim(ymin / pad_bottom, ymax * pad_top)
    return True, ymin / pad_bottom, ymax * pad_top


def set_ratio_xticks(ax, rmin, rmax):
    """Set x-axis ticks and limits for the ratio axis."""
    ax.set_xlim(rmin, rmax)
    i0 = int(math.ceil(rmin))
    i1 = int(math.floor(rmax))
    if i1 >= i0 and (i1 - i0) >= 3:
        ax.set_xticks(np.arange(i0, i1 + 1, 1.0))
        ax.set_xticks(np.arange(i0, i1 + 0.5, 0.5), minor=True)
    else:
        ax.set_xticks(np.linspace(rmin, rmax, 6))
    ax.tick_params(axis="x", which="major", labelsize=10)


# ===================== TABLE OUTPUT HELPERS =====================
def fcent_tag(fcent: float) -> str:
    """Return a filesystem-friendly tag for F_CENT, e.g. 1.00 -> 'fcent1p00'."""
    return f"fcent{fcent:.2f}".replace(".", "p")


def write_peak_json(outpath, meta, peaks_rows, rpeak_by_bin, overall_best):
    """Save peak-ratio results to a JSON file consumed by downstream scripts."""
    payload = {
        "meta":         meta,
        "R_peak_by_bin": rpeak_by_bin,
        "bins":         peaks_rows,
        "overall_max":  overall_best,
    }
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved peak table JSON: {outpath}")


def write_tables_xlsx(outpath, peaks_df, grid_df):
    """
    Write peak-ratio summary and the full external TDE grid to an Excel file.
    Applies number formatting and auto-widths if openpyxl is available.
    """
    if not _HAS_OPENPYXL:
        print(f"[warning] openpyxl not available; skipping XLSX output: {outpath}")
        print(f"          openpyxl import error: {_OPENPYXL_ERR}")
        return

    if (peaks_df is None or peaks_df.empty) and (grid_df is None or grid_df.empty):
        print(f"[warning] No tables to write to XLSX: {outpath}")
        return

    def _format_sheet(ws, numfmt_by_colname, freeze_cell="A2"):
        ws.freeze_panes = freeze_cell
        ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            f = copy(cell.font)
            f.bold = True
            cell.font = f
            a = copy(cell.alignment)
            a.horizontal = "center"
            cell.alignment = a

        header     = [c.value for c in ws[1]]
        name_to_col = {name: j + 1 for j, name in enumerate(header) if name is not None}

        for colname, fmt in numfmt_by_colname.items():
            if colname not in name_to_col:
                continue
            cidx = name_to_col[colname]
            for r in range(2, ws.max_row + 1):
                ws.cell(row=r, column=cidx).number_format = fmt

        for j in range(1, ws.max_column + 1):
            col_letter = get_column_letter(j)
            max_len    = 0
            for r in range(1, ws.max_row + 1):
                v = ws.cell(row=r, column=j).value
                if v is not None:
                    max_len = max(max_len, len(str(v)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 55)

    def _write(path):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            if peaks_df is not None and not peaks_df.empty:
                peaks_df.to_excel(writer, sheet_name="peaks", index=False)
            if grid_df is not None and not grid_df.empty:
                grid_df.to_excel(writer, sheet_name="ext_tot_grid", index=False)

            wb = writer.book

            if "peaks" in wb.sheetnames:
                _format_sheet(wb["peaks"], {
                    "bin_lo_log10M":       "0.00",
                    "bin_hi_log10M":       "0.00",
                    "R_peak":              "0.000",
                    "peak_total_ext_tdes": "0.00E+00",
                }, freeze_cell="A2")

            if "ext_tot_grid" in wb.sheetnames:
                fmts = {"R_center": "0.00"}
                for lab in MASS_LABELS_ALL:
                    fmts[lab] = "0.00E+00"
                _format_sheet(wb["ext_tot_grid"], fmts, freeze_cell="A2")

        print(f"Saved XLSX tables: {path}")

    try:
        _write(outpath)
    except PermissionError:
        ts   = time.strftime("%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(outpath)
        alt  = f"{root}_{ts}{ext}"
        print(f"[warning] XLSX appears open/locked. Writing to: {alt}")
        _write(alt)


# ===================== SCAN FOR ONE F_CENT VALUE =====================
def run_one_fcent(F_CENT, Zs, paths_by_z, phys_map, n_runs_nominal):
    """
    Execute the full V_kick/V_cent ratio scan for one central-boundary choice F_CENT.
    Saves XLSX and JSON tables and returns the physical external-TDE total grid
    with shape (N_MASS_ALL, N_R).
    """
    tag_fcent = fcent_tag(F_CENT)

    xlsx_out = os.path.join(BASE_DIR, f"{SCRIPT_STEM}_tables__{tag_fcent}.xlsx")
    json_out = os.path.join(BASE_DIR, f"{SCRIPT_STEM}_rpeak__{tag_fcent}.json")

    OUT_TAG = f"total_{n_runs_nominal:02d}runs" if n_runs_nominal > 1 else "total_single"
    OUT_TAG = f"{OUT_TAG}__{tag_fcent}__tdecay-{TDECAY_MODE}__physA__vcent"

    print("\n" + "=" * 86)
    print(f"[ratio_scan_vcent] START: F_CENT={F_CENT:.2f}  ({tag_fcent})")
    print("=" * 86)
    print(f"  R in [{RATIO_MIN:.2f}, {RATIO_MAX:.2f}],  DR={DR:.3f},  N_R={N_R}")
    print(f"  Ratio definition: R = V_kick / V_cent  (R_cent = F_CENT * R_e)")
    print(f"  Mass bins: {N_MASS_ALL}")
    if HAVE_PYARROW:
        print("  pyarrow available: reading only required Parquet columns.")
    else:
        print("  pyarrow not available: full Parquet reads (slower).")

    # Accumulate physical TDE totals across all snapshots
    ext_tde_tot = np.zeros((N_MASS_ALL, N_R), float)

    for z in Zs:
        paths = paths_by_z[z]
        print(f"\n=== [z={z:.2f}] {len(paths)} run file(s) ===")

        dt_to_z6_yr = float(pr.time_until_z6(z))
        if not np.isfinite(dt_to_z6_yr) or dt_to_z6_yr <= 0:
            print(f"  [warn] time_until_z6(z={z:.2f}) invalid ({dt_to_z6_yr}); skipping.")
            continue

        n_phys_by_m = np.zeros(N_MASS_ALL, dtype=float)
        for m in range(N_MASS_ALL):
            lo = float(MASS_EDGES_ALL[m])
            hi = float(MASS_EDGES_ALL[m + 1])
            n_phys_by_m[m] = float(phys_map.get(_bin_key(z, lo, hi), 0))

        ext_mean_runs    = []
        rep_cache_ready  = False
        rep_t_leave = rep_t_ext_max = rep_scale_base = rep_Vcent_over_Vesc0 = None

        for path in paths:
            try:
                if HAVE_PYARROW:
                    schema_cols  = set(pq.ParquetFile(path).schema.names)
                    cols_to_read = [c for c in NEEDED_CAND_COLS if c in schema_cols]
                    df = pd.read_parquet(path, columns=cols_to_read) if cols_to_read else pd.read_parquet(path)
                else:
                    df = pd.read_parquet(path)
            except Exception as e:
                print(f"  [warn] could not read {os.path.basename(path)}; skipping ({e})")
                continue

            if df is None or len(df) == 0:
                continue

            c_mstar = first_col(df, CAND["MSTAR"])
            c_mbh   = first_col(df, CAND["MBH"])
            c_re    = first_col(df, CAND["RE"])
            c_vesc  = first_col(df, CAND["VESC"])
            c_w     = first_col(df, CAND["W"])

            if any(x is None for x in [c_mstar, c_mbh, c_re, c_vesc]):
                raise KeyError(
                    "Missing required columns. Need equivalents of:\n"
                    "  Mstar_rem_Msun, Mrem_BH_Msun, Re_kpc, Vesc0_kms\n"
                    f"  Detected: MSTAR={c_mstar}, MBH={c_mbh}, RE={c_re}, VESC={c_vesc}"
                )

            df = df.rename(columns={c_mstar: "_MSTAR", c_mbh: "_MBH",
                                     c_re: "_RE",    c_vesc: "_VESC"})
            if c_w is not None:
                df = df.rename(columns={c_w: "_W"})
            else:
                df["_W"] = 1.0

            if not rep_cache_ready:
                rep_t_leave, rep_t_ext_max, rep_scale_base, rep_Vcent_over_Vesc0 = \
                    build_rep_times_for_snapshot(df, z, F_CENT)
                rep_cache_ready = True

                good = np.isfinite(rep_Vcent_over_Vesc0) & (rep_Vcent_over_Vesc0 > 0)
                if np.any(good):
                    med_ratio = float(np.nanmedian(rep_Vcent_over_Vesc0[good]))
                    print(f"  [rep] median V_cent / V_esc(0) across populated bins: {med_ratio:.4f}")

            Mstar = pd.to_numeric(df["_MSTAR"], errors="coerce").to_numpy(float)
            Mbh   = pd.to_numeric(df["_MBH"],   errors="coerce").to_numpy(float)
            Re    = pd.to_numeric(df["_RE"],     errors="coerce").to_numpy(float)
            Vesc0 = pd.to_numeric(df["_VESC"],   errors="coerce").to_numpy(float)
            W     = pd.to_numeric(df["_W"],      errors="coerce").to_numpy(float)

            logM   = np.log10(Mstar, where=(Mstar > 0), out=np.full_like(Mstar, np.nan))
            mass_i = np.digitize(logM, MASS_EDGES_ALL) - 1

            valid = (
                (mass_i >= 0) & (mass_i < N_MASS_ALL) &
                np.isfinite(Mbh)   & (Mbh   > 0) &
                np.isfinite(Re)    & (Re    > 0) &
                np.isfinite(Vesc0) & (Vesc0 > 0) &
                np.isfinite(W)     & (W     > 0)
            )
            if not np.any(valid):
                continue

            mass_i = mass_i[valid].astype(int)
            Mbh    = Mbh[valid]
            Re     = Re[valid]
            Vesc0  = Vesc0[valid]
            W      = W[valid]

            w_by_m = np.bincount(mass_i, weights=W, minlength=N_MASS_ALL).astype(float)
            denom  = np.where(w_by_m > 0, w_by_m, np.nan)[:, None]

            ext_tde_sum_run = np.zeros((N_MASS_ALL, N_R), float)

            # Pre-compute BH-only quantities (independent of ratio bin)
            Mbh_SI      = Mbh * pr.M_sun
            sigma_km_s, _ = pr.sigma_from_mbh(Mbh_SI)
            rt          = pr.r_t(Mbh_SI, M_STAR_SI, R_SUN_M)

            # Per-galaxy scaling of representative travel times via Re/Vesc0
            base  = rep_scale_base[mass_i]
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = (Re / Vesc0) / base
            scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

            Vcent_over_Vesc0 = rep_Vcent_over_Vesc0[mass_i]
            Vcent_over_Vesc0 = np.where(
                np.isfinite(Vcent_over_Vesc0) & (Vcent_over_Vesc0 > 0),
                Vcent_over_Vesc0, 1.0,
            )

            # Integration cap: full time to z = 6 per galaxy
            t_cap = np.where(
                np.isfinite(np.full_like(Mbh, dt_to_z6_yr)) & (dt_to_z6_yr > 0),
                dt_to_z6_yr, 0.0,
            )

            for rbin, rc in enumerate(RATIO_CENTERS):
                # V_kick = R * V_cent = R * (V_cent/V_esc(0)) * V_esc(0)
                V_kick    = rc * Vcent_over_Vesc0 * Vesc0
                V_kick    = np.where(np.isfinite(V_kick) & (V_kick >= 0), V_kick, 0.0)
                v_k_ms    = V_kick * pr.km

                rk        = pr.r_k(Mbh_SI, v_k_ms)
                r_eff_pc  = (pr.r_eff_from_rk_gamma1(rk) / pr.pc)

                Mb_coll_kg = pr.m_b_collisional(Mbh, sigma_km_s, r_eff_pc)
                cap_stars  = Mb_coll_kg / pr.M_sun
                cap        = np.where(np.isfinite(cap_stars) & (cap_stars > 0), cap_stars, 0.0)

                f_b = pr.f_b_from_mb(Mb_coll_kg, Mbh_SI)
                f_b = np.clip(np.where(np.isfinite(f_b), f_b, 0.0), 0.0, 1.0)

                rate_s  = pr.tde_rate_resonant(Mbh_SI, M_STAR_SI, rk, rt, v_k_ms, f_b)
                rate_yr = np.where(np.isfinite(rate_s) & (rate_s > 0), rate_s * SEC_PER_YEAR, 0.0)

                if TDECAY_MODE == "vkick":
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        t_decay_yr = 1.5e9 * (Mbh / 1.0e7) ** 2 * (V_kick / 1.0e3) ** (-3.0)
                    t_decay_yr = np.where(
                        (V_kick <= 0) & np.isfinite(Mbh), TDECAY_NO_DECAY_YR, t_decay_yr
                    )
                elif TDECAY_MODE == "mass_only":
                    t_decay_yr = 1.5e9 * (Mbh / 1.0e7) ** 2
                else:
                    raise ValueError(f"Unknown TDECAY_MODE: {TDECAY_MODE}")

                t_decay_yr = np.where(
                    np.isfinite(t_decay_yr) & (t_decay_yr > 0), t_decay_yr, 0.0
                )

                # Orbit segmentation: central and external exposure times.
                rep_leave = rep_t_leave[mass_i, rbin]
                t_leave   = rep_leave * scale
                t_leave   = np.where(np.isfinite(t_leave) & (t_leave > 0), t_leave, np.nan)

                leaves       = np.isfinite(t_leave) & (t_leave < t_cap)
                t_cent_pre   = np.where(leaves, t_leave, t_cap)

                t_ext    = np.zeros_like(t_cap)
                rem_time = np.maximum(0.0, t_cap - t_cent_pre)

                rep_ext_max = rep_t_ext_max[mass_i, rbin]
                t_ext_max   = rep_ext_max * scale

                has_ext_max = leaves & np.isfinite(t_ext_max) & (t_ext_max > 0)
                t_ext[has_ext_max]  = np.minimum(t_ext_max[has_ext_max], rem_time[has_ext_max])
                t_ext[leaves & ~has_ext_max] = rem_time[leaves & ~has_ext_max]

                # Integrate the exponential TDE rate over each orbit segment.
                N_cent_pre_unc = integral_exp(rate_yr, t_decay_yr, 0.0, t_cent_pre)
                t_leave_eff    = np.where(t_ext > 0, t_cent_pre, 0.0)
                N_ext_unc      = integral_exp(rate_yr, t_decay_yr, t_leave_eff, t_leave_eff + t_ext)

                # Apply the bound-star cap sequentially so external events cannot
                # exceed the remaining bound reservoir.
                N_cent_pre = np.minimum(N_cent_pre_unc, cap)
                rem        = np.maximum(0.0, cap - N_cent_pre)
                N_ext_fin  = np.where(
                    np.isfinite(np.minimum(N_ext_unc, rem)) & (np.minimum(N_ext_unc, rem) > 0),
                    np.minimum(N_ext_unc, rem), 0.0,
                )

                ext_tde_sum_run[:, rbin] += np.bincount(
                    mass_i, weights=W * N_ext_fin, minlength=N_MASS_ALL
                )

            ext_mean_run = np.where(np.isfinite(denom), ext_tde_sum_run / denom, np.nan)
            ext_mean_runs.append(ext_mean_run)

        if not ext_mean_runs:
            print("  -> no usable run files; skipping snapshot")
            continue

        ext_mean_z = np.nanmean(np.stack(ext_mean_runs, axis=0), axis=0)
        ext_mean_z = np.where(np.isfinite(ext_mean_z) & (ext_mean_z >= 0), ext_mean_z, 0.0)

        ext_tde_tot += ext_mean_z * n_phys_by_m.astype(float)[:, None]
        print(f"  -> used {len(ext_mean_runs)} run(s); added physical totals for this snapshot")

    # ===================== PEAK RATIOS =====================
    print("\n[Peak ratios] External TDE TOTAL expected TDEs — all mass bins:")
    print("  R_peak is defined as R = V_kick / V_cent.")
    overall_best = {"val": -np.inf, "R": np.nan, "bin": None}
    peaks_rows   = []
    rpeak_by_bin = {}

    for m in range(N_MASS_ALL):
        lab = MASS_LABELS_ALL[m]
        row = ext_tde_tot[m, :]
        if np.any(np.isfinite(row)) and np.nanmax(row) > 0:
            jj  = int(np.nanargmax(row))
            rpk = float(RATIO_CENTERS[jj])
            tpk = float(row[jj])
            print(f"  {lab}: R_peak = {rpk:.3f}   (peak total = {tpk:.6g})")
        else:
            rpk = float(RATIO_CENTERS[0])
            tpk = 0.0
            print(f"  {lab}: R_peak = {rpk:.3f}   (peak total = 0)")

        rpeak_by_bin[lab] = rpk
        peaks_rows.append({
            "bin_label":           lab,
            "bin_lo_log10M":       float(MASS_EDGES_ALL[m]),
            "bin_hi_log10M":       float(MASS_EDGES_ALL[m + 1]),
            "R_peak":              rpk,
            "peak_total_ext_tdes": tpk,
        })

        if np.isfinite(tpk) and tpk > overall_best["val"]:
            overall_best = {"val": tpk, "R": rpk, "bin": lab}

    if np.isfinite(overall_best["val"]) and overall_best["bin"] is not None:
        print(f"[Overall max] {overall_best['bin']}: R = {overall_best['R']:.3f}   "
              f"total = {overall_best['val']:.6g}")

    # ===================== SAVE TABLES =====================
    meta = {
        "script":           SCRIPT_STEM,
        "ratio_definition": "R = V_kick / V_cent",
        "F_CENT":           float(F_CENT),
        "TDECAY_MODE":      str(TDECAY_MODE),
        "RATIO_MIN":        float(RATIO_MIN),
        "RATIO_MAX":        float(RATIO_MAX),
        "DR":               float(DR),
        "R_centers":        [float(x) for x in RATIO_CENTERS],
        "parquet_root":     PARQUET_ROOT,
        "targets_file":     TARGETS_PQ,
        "out_tag":          OUT_TAG,
    }

    peaks_df = pd.DataFrame(peaks_rows)
    if overall_best["bin"] is not None:
        peaks_df = pd.concat([
            peaks_df,
            pd.DataFrame([{
                "bin_label":           "OVERALL_MAX",
                "bin_lo_log10M":       np.nan,
                "bin_hi_log10M":       np.nan,
                "R_peak":              float(overall_best["R"]),
                "peak_total_ext_tdes": float(overall_best["val"]),
            }])
        ], ignore_index=True)

    grid_df = pd.DataFrame({"R_center": RATIO_CENTERS})
    for m, lab in enumerate(MASS_LABELS_ALL):
        grid_df[lab] = ext_tde_tot[m, :]

    write_peak_json(json_out, meta, peaks_rows, rpeak_by_bin, overall_best)
    write_tables_xlsx(xlsx_out, peaks_df, grid_df)

    print(f"[ratio_scan_vcent] DONE: F_CENT={F_CENT:.2f}  -> tables saved ({tag_fcent})")
    return ext_tde_tot


# ===================== MAIN =====================
def main():
    """Run the ratio scan for all configured central-boundary choices and save the comparison figure."""
    plt.rcParams.update({
        "font.size":      12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    df_targets = load_bin_targets_physical()
    phys_map   = build_phys_target_map(df_targets)
    print(f"[targets] Loaded from: {TARGETS_PQ}")
    print(f"[targets] Entries: {len(phys_map):,}")

    Zs, paths_by_z   = discover_parquet_snapshots(PARQUET_ROOT)
    n_runs_nominal    = max(len(v) for v in paths_by_z.values())

    # Run scan for each F_CENT value
    results = []
    for fcent in F_CENT_LIST:
        ext_tde_tot = run_one_fcent(fcent, Zs, paths_by_z, phys_map, n_runs_nominal)
        results.append((fcent, ext_tde_tot))

    # ===================== COMBINED FIGURE =====================
    out_png = os.path.join(FIG_DIR, "external_tdes_vs_ratio_fcent_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.3), sharex=True)

    ext_tde_plots = [
        np.where(np.isfinite(ext_tde_tot) & (ext_tde_tot > 0), ext_tde_tot, np.nan)
        for _, ext_tde_tot in results
    ]

    all_pos = np.concatenate([p[np.isfinite(p)] for p in ext_tde_plots]) if ext_tde_plots else np.array([])
    all_pos = all_pos[all_pos > 0]
    if all_pos.size > 0:
        ymin = float(np.min(all_pos)) / 1.30
        ymax = float(np.max(all_pos)) * 1.10
        use_common_lim = True
    else:
        use_common_lim = False

    for ax, (fcent, _), ext_plot in zip(axes, results, ext_tde_plots):
        for m in range(N_MASS_ALL):
            y = ext_plot[m, :]
            line, = ax.plot(RATIO_CENTERS, y, lw=2.2, label=MASS_LABELS_ALL[m])

            finite = np.isfinite(y)
            if np.count_nonzero(finite) == 1:
                ax.scatter(
                    RATIO_CENTERS[finite],
                    y[finite],
                    s=35,
                    color=line.get_color(),
                    marker="o",
                    zorder=5,
                )
        ax.axvline(1.0, ls="--", lw=1.5, alpha=0.8)
        ax.set_title(
            f"Total expected external TDEs\n"
            r"$f_{\rm cent}=" + f"{fcent:.2f}$"
        )
        ax.set_xlabel(r"$\mathcal{R} = V_{\rm kick} / V_{\rm cent}$")
        set_ratio_xticks(ax, RATIO_MIN, RATIO_MAX)
        ax.grid(True, which="both", ls=":", alpha=0.7)

        if use_common_lim:
            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
        else:
            ok, _, _ = set_logy_positive(ax, ext_plot)
            if not ok:
                print(f"[plot] Warning: no positive external TDE totals for fcent={fcent:.2f}; "
                      "leaving y-axis linear.")

    axes[0].set_ylabel(r"$N_{\rm TDE}$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Host mass [log10 M*]",
        bbox_to_anchor=(1.01, 0.98),
        loc="upper left",
    )
    fig.tight_layout()
    save_fig(fig, out_png)
    plt.close(fig)

    print(f"\nSaved figure: {out_png}")


if __name__ == "__main__":
    main()
