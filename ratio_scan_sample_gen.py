#!/usr/bin/env python3
"""
ratio_scan_sample_gen.py
------------------------
Generate the galaxy-pair merger catalogues used by the downstream
V_kick/V_cent ratio scan.

For every merger snapshot, descendant stellar-mass bin and merger class, the script:
  1. Evaluates the galaxy population and merger rate at the snapshot redshift
     convention used by the no-delay model.
  2. Integrates over all valid progenitor masses and mass ratios that produce
     a descendant in the requested bin.
  3. Rounds the descendant-bin physical target and applies the Monte Carlo cap.
  4. Draws explicit primary/secondary galaxy pairs from the same two-dimensional
     rate distribution used for the target integral.
  5. Maps both galaxies to central BHs, merges the BHs and galaxies, and saves
     the descendant properties needed by ratio_scan_vcent.py.

Ordinary failed proposals are redrawn until each capped context is full.  The
saved row weight n_phys/n_samp is the only population rescaling applied later.

Outputs
-------
  ratio_scan_catalogue/runXX/data_z_*.parquet
  results_bin_targets/bin_targets_physical.parquet
"""

from __future__ import annotations

import os

# Limit BLAS thread counts before importing NumPy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

import merger_pair_sampling as mps
import physics_relations as pr


# ---------------------------------------------------------------------------
# Paths and Parquet support
# ---------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

CATALOGUE_DIR = os.path.join(BASE_DIR, "ratio_scan_catalogue")
BIN_TARGETS_DIR = os.path.join(BASE_DIR, "results_bin_targets")

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
        raise ImportError("A Parquet engine is required. Install pyarrow.") from exc


# ---------------------------------------------------------------------------
# Shared numerical setup
# ---------------------------------------------------------------------------
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.31, Ob0=0.048, Tcmb0=2.725 * u.K)

Z_EOR_INITIAL = 12.0
Z_START = 11.8
Z_END = 6.2
DZ_SNAP = -0.2

DZ_SLICE = 0.01
OMEGA_FULL = 4.0 * np.pi
MIN_COUNT = 1.0

N_RUNS = 10
BASE_SEED = 12345
SEED_STRIDE = 10_000
MAX_EVENTS_PER_BIN = 100_000

CONTROLS = mps.SamplingControls(
    max_events_per_context=MAX_EVENTS_PER_BIN,
    integration_steps=2048,
    max_tries_multiplier=200,
    max_tries_per_context=10_000_000,
)

PRINT_TARGETS = True


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def snapshot_file_tag(z: float) -> str:
    """Filename-safe redshift tag, for example 11.8 -> z_11_8."""
    z_tag = f"{z:.2f}".replace(".", "_")
    return "z_" + z_tag.rstrip("0").rstrip("_")


def shell_volume_mpc3(z: float, dz: float = DZ_SLICE) -> float:
    """Full-sky comoving volume of the thin GSMF counting shell."""
    dvc = cosmo.differential_comoving_volume(float(z)).to(u.Mpc**3 / u.sr).value
    return float(dvc * dz * OMEGA_FULL)


def build_redshift_grid() -> list[float]:
    """Return the EoR snapshot grid in descending redshift."""
    values = []
    z = Z_START
    while z >= Z_END - 1e-8:
        values.append(float(f"{z:.2f}"))
        z += DZ_SNAP
    return values


def build_snapshot_plan(z_values: list[float]) -> list[dict]:
    """Precompute deterministic targets and sampling contexts for all snapshots."""
    plan: list[dict] = []

    for idx, z_cur in enumerate(z_values):
        # Galaxy merger, BH coalescence, recoil, and TDE evolution are all tied
        # to the same snapshot convention. The rate is evaluated at the
        # interval midpoint, except for the first short interval below z=12.
        if idx == 0:
            z_rate = float(z_cur)
            t_prev = float(pr.age_of_universe_at_z(Z_EOR_INITIAL))
            t_cur = float(pr.age_of_universe_at_z(z_cur))
        else:
            z_prev = z_values[idx - 1]
            z_rate = 0.5 * (z_prev + z_cur)
            t_prev = float(pr.age_of_universe_at_z(z_prev))
            t_cur = float(pr.age_of_universe_at_z(z_cur))

        dt_gyr = (t_cur - t_prev) / 1.0e9
        volume = shell_volume_mpc3(z_cur)
        contexts, target_rows = mps.build_snapshot_contexts(
            z_snapshot=z_cur,
            z_rate=z_rate,
            dt_gyr=dt_gyr,
            shell_volume_mpc3=volume,
            min_primary_count=MIN_COUNT,
            controls=CONTROLS,
        )

        plan.append({
            "idx": idx,
            "z": float(z_cur),
            "z_rate": float(z_rate),
            "dt_gyr": float(dt_gyr),
            "shell_volume_mpc3": float(volume),
            "contexts": contexts,
            "target_rows": target_rows,
        })

    return plan


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame with the available Parquet engine."""
    codec = _PARQUET_CODEC.upper() if _PARQUET_ENGINE == "fastparquet" else _PARQUET_CODEC
    df.to_parquet(path, engine=_PARQUET_ENGINE, compression=codec, index=False)


def _downcast_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast bulk event values while preserving exact weights/bin edges."""
    out = df.copy()
    preserve_float64 = {
        "weight", "descendant_bin_lo", "descendant_bin_hi",
        "descendant_bin_lo_log10M", "descendant_bin_hi_log10M",
        "desc_bin_lo_log10M", "desc_bin_hi_log10M",
        "bin_lo_log10M", "bin_hi_log10M",
    }
    for col in out.select_dtypes(include=["float64"]).columns:
        if col not in preserve_float64:
            out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["int64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def save_remnant_catalogue(df: pd.DataFrame, out_base: str, run_tag: str) -> None:
    """Save one ratio-scan precursor catalogue."""
    out_dir = os.path.join(CATALOGUE_DIR, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_base}.parquet")

    desired_cols = [
        "event_id", "run_id", "z", "z_rate",
        "population_model_version",
        "merger_class",
        "descendant_bin_lo_log10M", "descendant_bin_hi_log10M",
        "desc_bin_lo_log10M", "desc_bin_hi_log10M",
        "bin_lo_log10M", "bin_hi_log10M",
        "Mstar_primary_Msun", "Mstar_secondary_Msun", "mu_star",
        "m1_BH_Msun", "m2_BH_Msun", "q", "Mrem_BH_Msun",
        "Mstar_rem_Msun", "Re_kpc", "log10_Mh_fire2", "Vesc0_kms",
        "weight",
    ]
    keep = [col for col in desired_cols if col in df.columns]
    df_save = pd.DataFrame(columns=keep) if df.empty else _downcast_for_storage(df[keep])
    _write_parquet(df_save, out_path)
    print(f"  Saved: {out_path}  ({len(df_save):,} rows)")


def save_bin_targets(snapshot_plan: list[dict]) -> None:
    """Save deterministic descendant-bin/class physical targets."""
    os.makedirs(BIN_TARGETS_DIR, exist_ok=True)
    rows = [row for snap in snapshot_plan for row in snap["target_rows"]]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = _downcast_for_storage(df)
    out_path = os.path.join(BIN_TARGETS_DIR, "bin_targets_physical.parquet")
    _write_parquet(df, out_path)
    print(f"[bin targets] Saved: {out_path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------
def print_context_summary(contexts: list[dict]) -> None:
    """Print physical and sampled targets by descendant bin and class."""
    if not PRINT_TARGETS:
        return
    print(f"  {'Descendant bin':>21}  {'class':>6}  {'N_phys':>12}  {'N_samp':>10}  {'weight':>10}")
    print("  " + "-" * 68)
    for ctx in contexts:
        lo = ctx["descendant_bin_lo_log10M"]
        hi = ctx["descendant_bin_hi_log10M"]
        print(
            f"  [{lo:6.3f}, {hi:6.3f}]  {ctx['merger_class']:>6}  "
            f"{ctx['n_events_phys']:12d}  {ctx['n_events_samp']:10d}  "
            f"{ctx['weight']:10.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate the ratio-scan galaxy-pair catalogues and target table."""
    z_values = build_redshift_grid()
    print("Precomputing galaxy-pair snapshot plan...")
    snapshot_plan = build_snapshot_plan(z_values)
    save_bin_targets(snapshot_plan)

    for run_idx in range(N_RUNS):
        run_tag = f"run{run_idx:02d}"
        seed_base = BASE_SEED + run_idx * SEED_STRIDE
        print("\n" + "=" * 72)
        print(f"RUN {run_tag}  (seed_base={seed_base})")
        print("=" * 72)

        for snap in snapshot_plan:
            z_cur = snap["z"]
            rng = np.random.default_rng(seed_base + snap["idx"])
            contexts = snap["contexts"]

            print(
                f"\n  z={z_cur:.2f} | z_rate={snap['z_rate']:.3f} | "
                f"dt={snap['dt_gyr']:.4e} Gyr | contexts={len(contexts)}"
            )
            if run_idx == 0:
                print_context_summary(contexts)

            df, summary = mps.fill_snapshot_contexts(
                contexts,
                z_snapshot=z_cur,
                rng=rng,
                controls=CONTROLS,
            )
            df = df.reset_index(drop=True)
            df.insert(0, "event_id", np.arange(len(df), dtype=np.int64))
            df.insert(1, "run_id", run_tag)

            for key, info in summary.items():
                if info["filled"] != info["target_samp"]:
                    raise RuntimeError(f"Underfilled context after sampling: {key}: {info}")

            print(f"  Generated {len(df):,} valid galaxy-pair remnants")
            out_base = f"data_{snapshot_file_tag(z_cur)}"
            save_remnant_catalogue(df, out_base, run_tag)

        print(f"\nFinished {run_tag}.")

    print("\nAll ratio-scan catalogues complete.")


if __name__ == "__main__":
    main()
