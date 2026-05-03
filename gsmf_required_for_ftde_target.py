#!/usr/bin/env python3
"""
gsmf_required_for_ftde_target.py
---------------------------------
Diagnostic what-if calculation for the GSMF normalisation required so that the
recoiling-remnant TDE channel would reach a target fractional contribution to
EoR hydrogen reionisation (f_TDE_target = 0.1 by default). This is not a
Monte Carlo rerun.

Because the total ionizing energy scales linearly with the number of
galaxies, and that number is directly proportional to phi_star, reaching
f_TDE_target requires boosting phi_star by a factor of

    boost = f_TDE_target / f_TDE_current

at every redshift and mass bin. The shape parameters (M_char, alpha) are
unchanged: only the comoving number density of galaxies is scaled.
This is a fixed-shape diagnostic: the Schechter shape parameters are held
fixed, and only phi_star is shifted.

Inputs
------
  - GSMF fit parameters from gsmf_fitting.py
  - thesis-result energy numbers hard-coded in this file

Outputs
-------
  figures/gsmf_required_for_ftde_target.png
      Comparison of the original FIRE-2-calibrated GSMF (solid) and the
      required GSMF (dashed) at integer redshifts z = 6-12.

  gsmf_required_for_ftde_target.txt
      Table of the original and required Schechter parameters, together
      with the boost factor and key energy figures.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
# Use a non-interactive backend so figures can be written on headless systems.
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow the script to find gsmf_fitting.py when run from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gsmf_fitting import get_gsmf_params, log10_schechter

# ---------------------------------------------------------------------------
# Reference simulation results
# ---------------------------------------------------------------------------

# Total ionising energy produced by external TDEs over the full EoR window, from thesis results.
E_ION_TOT_ERG = 1.4796011920675e+65   # erg  (Eq. results section)


# Total ionising energy required to reionise the IGM over the EoR shell, from thesis results.
E_ION_REQ_ERG = 4.80584958169175e+68   # erg  (Eq. results section)

# The resulting fractional contribution from the current simulation.
F_TDE_CURRENT = E_ION_TOT_ERG / E_ION_REQ_ERG

# Target contribution fraction for this analysis.
F_TDE_TARGET = 0.10

# ---------------------------------------------------------------------------
# Figure output directory (mirrors the convention in gsmf_fitting.py)
# ---------------------------------------------------------------------------
try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()

_FIG_DIR = os.path.join(_BASE_DIR, "figures")
os.makedirs(_FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_required_params(
    f_tde_current: float,
    f_tde_target: float,
) -> tuple[float, float, dict[int, dict]]:
    """
    Return the boost factor and the boosted Schechter parameters.

    The boost is applied only to log10_phi_star; log10_M_char, alpha, and
    shape_locked are left unchanged because they govern the shape of the
    galaxy population, not its overall abundance.

    Parameters
    ----------
    f_tde_current : fractional contribution from the current simulation.
    f_tde_target  : desired fractional contribution.

    Returns
    -------
    boost_factor        : multiplicative factor by which phi_star must increase.
    delta_log10_phi     : additive shift in log10(phi_star) (= log10(boost_factor)).
    boosted_params      : dict keyed by integer redshift with updated parameters.
    """
    boost_factor    = f_tde_target / f_tde_current
    delta_log10_phi = np.log10(boost_factor)

    original_params = get_gsmf_params()
    boosted_params: dict[int, dict] = {}

    for z, p in original_params.items():
        boosted_params[z] = {
            "log10_phi_star": p["log10_phi_star"] + delta_log10_phi,
            "log10_M_char":   p["log10_M_char"],
            "alpha":          p["alpha"],
            "shape_locked":   p["shape_locked"],
        }

    return boost_factor, delta_log10_phi, boosted_params


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Redshifts and colours chosen for the comparison plot.
# Three values (low, mid, high) keep the figure uncluttered while spanning
# the full EoR range.
_PLOT_REDSHIFTS = [6, 9, 12]
_COLORS = {6: "C0", 9: "C2", 12: "C3"}

# Mass and phi limits matching the simulation range and the existing GSMF plots.
_LOG10_M_LO, _LOG10_M_HI = 6.75, 9.90
_LOG10_PHI_LO, _LOG10_PHI_HI = -5.5, 1.5


def plot_gsmf_comparison(
    original_params: dict[int, dict],
    boosted_params:  dict[int, dict],
    boost_factor:    float,
    f_tde_target:    float,
    f_tde_current:   float,
) -> None:
    """
    Plot the original and required Schechter curves for a representative
    subset of redshifts.

    Solid lines show the FIRE-2-calibrated GSMF; dashed lines show the
    normalisation required to reach f_TDE = f_tde_target. Curves are
    labelled directly to avoid a cluttered legend. A text box reports the
    constant shift in log10(phi_star).
    """
    log10_M_grid = np.linspace(_LOG10_M_LO - 0.1, _LOG10_M_HI + 0.1, 600)
    delta_log10_phi = np.log10(boost_factor)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    for z in _PLOT_REDSHIFTS:
        color = _COLORS[z]

        popt_orig  = (original_params[z]["log10_phi_star"],
                      original_params[z]["log10_M_char"],
                      original_params[z]["alpha"])
        popt_boost = (boosted_params[z]["log10_phi_star"],
                      boosted_params[z]["log10_M_char"],
                      boosted_params[z]["alpha"])

        phi_orig  = log10_schechter(log10_M_grid, *popt_orig)
        phi_boost = log10_schechter(log10_M_grid, *popt_boost)

        ax.plot(log10_M_grid, phi_orig,  ls="-",  lw=2.2, color=color, alpha=0.90)
        ax.plot(log10_M_grid, phi_boost, ls="--", lw=2.2, color=color, alpha=0.90)

        # Direct label on the original curve at a fixed stellar mass.
        # The label sits just above the curve at log10_M = 7.4.
        label_x = 7.4
        label_y = float(log10_schechter(np.array([label_x]), *popt_orig)[0])
        if _LOG10_PHI_LO < label_y < _LOG10_PHI_HI:
            ax.text(
                label_x, label_y + 0.18,
                rf"$z={z}$",
                color=color, fontsize=10, ha="center", va="bottom",
                fontweight="bold",
            )

    # Style legend (solid = original, dashed = required): placed in the
    # lower-left where no curves reach within the clipped y-range.
    ax.plot([], [], ls="-",  lw=2, color="grey", label="Original FIRE-2 GSMF")
    ax.plot([], [], ls="--", lw=2, color="grey",
            label=rf"Required GSMF ($f_{{\rm TDE}}={f_tde_target:.2f}$)")
    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    # Text box summarising the global normalisation shift.
    # Placed in the upper-right corner where no curves reach within the
    # clipped axes. An arrow would be misleading because the vertical gap
    # between original and boosted curves varies with stellar mass (the
    # Schechter shape is nonlinear); the shift is constant only in log10(phi_star).
    info_text = (
        rf"Uniform $\phi_\star$ shift: $+{delta_log10_phi:.2f}$ dex"
        "\n"
        rf"(i.e. $\phi_\star \times {boost_factor:.0f}$ at all $z$)"
    )
    ax.text(
        0.97, 0.97, info_text,
        transform=ax.transAxes,
        fontsize=9.5, va="top", ha="right",
        color="dimgrey",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="lightgrey", alpha=0.9),
    )

    ax.set_xlim(_LOG10_M_LO - 0.05, _LOG10_M_HI + 0.05)
    ax.set_ylim(_LOG10_PHI_LO, _LOG10_PHI_HI)
    ax.set_xlabel(r"$\log_{10}(M_\star\,/\,M_\odot)$", fontsize=13)
    ax.set_ylabel(r"$\log_{10}\,\phi\;[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$", fontsize=13)
    ax.set_title(
        rf"GSMF required for $f_{{\rm TDE}}={f_tde_target:.2f}$"
        rf" (original $f_{{\rm TDE}}={f_tde_current:.2e}$)",
        fontsize=11,
    )
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()

    outpath = os.path.join(_FIG_DIR, "gsmf_required_for_ftde_target.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def save_results_txt(
    original_params:  dict[int, dict],
    boosted_params:   dict[int, dict],
    boost_factor:     float,
    delta_log10_phi:  float,
    f_tde_current:    float,
    f_tde_target:     float,
) -> None:
    """
    Write a plain-text summary of the boost factor and original/required
    Schechter parameters.
    """
    outpath = os.path.join(_BASE_DIR, "gsmf_required_for_ftde_target.txt")

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("GSMF normalisation required to reach f_TDE = {:.2f}".format(f_tde_target))
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  E_ion_tot (current simulation)  = {E_ION_TOT_ERG:.3e} erg")
    lines.append(f"  E_ion_req (EoR full-sky shell)   = {E_ION_REQ_ERG:.3e} erg")
    lines.append(f"  f_TDE (current)                  = {f_tde_current:.3e}")
    lines.append(f"  f_TDE (target)                   = {f_tde_target:.2f}")
    lines.append(f"  Required boost factor             = {boost_factor:.2f}")
    lines.append(f"  Shift in log10(phi_star)          = +{delta_log10_phi:.4f} dex")
    lines.append("")
    lines.append(
        "  Interpretation: to reach f_TDE = {:.2f}, the comoving galaxy number".format(f_tde_target)
    )
    lines.append(
        f"  density would need to be ~{boost_factor:.0f}x higher at every redshift and"
    )
    lines.append("  mass bin, with the shape of the GSMF unchanged.")
    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"{'z':>4}  {'log10_phi* (orig)':>18}  {'log10_phi* (req)':>17}"
        f"  {'log10_M_char':>13}  {'alpha':>7}  {'shape_lock':>10}"
    )
    lines.append("-" * 72)

    for z in sorted(original_params.keys()):
        p_orig  = original_params[z]
        p_boost = boosted_params[z]
        lock    = "yes" if p_orig["shape_locked"] else "no"
        lines.append(
            f"{z:>4}  {p_orig['log10_phi_star']:>18.4f}  {p_boost['log10_phi_star']:>17.4f}"
            f"  {p_orig['log10_M_char']:>13.4f}  {p_orig['alpha']:>7.4f}  {lock:>10}"
        )

    lines.append("-" * 72)
    lines.append("")

    with open(outpath, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute the fixed-shape GSMF boost and write the diagnostic figure and text table."""
    boost_factor, delta_log10_phi, boosted_params = compute_required_params(
        f_tde_current=F_TDE_CURRENT,
        f_tde_target=F_TDE_TARGET,
    )

    original_params = get_gsmf_params()

    print("\n=== GSMF required for f_TDE = {:.2f} ===\n".format(F_TDE_TARGET))
    print(f"  Current f_TDE             : {F_TDE_CURRENT:.3e}")
    print(f"  Target  f_TDE             : {F_TDE_TARGET:.2f}")
    print(f"  Required phi_star boost   : x{boost_factor:.1f}")
    print(f"  Shift in log10(phi_star)  : +{delta_log10_phi:.4f} dex")
    print()

    print(
        f"  {'z':>4}  {'log10_phi* (orig)':>18}  {'log10_phi* (req)':>17}"
        f"  {'log10_M_char':>13}  {'alpha':>7}"
    )
    print("  " + "-" * 68)
    for z in sorted(original_params.keys()):
        p_o = original_params[z]
        p_b = boosted_params[z]
        print(
            f"  {z:>4}  {p_o['log10_phi_star']:>18.4f}  {p_b['log10_phi_star']:>17.4f}"
            f"  {p_o['log10_M_char']:>13.4f}  {p_o['alpha']:>7.4f}"
        )
    print()

    plot_gsmf_comparison(
        original_params=original_params,
        boosted_params=boosted_params,
        boost_factor=boost_factor,
        f_tde_target=F_TDE_TARGET,
        f_tde_current=F_TDE_CURRENT,
    )

    save_results_txt(
        original_params=original_params,
        boosted_params=boosted_params,
        boost_factor=boost_factor,
        delta_log10_phi=delta_log10_phi,
        f_tde_current=F_TDE_CURRENT,
        f_tde_target=F_TDE_TARGET,
    )


if __name__ == "__main__":
    main()
