#!/usr/bin/env python3
"""
gsmf_fitting.py
---------------
Fits a single-Schechter galaxy stellar mass function (GSMF) to the
completeness-limited FIRE-2 simulation bins (Ma et al. 2018, Appendix C)
at integer redshifts z = 6-12.

At z >= 11, the FIRE-2 data poorly constrain the Schechter knee, so a
shape-lock is applied: M_char and alpha are fixed to their z = 10 values
and only the normalisation phi_star is refitted.

Public interface
----------------
get_gsmf_params()
    Returns a dict keyed by integer redshift with the three fitted
    Schechter parameters and a flag indicating whether the shape-lock
    was applied. Safe to call from other modules; the fit runs once
    and the result is cached.

log10_schechter(log10_M, log10_phi_star, log10_M_char, alpha)
    Evaluates log10(phi) of the per-dex Schechter function at the
    given stellar masses.

Outputs
-------
figures/gsmf_schechter_fits.png
figures/gsmf_schechter_fits_interpolated.png

Running the script directly prints the fitted parameters and regenerates the figures.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit

# Use a non-interactive backend so figures can be written on headless systems.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

try:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE_DIR = os.getcwd()

_FIG_DIR = os.path.join(_BASE_DIR, "figures")
os.makedirs(_FIG_DIR, exist_ok=True)


def _save_figure(fig: plt.Figure, filename: str, dpi: int = 200) -> str:
    """Save a Matplotlib figure to figures/ and close it."""
    outpath = os.path.join(_FIG_DIR, filename)
    existed = os.path.exists(outpath)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(("Updated: " if existed else "Saved:  ") + outpath)
    return outpath


# ---------------------------------------------------------------------------
# Schechter function
# ---------------------------------------------------------------------------

def log10_schechter(
    log10_M: np.ndarray,
    log10_phi_star: float,
    log10_M_char: float,
    alpha: float,
) -> np.ndarray:
    """
    Evaluate log10(phi) for the per-dex single-Schechter GSMF.

    Parameters
    ----------
    log10_M        : log10 of stellar mass [M_sun]
    log10_phi_star : log10 of the normalisation [Mpc^-3 dex^-1]
    log10_M_char   : log10 of the characteristic mass [M_sun]
    alpha          : low-mass slope

    Returns
    -------
    log10(phi) [Mpc^-3 dex^-1]
    """
    M        = 10.0 ** np.asarray(log10_M, dtype=float)
    M_char   = 10.0 ** float(log10_M_char)
    phi_star = 10.0 ** float(log10_phi_star)

    x   = M / M_char
    phi = np.log(10.0) * phi_star * (x ** (alpha + 1.0)) * np.exp(-x)

    # Guard against log10(0) from floating-point underflow.
    phi = np.maximum(phi, 1e-300)
    return np.log10(phi)


# ---------------------------------------------------------------------------
# FIRE-2 completeness-limited binned number densities (Ma et al. 2018, App. C)
# ---------------------------------------------------------------------------

_FIRE2_DATA: Dict[int, Dict[str, np.ndarray]] = {
    6:  {"log10_M":   np.array([4.48, 5.13, 5.78, 6.43, 7.08, 7.74, 8.39, 9.04, 9.69]),
         "log10_phi": np.array([0.62, 0.35, -0.09, -0.56, -1.12, -1.66, -2.32, -2.88, -3.98])},
    7:  {"log10_M":   np.array([4.42, 5.04, 5.65, 6.26, 6.88, 7.49, 8.11, 8.72, 9.33]),
         "log10_phi": np.array([0.60, 0.32, -0.16, -0.63, -1.20, -1.68, -2.32, -2.99, -4.00])},
    8:  {"log10_M":   np.array([4.57, 5.29, 6.00, 6.72, 7.43, 8.15, 8.86]),
         "log10_phi": np.array([0.48, -0.01, -0.61, -1.31, -1.89, -2.64, -3.89])},
    9:  {"log10_M":   np.array([4.50, 5.17, 5.84, 6.50, 7.17, 7.84, 8.51]),
         "log10_phi": np.array([0.39, -0.11, -0.69, -1.38, -2.00, -2.69, -3.66])},
    10: {"log10_M":   np.array([4.41, 5.01, 5.62, 6.22, 6.83, 7.43, 8.04]),
         "log10_phi": np.array([0.32, -0.12, -0.74, -1.39, -1.98, -2.54, -3.44])},
    11: {"log10_M":   np.array([4.46, 5.10, 5.74, 6.38, 7.03, 7.67]),
         "log10_phi": np.array([0.21, -0.44, -1.10, -1.87, -2.79, -2.90])},
    12: {"log10_M":   np.array([4.42, 5.04, 5.65, 6.26, 6.88, 7.49]),
         "log10_phi": np.array([0.07, -0.52, -1.33, -1.97, -2.81, -3.25])},
}

# z = 10 is treated as the last redshift where the FIRE-2 bins still constrain
# the Schechter knee. Its fitted M_char and alpha values are reused for
# z = 11 and z = 12.
_SHAPE_LOCK_REF_Z = 10
_SHAPE_LOCK_ZS    = {11, 12}


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _fit_schechter_free(
    log10_M: np.ndarray,
    log10_phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit log10_phi_star, log10_M_char, and alpha freely in log-space."""
    p0     = (-1.5, 9.0, -1.8)
    bounds = ((-12.0, 5.0, -3.0), (2.0, 15.0, -0.5))
    popt, pcov = curve_fit(
        log10_schechter,
        np.asarray(log10_M, dtype=float),
        np.asarray(log10_phi, dtype=float),
        p0=p0,
        bounds=bounds,
        maxfev=20_000,
    )
    return popt, pcov


def _fit_schechter_phi_only(
    log10_M: np.ndarray,
    log10_phi: np.ndarray,
    fixed_log10_M_char: float,
    fixed_alpha: float,
) -> Tuple[float, float]:
    """
    Fit only log10_phi_star while holding log10_M_char and alpha fixed to
    the reference shape.
    Returns (log10_phi_star, variance).
    """
    def _schechter_fixed_shape(log10_M: np.ndarray, log10_phi_star: float) -> np.ndarray:
        return log10_schechter(log10_M, log10_phi_star, fixed_log10_M_char, fixed_alpha)

    popt, pcov = curve_fit(
        _schechter_fixed_shape,
        np.asarray(log10_M, dtype=float),
        np.asarray(log10_phi, dtype=float),
        p0=[-3.5],
        bounds=([-12.0], [2.0]),
        maxfev=20_000,
    )
    return float(popt[0]), float(pcov[0, 0])


# ---------------------------------------------------------------------------
# Main fitting routine (result is cached after first call)
# ---------------------------------------------------------------------------

_cached_params: Dict[int, Dict] | None = None


def get_gsmf_params() -> Dict[int, Dict]:
    """
    Return the fitted single-Schechter GSMF parameters for z = 6-12.

    The fit is computed once on the first call and cached; subsequent
    calls return immediately without recomputing.

    Returns
    -------
    dict keyed by integer redshift, each entry containing:
        log10_phi_star : float  -- log10 normalisation [Mpc^-3 dex^-1]
        log10_M_char   : float  -- log10 characteristic mass [M_sun]
        alpha          : float  -- low-mass slope
        shape_locked   : bool   -- True if shape was fixed to z = 10 values
    """
    global _cached_params
    if _cached_params is not None:
        return _cached_params

    params: Dict[int, Dict] = {}

    # Step 1: fit all redshifts with all three Schechter parameters free.
    for z, d in _FIRE2_DATA.items():
        popt, _ = _fit_schechter_free(d["log10_M"], d["log10_phi"])
        params[z] = {
            "log10_phi_star": float(popt[0]),
            "log10_M_char":   float(popt[1]),
            "alpha":          float(popt[2]),
            "shape_locked":   False,
        }

    # Step 2: refit the shape-locked redshifts using the z = 10 shape.
    ref = params[_SHAPE_LOCK_REF_Z]
    for z in _SHAPE_LOCK_ZS:
        d = _FIRE2_DATA[z]
        log10_phi_star, _ = _fit_schechter_phi_only(
            d["log10_M"], d["log10_phi"],
            fixed_log10_M_char=ref["log10_M_char"],
            fixed_alpha=ref["alpha"],
        )
        params[z] = {
            "log10_phi_star": log10_phi_star,
            "log10_M_char":   ref["log10_M_char"],
            "alpha":          ref["alpha"],
            "shape_locked":   True,
        }

    _cached_params = params
    return _cached_params


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_gsmf_params() -> None:
    """Print the fitted Schechter parameters for all redshifts."""
    params = get_gsmf_params()
    for z in sorted(params):
        p    = params[z]
        flag = "  [shape locked to z=10]" if p["shape_locked"] else ""
        print(f"z = {z}{flag}")
        print(f"  log10(phi_star) = {p['log10_phi_star']:.3f}"
              f"   (phi_star = {10**p['log10_phi_star']:.3e} Mpc^-3 dex^-1)")
        print(f"  log10(M_char)   = {p['log10_M_char']:.3f}"
              f"   (M_char   = {10**p['log10_M_char']:.3e} M_sun)")
        print(f"  alpha           = {p['alpha']:.3f}")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_COLORS  = {6: "k",  7: "C0", 8: "C1", 9: "C2", 10: "C3", 11: "C4", 12: "C5"}
_MARKERS = {6: "o",  7: "D",  8: "s",  9: "^",  10: "v",  11: "P",  12: "X"}


def plot_gsmf_fits() -> None:
    """Plot the Schechter fits against the FIRE-2 data points."""
    params = get_gsmf_params()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for z in sorted(_FIRE2_DATA):
        d    = _FIRE2_DATA[z]
        p    = params[z]
        popt = (p["log10_phi_star"], p["log10_M_char"], p["alpha"])

        log10_M_plot  = np.linspace(float(d["log10_M"].min()) - 0.3,
                                    float(d["log10_M"].max()) + 0.3, 400)
        log10_phi_fit = log10_schechter(log10_M_plot, *popt)

        label = f"z = {z}" + (" (shape = z=10)" if p["shape_locked"] else "")
        color = _COLORS.get(z, "C7")

        ax.plot(log10_M_plot, log10_phi_fit, ls="--", lw=2, color=color,
                label=label, alpha=0.9)
        ax.scatter(d["log10_M"], d["log10_phi"], s=49, marker=_MARKERS.get(z, "o"),
                   facecolors="none", edgecolors=color, linewidths=2.0, zorder=3)

    ax.set_xlabel(r"$\log_{10}(M_\star\,/\,M_\odot)$")
    ax.set_ylabel(r"$\log_{10}\,\phi\;[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.set_title("FIRE-2 GSMF — single-Schechter fits (z = 6-12)")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(title="Redshift")
    fig.tight_layout()

    _save_figure(fig, "gsmf_schechter_fits.png")


def plot_gsmf_interpolated() -> None:
    """
    Plot fitted and PCHIP-interpolated GSMF curves side by side.
    Requires schechter_log10phi_perdex_at_z from physics_relations.py.
    """
    try:
        from physics_relations import schechter_log10phi_perdex_at_z
    except ImportError as err:
        print(f"[Warning] Could not import from physics_relations.py -- skipping interpolation plot.\n  {err}")
        return

    # Mix of fitted (integer) and interpolated (non-integer) redshifts.
    z_show       = [6.5, 7.0, 8.5, 10.0, 10.5, 11.0, 11.5, 12.0]
    fitted_zs    = set(_FIRE2_DATA.keys())
    log10_M_grid = np.linspace(4.0, 10.0, 500)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    for z in z_show:
        log10_phi = schechter_log10phi_perdex_at_z(log10_M_grid, float(z), method="pchip")
        is_fitted = (abs(z - round(z)) < 1e-9) and (int(round(z)) in fitted_zs)

        if is_fitted:
            ax.plot(log10_M_grid, log10_phi, lw=2.2,
                    label=rf"$z = {z:g}$ (fit)")
        else:
            ax.plot(log10_M_grid, log10_phi, lw=2.2, ls="--", alpha=0.9,
                    label=rf"$z = {z:g}$ (interpolated)")

    ax.set_xlabel(r"$\log_{10}(M_\star\,/\,M_\odot)$")
    ax.set_ylabel(r"$\log_{10}\,\phi\;[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.set_title("FIRE-2 GSMF — fits and PCHIP interpolations (z = 6-12)")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(title="Redshift", ncol=2)
    ax.set_ylim(-6, 1.5)
    fig.tight_layout()

    _save_figure(fig, "gsmf_schechter_fits_interpolated.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_gsmf_params()
    plot_gsmf_fits()
    plot_gsmf_interpolated()
