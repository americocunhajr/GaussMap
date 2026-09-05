#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig07_orbit_averaged_expansion_map.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Jul--Aug 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Orbit-averaged expansion map in parameter space
#
#  Computes the finite-orbit average of log|T'| over the (A,B) plane and superimposes the analytical bifurcation and explicit Collet--Eckmann curves.
#
#  Mathematical model:
#
#      T(y) = A + B exp(-y^2),    B > 0.
#
#  The numerical and plotting parameters below are the finalized values
#  used to reproduce the corresponding figure in Ref. [1].  Each script
#  is self-contained and writes both PNG and PDF versions to ../output/.
#
#  Software requirements:
#      Python 3.10+; NumPy; Matplotlib
#      SciPy is additionally required by Figure 9.
#
#  Reference:
#
#  [1] A. Cunha Jr
#      The Generalized Gaussian Iterated Map
#      Manuscript prepared for submission to the SIAM Journal on
#      Applied Dynamical Systems (SIADS), 2026.
# -----------------------------------------------------------------

"""
Orbit-averaged expansion map for the normalized Gaussian iterated map
=====================================================================

This script generates a publication-quality parameter-space map for

    T(y) = A + B exp(-y^2),

showing the finite-orbit average

    lambda_N = (1/N) sum log |T'(y_n)|

along the critical orbit y_0 = 0.

The analytical saddle-node, period-doubling, contraction, and explicit
Collet--Eckmann curves are superimposed using a consistent color scheme.

Programmer:
    Prof. Americo Cunha Jr (LNCC & UERJ)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from pathlib import Path


# ================================================================
# OUTPUT AND USER CONTROLS
# ================================================================

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig07.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig07.pdf"
DPI = 360

A_MIN, A_MAX = -5.0, 2.0
B_MIN, B_MAX = 0.05, 8.0

NA, NB = 720, 640
N_TRANSIENT = 1800
N_SAMPLE = 1800

DISPLAY_MIN, DISPLAY_MAX = -2.0, 0.8


# ================================================================
# TYPOGRAPHY
# ================================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10.5,
    "figure.titlesize": 18,
})


# ================================================================
# MAP AND ANALYTICAL CURVES
# ================================================================

def T(y, A, B):
    """Normalized Gaussian map."""
    return A + B*np.exp(-y*y)


def dT(y, B):
    """Derivative of the normalized Gaussian map."""
    return -2.0*B*y*np.exp(-y*y)


def saddle_node_curve(y):
    """Analytical saddle-node curve."""
    A = y + 1.0/(2.0*y)
    B = -np.exp(y*y)/(2.0*y)
    return A, B


def period_doubling_curve(y):
    """Analytical fixed-point period-doubling curve."""
    A = y - 1.0/(2.0*y)
    B = np.exp(y*y)/(2.0*y)
    return A, B


def collet_eckmann_curve(q):
    """Explicit Collet--Eckmann parameter curve."""
    em = np.exp(-q*q)
    A = -q*(1.0 + em)/(1.0 - em)
    B = 2.0*q/(1.0 - em)
    return A, B


# ================================================================
# ORBIT-AVERAGED EXPANSION MAP
# ================================================================

def compute_expansion_map(A_values, B_values):
    """
    Compute the finite-orbit average

        lambda_N = (1/N) sum log |T'(y_n)|

    along the critical orbit y_0 = 0.
    """
    AA, BB = np.meshgrid(A_values, B_values)
    y = np.zeros_like(AA)

    for _ in range(N_TRANSIENT):
        y = T(y, AA, BB)

    expansion = np.zeros_like(AA)
    tiny = np.finfo(float).tiny

    for _ in range(N_SAMPLE):
        derivative = np.maximum(np.abs(dT(y, BB)), tiny)
        expansion += np.log(derivative)
        y = T(y, AA, BB)

    return expansion/N_SAMPLE


def outlined_curve(
    ax,
    x,
    y,
    *,
    color,
    linewidth,
    linestyle="-",
    label=None,
    zorder=10,
):
    """
    Plot a colored analytical curve with a thin white halo.
    """
    line, = ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
    )

    line.set_path_effects([
        pe.Stroke(linewidth=linewidth + 2.6, foreground="white"),
        pe.Normal(),
    ])

    return line


# ================================================================
# MAIN FIGURE
# ================================================================

def main():
    A_values = np.linspace(A_MIN, A_MAX, NA)
    B_values = np.linspace(B_MIN, B_MAX, NB)

    expansion = compute_expansion_map(A_values, B_values)
    expansion_display = np.clip(expansion, DISPLAY_MIN, DISPLAY_MAX)

    # Soft, publication-oriented diverging map:
    # deep blue -> pale neutral -> warm red.
    cmap = LinearSegmentedColormap.from_list(
        "expansion_map",
        [
            "#17365d",
            "#4f81bd",
            "#c6d9f1",
            "#f7f7f7",
            "#f4c7a1",
            "#d6604d",
            "#8b1a1a",
        ],
        N=256,
    )

    norm = TwoSlopeNorm(
        vmin=DISPLAY_MIN,
        vcenter=0.0,
        vmax=DISPLAY_MAX,
    )

    fig, ax = plt.subplots(
        figsize=(11.4, 8.6),
        constrained_layout=True,
    )

    im = ax.imshow(
        expansion_display,
        origin="lower",
        extent=[A_MIN, A_MAX, B_MIN, B_MAX],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )

    # Zero-expansion contour
    zero_contour = ax.contour(
        A_values,
        B_values,
        expansion,
        levels=[0.0],
        colors="black",
        linewidths=1.1,
        linestyles="-",
        zorder=8,
    )

    # Apply a white halo when supported by the installed Matplotlib version.
    try:
        zero_contour.set_path_effects([
            pe.Stroke(linewidth=2.8, foreground="white"),
            pe.Normal(),
        ])
    except AttributeError:
        pass

    # Saddle-node curve
    y_sn = np.linspace(-3.6, -0.10, 7000)
    A_sn, B_sn = saddle_node_curve(y_sn)
    mask_sn = (
        (A_sn >= A_MIN) & (A_sn <= A_MAX)
        & (B_sn >= B_MIN) & (B_sn <= B_MAX)
    )

    outlined_curve(
        ax,
        A_sn[mask_sn],
        B_sn[mask_sn],
        color="#2166ac",
        linewidth=2.3,
        label="saddle-node",
    )

    # Period-doubling curve
    y_pd = np.linspace(0.06, 2.5, 7000)
    A_pd, B_pd = period_doubling_curve(y_pd)
    mask_pd = (
        (A_pd >= A_MIN) & (A_pd <= A_MAX)
        & (B_pd >= B_MIN) & (B_pd <= B_MAX)
    )

    outlined_curve(
        ax,
        A_pd[mask_pd],
        B_pd[mask_pd],
        color="#b2182b",
        linewidth=2.3,
        label="fixed-point period doubling",
    )

    # Global contraction threshold
    Bc = np.sqrt(np.e/2.0)
    contraction = ax.axhline(
        Bc,
        color="#1b7837",
        linestyle="--",
        linewidth=2.0,
        label=r"$B=\sqrt{e/2}$",
        zorder=10,
    )
    contraction.set_path_effects([
        pe.Stroke(linewidth=4.4, foreground="white"),
        pe.Normal(),
    ])

    # Explicit Collet--Eckmann curve
    q_star = 1.52861472656227
    q = np.linspace(0.05, 0.999*q_star, 2600)
    A_ce, B_ce = collet_eckmann_curve(q)
    mask_ce = (
        (A_ce >= A_MIN) & (A_ce <= A_MAX)
        & (B_ce >= B_MIN) & (B_ce <= B_MAX)
    )

    outlined_curve(
        ax,
        A_ce[mask_ce],
        B_ce[mask_ce],
        color="#762a83",
        linewidth=2.2,
        linestyle="-.",
        label="explicit Collet--Eckmann curve",
        zorder=11,
    )

    # Wedge boundaries: -B <= A <= 0
    B_wedge = np.linspace(B_MIN, B_MAX, 900)

    left_boundary, = ax.plot(
        -B_wedge,
        B_wedge,
        color="0.20",
        linestyle=":",
        linewidth=1.4,
        zorder=9,
    )
    left_boundary.set_path_effects([
        pe.Stroke(linewidth=3.4, foreground="white"),
        pe.Normal(),
    ])

    right_boundary = ax.axvline(
        0.0,
        color="0.20",
        linestyle=":",
        linewidth=1.4,
        zorder=9,
    )
    right_boundary.set_path_effects([
        pe.Stroke(linewidth=3.4, foreground="white"),
        pe.Normal(),
    ])

    ax.set_xlim(A_MIN, A_MAX)
    ax.set_ylim(B_MIN, B_MAX)
    ax.set_xlabel(r"$A$")
    ax.set_ylabel(r"$B$")
    ax.set_title(
        r"Orbit-averaged expansion rate along the critical orbit $y_0=0$",
        pad=10,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
        fraction=0.05,
    )
    cbar.set_label(
        r"orbit-averaged expansion rate $\lambda_N$",
        fontsize=13.5,
    )
    cbar.ax.tick_params(labelsize=11.5)
    cbar.ax.axhline(
        0.0,
        color="black",
        linewidth=1.2,
    )

    legend_handles = [
        Line2D([0], [0], color="#2166ac", linewidth=2.3,
               label="saddle-node"),
        Line2D([0], [0], color="#b2182b", linewidth=2.3,
               label="fixed-point period doubling"),
        Line2D([0], [0], color="#762a83", linestyle="-.",
               linewidth=2.2, label="explicit Collet--Eckmann curve"),
        Line2D([0], [0], color="#1b7837", linestyle="--",
               linewidth=2.0, label=r"$B=\sqrt{e/2}$"),
        Line2D([0], [0], color="black", linewidth=1.2,
               label=r"$\lambda_N=0$ contour"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.10),
        ncol=2,
        frameon=True,
        framealpha=0.98,
        borderpad=0.8,
        columnspacing=1.5,
        handlelength=2.6,
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=DPI,
        bbox_inches="tight",
    )
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
