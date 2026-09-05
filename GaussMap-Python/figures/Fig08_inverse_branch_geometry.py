#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig08_inverse_branch_geometry.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Aug 03, 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Geometry of the explicit inverse branches
#
#  Visualizes the two real inverse branches of the normalized Gaussian map and their coalescence at the critical value.
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
Geometry of the inverse branches of the normalized Gaussian map
===============================================================

This script generates a publication-quality two-panel figure for

    T(y) = A + B exp(-y^2),

illustrating the two inverse branches

    y_-(z) = -sqrt[-log((z-A)/B)],
    y_+(z) =  sqrt[-log((z-A)/B)],

defined for A < z < A+B.

Panel (a) shows how horizontal levels intersect the two monotone branches of
the Gaussian map.

Panel (b) shows the inverse branches explicitly as functions of z and their
coalescence at the critical value z=A+B.

Programmer:
    Prof. Americo Cunha Jr (LNCC & UERJ)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# ================================================================
# OUTPUT AND USER CONTROLS
# ================================================================

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig08.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig08.pdf"
DPI = 360

A = -1.0
B = 2.25

Y_MIN = -2.25
Y_MAX = 2.25
N_POINTS = 5000

# Representative horizontal levels used in panel (a).
Z_LEVELS = (0.05, 0.72)


# ================================================================
# TYPOGRAPHY
# ================================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 18,
})


# ================================================================
# MAP AND INVERSE BRANCHES
# ================================================================

def T(y, A, B):
    """Normalized Gaussian map."""
    return A + B*np.exp(-y*y)


def inverse_plus(z, A, B):
    """Positive inverse branch."""
    return np.sqrt(-np.log((z-A)/B))


def inverse_minus(z, A, B):
    """Negative inverse branch."""
    return -np.sqrt(-np.log((z-A)/B))


# ================================================================
# MAIN FIGURE
# ================================================================

def main():
    critical_value = A+B

    map_color = "#0072B2"
    negative_branch_color = "#009E73"
    positive_branch_color = "#D55E00"
    level_colors = ("#CC79A7", "#E69F00")
    neutral_color = "#4D4D4D"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 5.9),
        constrained_layout=True
    )

    # ------------------------------------------------------------
    # Panel (a): preimages on the graph of T
    # ------------------------------------------------------------
    ax = axes[0]

    y = np.linspace(Y_MIN, Y_MAX, N_POINTS)
    z = T(y, A, B)

    negative = y <= 0.0
    positive = y >= 0.0

    ax.plot(
        y[negative],
        z[negative],
        color=negative_branch_color,
        linewidth=2.6,
        label=r"left monotone branch"
    )

    ax.plot(
        y[positive],
        z[positive],
        color=positive_branch_color,
        linewidth=2.6,
        label=r"right monotone branch"
    )

    ax.axvline(
        0.0,
        color=neutral_color,
        linestyle=":",
        linewidth=1.2,
        alpha=0.80
    )

    ax.scatter(
        [0.0],
        [critical_value],
        s=74,
        facecolor=map_color,
        edgecolor="white",
        linewidth=1.0,
        zorder=7
    )

    ax.annotate(
        r"$y_{\rm cr}=0$",
        xy=(0.0, critical_value),
        xytext=(28, -24),
        textcoords="offset points",
        fontsize=10.3,
        color=map_color,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.95,
            "color": map_color
        },
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": map_color,
            "alpha": 0.93
        }
    )

    for index, (z_level, color) in enumerate(
        zip(Z_LEVELS, level_colors),
        start=1
    ):
        y_minus = inverse_minus(z_level, A, B)
        y_plus = inverse_plus(z_level, A, B)

        ax.axhline(
            z_level,
            color=color,
            linestyle="--",
            linewidth=1.45,
            alpha=0.90
        )

        ax.scatter(
            [y_minus, y_plus],
            [z_level, z_level],
            s=60,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=7
        )

        offset_y = 12 if index == 1 else -22

        ax.annotate(
            rf"$y_-(z_{index})$",
            xy=(y_minus, z_level),
            xytext=(-4, offset_y),
            textcoords="offset points",
            ha="right",
            va="bottom" if offset_y > 0 else "top",
            fontsize=9.8,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86
            }
        )

        ax.annotate(
            rf"$y_+(z_{index})$",
            xy=(y_plus, z_level),
            xytext=(4, offset_y),
            textcoords="offset points",
            ha="left",
            va="bottom" if offset_y > 0 else "top",
            fontsize=9.8,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86
            }
        )

    ax.set_xlim(Y_MIN, Y_MAX)
    ax.set_ylim(A-0.08, critical_value+0.15)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z=T(y)$")
    ax.set_title(
        r"(a) Two preimages of an admissible level $z$",
        pad=9
    )
    ax.grid(True, alpha=0.17)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, -0.25),
        ncol=2,
        frameon=True,
        framealpha=0.97
    )

    # ------------------------------------------------------------
    # Panel (b): explicit inverse branches
    # ------------------------------------------------------------
    ax = axes[1]

    eps = 2.0e-4
    z_values = np.linspace(
        A+eps,
        critical_value-eps,
        N_POINTS
    )

    y_minus_values = inverse_minus(z_values, A, B)
    y_plus_values = inverse_plus(z_values, A, B)

    ax.axvspan(
        A,
        critical_value,
        color="#F2F2F2",
        alpha=0.72,
        zorder=0
    )

    ax.plot(
        z_values,
        y_minus_values,
        color=negative_branch_color,
        linewidth=2.6,
        label=r"$y_-(z)$"
    )

    ax.plot(
        z_values,
        y_plus_values,
        color=positive_branch_color,
        linewidth=2.6,
        label=r"$y_+(z)$"
    )

    ax.axhline(
        0.0,
        color=neutral_color,
        linestyle=":",
        linewidth=1.2,
        alpha=0.80
    )

    ax.axvline(
        critical_value,
        color=map_color,
        linestyle="--",
        linewidth=1.45,
        alpha=0.90
    )

    ax.scatter(
        [critical_value],
        [0.0],
        s=74,
        facecolor=map_color,
        edgecolor="white",
        linewidth=1.0,
        zorder=7
    )

    ax.annotate(
        r"$y_-(A+B)=y_+(A+B)=0$",
        xy=(critical_value, 0.0),
        xytext=(-170, 28),
        textcoords="offset points",
        fontsize=10.2,
        color=map_color,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.95,
            "color": map_color
        },
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": map_color,
            "alpha": 0.93
        }
    )

    for index, (z_level, color) in enumerate(
        zip(Z_LEVELS, level_colors),
        start=1
    ):
        ym = inverse_minus(z_level, A, B)
        yp = inverse_plus(z_level, A, B)

        ax.axvline(
            z_level,
            color=color,
            linestyle="--",
            linewidth=1.15,
            alpha=0.72
        )

        ax.scatter(
            [z_level, z_level],
            [ym, yp],
            s=54,
            facecolor="white",
            edgecolor=color,
            linewidth=1.7,
            zorder=7
        )

    ax.set_xlim(A-0.08, critical_value+0.18)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"inverse value $y_\pm(z)$")
    ax.set_title(
        r"(b) Explicit inverse branches",
        pad=9
    )
    ax.grid(True, alpha=0.17)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, -0.25),
        ncol=2,
        frameon=True,
        framealpha=0.97
    )

    ax.text(
        0.035,
        0.955,
        r"$y_\pm(z)=\pm\sqrt{-\ln\!\left(\frac{z-A}{B}\right)}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.6,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": neutral_color,
            "alpha": 0.93
        }
    )

    fig.suptitle(
        r"Geometry of the inverse branches of "
        r"$T(y)=A+B e^{-y^2}$"
        + rf" for $A={A:.2f}$ and $B={B:.2f}$"
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=DPI,
        bbox_inches="tight"
    )
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    main()
