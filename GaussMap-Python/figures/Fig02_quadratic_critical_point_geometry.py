#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig02_quadratic_critical_point_geometry.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Aug 03, 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Quadratic geometry near the critical point
#
#  Compares the exact normalized Gaussian map with its quadratic Taylor model near the critical point and verifies the quartic leading error.
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
Quadratic geometry near the critical point of the Gaussian map
==============================================================

This script generates a publication-quality two-panel figure for the
normalized Gaussian map

    T(y) = A + B exp(-y^2),

and its quadratic approximation near the critical point y_cr = 0,

    T_quad(y) = A + B - B y^2.

Panel (a) compares the exact map and the quadratic approximation in a
neighborhood of the critical point.

Panel (b) shows the absolute approximation error and compares it with the
quartic asymptotic term (B/2) y^4, confirming the expansion

    T(y) = A + B - B y^2 + (B/2) y^4 + O(y^6).

Programmer:
    Prof. Americo Cunha Jr (LNCC & UERJ)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ================================================================
# OUTPUT AND USER CONTROLS
# ================================================================

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig02.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig02.pdf"
DPI = 360

# Representative parameters.
A = -1.0
B = 2.25

# Local observation window.
Y_MIN = -0.90
Y_MAX = 0.90
N_POINTS = 4000


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
# MAP AND LOCAL APPROXIMATIONS
# ================================================================

def T(y, A, B):
    """Normalized Gaussian map."""
    return A + B*np.exp(-y*y)


def T_quadratic(y, A, B):
    """Quadratic Taylor approximation at y=0."""
    return A + B - B*y*y


def quartic_error_model(y, B):
    """Leading-order error magnitude."""
    return 0.5*B*y**4


# ================================================================
# MAIN FIGURE
# ================================================================

def main():
    y = np.linspace(Y_MIN, Y_MAX, N_POINTS)

    exact = T(y, A, B)
    quadratic = T_quadratic(y, A, B)
    error = np.abs(exact-quadratic)
    quartic = quartic_error_model(y, B)

    exact_color = "#0072B2"
    quadratic_color = "#D55E00"
    quartic_color = "#009E73"
    critical_color = "#CC79A7"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.7),
        constrained_layout=True
    )

    # ------------------------------------------------------------
    # Panel (a): local geometry
    # ------------------------------------------------------------
    ax = axes[0]

    ax.plot(
        y,
        exact,
        color=exact_color,
        linewidth=2.7,
        label=r"exact map $T(y)$"
    )

    ax.plot(
        y,
        quadratic,
        color=quadratic_color,
        linestyle="--",
        linewidth=2.3,
        label=r"quadratic approximation $T_{\rm quad}(y)$"
    )

    critical_value = A+B

    ax.scatter(
        [0.0],
        [critical_value],
        s=76,
        facecolor=critical_color,
        edgecolor="white",
        linewidth=1.0,
        zorder=6
    )

    ax.annotate(
        r"$y_{\rm cr}=0,\quad T(0)=A+B$",
        xy=(0.0, critical_value),
        xytext=(26, -32),
        textcoords="offset points",
        fontsize=10.4,
        color="#6A3D73",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
            "color": "#6A3D73"
        },
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": critical_color,
            "alpha": 0.93
        }
    )

    ax.axvline(
        0.0,
        color="#4D4D4D",
        linestyle=":",
        linewidth=1.2,
        alpha=0.80
    )

    ax.set_xlim(Y_MIN, Y_MAX)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$T(y)$")
    ax.set_title(
        "(a) Exact map and quadratic local model",
        pad=9
    )
    ax.grid(True, alpha=0.17)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=True,
        framealpha=0.97
    )

    ax.text(
        0.025,
        0.965,
        rf"$A={A:.2f},\ B={B:.2f}$"
        + "\n"
        + r"$T''(0)=-2B$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.3,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": exact_color,
            "alpha": 0.93
        }
    )

    # ------------------------------------------------------------
    # Panel (b): quartic remainder
    # ------------------------------------------------------------
    ax = axes[1]

    positive = np.abs(y) > 1.0e-6

    ax.semilogy(
        np.abs(y[positive]),
        error[positive],
        color=quadratic_color,
        linewidth=2.4,
        label=r"$|T(y)-T_{\rm quad}(y)|$"
    )

    ax.semilogy(
        np.abs(y[positive]),
        quartic[positive],
        color=quartic_color,
        linestyle="--",
        linewidth=2.2,
        label=r"leading term $\frac{B}{2}|y|^4$"
    )

    ax.set_xlim(0.0, Y_MAX)
    ax.set_xlabel(r"$|y|$")
    ax.set_ylabel(r"absolute approximation error")
    ax.set_title(
        r"(b) Quartic scaling of the Taylor remainder",
        pad=9
    )
    ax.grid(True, which="both", alpha=0.17)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.97
    )

    ax.text(
        0.04,
        0.94,
        r"$T(y)=A+B-By^2+\frac{B}{2}y^4+\mathcal{O}(y^6)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "edgecolor": quartic_color,
            "alpha": 0.93
        }
    )

    fig.suptitle(
        r"Quadratic geometry near the critical point of "
        r"$T(y)=A+B e^{-y^2}$"
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
