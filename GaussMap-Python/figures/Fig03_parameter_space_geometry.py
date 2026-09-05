#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig03_parameter_space_geometry.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Jul 23, 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Analytical parameter-space bifurcation geometry
#
#  Plots the analytical saddle-node and period-doubling loci, the global contraction threshold, the invariant-interval wedge, the cusp point, and the explicit Collet--Eckmann curve.
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
Standalone analytical parameter-plane figure used as Figure 3.

The formulas are the exact closed-form loci derived in the paper:

    A_sn(y) = y + 1/(2y),     B_sn(y) = -exp(y^2)/(2y), y<0,
    A_pd(y) = y - 1/(2y),     B_pd(y) =  exp(y^2)/(2y), y>0.

The explicit Collet--Eckmann curve is parameterized by q in (0,q_*), where
q_* solves exp(q_*^2)=1+4q_*^2.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig03.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig03.pdf"
DPI = 360

A_MIN, A_MAX = -5.0, 2.0
B_MIN, B_MAX = 0.02, 8.0

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10.2,
    "figure.titlesize": 18,
})


def saddle_node_curve(y):
    return y + 1.0/(2.0*y), -np.exp(y*y)/(2.0*y)


def period_doubling_curve(y):
    return y - 1.0/(2.0*y), np.exp(y*y)/(2.0*y)


def collet_eckmann_curve(q):
    em = np.exp(-q*q)
    return -q*(1.0+em)/(1.0-em), 2.0*q/(1.0-em)


def main():
    fig, ax = plt.subplots(figsize=(9.0, 6.9), constrained_layout=True)

    B_fill = np.linspace(B_MIN, B_MAX, 1200)
    Bc = np.sqrt(np.e/2.0)

    # Region A<=0<=A+B, equivalently -B<=A<=0, where the critical point
    # belongs to the invariant interval I=[A,A+B].
    ax.fill_betweenx(B_fill, -B_fill, np.zeros_like(B_fill),
                     color="#d7e8f3", alpha=0.75,
                     label=r"$0\in I=[A,A+B]$")

    # Light shading below the rigorous global contraction threshold.
    ax.axhspan(B_MIN, min(Bc, B_MAX), color="#ececec", alpha=0.75)
    ax.axhline(Bc, color="#1b7837", ls="--", lw=1.8,
               label=r"$B=\sqrt{e/2}$")

    y_sn = np.linspace(-3.6, -0.10, 7000)
    A_sn, B_sn = saddle_node_curve(y_sn)
    m = ((A_sn>=A_MIN)&(A_sn<=A_MAX)&(B_sn>=B_MIN)&(B_sn<=B_MAX))
    ax.plot(A_sn[m], B_sn[m], color="#2166ac", lw=2.5, label="saddle-node")

    y_pd = np.linspace(0.06, 2.5, 7000)
    A_pd, B_pd = period_doubling_curve(y_pd)
    m = ((A_pd>=A_MIN)&(A_pd<=A_MAX)&(B_pd>=B_MIN)&(B_pd<=B_MAX))
    ax.plot(A_pd[m], B_pd[m], color="#b2182b", lw=2.5,
            label="fixed-point period doubling")

    q_star = 1.52861472656227
    q = np.linspace(0.05, 0.999*q_star, 2600)
    A_ce, B_ce = collet_eckmann_curve(q)
    m = ((A_ce>=A_MIN)&(A_ce<=A_MAX)&(B_ce>=B_MIN)&(B_ce<=B_MAX))
    ax.plot(A_ce[m], B_ce[m], color="#762a83", ls="-.", lw=2.2,
            label="explicit Collet--Eckmann curve")

    # Cusp point of the fixed-point equation.
    A_tip = -np.sqrt(2.0)
    ax.scatter([A_tip], [Bc], s=60, color="black", zorder=7)
    annotation = ax.annotate(
        r"$(-\sqrt{2},\sqrt{e/2})$" + "\n" + rf"$\approx({A_tip:.4f},{Bc:.4f})$",
        xy=(A_tip, Bc), xytext=(-58, 66), textcoords="offset points",
        ha="center", va="bottom", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                  edgecolor="black", linewidth=1.2, alpha=0.98),
        arrowprops=dict(arrowstyle="-|>", linewidth=2.1, color="black",
                        mutation_scale=16, shrinkA=4, shrinkB=4,
                        connectionstyle="arc3,rad=-0.08"), zorder=12)
    annotation.arrow_patch.set_path_effects([
        pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()
    ])

    ax.set_xlim(A_MIN, A_MAX); ax.set_ylim(B_MIN, B_MAX)
    ax.set_xlabel(r"$A$"); ax.set_ylabel(r"$B$")
    ax.set_title(r"Parameter-space geometry of the normalized Gaussian family "
                 r"$T(y)=A+B e^{-y^2}$")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower left", frameon=True, framealpha=0.96)

    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
