#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig05_cobweb_evolution.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Aug 03, 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Cobweb evolution along the slice A=-1
#
#  Generates cobweb diagrams for the same four dynamical regimes used in the critical-orbit time-series figure.
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
Cobweb evolution for the normalized Gaussian map
=================================================

Generates a four-panel publication-quality cobweb figure for

    T(y) = A + B exp(-y^2)

along the slice A = -1.

Programmer:
    Prof. Americo Cunha Jr (LNCC & UERJ)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig05.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig05.pdf"
DPI = 360

A = -1.0
X_MIN, X_MAX = -1.15, 1.35
Y_MIN, Y_MAX = -1.15, 1.35

CASES = [
    (0.90, "(a) Attracting fixed point", "period 1", 28, "#0072B2"),
    (1.75, "(b) Attracting period-two orbit", "period 2", 34, "#009E73"),
    (2.05, "(c) Attracting period-four orbit", "period 4", 42, "#E69F00"),
    (2.25, "(d) Irregular non-periodic orbit", "expanding regime", 58, "#CC79A7"),
]

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 18,
})

def T(y, A, B):
    return A + B*np.exp(-y*y)

def dT(y, B):
    return -2.0*B*y*np.exp(-y*y)

def orbit(A, B, y0, n_iter):
    values = np.empty(n_iter + 1)
    values[0] = y0
    for n in range(n_iter):
        values[n+1] = T(values[n], A, B)
    return values

def fixed_points(A, B):
    grid = np.linspace(X_MIN, X_MAX, 25000)
    f = T(grid, A, B) - grid
    roots = []
    for i in range(len(grid)-1):
        if f[i] == 0:
            roots.append(grid[i])
        elif f[i]*f[i+1] < 0:
            left, right = grid[i], grid[i+1]
            fl = f[i]
            for _ in range(60):
                mid = 0.5*(left+right)
                fm = T(mid, A, B)-mid
                if fl*fm <= 0:
                    right = mid
                else:
                    left = mid
                    fl = fm
            roots.append(0.5*(left+right))
    out = []
    for r in roots:
        if not out or abs(r-out[-1]) > 1e-8:
            out.append(r)
    return np.asarray(out)

def draw_panel(ax, B, title, regime, n_iter, cobweb_color):
    x = np.linspace(X_MIN, X_MAX, 3000)
    y = T(x, A, B)
    orb = orbit(A, B, 0.0, n_iter)

    ax.axvspan(A, A+B, color="#DCE6F1", alpha=0.42, zorder=0)
    ax.plot(x, y, color="#0072B2", linewidth=2.5, zorder=3)
    ax.plot(x, x, color="#4D4D4D", linestyle="--", linewidth=1.6, zorder=2)

    for n in range(n_iter):
        alpha = 0.95 if n < 8 else 0.50
        lw = 1.35 if n < 8 else 1.0
        y0, y1 = orb[n], orb[n+1]
        ax.plot([y0, y0], [y0, y1], color=cobweb_color,
                linewidth=lw, alpha=alpha, zorder=4)
        ax.plot([y0, y1], [y1, y1], color=cobweb_color,
                linewidth=lw, alpha=alpha, zorder=4)

    ax.scatter([0], [0], s=55, facecolor="black",
               edgecolor="white", linewidth=0.8, zorder=7)
    ax.annotate(r"$y_0=0$", xy=(0, 0), xytext=(-38, 12),
                textcoords="offset points", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.14",
                          facecolor="white", edgecolor="0.55", alpha=0.92))

    for r in fixed_points(A, B):
        stable = abs(dT(r, B)) < 1
        ax.scatter([r], [r], s=58,
                   facecolor="#009E73" if stable else "white",
                   edgecolor="#009E73" if stable else "#D55E00",
                   linewidth=1.7, zorder=8)

    ax.scatter([0], [A+B], s=54, facecolor="#E69F00",
               edgecolor="white", linewidth=0.8, zorder=8)

    ax.text(0.025, 0.965,
            regime + "\n" + rf"$A={A:.2f},\ B={B:.2f}$" +
            "\n" + rf"$N={n_iter}$ iterates",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.2,
            bbox=dict(boxstyle="round,pad=0.28",
                      facecolor="white", edgecolor=cobweb_color,
                      linewidth=1.0, alpha=0.93), zorder=10)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$y_n$")
    ax.set_ylabel(r"$y_{n+1}$")
    ax.set_title(title, pad=9)
    ax.grid(True, alpha=0.18)

def main():
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.3),
                             constrained_layout=True)

    for ax, case in zip(axes.ravel(), CASES):
        draw_panel(ax, *case)

    handles = [
        Line2D([0], [0], color="#0072B2", linewidth=2.5, label=r"$T(y)$"),
        Line2D([0], [0], color="#4D4D4D", linestyle="--",
               linewidth=1.6, label=r"$y_{n+1}=y_n$"),
        Line2D([0], [0], color="#7A5195", linewidth=1.4,
               label="cobweb trajectory"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor="#009E73", markeredgecolor="#009E73",
               markersize=7, label="stable fixed point"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor="white", markeredgecolor="#D55E00",
               markeredgewidth=1.5, markersize=7,
               label="unstable fixed point"),
    ]

    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=False, columnspacing=1.5, handlelength=2.6)

    fig.suptitle(
        r"Cobweb evolution of the normalized Gaussian map "
        r"$T(y)=A+B e^{-y^2}$ along the slice $A=-1$"
    )

    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
