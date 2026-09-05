#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig04_critical_orbit_evolution.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Aug 03, 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Evolution of the critical orbit
#
#  Generates the four time-series panels along A=-1 showing a fixed point, period-two and period-four attracting cycles, and a numerically irregular expanding critical orbit.
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
Critical-orbit evolution for the normalized Gaussian map
========================================================

This script generates a four-panel publication-quality figure for

    T(y) = A + B exp(-y^2),

along the slice A = -1. The panels show an attracting fixed point,
attracting period-two and period-four cycles, and an irregular expanding
critical orbit.

Programmer:
    Prof. Americo Cunha Jr (LNCC & UERJ)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig04.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig04.pdf"
DPI = 360

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 18,
})

def T(y, A, B):
    return A + B*np.exp(-y**2)

def dT(y, B):
    return -2.0*B*y*np.exp(-y**2)

def critical_orbit(A, B, n_iter):
    orbit = np.empty(n_iter + 1)
    orbit[0] = 0.0
    for n in range(n_iter):
        orbit[n+1] = T(orbit[n], A, B)
    return orbit

def orbit_averaged_expansion_rate(A, B, n_transient=2500, n_sample=4000):
    y = 0.0
    for _ in range(n_transient):
        y = T(y, A, B)
    total = 0.0
    tiny = np.finfo(float).tiny
    for _ in range(n_sample):
        total += np.log(max(abs(dT(y, B)), tiny))
        y = T(y, A, B)
    return total/n_sample

def asymptotic_levels(A, B, n_transient=3000, n_keep=64, decimals=10):
    y = 0.0
    for _ in range(n_transient):
        y = T(y, A, B)
    tail = np.empty(n_keep)
    for k in range(n_keep):
        y = T(y, A, B)
        tail[k] = y
    return np.unique(np.round(tail, decimals=decimals))

def main():
    A = -1.0
    cases = [
        dict(B=0.90, title="(a) Attracting fixed point",
             regime="period 1", n_iter=60, mark_levels=True, color="#2166ac"),
        dict(B=1.75, title="(b) Attracting period-two orbit",
             regime="period 2", n_iter=70, mark_levels=True, color="#1b9e77"),
        dict(B=2.05, title="(c) Attracting period-four orbit",
             regime="period 4", n_iter=90, mark_levels=True, color="#d95f02"),
        dict(B=2.25, title="(d) Irregular expanding critical orbit",
             regime="non-periodic expanding regime", n_iter=180,
             mark_levels=False, color="#7570b3"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.8), constrained_layout=True)

    for ax, case in zip(axes.ravel(), cases):
        B = case["B"]
        color = case["color"]
        orbit = critical_orbit(A, B, case["n_iter"])
        n = np.arange(orbit.size)

        transient_cut = min(20, case["n_iter"]//3)
        ax.axvspan(0, transient_cut, color="#d9d9d9", alpha=0.42, zorder=0)
        ax.axvline(transient_cut, color="0.35", linestyle=":", linewidth=1.3)

        ax.plot(n, orbit, color=color, linewidth=1.55, alpha=0.95, zorder=3)
        ax.scatter(n, orbit, s=19, color=color, edgecolors="white",
                   linewidths=0.35, zorder=4)
        ax.scatter([0], [0], s=68, facecolors="black", edgecolors="white",
                   linewidths=0.8, zorder=6)

        if case["mark_levels"]:
            levels = asymptotic_levels(A, B)
            for level in levels:
                ax.axhline(level, color=color, linestyle="--",
                           linewidth=1.05, alpha=0.62)
            for j, level in enumerate(levels[:4]):
                ax.text(
                    0.985, level, rf"$y^*_{{{j+1}}}={level:.3f}$",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=9.5, color="0.20",
                    bbox=dict(boxstyle="round,pad=0.12",
                              facecolor="white", edgecolor="none", alpha=0.84)
                )

        rate = orbit_averaged_expansion_rate(A, B)
        info = (
            case["regime"] + "\n"
            + rf"$A={A:.2f},\ B={B:.2f}$" + "\n"
            + rf"$\lambda_N={rate:.3f}$"
        )
        ax.text(
            0.025, 0.965, info,
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=color, linewidth=1.0, alpha=0.92),
            zorder=8
        )

        ax.text(0.02, 0.04, "transient", transform=ax.transAxes,
                fontsize=9.5, color="0.35", ha="left", va="bottom")

        ax.set_xlabel(r"iteration $n$")
        ax.set_ylabel(r"$y_n=T^n(0)$")
        ax.set_title(case["title"], pad=9)
        ax.grid(True, alpha=0.20)

        ymin = min(np.min(orbit), -0.05)
        ymax = max(np.max(orbit), 0.05)
        span = ymax - ymin
        ax.set_ylim(ymin - 0.10*span, ymax + 0.10*span)

    fig.suptitle(
        r"Evolution of the critical orbit $y_n=T^n(0)$ along the slice $A=-1$"
    )
    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
