#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
#  Fig09_empirical_density_postcritical_peaks.py
# -----------------------------------------------------------------
#  Programmer: Americo Cunha Jr
#  Affiliations: Laboratorio Nacional de Computacao Cientifica (LNCC)
#                Universidade do Estado do Rio de Janeiro (UERJ)
#
#  Originally programmed in: Jul--Aug 2026
#           Last updated in: Sep 04, 2026
# -----------------------------------------------------------------
#  Long-orbit empirical density and post-critical peak matching
#
#  Computes a two-million-iterate empirical density, detects prominent peaks, and matches them with nearby post-critical images.
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
Histogram estimate of the invariant density and identified singular peaks
=========================================================================

This script generates a publication-quality figure for the normalized
Gaussian map

    T(y) = A + B exp(-y^2),

using only a long-orbit histogram estimate of the invariant density.

The script:
    1. computes a long critical orbit after a transient;
    2. estimates the invariant density with a moderate number of bins;
    3. detects statistically prominent histogram peaks;
    4. compares each detected peak with higher-order post-critical images
       c_k = T^k(0);
    5. labels peaks that can be associated with a nearby c_k.

No Ulam approximation is used.

Programmer:
    Prof. Americo Cunha Jr (LNCC & UERJ)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.signal import find_peaks


# ================================================================
# OUTPUT AND USER CONTROLS
# ================================================================

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "Fig09.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "Fig09.pdf"
DPI = 360

A = -1.0
B = 2.25

N_TRANSIENT = 50_000
N_ORBIT = 2_000_000

# Moderate resolution to reduce histogram variance while preserving peaks.
N_BINS = 360

# Post-critical orbit used for peak identification.
N_CRITICAL_IMAGES = 150

# Peak-detection controls.
PEAK_PROMINENCE_FRACTION = 0.055
PEAK_MINIMUM_DISTANCE_BINS = 7
PEAK_MINIMUM_HEIGHT_FRACTION = 0.10

# A detected peak is associated with c_k only if the distance is smaller
# than this multiple of the histogram bin width.
MATCH_TOLERANCE_IN_BINS = 2.5

# Maximum number of labelled peaks to keep the figure readable.
MAX_LABELLED_PEAKS = 14

# Plot-window controls. The left limit is inferred from the first populated
# histogram bin, while the right limit is extended slightly beyond A+B so
# that the boundary peak and its annotation remain fully visible.
LEFT_SUPPORT_MARGIN_BINS = 4.0
RIGHT_SUPPORT_MARGIN = 0.045


# ================================================================
# TYPOGRAPHY
# ================================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10.5,
    "figure.titlesize": 18,
})


# ================================================================
# MAP AND ORBITS
# ================================================================

def T(y, A, B):
    """Normalized Gaussian map."""
    return A + B*np.exp(-y*y)


def critical_orbit_images(A, B, n_images):
    """Return c_k=T^k(0), k=1,...,n_images."""
    values = np.empty(n_images)
    y = 0.0

    for k in range(n_images):
        y = T(y, A, B)
        values[k] = y

    return values


def orbit_histogram(A, B, edges):
    """Estimate the physical invariant density from one long orbit."""
    y = 0.0

    for _ in range(N_TRANSIENT):
        y = T(y, A, B)

    counts = np.zeros(len(edges)-1, dtype=np.int64)

    chunk = 200_000
    remaining = N_ORBIT

    while remaining > 0:
        m = min(chunk, remaining)
        values = np.empty(m)

        for j in range(m):
            y = T(y, A, B)
            values[j] = y

        counts += np.histogram(values, bins=edges)[0]
        remaining -= m

    widths = np.diff(edges)
    density = counts/(counts.sum()*widths)

    return density


# ================================================================
# PEAK IDENTIFICATION
# ================================================================

def detect_prominent_peaks(density):
    """
    Detect prominent peaks in the histogram density.

    The thresholds are defined relative to the maximum density so that the
    procedure remains scale-independent.
    """
    maximum = np.max(density)

    peak_indices, properties = find_peaks(
        density,
        prominence=PEAK_PROMINENCE_FRACTION*maximum,
        height=PEAK_MINIMUM_HEIGHT_FRACTION*maximum,
        distance=PEAK_MINIMUM_DISTANCE_BINS
    )

    return peak_indices, properties


def match_peaks_to_critical_images(
    peak_positions,
    critical_images,
    tolerance,
):
    """
    Match each peak to the nearest post-critical image c_k.

    Returns
    -------
    matches : list of dict
        One dictionary per peak containing the nearest index, value,
        distance, and whether the tolerance criterion is satisfied.
    """
    matches = []

    for peak_position in peak_positions:
        distances = np.abs(critical_images-peak_position)
        nearest = int(np.argmin(distances))

        matches.append({
            "critical_index": nearest+1,
            "critical_value": critical_images[nearest],
            "distance": distances[nearest],
            "accepted": distances[nearest] <= tolerance,
        })

    return matches


# ================================================================
# MAIN FIGURE
# ================================================================

def main():
    interval_left = A
    interval_right = A+B

    edges = np.linspace(
        interval_left,
        interval_right,
        N_BINS+1
    )
    centers = 0.5*(edges[:-1] + edges[1:])
    widths = np.diff(edges)
    bin_width = widths[0]

    density = orbit_histogram(A, B, edges)

    # Determine an empirical plotting interval from populated histogram bins.
    populated = np.flatnonzero(density > 0.0)

    if populated.size:
        empirical_left = edges[populated[0]]
    else:
        empirical_left = interval_left

    plot_left = max(
        interval_left,
        empirical_left - LEFT_SUPPORT_MARGIN_BINS*bin_width
    )
    plot_right = interval_right + RIGHT_SUPPORT_MARGIN

    critical_images = critical_orbit_images(
        A,
        B,
        N_CRITICAL_IMAGES
    )

    peak_indices, peak_properties = detect_prominent_peaks(density)
    peak_positions = centers[peak_indices]
    peak_heights = density[peak_indices]

    match_tolerance = MATCH_TOLERANCE_IN_BINS*bin_width
    matches = match_peaks_to_critical_images(
        peak_positions,
        critical_images,
        match_tolerance
    )

    # Rank peaks by prominence and retain the most informative labels.
    prominences = peak_properties["prominences"]
    ranking = np.argsort(prominences)[::-1]
    labelled_indices = set(
        ranking[:min(MAX_LABELLED_PEAKS, len(ranking))].tolist()
    )

    histogram_face = "#9ecae1"
    histogram_edge = "#3182bd"
    matched_color = "#1b7837"
    unmatched_color = "#b2182b"

    fig, ax = plt.subplots(
        figsize=(12.4, 7.7),
        constrained_layout=True
    )

    ax.bar(
        centers,
        density,
        width=0.92*widths,
        color=histogram_face,
        edgecolor=histogram_edge,
        linewidth=0.35,
        alpha=0.72,
        align="center",
        label=rf"long-orbit histogram ({N_BINS} bins)",
        zorder=1
    )

    # Mark the detected peaks.
    for j, (
        peak_index,
        peak_position,
        peak_height,
        match
    ) in enumerate(zip(
        peak_indices,
        peak_positions,
        peak_heights,
        matches
    )):
        accepted = match["accepted"]
        marker_color = matched_color if accepted else unmatched_color

        ax.scatter(
            [peak_position],
            [peak_height],
            s=48,
            facecolor="white",
            edgecolor=marker_color,
            linewidth=1.6,
            zorder=5
        )

        if accepted:
            vertical = ax.axvline(
                match["critical_value"],
                color=matched_color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.70,
                zorder=2
            )
            vertical.set_path_effects([
                pe.Stroke(linewidth=2.6, foreground="white"),
                pe.Normal()
            ])

        if j in labelled_indices:
            if accepted:
                label = (
                    rf"$c_{{{match['critical_index']}}}$"
                    + "\n"
                    + rf"$y\approx{peak_position:.3f}$"
                )
            else:
                label = (
                    "unmatched peak"
                    + "\n"
                    + rf"$y\approx{peak_position:.3f}$"
                )

            # Alternate label orientation to reduce overlap.
            horizontal_offset = 8 if j % 2 == 0 else -8
            alignment = "left" if horizontal_offset > 0 else "right"
            vertical_offset = 18 + 10*(j % 3)

            ax.annotate(
                label,
                xy=(peak_position, peak_height),
                xytext=(horizontal_offset, vertical_offset),
                textcoords="offset points",
                ha=alignment,
                va="bottom",
                fontsize=9.3,
                color=marker_color,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": marker_color,
                    "linewidth": 0.8,
                    "alpha": 0.94
                },
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 0.85,
                    "color": marker_color
                },
                zorder=7
            )

    ax.set_xlim(plot_left, plot_right)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"invariant density $\rho(y)$")
    ax.set_title(
        r"Histogram estimate and identified singular peaks "
        r"of the invariant density",
        pad=10
    )
    ax.axvline(
        interval_right,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        alpha=0.70,
        zorder=2
    )

    ax.text(
        interval_right,
        0.035,
        r"$A+B$",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=10.0,
        color="0.30",
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86
        }
    )

    ax.grid(True, alpha=0.18)

    legend_handles = [
        Line2D(
            [0], [0],
            color=histogram_edge,
            linewidth=7,
            alpha=0.45,
            label=rf"histogram estimate ({N_BINS} bins)"
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=matched_color,
            markeredgewidth=1.6,
            markersize=7,
            label=r"peak matched to a post-critical image $c_k$"
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=unmatched_color,
            markeredgewidth=1.6,
            markersize=7,
            label="prominent unmatched peak"
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.12),
        ncol=3,
        frameon=True,
        framealpha=0.98,
        borderpad=0.75,
        columnspacing=1.4,
        handlelength=2.6
    )

    ax.text(
        0.015,
        0.965,
        (
            rf"$A={A:.2f},\ B={B:.2f}$"
            + "\n"
            + rf"$N_{{\rm orbit}}={N_ORBIT:,}$"
            + "\n"
            + rf"display window $=[{plot_left:.3f},{plot_right:.3f}]$"
            + "\n"
            + rf"matching tolerance $={MATCH_TOLERANCE_IN_BINS:.1f}$ bins"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.3,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.45",
            "alpha": 0.94
        }
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=DPI,
        bbox_inches="tight"
    )
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")

    plt.close(fig)

    print("Detected peaks:")
    for peak_position, peak_height, match in zip(
        peak_positions,
        peak_heights,
        matches
    ):
        print({
            "peak_position": float(peak_position),
            "peak_height": float(peak_height),
            "nearest_c_index": int(match["critical_index"]),
            "nearest_c_value": float(match["critical_value"]),
            "distance": float(match["distance"]),
            "accepted": bool(match["accepted"]),
        })


if __name__ == "__main__":
    main()
