#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the finalized figure-generation scripts for the GaussMap paper."""

from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"

SCRIPTS = {
    1: "Fig01_representative_geometries.py",
    2: "Fig02_quadratic_critical_point_geometry.py",
    3: "Fig03_parameter_space_geometry.py",
    4: "Fig04_critical_orbit_evolution.py",
    5: "Fig05_cobweb_evolution.py",
    6: "Fig06_bifurcation_diagrams.py",
    7: "Fig07_orbit_averaged_expansion_map.py",
    8: "Fig08_inverse_branch_geometry.py",
    9: "Fig09_empirical_density_postcritical_peaks.py",
    10: "Fig10_perron_frobenius_evolution.py",
    11: "Fig11_singularity_propagation.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce manuscript figures for The Generalized Gaussian Iterated Map."
    )
    parser.add_argument(
        "--figures", type=int, nargs="+", choices=range(1, 12),
        help="Figure numbers to generate. Default: all figures 1--11."
    )
    args = parser.parse_args()
    selected = args.figures if args.figures else list(SCRIPTS)

    for number in selected:
        script = FIGDIR / SCRIPTS[number]
        print(f"[GaussMap] Generating Figure {number}: {script.name}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

    print("[GaussMap] Finished. Outputs are in:", ROOT / "output")


if __name__ == "__main__":
    main()
