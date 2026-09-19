#!/usr/bin/env python3
"""
Corrected tests for the universality and emergent-gravity correlations.

Supersedes two results previously reported in the README:

  "Ising and Heisenberg produce the same emergent geometry curve
   (Pearson r = 0.89)"
  "universal coupling r = 0.9987"  (effective gravitational constant)

Both were bare correlation coefficients with no confidence interval, no
null distribution and no leverage check. See curve_comparison.py for why
neither number means what it appears to mean.

Usage:  python src/analysis/12_universality_and_gravity_validation.py

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curve_comparison import (  # noqa: E402
    format_curve_report,
    jackknife_r,
    null_r_distribution,
    partial_correlation,
    pearson_ci,
    scaling_collapse,
    shot_noise_sigma,
)

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
RULE = "=" * 72

LAMBDAS = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
SHOTS = 8192
N_PAIRS = 4

# Per-basis sweeps used by the gravity analysis (10_gravity_analysis.py).
ISING_BASES = {
    "Z": [0.0061, 0.0126, 0.0371, 0.0960, 0.1426, 0.1714, 0.1394, 0.1117],
    "X": [0.0079, 0.0095, 0.0080, 0.0145, 0.0240, 0.0169, 0.0207, 0.0097],
    "Y": [0.0077, 0.0129, 0.0273, 0.0253, 0.0207, 0.0198, 0.0102, 0.0182],
}
XY_BASES = {
    "Z": [0.0064, 0.0129, 0.0164, 0.0081, 0.0335, 0.0362, 0.0205, 0.0081],
    "X": [0.0067, 0.0303, 0.0467, 0.0880, 0.0381, 0.0664, 0.0737, 0.0319],
    "Y": [0.0076, 0.0143, 0.0281, 0.0409, 0.0279, 0.0450, 0.0518, 0.0089],
}


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def load_universality() -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(ROOT, "results", "universality_test_results.json")
    with open(path) as fh:
        data = json.load(fh)
    ising = np.array([v["avg_C"] for v in data["ising_sweep"]["values"]])
    heis = np.array([v["avg_C"] for v in data["heisenberg_sweep"]["values"]])
    return ising, heis


def main() -> None:
    print(RULE)
    print("CORRECTED UNIVERSALITY AND EMERGENT-GRAVITY TESTS")
    print("Andrew Thurlow | 528 Labs")
    print(RULE)

    sigma = shot_noise_sigma(SHOTS, N_PAIRS)
    print(f"\nShot-noise sigma on each swept point: {sigma:.4f}"
          f"  ({SHOTS} shots, {N_PAIRS} pairs averaged)")

    ising, heis = load_universality()
    out: dict = {"shot_noise_sigma": sigma}

    # ------------------------------------------------------------------
    banner("PART 1 - UNIVERSALITY: IS r = 0.89 EVIDENCE OF ANYTHING?")
    # ------------------------------------------------------------------
    print("\nIsing and Heisenberg cross-field correlation sweeps:\n")
    print(f"  {'lambda':<10}{'Ising':<12}{'Heisenberg':<12}")
    print("  " + "-" * 32)
    for i, lam in enumerate(LAMBDAS):
        print(f"  {lam:<10}{ising[i]:<12.4f}{heis[i]:<12.4f}")

    r_res = pearson_ci(ising, heis, sigma)
    null = null_r_distribution(ising, LAMBDAS, sigma)
    collapse = scaling_collapse(LAMBDAS, ising, heis, sigma)

    print()
    print(format_curve_report(r_res, null, collapse,
                              "Ising vs Heisenberg:"))

    beats = float(np.mean(null["samples"] >= r_res["r"]))
    print(
        "\nBoth sweeps rise from zero, peak and fall. Any two curves of that"
        "\nshape correlate strongly, so r near 0.9 is the baseline expectation"
        "\nrather than a discovery. Unrelated unimodal curves with a randomly"
        f"\nplaced peak reach r >= {r_res['r']:.2f} about {100 * beats:.0f}% of the time."
    )

    i_peak = float(LAMBDAS[int(np.argmax(ising))])
    h_peak = float(LAMBDAS[int(np.argmax(heis))])
    print(
        f"\nThe curves also peak in different places: Ising at lambda = {i_peak},"
        f"\nHeisenberg at lambda = {h_peak}. The README noted this while still"
        "\ncalling the curves the same. They are not the same curve."
    )
    print(
        f"\nThe collapse test is the real check. chi2/dof ="
        f" {collapse['chi2_per_dof']:.1f} against a"
        "\nvalue near 1 for genuinely universal curves, with the worst point"
        f" {collapse['max_residual_sigma']:.1f} sigma\noff. Universality is not supported by this data."
    )
    out["universality"] = {
        "r": r_res["r"], "r_ci": r_res["ci"],
        "null_median_r": null["median"], "null_p95_r": null["p95"],
        "fraction_of_unrelated_curves_reaching_r": beats,
        "ising_peak_lambda": i_peak, "heisenberg_peak_lambda": h_peak,
        "chi2_per_dof": collapse["chi2_per_dof"],
        "best_scale": collapse["best_scale"],
        "max_residual_sigma": collapse["max_residual_sigma"],
        "collapses": collapse["collapses"],
    }

    # ------------------------------------------------------------------
    banner("PART 2 - EMERGENT GRAVITY: r = 0.9987 IS MOSTLY DEFINITIONAL")
    # ------------------------------------------------------------------
    lam_nz = LAMBDAS[1:]
    tr_i = np.array([sum(ISING_BASES[b][i] for b in "ZXY")
                     for i in range(1, len(LAMBDAS))])
    tr_x = np.array([sum(XY_BASES[b][i] for b in "ZXY")
                     for i in range(1, len(LAMBDAS))])
    g_i = 1.0 / (4 * lam_nz * tr_i)
    g_x = 1.0 / (4 * lam_nz * tr_x)
    inv_lam = 1.0 / lam_nz

    print("\nG_eff = 1 / (4 * lambda * Tr G), computed for both Hamiltonians:\n")
    print(f"  {'lambda':<10}{'Ising G':<14}{'XY G':<14}{'1/lambda':<12}")
    print("  " + "-" * 48)
    for i, lam in enumerate(lam_nz):
        print(f"  {lam:<10}{g_i[i]:<14.2f}{g_x[i]:<14.2f}{inv_lam[i]:<12.2f}")

    # The decisive check: replace both measured inputs with constants, so
    # nothing is measured at all, and see what r the definition alone yields.
    g_const_i = 1.0 / (4 * lam_nz * np.full_like(lam_nz, tr_i.mean()))
    g_const_x = 1.0 / (4 * lam_nz * np.full_like(lam_nz, tr_x.mean()))
    r_definitional = float(np.corrcoef(g_const_i, g_const_x)[0, 1])

    pc = partial_correlation(g_i, g_x, inv_lam)
    jk = jackknife_r(g_i, g_x)
    # Same comparison on a log scale, where the two-orders-of-magnitude
    # dynamic range no longer lets the smallest lambda dominate.
    r_log = float(np.corrcoef(np.log(g_i), np.log(g_x))[0, 1])
    pc_log = partial_correlation(np.log(g_i), np.log(g_x), np.log(inv_lam))

    print(f"\n  r with BOTH measured inputs replaced by constants"
          f" = {r_definitional:.4f}")
    print("  (nothing is measured in that line: it is 1/lambda against"
          " 1/lambda.\n   Any r at or below it carries no measured"
          " information at all.)\n")
    print(f"  raw r (as published)                 = {pc['raw_r']:.4f}")
    print(f"  r of Ising G with plain 1/lambda      = {pc['r_x_with_z']:.4f}")
    print(f"  r of XY G with plain 1/lambda         = {pc['r_y_with_z']:.4f}")
    print(f"  partial r, 1/lambda removed           = {pc['partial_r']:.4f}")
    print(f"  r on a log scale                      = {r_log:.4f}")
    print(f"  partial r on a log scale              = {pc_log['partial_r']:.4f}")
    print(
        "\nBoth series are defined as one over lambda times a measured"
        "\nquantity, so they are forced to share a 1/lambda factor. Replacing"
        "\nboth measurements with constants gives r = 1.0000 exactly: the"
        "\ndefinition alone produces a perfect correlation before any data is"
        "\ncollected. The published 0.9987 sits below that ceiling, so the"
        "\nmeasurement only degrades an agreement that was already guaranteed."
        "\nThe correlation is evidence about the formula, not about the physics."
    )

    print(f"\n  Leverage (drop one point and recompute):")
    print(f"  {'dropped lambda':<18}{'r':<10}")
    print("  " + "-" * 28)
    for d in jk["per_point"]:
        print(f"  {lam_nz[d['dropped_index']]:<18}{d['r']:<10.4f}")
    print(f"\n  full r {jk['full_r']:.4f}, falls to {jk['min_r']:.4f} when the"
          f" lambda = {lam_nz[jk['most_influential_index']]} point is dropped")

    # The measured quantity, without the imposed 1/lambda
    tr_r = pearson_ci(tr_i, tr_x, sigma * np.sqrt(3))
    tr_null = null_r_distribution(tr_i, lam_nz, sigma * np.sqrt(3))
    tr_collapse = scaling_collapse(lam_nz, tr_i, tr_x, sigma * np.sqrt(3))
    print("\n  Correlating the measured part alone, Tr G, with no 1/lambda:")
    print(format_curve_report(tr_r, tr_null, tr_collapse))
    print(
        "\n  That is the honest version of the comparison, and it does not"
        "\n  support a universal coupling."
    )

    out["gravity"] = {
        "raw_r": pc["raw_r"],
        "r_with_measurements_replaced_by_constants": r_definitional,
        "raw_r_log_scale": r_log,
        "partial_r_log_scale": pc_log["partial_r"],
        "partial_r_controlling_for_inv_lambda": pc["partial_r"],
        "r_ising_G_with_inv_lambda": pc["r_x_with_z"],
        "r_xy_G_with_inv_lambda": pc["r_y_with_z"],
        "jackknife_min_r": jk["min_r"],
        "jackknife_swing": jk["swing"],
        "trace_only_r": tr_r["r"],
        "trace_only_r_ci": tr_r["ci"],
        "trace_only_chi2_per_dof": tr_collapse["chi2_per_dof"],
    }

    # ------------------------------------------------------------------
    banner("PART 3 - CONTROL: DOES THE COLLAPSE TEST ACCEPT REAL UNIVERSALITY?")
    # ------------------------------------------------------------------
    print(
        "\nA test that rejects everything is as useless as one that accepts"
        "\neverything. Feeding it two curves that ARE the same shape, differing"
        "\nonly by amplitude and coupling scale plus shot noise:\n"
    )
    rng = np.random.default_rng(5)
    shape = np.exp(-((LAMBDAS - 0.8) ** 2) / (2 * 0.5 ** 2))
    accepted = 0
    trials = 200
    for t in range(trials):
        a = 0.17 * shape + rng.normal(0, sigma, LAMBDAS.size)
        warped = np.interp(LAMBDAS / 1.4, LAMBDAS, shape, left=shape[0],
                           right=shape[-1])
        b = 0.12 * warped + rng.normal(0, sigma, LAMBDAS.size)
        accepted += scaling_collapse(LAMBDAS, a, b, sigma)["collapses"]
    print(f"  genuinely universal pairs accepted : {100 * accepted / trials:.0f}%")
    print(f"  Ising vs Heisenberg                : "
          f"{'accepted' if collapse['collapses'] else 'rejected'} "
          f"(chi2/dof = {collapse['chi2_per_dof']:.1f})")
    out["control_acceptance_rate"] = accepted / trials

    dest = os.path.join(ROOT, "results", "corrected_curve_comparisons.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{RULE}\nWritten: {os.path.relpath(dest, os.getcwd())}\n{RULE}")


if __name__ == "__main__":
    main()
