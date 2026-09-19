#!/usr/bin/env python3
"""
Corrected baseline statistics and metric-axiom tests.

Supersedes two results previously reported in the README:

  "Coupling ratios from 8x to 27x above the uncoupled baseline"
  "Triangle inequality (100% satisfaction)"

Both were artefacts of how they were computed. See ``correlation_stats.py``
and ``metric_axioms.py`` for the specifics. This script re-runs both
analyses correctly and prints a negative control for each, so that a reader
can see the tests are capable of failing.

Hardware raw counts are not archived in this repository, so the numbers
below come from exact simulation of the same circuit, sampled at the same
shot count used on hardware. Point the loader at a results file containing
raw counts to reproduce this for a hardware run.

Usage:  python src/analysis/11_baseline_and_metric_validation.py

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correlation_stats import (  # noqa: E402
    compare_to_baseline,
    cross_chain_pairs,
    format_comparison,
    connected_correlation_matrix,
)
from metric_axioms import (  # noqa: E402
    check_axioms,
    check_triangle_inequality,
    discriminating_power,
    distance_matrix,
    format_axiom_report,
)
from two_chain_model import (  # noqa: E402
    build_two_chain_circuit,
    exact_correlation_matrix,
    sample_counts,
)

N_CHAIN = 4
N_TOTAL = 2 * N_CHAIN
J_INTRA = 1.0
TROTTER_STEPS = 6
DT = 0.3
SHOTS = 8192
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

RULE = "=" * 72


def banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    print(RULE)
    print("CORRECTED BASELINE STATISTICS AND METRIC AXIOM TESTS")
    print("Andrew Thurlow | 528 Labs")
    print(RULE)
    print(
        "\nSource: exact simulation of the two-chain circuit, sampled at"
        f" {SHOTS} shots.\nHardware raw counts are not archived in results/,"
        " so hardware runs cannot yet\nbe re-analysed with these statistics."
    )

    pairs = cross_chain_pairs(N_CHAIN)

    # ------------------------------------------------------------------
    banner("PART 1 - THE DECOUPLED BASELINE IS A NOISE FLOOR")
    # ------------------------------------------------------------------
    qc_off = build_two_chain_circuit(N_CHAIN, J_INTRA, 0.0, TROTTER_STEPS, DT)
    exact_off = exact_correlation_matrix(qc_off)
    exact_cross_off = [exact_off[i, j] for i, j in pairs]

    print("\nExact cross-chain correlations with the coupling removed:")
    print("  " + "  ".join(f"{v:+.2e}" for v in exact_cross_off))
    print(
        "\nThese are zero to machine precision. With no inter-chain term the"
        "\nchains are in a product state, so the true decoupled correlation is"
        "\nexactly zero and any measured value is pure shot noise."
        "\n\nThe original 'coupling ratio' divides by that number. It therefore"
        "\nmeasures one over the noise floor, which is why the same experiment"
        "\ngave 95.7x on Torino and 13.7x on Fez."
    )

    counts_off = sample_counts(qc_off, SHOTS, seed=11)
    results = {}

    print("\nCoupled runs against that baseline:\n")
    for lam in [0.5, 1.0, 2.0]:
        qc_on = build_two_chain_circuit(N_CHAIN, J_INTRA, lam, TROTTER_STEPS, DT)
        counts_on = sample_counts(qc_on, SHOTS, seed=100 + int(lam * 10))
        cmp_ = compare_to_baseline(counts_on, counts_off, N_TOTAL, pairs)
        print(format_comparison(cmp_, f"lambda = {lam}"))
        print()
        results[f"lambda_{lam}"] = {
            k: v for k, v in cmp_.items() if not k.startswith("_")
        }

    print(
        "The difference and its z-score are stable and meaningful. The ratio"
        "\nis not: its denominator is consistent with zero, so it is reported"
        "\nas a lower bound only."
    )

    # ------------------------------------------------------------------
    banner("PART 2 - HOW THE RATIO SCALES WITH SHOT COUNT")
    # ------------------------------------------------------------------
    print(
        "\nIf the ratio were physical it would not depend on how long we"
        "\nmeasured. It scales as sqrt(shots), because that is how the noise"
        "\nfloor shrinks.\n"
    )
    qc_on = build_two_chain_circuit(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)
    print(f"  {'shots':>8}  {'ratio':>10}   {'difference':>12}")
    print("  " + "-" * 36)
    scaling = []
    for shots in [1024, 4096, 16384, 65536]:
        c_on = sample_counts(qc_on, shots, seed=7)
        c_off = sample_counts(qc_off, shots, seed=8)
        cmp_ = compare_to_baseline(c_on, c_off, N_TOTAL, pairs, n_boot=800)
        print(f"  {shots:>8}  {cmp_['ratio_point']:>10.1f}   "
              f"{cmp_['difference']:>+12.4f}")
        scaling.append({"shots": shots, "ratio": cmp_["ratio_point"],
                        "difference": cmp_["difference"]})
    print(
        "\nThe ratio climbs with shot count while the difference stays put."
        "\nThe difference is the physical quantity."
    )

    # ------------------------------------------------------------------
    banner("PART 3 - TRIANGLE INEQUALITY ON REAL SITE DISTANCES")
    # ------------------------------------------------------------------
    print(
        "\nThe original test compared scalars on a line using |x_i - x_j|,"
        "\nwhich satisfies the triangle inequality as an identity. It returns"
        "\n100% for any input, including random numbers.\n"
        "\nThe test below builds an 8 x 8 site-to-site distance matrix from"
        "\nthe measured correlations and checks every triple, so each"
        "\nconstraint is real and can fail.\n"
    )

    axiom_results = {}
    for lam in [0.5, 1.0]:
        qc = build_two_chain_circuit(N_CHAIN, J_INTRA, lam, TROTTER_STEPS, DT)
        counts = sample_counts(qc, SHOTS, seed=200 + int(lam * 10))
        corr = connected_correlation_matrix(counts, N_TOTAL)
        for kind in ("inverse", "log"):
            res = check_axioms(corr, kind=kind)
            print(format_axiom_report(res, f"lambda = {lam},  d = "
                                           + ("1/|C|" if kind == "inverse"
                                              else "-log|C|")))
            print()
            axiom_results[f"lambda_{lam}_{kind}"] = {
                "pct": res["triangle"]["pct"],
                "violated": res["triangle"]["violated"],
                "total": res["triangle"]["total"],
                "identity_of_indiscernibles": res["identity_of_indiscernibles"],
                "is_metric": res["is_metric"],
            }

    # ------------------------------------------------------------------
    banner("PART 4 - NEGATIVE CONTROL: CAN THE TEST FAIL?")
    # ------------------------------------------------------------------
    print(
        "\nScrambling the correlations destroys any geometric structure while"
        "\nkeeping the same set of values. A test with discriminating power"
        "\nshould reject at least some scrambled inputs.\n"
    )
    qc = build_two_chain_circuit(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)
    corr = connected_correlation_matrix(sample_counts(qc, SHOTS, seed=42), N_TOTAL)
    controls = {}
    for kind in ("inverse", "log"):
        dp = discriminating_power(corr, kind=kind, n_trials=200)
        print(f"  d = {'1/|C|' if kind == 'inverse' else '-log|C|'}")
        print(f"    measured data        : {dp['observed_pct']:.1f}% of triples satisfied")
        print(f"    scrambled data       : {dp['scrambled_mean_pct']:.1f}% mean, "
              f"{dp['scrambled_min_pct']:.1f}% worst case")
        print(f"    test can fail        : {dp['test_can_fail']}")
        print()
        controls[kind] = dp

    print("  For comparison, the original coupling-space test:")
    rng = np.random.default_rng(0)
    for trial in range(3):
        vals = rng.random(8)
        sat = tot = 0
        for a in range(8):
            for b in range(a + 1, 8):
                for c in range(b + 1, 8):
                    x, y, z = (abs(vals[a] - vals[b]), abs(vals[b] - vals[c]),
                               abs(vals[a] - vals[c]))
                    tot += 1
                    if x + y >= z and x + z >= y and y + z >= x:
                        sat += 1
        print(f"    random input, trial {trial + 1}: {100 * sat / tot:.1f}% satisfied")
    print("    the test cannot fail, so its 100% carried no information")

    # ------------------------------------------------------------------
    out = {
        "source": "exact simulation of the two-chain circuit",
        "shots": SHOTS,
        "note": (
            "Supersedes the coupling-ratio and triangle-inequality results in "
            "earlier versions of the README. Hardware raw counts are not "
            "archived, so hardware runs cannot yet be re-analysed."
        ),
        "baseline_comparison": {
            k: {
                "difference": v["difference"],
                "difference_ci": v["difference_ci"],
                "z_score": v["z_score"],
                "ratio_lower_bound": v["ratio_lower_bound"],
                "baseline_consistent_with_zero": v["baseline_consistent_with_zero"],
            }
            for k, v in results.items()
        },
        "ratio_vs_shots": scaling,
        "metric_axioms": axiom_results,
        "negative_control": {
            k: {kk: vv for kk, vv in v.items()} for k, v in controls.items()
        },
    }
    dest = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "results", "corrected_statistics.json",
    )
    dest = os.path.normpath(dest)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{RULE}\nWritten: {os.path.relpath(dest, os.getcwd())}\n{RULE}")


if __name__ == "__main__":
    main()
