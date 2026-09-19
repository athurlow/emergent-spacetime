#!/usr/bin/env python3
"""
Corrected analysis of the "spacetime tearing" result.

Supersedes the claim previously in the README:

  "Removing coupling destroys geometry. Spacetime tearing, 62-92%
   correlation collapse upon decoupling, confirmed across all experiments,
   consistent with Van Raamsdonk's disconnection prediction."

Two separate problems, one statistical and one physical. See
tearing_analysis.py for the statistics. The physical problem is the more
serious of the two and is demonstrated in Part 3 below.

Usage:  python src/analysis/13_tearing_validation.py

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tearing_analysis import (  # noqa: E402
    format_tearing_report,
    measurable_reduction_ceiling,
    noise_only_mean_abs,
    tearing_summary,
)
from two_chain_model import (  # noqa: E402
    bipartite_entropy,
    build_torn_circuit,
    build_two_chain_circuit,
    exact_correlation_matrix,
)

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
RULE = "=" * 72
SHOTS = 8192
N_CHAIN = 4
J_INTRA = 1.0
DT = 0.3
TROTTER_STEPS = 6


def banner(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def cross_mean(qc) -> float:
    c = exact_correlation_matrix(qc)
    return float(np.mean([abs(c[i, i + N_CHAIN]) for i in range(N_CHAIN)]))


def main() -> None:
    print(RULE)
    print("CORRECTED SPACETIME-TEARING ANALYSIS")
    print("Andrew Thurlow | 528 Labs")
    print(RULE)

    out: dict = {}

    # ------------------------------------------------------------------
    banner("PART 1 - THE REDUCTION PERCENTAGE IS CEILING-LIMITED")
    # ------------------------------------------------------------------
    path = os.path.join(ROOT, "results", "ibm_torino_results.json")
    with open(path) as fh:
        t = json.load(fh)["summary"]["spacetime_tearing"]
    coupled = [t[f"q{i}_connected"] for i in range(4)]
    torn = [t[f"q{i}_torn"] for i in range(4)]

    null = noise_only_mean_abs(SHOTS, 1)
    print(f"\nPer-pair shot noise sigma: {null['per_pair_sigma']:.4f}\n")
    print(f"  {'pair':<8}{'coupled':<12}{'torn':<12}{'torn in sigma':<16}"
          f"{'published':<12}")
    print("  " + "-" * 58)
    for i in range(4):
        print(f"  q{i:<7}{coupled[i]:<12.4f}{torn[i]:<12.4f}"
              f"{torn[i] / null['per_pair_sigma']:<16.1f}"
              f"{t[f'q{i}_reduction_pct']:<12.1f}")

    res = tearing_summary(coupled, torn, SHOTS)
    print()
    print(format_tearing_report(res, "Aggregated over the four pairs:"))
    print(
        "\nmean|C| is biased upward by sigma*sqrt(2/pi), so the reduction is"
        "\nbiased downward and cannot reach 100% even when the residual is"
        f"\nexactly zero. The cap here is {res['ceiling_pct']:.1f}%, which sits inside the"
        "\npublished 62-92% range: much of that range is the cap, not physics."
    )
    out["torino_4x4"] = {k: v for k, v in res.items() if k != "residual"}
    out["torino_4x4"]["residual"] = {
        k: v for k, v in res["residual"].items() if k != "samples"
    }

    # ------------------------------------------------------------------
    banner("PART 2 - THE 16-REGION SPREAD CANNOT BE DIAGNOSED AS ARCHIVED")
    # ------------------------------------------------------------------
    path = os.path.join(ROOT, "results", "ibm_fullchip_128q_results.json")
    with open(path) as fh:
        chip = json.load(fh)
    regions = [r["reduction_pct"] for r in chip["tearing"]["per_region"]]
    coupled_avg = chip["coupling_effect"]["coupled_avg_C"]
    print(f"\n16 parallel regions, reductions from {min(regions):.1f}% to"
          f" {max(regions):.1f}%, reported as a\nrange of behaviour across the"
          " chip.\n")
    ceiling_avg = measurable_reduction_ceiling(coupled_avg, SHOTS, N_CHAIN)
    print(f"  chip-average coupled mean|C|      : {coupled_avg:.4f}")
    print(f"  ceiling at that signal strength   : {ceiling_avg:.1f}%")
    print(f"  reported chip-average reduction   : "
          f"{chip['tearing']['average_reduction_pct']:.1f}%")
    print(
        "\n  The chip average sits well below the ceiling, so unlike the"
        "\n  4-qubit runs the cap is not what limits it here. Each region has"
        "\n  its own ceiling set by its own coupled signal strength, and a"
        "\n  weaker region can report a smaller reduction with no difference"
        "\n  in physics. Per-region coupled values are not archived, so the"
        "\n  28.9% to 93.3% spread cannot be separated into ceiling effects,"
        "\n  qubit quality and real residual. It should not be quoted as a"
        "\n  range of physical behaviour until it can."
    )
    out["fullchip_128q"] = {
        "reported_range_pct": [min(regions), max(regions)],
        "chip_average_reduction_pct": chip["tearing"]["average_reduction_pct"],
        "ceiling_at_chip_average_signal_pct": ceiling_avg,
    }

    # ------------------------------------------------------------------
    banner("PART 3 - TEARING DOES NOT REMOVE ENTANGLEMENT")
    # ------------------------------------------------------------------
    print(
        "\nThe torn circuit couples for half the Trotter steps, then evolves"
        "\nwith intra-chain gates only. That second half is a product of local"
        "\nunitaries U_A (x) U_B, which cannot change entanglement across the"
        "\nA:B cut. This is a theorem, not an approximation.\n"
    )
    full = build_two_chain_circuit(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)
    torn_qc = build_torn_circuit(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)
    stage = build_two_chain_circuit(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS // 2, DT)

    rows = [
        ("coupled throughout (6 steps)", full),
        ("torn (3 coupled + 3 local)", torn_qc),
        ("its own coupling stage (3 steps)", stage),
    ]
    print(f"  {'circuit':<36}{'cross |C|':<14}{'S(A:B) bits':<14}")
    print("  " + "-" * 64)
    for name, qc in rows:
        print(f"  {name:<36}{cross_mean(qc):<14.4f}"
              f"{bipartite_entropy(qc, N_CHAIN):<14.6f}")

    d_ent = abs(bipartite_entropy(torn_qc, N_CHAIN)
                - bipartite_entropy(stage, N_CHAIN))
    d_corr = abs(cross_mean(torn_qc) - cross_mean(stage))
    print(f"\n  tearing changes entanglement by : {d_ent:.2e} bits")
    print(f"  tearing changes the correlator by: {d_corr:.4f}"
          f"  ({100 * d_corr / cross_mean(stage):.0f}%)")
    print(
        "\n  The correlator collapses while the entanglement is frozen. The"
        "\n  measured collapse is a basis-dependent observable rotating under"
        "\n  local evolution, not entanglement being destroyed."
    )
    print(
        "\n  Note also that the torn state carries MORE A:B entanglement than"
        "\n  the 6-step coupled run it was compared against"
        f" ({bipartite_entropy(torn_qc, N_CHAIN):.3f} vs"
        f" {bipartite_entropy(full, N_CHAIN):.3f} bits),\n  because it ran fewer coupling steps."
    )
    out["entanglement"] = {
        "S_coupled_6_steps": bipartite_entropy(full, N_CHAIN),
        "S_torn": bipartite_entropy(torn_qc, N_CHAIN),
        "S_own_coupling_stage": bipartite_entropy(stage, N_CHAIN),
        "entropy_change_from_tearing_bits": d_ent,
        "correlator_change_from_tearing": d_corr,
    }

    # ------------------------------------------------------------------
    banner("PART 4 - THE RESIDUAL IS REAL AND SIMULATION PREDICTS IT")
    # ------------------------------------------------------------------
    sim_torn = cross_mean(torn_qc)
    print(f"\n  simulated torn cross |C| : {sim_torn:.4f}")
    print(f"  measured  torn cross |C| : {res['torn_mean_abs']:.4f}")
    print(f"  noise alone would give   : {res['residual']['noise_only_expected']:.4f}")
    print(
        f"\n  The measurement rejects zero at p = {res['residual']['p_value_vs_zero']:.4f} and lands near the"
        "\n  simulated value, so the surviving correlation is real rather than"
        "\n  a noise artefact. Decoupling suppresses the correlator; it does"
        "\n  not eliminate it."
    )
    out["residual_vs_simulation"] = {
        "simulated": sim_torn,
        "measured": res["torn_mean_abs"],
        "noise_only": res["residual"]["noise_only_expected"],
        "p_value_vs_zero": res["residual"]["p_value_vs_zero"],
    }

    # ------------------------------------------------------------------
    banner("WHAT THE EXPERIMENT DOES AND DOES NOT SHOW")
    # ------------------------------------------------------------------
    print("""
  DOES show: removing the inter-chain coupling suppresses the cross-field
  ZZ correlator by roughly 84%, reproducibly, leaving a small residual that
  simulation predicts.

  DOES NOT show: that entanglement is removed, or that the emergent space
  is disconnected. The entanglement across the A:B cut is mathematically
  unchanged by the tearing step. Van Raamsdonk's prediction concerns
  entanglement, so this experiment cannot test it as built.

  The useful lesson is the one the comparison forces: a single-basis ZZ
  correlator is not a proxy for entanglement. Testing disconnection needs
  an entanglement measure, such as negativity or a witness, which requires
  measurements in more than one basis.
""")

    dest = os.path.join(ROOT, "results", "corrected_tearing.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"{RULE}\nWritten: {os.path.relpath(dest, os.getcwd())}\n{RULE}")


if __name__ == "__main__":
    main()
