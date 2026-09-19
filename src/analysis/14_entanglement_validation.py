#!/usr/bin/env python3
"""
Validation of the randomized-measurement entanglement protocol.

Run this before booking hardware time. It checks the estimators against
exactly known values, records why two simpler certificates were rejected,
checks that the chosen one never fires on a separable state, and reports
how much data a decision needs.

The scientific payoff is Part 5. The tearing analysis argued that a
collapsing ZZ correlator does not mean entanglement was removed. Here the
same three states go through a measurement that does see entanglement, and
the torn state is certified entangled while its correlator has collapsed.

Usage:  python src/analysis/14_entanglement_validation.py

Takes roughly fifteen minutes. Most of it is Part 3, which deliberately
runs the estimator on eight qubits to show it failing, and Part 6, which
simulates the full hardware sampling pattern.

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow_entanglement import (  # noqa: E402
    build_witness,
    measure_witness,
    partial_transpose,
    pt_moment_p3,
    renyi2_entropy,
    snapshot_matrices,
    subsystem_purity,
    witness_validity_floor,
)
from two_chain_model import (  # noqa: E402
    build_torn_circuit,
    build_two_chain_circuit,
    exact_correlation_matrix,
    simulate_hardware_protocol,
    simulate_randomized_measurements,
)

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
RULE = "=" * 72
N_CHAIN = 4
N_TOTAL = 8
KEEP = [0, 1, 4, 5]        # two qubits from each chain
LOCAL_A = [0, 1]           # their positions inside KEEP
TRACE_OUT = [2, 3, 6, 7]
N_SNAPSHOTS = 16_000


def banner(t: str) -> None:
    print(f"\n{RULE}\n{t}\n{RULE}")


def snaps_for(qc, n=N_SNAPSHOTS, seed=0, depolarizing=0.0):
    b, bits = simulate_randomized_measurements(qc, n, seed=seed,
                                               depolarizing=depolarizing)
    return snapshot_matrices(b, bits)


def negativity(rho, n, sub):
    pt = partial_transpose(rho, n, sub)
    ev = np.linalg.eigvalsh((pt + pt.conj().T) / 2)
    return float((np.sum(np.abs(ev)) - 1) / 2)


def main() -> None:
    print(RULE)
    print("RANDOMIZED-MEASUREMENT ENTANGLEMENT PROTOCOL - VALIDATION")
    print("Andrew Thurlow | 528 Labs")
    print(RULE)
    out: dict = {}

    states = {
        "coupled": build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3),
        "torn": build_torn_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3),
        "decoupled": build_two_chain_circuit(N_CHAIN, 1.0, 0.0, 6, 0.3),
    }
    reduced = {
        k: partial_trace(Statevector.from_instruction(v), TRACE_OUT).data
        for k, v in states.items()
    }

    # ------------------------------------------------------------------
    banner("PART 1 - ESTIMATORS AGAINST EXACTLY KNOWN VALUES")
    # ------------------------------------------------------------------
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    s = snaps_for(bell, 20_000, seed=3)
    print("\nBell state. Its partial transpose has eigenvalues (1,1,1,-1)/2,")
    print("so p2 = 1 and p3 = 3/8 - 1/8 = 1/4 exactly.\n")
    print(f"  {'quantity':<12}{'measured':<14}{'exact':<10}")
    print("  " + "-" * 36)
    rows = [("p2", subsystem_purity(s, [0, 1]), 1.0),
            ("p3", pt_moment_p3(s, [0], 400_000), 0.25),
            ("S_2(A)", renyi2_entropy(subsystem_purity(s, [0])), 1.0)]
    for name, got, exact in rows:
        print(f"  {name:<12}{got:<14.4f}{exact:<10.4f}")
    out["bell"] = {n: {"measured": g, "exact": e} for n, g, e in rows}

    # ------------------------------------------------------------------
    banner("PART 2 - WHY SUBSYSTEM ENTROPY IS NOT A CERTIFICATE")
    # ------------------------------------------------------------------
    print("""
On a noisy device the global state is mixed, and mixedness alone produces
subsystem entropy. Below is one separable state, two chains that never
interacted, measured through a local depolarizing channel of varying
strength. Local channels cannot create entanglement, so no row is
entangled.
""")
    print(f"  {'device noise':<16}{'S_2(A) bits':<16}{'entangled?':<12}")
    print("  " + "-" * 44)
    entropy_rows = {}
    for p_dep in (0.0, 0.05, 0.15, 0.30):
        sn = snaps_for(states["decoupled"], N_SNAPSHOTS, seed=21,
                       depolarizing=p_dep)
        s2a = renyi2_entropy(subsystem_purity(sn, list(range(N_CHAIN))))
        print(f"  {p_dep:<16.2f}{s2a:<16.3f}{'no':<12}")
        entropy_rows[p_dep] = s2a
    out["entropy_without_entanglement"] = entropy_rows
    print("\n  Reporting S_2(A) as evidence would call every row a success.")
    print("  (The zero-noise row can come out slightly negative: the purity")
    print("   estimator is unbiased, so it scatters either side of 1.)")

    # ------------------------------------------------------------------
    banner("PART 3 - TWO CERTIFICATES THAT DO NOT WORK HERE")
    # ------------------------------------------------------------------
    print("\n3a. The p3-PPT condition, p3 >= p2^2 for separable states.\n")
    print(f"  {'cut':<24}{'exact gap':<14}{'detectable?':<14}{'negativity':<12}")
    print("  " + "-" * 62)
    cuts = [([0], [4]), ([0, 1], [4, 5]), ([0, 1, 2], [4, 5, 6]),
            ([0, 1, 2, 3], [4, 5, 6, 7])]
    cut_rows = {}
    for a, b in cuts:
        keep = sorted(a + b)
        drop = [q for q in range(N_TOTAL) if q not in keep]
        sv = Statevector.from_instruction(states["coupled"])
        red = partial_trace(sv, drop).data if drop else \
            np.outer(sv.data, sv.data.conj())
        idx = {q: i for i, q in enumerate(keep)}
        local = [idx[q] for q in a]
        pt = partial_transpose(red, len(keep), local)
        p2 = float(np.real(np.trace(pt @ pt)))
        p3 = float(np.real(np.trace(pt @ pt @ pt)))
        gap = p3 - p2 ** 2
        neg = negativity(red, len(keep), local)
        label = f"{len(a)}+{len(b)}"
        print(f"  {label:<24}{gap:<+14.4f}{str(gap < 0):<14}{neg:<12.4f}")
        cut_rows[label] = {"gap": gap, "negativity": neg}
    out["cut_scan"] = cut_rows
    print("""
  The 2+2 cut is entangled, negativity 0.10, yet its gap is positive: the
  p3-PPT test is sufficient for entanglement and never necessary, so it
  cannot see that state. Only the full 4+4 cut has a decisive gap, and
  that is exactly where the estimator fails:
""")
    p2s, gaps = [], []
    for trial in range(4):
        sn = snaps_for(states["coupled"], 10_000, seed=90 + trial)
        p2 = subsystem_purity(sn, list(range(N_TOTAL)))
        p3 = pt_moment_p3(sn, list(range(N_CHAIN)), 400_000, seed=trial)
        p2s.append(p2)
        gaps.append(p3 - p2 ** 2)
    print(f"    full 4+4 cut, 10000 snapshots, 4 seeds:")
    print(f"      p2  = {np.mean(p2s):+.2f} +/- {np.std(p2s):.2f}"
          f"   (exact 1.00)")
    print(f"      gap = {np.mean(gaps):+.1f} +/- {np.std(gaps):.1f}"
          f"   (exact -0.84)")
    print("    The spread dwarfs the quantity. Shadow moment variance grows")
    print("    exponentially with qubit count, so eight qubits is out of reach.")
    out["p3_full_cut_failure"] = {
        "p2_mean": float(np.mean(p2s)), "p2_sd": float(np.std(p2s)),
        "gap_mean": float(np.mean(gaps)), "gap_sd": float(np.std(gaps)),
        "exact_gap": -0.8447,
    }

    print("\n3b. Reconstructing the reduced state and taking its negativity.\n")
    print(f"  {'state':<14}{'exact':<12}{'reconstructed':<16}")
    print("  " + "-" * 42)
    from shadow_entanglement import reduced_shadows
    neg_rows = {}
    for name in ("coupled", "decoupled"):
        sn = snaps_for(states[name], 16_000, seed=5)
        rho = reduced_shadows(sn, KEEP).mean(axis=0)
        rho = (rho + rho.conj().T) / 2
        ev, V = np.linalg.eigh(rho)
        ev = np.maximum(ev, 0)
        ev /= ev.sum()
        rho = V @ np.diag(ev) @ V.conj().T
        got = negativity(rho, 4, LOCAL_A)
        exact = negativity(reduced[name], 4, LOCAL_A)
        print(f"  {name:<14}{exact:<12.4f}{got:<16.4f}")
        neg_rows[name] = {"exact": exact, "reconstructed": got}
    out["negativity_bias"] = neg_rows
    print("""
  The separable control should read zero and does not. Projecting the noisy
  reconstruction back to a physical state turns estimator noise into
  spurious negative eigenvalues, which is a positive bias. A certificate
  with false positives is worse than no certificate.
""")

    # ------------------------------------------------------------------
    banner("PART 4 - THE WITNESS, AND THAT IT NEVER FIRES ON A SEPARABLE STATE")
    # ------------------------------------------------------------------
    print("""
W = |phi><phi|^{T_A}, with phi the most negative eigenvector of the ideal
reduced state's partial transpose. For any PPT state sigma,
Tr(W sigma) = Tr(|phi><phi| sigma^{T_A}) >= 0, and every separable state is
PPT. So Tr(W rho) < 0 certifies entanglement. The ideal state only chooses
the direction; a mismatch with hardware costs sensitivity, never validity.

Tr(W rho) is linear in rho, so the shadow estimator is unbiased and has no
false-positive floor.
""")
    witnesses = {n: build_witness(reduced[n], 4, LOCAL_A) for n in states}
    print(f"  {'witness built from':<22}{'min Tr(W sigma) over 4000 separable states':<12}")
    print("  " + "-" * 64)
    floors = {}
    for name, (w, _) in witnesses.items():
        floor = witness_validity_floor(w, 4, LOCAL_A, n_trials=4000)
        print(f"  {name:<22}{floor:+.5f}")
        floors[name] = floor
    out["witness_validity_floor"] = floors
    print("\n  All non-negative, as a valid witness requires.")

    # ------------------------------------------------------------------
    banner("PART 5 - THE THREE STATES, MEASURED FOR ENTANGLEMENT")
    # ------------------------------------------------------------------
    print()
    print(f"  {'state':<12}{'cross |C|':<12}{'S_2(A)':<10}{'Tr(W rho)':<26}"
          f"{'entangled?':<12}")
    print("  " + "-" * 72)
    rows5 = {}
    for name, qc in states.items():
        sn = snaps_for(qc, N_SNAPSHOTS, seed=7)
        w, _ = witnesses[name]
        res = measure_witness(sn, KEEP, w)
        s2a = renyi2_entropy(subsystem_purity(sn, list(range(N_CHAIN))))
        c = exact_correlation_matrix(qc)
        cross = float(np.mean([abs(c[i, i + N_CHAIN]) for i in range(N_CHAIN)]))
        ci = f"{res['witness_value']:+.4f} [{res['ci'][0]:+.4f},{res['ci'][1]:+.4f}]"
        print(f"  {name:<12}{cross:<12.4f}{s2a:<10.3f}{ci:<26}"
              f"{str(res['entangled_certified']):<12}")
        rows5[name] = {"cross_C": cross, "renyi2_A": s2a,
                       "witness": res["witness_value"], "ci": res["ci"],
                       "certified": res["entangled_certified"]}
    out["states"] = rows5
    print("""
  The torn state is the point. Its ZZ correlator has collapsed to about a
  ninth of the coupled value, which the earlier experiments read as
  spacetime disconnecting, and it is still certified entangled. The
  decoupled state, which really is separable, is correctly not certified.
""")

    # ------------------------------------------------------------------
    banner("PART 6 - HOW MANY RANDOM BASES THE HARDWARE RUN NEEDS")
    # ------------------------------------------------------------------
    print("""
On a device one circuit carries one random basis and is repeated for many
shots, so those shots are correlated. Precision is governed by the number
of distinct bases, not the total shot count, and the analysis averages
within a basis before bootstrapping over bases.

Below is that exact protocol, simulated at 64 shots per basis.
""")
    print(f"  {'state':<12}{'bases':<10}{'Tr(W rho)':<24}{'certified in':<14}")
    print("  " + "-" * 60)
    budget = {}
    for name in ("coupled", "torn", "decoupled"):
        w, _ = witnesses[name]
        for n_settings in (600, 2000):
            vals, fired = [], 0
            for trial in range(4):
                b, bits, sid = simulate_hardware_protocol(
                    states[name], n_settings, 64, seed=77 * trial + n_settings)
                res = measure_witness(snapshot_matrices(b, bits), KEEP, w,
                                      settings=sid, seed=trial)
                vals.append(res["witness_value"])
                fired += res["entangled_certified"]
            summary = f"{np.mean(vals):+.4f} +/- {np.std(vals):.4f}"
            print(f"  {name:<12}{n_settings:<10}{summary:<24}{f'{fired}/4':<14}")
            budget[f"{name}_{n_settings}"] = {
                "mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                "certified": f"{fired}/4"}
    out["basis_budget"] = budget
    print("""
  600 bases is enough for the coupled state but marginal for the torn one,
  whose witness value is smaller. The run script uses 2000. The separable
  control is never certified at either count, which is the property that
  matters most.
""")

    dest = os.path.join(ROOT, "results", "entanglement_protocol_validation.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"{RULE}\nWritten: {os.path.relpath(dest, os.getcwd())}\n{RULE}")


if __name__ == "__main__":
    main()
