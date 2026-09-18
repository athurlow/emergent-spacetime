#!/usr/bin/env python3
"""
Metric-axiom tests for an emergent distance built from a correlation matrix.

Replaces the triangle-inequality test used in the original analysis. That
test compared scalars on a line:

    d(lambda_i, lambda_j) = | Tr G(lambda_i) - Tr G(lambda_j) |

For real numbers, |a - c| <= |a - b| + |b - c| is an identity. The test
therefore returns 100% for any input whatsoever, including random noise,
and the reported "100% of testable triangles satisfied" carried no
information about the emergent geometry. The per-site triangle list in the
original script was written but never iterated.

The tests here operate on a genuine N x N site-to-site distance matrix
derived from the measured correlation matrix, so every triple is a real
constraint that can fail. ``discriminating_power`` demonstrates that it
does fail on scrambled input, which is the property the original test
lacked.

Two distance definitions are supported:

    inverse : d(i,j) = 1 / |C(i,j)|        (as used in the paper)
    log     : d(i,j) = -log(|C(i,j)|)      (mutual-information-like)

Note that the inverse definition does not satisfy the identity of
indiscernibles: d(i,i) = 1/|C(i,i)| is finite and non-zero, so it is a
semi-metric at best. ``check_axioms`` reports this explicitly rather than
silently zeroing the diagonal.

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "distance_matrix",
    "check_triangle_inequality",
    "check_axioms",
    "discriminating_power",
    "format_axiom_report",
]

EPS = 1e-12


def distance_matrix(corr: np.ndarray, kind: str = "inverse",
                    floor: float = 1e-6) -> np.ndarray:
    """Distance matrix from a connected correlation matrix.

    ``floor`` clamps |C| from below so that a correlation buried in the
    noise maps to a large but finite distance instead of infinity.
    """
    c = np.abs(np.asarray(corr, dtype=float))
    c = np.maximum(c, floor)
    if kind == "inverse":
        d = 1.0 / c
    elif kind == "log":
        d = -np.log(c)
    else:
        raise ValueError(f"unknown distance kind: {kind!r}")
    return d


def check_triangle_inequality(d: np.ndarray,
                              tol: float = 1e-9) -> dict:
    """Test d(i,k) <= d(i,j) + d(j,k) over all distinct site triples.

    Every unordered triple contributes three constraints, one per choice of
    intermediate vertex. The diagonal is excluded, so this tests the
    geometry between sites and not a point against itself.
    """
    n = d.shape[0]
    if d.shape[0] != d.shape[1]:
        raise ValueError("distance matrix must be square")

    total = 0
    satisfied = 0
    violations = []
    worst = 0.0

    for i, j, k in itertools.combinations(range(n), 3):
        for a, b, c in ((i, j, k), (j, i, k), (i, k, j)):
            # constraint: d(a,c) <= d(a,b) + d(b,c)
            direct = d[a, c]
            detour = d[a, b] + d[b, c]
            total += 1
            excess = direct - detour
            if excess <= tol:
                satisfied += 1
            else:
                violations.append(
                    {"path": (a, b, c), "direct": float(direct),
                     "detour": float(detour), "excess": float(excess)}
                )
                worst = max(worst, float(excess / max(detour, EPS)))

    violations.sort(key=lambda v: -v["excess"])
    return {
        "total": total,
        "satisfied": satisfied,
        "violated": total - satisfied,
        "pct": 100.0 * satisfied / total if total else float("nan"),
        "worst_relative_excess": worst,
        "violations": violations[:10],
    }


def check_axioms(corr: np.ndarray, kind: str = "inverse",
                 floor: float = 1e-6, tol: float = 1e-9) -> dict:
    """Test all four metric axioms on a correlation-derived distance."""
    d = distance_matrix(corr, kind=kind, floor=floor)
    n = d.shape[0]
    off = ~np.eye(n, dtype=bool)

    non_negative = bool((d[off] >= -tol).all())
    symmetric = bool(np.allclose(d, d.T, atol=1e-9))

    diag = np.diag(d)
    identity_holds = bool(np.all(np.abs(diag) <= tol))

    # Distinct sites must be at non-zero distance for a true metric.
    distinct_separated = bool((d[off] > tol).all())

    triangle = check_triangle_inequality(d, tol=tol)

    return {
        "kind": kind,
        "n_sites": n,
        "non_negativity": non_negative,
        "symmetry": symmetric,
        "identity_of_indiscernibles": identity_holds,
        "self_distance_max": float(np.max(np.abs(diag))),
        "distinct_sites_separated": distinct_separated,
        "triangle": triangle,
        "is_metric": bool(
            non_negative and symmetric and identity_holds
            and distinct_separated and triangle["violated"] == 0
        ),
    }


def discriminating_power(corr: np.ndarray, kind: str = "inverse",
                         n_trials: int = 200, floor: float = 1e-6,
                         seed: int | None = 0) -> dict:
    """Negative control: does the triangle test reject scrambled input?

    Shuffles the off-diagonal correlations, destroying any geometric
    structure while preserving the value distribution, and reports the pass
    rate. A test with no discriminating power scores 100% here too, which
    is exactly what the original coupling-space test did.
    """
    rng = np.random.default_rng(seed)
    n = corr.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = np.abs(corr)[iu]

    observed = check_triangle_inequality(
        distance_matrix(corr, kind=kind, floor=floor)
    )["pct"]

    scores = []
    for _ in range(n_trials):
        shuffled = np.zeros_like(corr, dtype=float)
        perm = rng.permutation(vals)
        shuffled[iu] = perm
        shuffled = shuffled + shuffled.T
        np.fill_diagonal(shuffled, np.diag(corr))
        scores.append(
            check_triangle_inequality(
                distance_matrix(shuffled, kind=kind, floor=floor)
            )["pct"]
        )
    scores = np.array(scores)
    return {
        "observed_pct": observed,
        "scrambled_mean_pct": float(scores.mean()),
        "scrambled_min_pct": float(scores.min()),
        "scrambled_max_pct": float(scores.max()),
        "test_can_fail": bool(scores.min() < 100.0),
        "observed_beats_scrambled": bool(observed > scores.mean()),
    }


def format_axiom_report(result: dict, label: str = "") -> str:
    """Human-readable block for a check_axioms result."""
    t = result["triangle"]
    mark = lambda ok: "PASS" if ok else "FAIL"
    lines = []
    if label:
        lines.append(label)
    lines.append(f"  distance definition           : {result['kind']}")
    lines.append(f"  sites                         : {result['n_sites']}")
    lines.append(f"  non-negativity                : {mark(result['non_negativity'])}")
    lines.append(f"  symmetry                      : {mark(result['symmetry'])}")
    lines.append(
        f"  identity of indiscernibles    : {mark(result['identity_of_indiscernibles'])}"
        f"   (max self-distance {result['self_distance_max']:.3f})"
    )
    lines.append(
        f"  distinct sites separated      : {mark(result['distinct_sites_separated'])}"
    )
    lines.append(
        f"  triangle inequality           : {t['satisfied']}/{t['total']} "
        f"({t['pct']:.1f}%)"
    )
    if t["violated"]:
        w = t["violations"][0]
        lines.append(
            f"    worst violation: d{w['path'][0], w['path'][2]} = {w['direct']:.1f} "
            f"exceeds d{w['path'][0], w['path'][1]} + d{w['path'][1], w['path'][2]} "
            f"= {w['detour']:.1f}"
        )
    lines.append(f"  valid metric overall          : {mark(result['is_metric'])}")
    return "\n".join(lines)
