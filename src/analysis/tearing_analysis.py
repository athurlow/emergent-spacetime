#!/usr/bin/env python3
"""
Corrected statistics and interpretation for the "spacetime tearing" result.

The original statistic was

    reduction = (1 - mean_i |C_torn(i)| / mean_i |C_coupled(i)|) * 100

reported per experiment as 62% to 92%. Three problems:

1. The numerator sits at or near the shot-noise floor, so the percentage is
   partly reading out where the noise landed rather than a physical
   gradient. Across the four pairs of the first Torino run the torn values
   span 0.8 to 2.9 sigma.

2. mean|C| is biased upward, by sigma*sqrt(2/pi), so the reduction is
   biased downward and is capped below 100% even when the residual
   correlation is exactly zero. ``measurable_reduction_ceiling`` computes
   that cap. For the published coupled values it sits at 90% to 95%, which
   is most of the reported 62-92% range.

3. No uncertainty was propagated, so a spread that is largely noise was
   presented as structure varying across qubit pairs and experiments.

The deeper problem is not statistical. The torn circuit couples for half
the Trotter steps and then evolves under intra-chain gates only. That
second half is a product of local unitaries U_A (x) U_B, which provably
cannot change the entanglement across the A:B cut. Simulation confirms it:
the torn state and its own coupling stage have identical A:B entropy to
1e-14 bits, while the ZZ correlator falls by 88%.

So the measured collapse is a basis-dependent correlator rotating under
local evolution, not entanglement being removed. Van Raamsdonk's
disconnection prediction is about entanglement, so this experiment cannot
bear on it either way. What it does show is that a single-basis ZZ
correlator is not a proxy for entanglement.

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "measurable_reduction_ceiling",
    "noise_only_mean_abs",
    "residual_upper_bound",
    "tearing_summary",
    "format_tearing_report",
]


def noise_only_mean_abs(shots: int, n_pairs: int) -> dict:
    """Distribution of mean|C| when the true correlation is exactly zero."""
    sigma = 1.0 / np.sqrt(shots)
    rng = np.random.default_rng(0)
    sims = np.abs(rng.normal(0, sigma, size=(200_000, n_pairs))).mean(axis=1)
    return {
        "per_pair_sigma": float(sigma),
        "expected": float(sims.mean()),
        "p95": float(np.percentile(sims, 95)),
        "samples": sims,
    }


def measurable_reduction_ceiling(coupled_mean_abs: float, shots: int,
                                 n_pairs: int) -> float:
    """Largest reduction the mean|C| estimator can report, as a percentage.

    Reached when the residual correlation is exactly zero, because the
    estimator still returns the positive bias of the absolute value.
    """
    floor = noise_only_mean_abs(shots, n_pairs)["expected"]
    return float((1.0 - floor / coupled_mean_abs) * 100.0)


def residual_upper_bound(torn_pairs, shots: int,
                         confidence: float = 0.95) -> dict:
    """One-sided upper bound on the residual cross-field correlation.

    When the torn measurement is consistent with zero, a bound is the
    honest summary: the data limits how large any surviving correlation can
    be, and does not establish a particular non-zero value.
    """
    torn = np.abs(np.asarray(torn_pairs, dtype=float))
    n = torn.size
    sigma = 1.0 / np.sqrt(shots)
    observed = float(torn.mean())

    null = noise_only_mean_abs(shots, n)
    p_value = float((null["samples"] >= observed).mean())

    # Upper bound: largest true |C| whose sampling distribution still puts
    # the observed mean in its lower tail at the stated confidence.
    rng = np.random.default_rng(1)
    grid = np.linspace(0.0, observed + 6 * sigma / np.sqrt(n), 400)
    bound = float(grid[-1])
    for true_c in grid:
        sims = np.abs(rng.normal(true_c, sigma, size=(4000, n))).mean(axis=1)
        if float((sims <= observed).mean()) < 1.0 - confidence:
            bound = float(true_c)
            break
    return {
        "observed_mean_abs": observed,
        "noise_only_expected": null["expected"],
        "noise_only_p95": null["p95"],
        "p_value_vs_zero": p_value,
        "consistent_with_zero": bool(p_value > 0.05),
        "upper_bound": bound,
        "confidence": confidence,
    }


def tearing_summary(coupled_pairs, torn_pairs, shots: int,
                    confidence: float = 0.95, n_boot: int = 4000,
                    seed: int | None = 0) -> dict:
    """Reduction with an interval, against the ceiling the estimator allows."""
    coupled = np.abs(np.asarray(coupled_pairs, dtype=float))
    torn = np.abs(np.asarray(torn_pairs, dtype=float))
    n = coupled.size
    sigma = 1.0 / np.sqrt(shots)

    c_mean, t_mean = float(coupled.mean()), float(torn.mean())
    reduction = (1.0 - t_mean / c_mean) * 100.0

    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        c = np.abs(rng.normal(coupled, sigma)).mean()
        t = np.abs(rng.normal(torn, sigma)).mean()
        reps[b] = (1.0 - t / c) * 100.0
    alpha = (1 - confidence) / 2

    ceiling = measurable_reduction_ceiling(c_mean, shots, n)
    residual = residual_upper_bound(torn, shots, confidence)

    return {
        "coupled_mean_abs": c_mean,
        "torn_mean_abs": t_mean,
        "reduction_pct": float(reduction),
        "reduction_ci": (float(np.percentile(reps, 100 * alpha)),
                         float(np.percentile(reps, 100 * (1 - alpha)))),
        "ceiling_pct": ceiling,
        "reduction_is_at_ceiling": bool(reduction >= ceiling - 2.0),
        "residual": residual,
        "shots": shots,
        "n_pairs": int(n),
        "confidence": confidence,
    }


def format_tearing_report(res: dict, label: str = "") -> str:
    """Human-readable block for a tearing_summary result."""
    pct = int(round(res["confidence"] * 100))
    r = res["residual"]
    lines = []
    if label:
        lines.append(label)
    lines.append(f"  coupled   mean|C| = {res['coupled_mean_abs']:.4f}")
    lines.append(f"  torn      mean|C| = {res['torn_mean_abs']:.4f}"
                 f"   (noise alone would give {r['noise_only_expected']:.4f})")
    lines.append(
        f"  reduction         = {res['reduction_pct']:.1f}%  "
        f"[{res['reduction_ci'][0]:.1f}%, {res['reduction_ci'][1]:.1f}%]"
    )
    lines.append(
        f"  ceiling for this estimator = {res['ceiling_pct']:.1f}%"
        "   (what a residual of exactly zero would report)"
    )
    if r["consistent_with_zero"]:
        lines.append(
            f"  residual correlation is consistent with zero; "
            f"{pct}% upper bound {r['upper_bound']:.4f}"
        )
    else:
        lines.append(
            f"  residual correlation is a real non-zero "
            f"(p = {r['p_value_vs_zero']:.4f} against zero), not just noise"
        )
    return "\n".join(lines)
