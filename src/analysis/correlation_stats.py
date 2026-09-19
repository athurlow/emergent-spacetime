#!/usr/bin/env python3
"""
Uncertainty-aware statistics for connected ZZ correlators measured on hardware.

Replaces the bare "coupling ratio" headline used in the original analysis
scripts. That ratio was

    ratio = mean_i |C_coupled(i)| / mean_i |C_uncoupled(i)|

and it has three defects:

1. The denominator estimates a quantity that is exactly zero. With no
   inter-chain coupling the two chains are in a product state, so the true
   cross-chain connected correlator vanishes and the measured value is a
   pure shot-noise fluctuation. The ratio is therefore 1/noise, not a
   property of the state. It scales as sqrt(shots) and varies between
   backends purely with their noise floor.

2. mean_i |C_i| is biased upward when the true correlations are near zero,
   because E|X| > 0 for any zero-mean X. The bias is sigma*sqrt(2/pi),
   which is the entire signal in the uncoupled case.

3. No uncertainty was propagated anywhere, so a ratio of 95.7 and a ratio
   of 13.7 for the same experiment on two backends looked like two
   findings rather than one noise floor measured twice.

The functions here report the signed mean (unbiased), a bootstrap
confidence interval, the coupled-minus-uncoupled difference with its own
interval, and a ratio that is presented as a one-sided lower bound
whenever the denominator is consistent with zero.

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "counts_to_arrays",
    "connected_correlation_matrix",
    "cross_chain_pairs",
    "bootstrap_pair_means",
    "summarise_pairs",
    "compare_to_baseline",
    "format_comparison",
]

DEFAULT_BOOTSTRAP = 4000
DEFAULT_CONFIDENCE = 0.95


def counts_to_arrays(counts: dict, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Qiskit counts dict into eigenvalue and weight arrays.

    Returns ``(z, w)`` where ``z[s, i]`` is the +1/-1 Z eigenvalue of qubit
    ``i`` in outcome ``s`` and ``w[s]`` is that outcome's shot count.
    Bitstrings are little-endian, matching Qiskit and the original scripts.
    """
    keys = list(counts.keys())
    if not keys:
        raise ValueError("empty counts dict")

    w = np.array([counts[k] for k in keys], dtype=float)
    z = np.empty((len(keys), n_qubits), dtype=float)
    for s, key in enumerate(keys):
        bits = key.replace(" ", "")
        if len(bits) < n_qubits:
            raise ValueError(
                f"bitstring {key!r} has {len(bits)} bits, expected >= {n_qubits}"
            )
        for i in range(n_qubits):
            z[s, i] = 1.0 - 2.0 * int(bits[len(bits) - 1 - i])
    return z, w


def _connected_from_arrays(z: np.ndarray, w: np.ndarray, pairs) -> np.ndarray:
    """Connected correlators C_ij = <Z_i Z_j> - <Z_i><Z_j> for each pair."""
    total = w.sum()
    mean_z = (w[:, None] * z).sum(axis=0) / total
    out = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        zz = float((w * z[:, i] * z[:, j]).sum() / total)
        out[k] = zz - mean_z[i] * mean_z[j]
    return out


def connected_correlation_matrix(counts: dict, n_qubits: int) -> np.ndarray:
    """Full connected correlation matrix from counts.

    The diagonal is set to 1 - <Z_i>^2, matching the original scripts.
    """
    z, w = counts_to_arrays(counts, n_qubits)
    total = w.sum()
    mean_z = (w[:, None] * z).sum(axis=0) / total
    zz = (z * w[:, None]).T @ z / total
    corr = zz - np.outer(mean_z, mean_z)
    np.fill_diagonal(corr, 1.0 - mean_z ** 2)
    return corr


def cross_chain_pairs(n_chain: int) -> list[tuple[int, int]]:
    """Corresponding cross-chain pairs (i_A, i_B) for a two-chain layout."""
    return [(i, i + n_chain) for i in range(n_chain)]


def bootstrap_pair_means(
    counts: dict,
    n_qubits: int,
    pairs,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int | None = 0,
) -> tuple[float, np.ndarray]:
    """Bootstrap the pair-averaged signed correlator.

    Resamples the multinomial shot distribution, which propagates both the
    <Z_i Z_j> and <Z_i><Z_j> uncertainty and their covariance without
    needing a delta-method approximation.

    Returns ``(point_estimate, replicates)``.
    """
    z, w = counts_to_arrays(counts, n_qubits)
    point = float(_connected_from_arrays(z, w, pairs).mean())

    total = int(round(w.sum()))
    probs = w / w.sum()
    rng = np.random.default_rng(seed)
    resampled = rng.multinomial(total, probs, size=n_boot).astype(float)

    reps = np.empty(n_boot)
    for b in range(n_boot):
        reps[b] = _connected_from_arrays(z, resampled[b], pairs).mean()
    return point, reps


def _ci(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(samples, 100 * alpha)),
        float(np.percentile(samples, 100 * (1 - alpha))),
    )


def summarise_pairs(
    counts: dict,
    n_qubits: int,
    pairs,
    n_boot: int = DEFAULT_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | None = 0,
) -> dict:
    """Signed pair-averaged correlator with a bootstrap interval.

    ``mean_abs`` is reported alongside for comparison with the original
    scripts, together with ``mean_abs_noise_bias``: the value that
    statistic would take on average if every true correlation were zero.
    """
    point, reps = bootstrap_pair_means(counts, n_qubits, pairs, n_boot, seed)
    lo, hi = _ci(reps, confidence)
    se = float(reps.std(ddof=1))

    z, w = counts_to_arrays(counts, n_qubits)
    per_pair = _connected_from_arrays(z, w, pairs)
    shots = float(w.sum())
    # Shot-noise scale for a single correlator, and the resulting positive
    # bias in mean|C| when the truth is zero.
    per_pair_se = 1.0 / np.sqrt(shots)

    return {
        "shots": shots,
        "per_pair": per_pair.tolist(),
        "mean_signed": point,
        "se": se,
        "ci": (lo, hi),
        "confidence": confidence,
        "consistent_with_zero": bool(lo <= 0.0 <= hi),
        "mean_abs": float(np.abs(per_pair).mean()),
        "mean_abs_noise_bias": float(per_pair_se * np.sqrt(2.0 / np.pi)),
        "_reps": reps,
    }


def compare_to_baseline(
    coupled_counts: dict,
    uncoupled_counts: dict,
    n_qubits: int,
    pairs,
    n_boot: int = DEFAULT_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | None = 0,
) -> dict:
    """Compare a coupled run against its decoupled baseline.

    The primary statistic is the difference, which is well defined even
    when the baseline is zero. The ratio is reported only as supporting
    detail, and is downgraded to a one-sided lower bound when the
    denominator's interval covers zero, because the ratio is then unbounded
    above.
    """
    cs = summarise_pairs(coupled_counts, n_qubits, pairs, n_boot, confidence, seed)
    us = summarise_pairs(uncoupled_counts, n_qubits, pairs, n_boot, confidence,
                         None if seed is None else seed + 1)

    diff_reps = cs["_reps"] - us["_reps"]
    diff = cs["mean_signed"] - us["mean_signed"]
    diff_lo, diff_hi = _ci(diff_reps, confidence)
    diff_se = float(diff_reps.std(ddof=1))
    z_score = diff / diff_se if diff_se > 0 else float("inf")

    # Ratio of magnitudes, bootstrapped jointly. Circuits are separate jobs,
    # so the two resamplings are independent.
    denom = np.abs(us["_reps"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_reps = np.abs(cs["_reps"]) / denom
    ratio_reps = ratio_reps[np.isfinite(ratio_reps)]

    alpha = 1.0 - confidence
    ratio_lower = float(np.percentile(ratio_reps, 100 * alpha)) if ratio_reps.size else float("nan")
    ratio_point = (
        abs(cs["mean_signed"]) / abs(us["mean_signed"])
        if us["mean_signed"] != 0
        else float("inf")
    )

    return {
        "coupled": cs,
        "uncoupled": us,
        "difference": diff,
        "difference_se": diff_se,
        "difference_ci": (diff_lo, diff_hi),
        "z_score": float(z_score),
        "significant": bool(diff_lo > 0.0),
        "baseline_consistent_with_zero": us["consistent_with_zero"],
        "ratio_point": ratio_point,
        "ratio_lower_bound": ratio_lower,
        "ratio_is_bounded": not us["consistent_with_zero"],
        "confidence": confidence,
    }


def format_comparison(result: dict, label: str = "") -> str:
    """Human-readable block for a compare_to_baseline result."""
    c, u = result["coupled"], result["uncoupled"]
    pct = int(round(result["confidence"] * 100))
    lines = []
    if label:
        lines.append(label)
    lines.append(
        f"  coupled     C = {c['mean_signed']:+.4f}  "
        f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]  ({pct}% CI, {int(c['shots'])} shots)"
    )
    lines.append(
        f"  decoupled   C = {u['mean_signed']:+.4f}  "
        f"[{u['ci'][0]:+.4f}, {u['ci'][1]:+.4f}]"
        + ("   consistent with zero" if u["consistent_with_zero"] else "")
    )
    lines.append(
        f"  difference    = {result['difference']:+.4f}  "
        f"[{result['difference_ci'][0]:+.4f}, {result['difference_ci'][1]:+.4f}]  "
        f"z = {result['z_score']:.1f}"
    )
    if result["ratio_is_bounded"]:
        lines.append(f"  ratio         = {result['ratio_point']:.1f}x")
    else:
        lines.append(
            f"  ratio         > {result['ratio_lower_bound']:.1f}x "
            f"({pct}% lower bound; denominator consistent with zero, so the"
            " ratio is unbounded above and is not a physical quantity)"
        )
    lines.append(
        f"  note: mean|C| on the decoupled run is {u['mean_abs']:.4f}; pure shot"
        f" noise alone would give {u['mean_abs_noise_bias']:.4f}"
    )
    return "\n".join(lines)
