#!/usr/bin/env python3
"""
Tools for comparing two measured curves without over-reading Pearson r.

Two results in this repository were reported as bare correlation
coefficients:

    universality   r = 0.89    Ising vs Heisenberg lambda sweeps
    emergent G     r = 0.9987  Ising vs XY effective gravitational constant

Neither number carried a confidence interval, a null distribution, or a
leverage check, and in both cases r is close to uninformative:

1. Both lambda sweeps rise from zero, peak, and fall. Any two curves of
   that shape correlate strongly whatever physics produced them, so a high
   r is not evidence that the two Hamiltonians agree. ``null_r_distribution``
   calibrates what r is worth by drawing curves that are deliberately not
   universal.

2. The two G_eff series are both defined as G = 1/(4 * lambda * eta). They
   therefore share a 1/lambda factor that was imposed by the definition
   rather than measured. Correlating them mostly measures that shared
   factor. ``partial_correlation`` removes it; ``jackknife_r`` shows how
   much of what remains rests on one or two points.

The positive test for universality is not a correlation at all. It is
whether the curves collapse onto a common shape once a non-universal
amplitude and coupling scale are allowed, with residuals consistent with
measurement noise. That is ``scaling_collapse``.

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pearson_ci",
    "null_r_distribution",
    "partial_correlation",
    "jackknife_r",
    "scaling_collapse",
    "shot_noise_sigma",
    "format_curve_report",
]


def shot_noise_sigma(shots: int, n_pairs: int) -> float:
    """Standard error of a pair-averaged connected correlator."""
    return 1.0 / (np.sqrt(shots) * np.sqrt(n_pairs))


def _r(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pearson_ci(x, y, sigma: float, n_boot: int = 4000,
               confidence: float = 0.95, seed: int | None = 0) -> dict:
    """Pearson r with an interval that propagates measurement noise.

    Each point is jittered by its own shot noise, which is the dominant
    uncertainty here, rather than resampling points with replacement. With
    only eight points a case-resampling bootstrap is badly behaved.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        reps[b] = _r(x + rng.normal(0, sigma, x.size),
                     y + rng.normal(0, sigma, y.size))
    reps = reps[np.isfinite(reps)]
    alpha = (1 - confidence) / 2
    return {
        "r": _r(x, y),
        "ci": (float(np.percentile(reps, 100 * alpha)),
               float(np.percentile(reps, 100 * (1 - alpha)))),
        "sd": float(reps.std(ddof=1)),
        "n_points": int(x.size),
        "confidence": confidence,
    }


def null_r_distribution(reference, lambdas, sigma: float,
                        n_trials: int = 20000, seed: int | None = 0) -> dict:
    """What r is worth, given that both curves rise and fall.

    Draws smooth unimodal curves with a randomly placed peak, random width
    and random amplitude, adds shot noise, and correlates each against the
    reference. These curves share no physics with the reference beyond the
    generic rise-then-fall shape, so the resulting distribution is the null
    that a claimed universality has to beat.
    """
    reference = np.asarray(reference, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    rng = np.random.default_rng(seed)
    span = lam.max() - lam.min()

    out = np.empty(n_trials)
    for t in range(n_trials):
        peak = rng.uniform(lam.min() + 0.1 * span, lam.max())
        width = rng.uniform(0.15 * span, 0.7 * span)
        amp = rng.uniform(0.5, 1.5) * np.abs(reference).max()
        curve = amp * np.exp(-((lam - peak) ** 2) / (2 * width ** 2))
        curve = curve + rng.normal(0, sigma, lam.size)
        out[t] = _r(reference, curve)
    out = out[np.isfinite(out)]
    return {
        "samples": out,
        "median": float(np.median(out)),
        "p90": float(np.percentile(out, 90)),
        "p95": float(np.percentile(out, 95)),
    }


def partial_correlation(x, y, z) -> dict:
    """Correlation of x and y after removing the shared dependence on z.

    Used to strip the definitional 1/lambda factor out of the G_eff
    comparison. Both series are built as 1/(4*lambda*eta), so they are
    guaranteed to track each other through lambda alone.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    def resid(v):
        A = np.vstack([z, np.ones_like(z)]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef

    rx, ry = resid(x), resid(y)
    return {
        "raw_r": _r(x, y),
        "partial_r": _r(rx, ry),
        "r_x_with_z": _r(x, z),
        "r_y_with_z": _r(y, z),
    }


def jackknife_r(x, y) -> dict:
    """Leverage check: recompute r with each point removed in turn."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    full = _r(x, y)
    drops = []
    for i in range(x.size):
        keep = np.arange(x.size) != i
        drops.append({"dropped_index": i, "r": _r(x[keep], y[keep])})
    values = np.array([d["r"] for d in drops])
    worst = int(np.argmin(values))
    return {
        "full_r": full,
        "per_point": drops,
        "min_r": float(values.min()),
        "max_r": float(values.max()),
        "most_influential_index": worst,
        "swing": float(full - values.min()),
    }


def scaling_collapse(lambdas, curve_a, curve_b, sigma: float,
                     n_scan: int = 400) -> dict:
    """Test universality properly: do the curves collapse onto one shape?

    Fits ``curve_b(lambda) ~ amplitude * curve_a(lambda / scale)`` by
    scanning the coupling rescaling and solving for the amplitude in closed
    form, then reports chi-squared per degree of freedom against the known
    measurement noise.

    chi2/dof near 1 means the two curves are the same shape to within
    measurement error, which is what universality asserts. A large value
    means they are genuinely different curves, however well they correlate.
    """
    lam = np.asarray(lambdas, dtype=float)
    a = np.asarray(curve_a, dtype=float)
    b = np.asarray(curve_b, dtype=float)

    best = None
    for scale in np.linspace(0.3, 3.0, n_scan):
        warped = np.interp(lam / scale, lam, a, left=a[0], right=a[-1])
        denom = float(warped @ warped)
        if denom <= 0:
            continue
        amp = float(warped @ b) / denom
        resid = b - amp * warped
        chi2 = float(np.sum((resid / sigma) ** 2))
        if best is None or chi2 < best["chi2"]:
            best = {"chi2": chi2, "scale": float(scale), "amplitude": amp,
                    "residuals": resid}

    dof = max(lam.size - 2, 1)  # two fitted parameters
    chi2_dof = best["chi2"] / dof
    return {
        "chi2": best["chi2"],
        "dof": dof,
        "chi2_per_dof": chi2_dof,
        "best_scale": best["scale"],
        "best_amplitude": best["amplitude"],
        "max_residual_sigma": float(np.max(np.abs(best["residuals"])) / sigma),
        "collapses": bool(chi2_dof < 2.0),
        "sigma": sigma,
    }


def format_curve_report(r_result: dict, null: dict, collapse: dict,
                        label: str = "") -> str:
    """Human-readable block combining r, its null, and the collapse test."""
    pct = int(round(r_result["confidence"] * 100))
    beats = float(np.mean(null["samples"] >= r_result["r"]))
    lines = []
    if label:
        lines.append(label)
    lines.append(
        f"  Pearson r            = {r_result['r']:.3f}  "
        f"[{r_result['ci'][0]:.3f}, {r_result['ci'][1]:.3f}]  "
        f"({pct}% CI, {r_result['n_points']} points)"
    )
    lines.append(
        f"  unrelated curves     : median r = {null['median']:.2f}, "
        f"{100 * beats:.0f}% of them reach {r_result['r']:.2f} or better"
    )
    lines.append(
        f"  scaling collapse     : chi2/dof = {collapse['chi2_per_dof']:.1f} "
        f"(scale {collapse['best_scale']:.2f}, "
        f"worst point {collapse['max_residual_sigma']:.1f} sigma)"
    )
    lines.append(
        f"  curves are the same shape to within measurement error: "
        f"{'YES' if collapse['collapses'] else 'NO'}"
    )
    return "\n".join(lines)
