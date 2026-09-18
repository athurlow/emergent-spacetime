#!/usr/bin/env python3
"""Tests for the corrected baseline statistics and metric-axiom tests."""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))

from correlation_stats import (  # noqa: E402
    compare_to_baseline,
    connected_correlation_matrix,
    counts_to_arrays,
    cross_chain_pairs,
    summarise_pairs,
)
from metric_axioms import (  # noqa: E402
    check_axioms,
    check_triangle_inequality,
    discriminating_power,
    distance_matrix,
)
from two_chain_model import (  # noqa: E402
    build_two_chain_circuit,
    exact_correlation_matrix,
    sample_counts,
)

N_CHAIN = 4
N_TOTAL = 8


# ----------------------------------------------------------------------
# correlation statistics
# ----------------------------------------------------------------------

def test_counts_to_arrays_is_little_endian():
    # '00000001' means qubit 0 is excited, so Z_0 = -1 and the rest +1.
    z, w = counts_to_arrays({"00000001": 10}, N_TOTAL)
    assert w[0] == 10
    assert z[0, 0] == -1.0
    assert (z[0, 1:] == 1.0).all()


def test_perfectly_correlated_pair_gives_unit_correlation():
    counts = {"00000000": 500, "00000011": 500}
    corr = connected_correlation_matrix(counts, N_TOTAL)
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-12)


def test_anticorrelated_pair_gives_minus_one():
    counts = {"00000001": 500, "00000010": 500}
    corr = connected_correlation_matrix(counts, N_TOTAL)
    assert corr[0, 1] == pytest.approx(-1.0, abs=1e-12)


def test_sampled_correlations_converge_to_exact():
    qc = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    exact = exact_correlation_matrix(qc)
    counts = sample_counts(qc, shots=400_000, seed=3)
    measured = connected_correlation_matrix(counts, N_TOTAL)
    pairs = cross_chain_pairs(N_CHAIN)
    for i, j in pairs:
        assert measured[i, j] == pytest.approx(exact[i, j], abs=0.01)


def test_decoupled_cross_chain_correlation_is_exactly_zero():
    """The premise behind the fix: the baseline has no signal to measure."""
    qc = build_two_chain_circuit(N_CHAIN, 1.0, 0.0, 6, 0.3)
    exact = exact_correlation_matrix(qc)
    for i, j in cross_chain_pairs(N_CHAIN):
        assert exact[i, j] == pytest.approx(0.0, abs=1e-12)


def test_baseline_is_usually_flagged_consistent_with_zero():
    """Coverage is a rate, not a guarantee for any single run.

    The true decoupled correlation is zero, so a nominal 95% interval should
    cover zero about 95% of the time. Individual decoupled runs land outside
    it roughly one time in twenty, which is itself why a single hardware
    baseline is not a safe denominator.
    """
    qc_off = build_two_chain_circuit(N_CHAIN, 1.0, 0.0, 6, 0.3)
    pairs = cross_chain_pairs(N_CHAIN)
    flags = [
        summarise_pairs(sample_counts(qc_off, 8192, seed=s), N_TOTAL,
                        pairs, n_boot=300, seed=s)["consistent_with_zero"]
        for s in range(60)
    ]
    assert 0.85 <= float(np.mean(flags)) <= 1.0


def test_baseline_spread_matches_shot_noise():
    """The decoupled estimate scatters by exactly the shot-noise width."""
    qc_off = build_two_chain_circuit(N_CHAIN, 1.0, 0.0, 6, 0.3)
    pairs = cross_chain_pairs(N_CHAIN)
    pts = [
        summarise_pairs(sample_counts(qc_off, 8192, seed=s), N_TOTAL,
                        pairs, n_boot=50, seed=s)["mean_signed"]
        for s in range(80)
    ]
    predicted = 1.0 / np.sqrt(8192) / np.sqrt(len(pairs))
    assert np.mean(pts) == pytest.approx(0.0, abs=3 * predicted)
    assert np.std(pts) == pytest.approx(predicted, rel=0.5)


def test_ratio_is_reported_as_lower_bound_when_baseline_is_noise():
    qc_on = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    qc_off = build_two_chain_circuit(N_CHAIN, 1.0, 0.0, 6, 0.3)
    res = compare_to_baseline(
        sample_counts(qc_on, 8192, seed=6),
        sample_counts(qc_off, 8192, seed=7),
        N_TOTAL, cross_chain_pairs(N_CHAIN), n_boot=600,
    )
    assert res["baseline_consistent_with_zero"]
    assert not res["ratio_is_bounded"]
    assert res["significant"]
    assert res["difference_ci"][0] > 0


def test_difference_is_shot_stable_while_ratio_is_not():
    """The ratio tracks the noise floor; the difference tracks the physics."""
    qc_on = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    qc_off = build_two_chain_circuit(N_CHAIN, 1.0, 0.0, 6, 0.3)
    pairs = cross_chain_pairs(N_CHAIN)
    low = compare_to_baseline(
        sample_counts(qc_on, 1024, seed=6), sample_counts(qc_off, 1024, seed=7),
        N_TOTAL, pairs, n_boot=400,
    )
    high = compare_to_baseline(
        sample_counts(qc_on, 65536, seed=6), sample_counts(qc_off, 65536, seed=7),
        N_TOTAL, pairs, n_boot=400,
    )
    assert high["difference"] == pytest.approx(low["difference"], abs=0.02)
    assert high["ratio_point"] > 2 * low["ratio_point"]


def test_bootstrap_interval_covers_the_exact_value():
    qc = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    exact = exact_correlation_matrix(qc)
    pairs = cross_chain_pairs(N_CHAIN)
    truth = float(np.mean([exact[i, j] for i, j in pairs]))
    covered = 0
    trials = 20
    for s in range(trials):
        summary = summarise_pairs(
            sample_counts(qc, 8192, seed=1000 + s), N_TOTAL, pairs, n_boot=400,
        )
        lo, hi = summary["ci"]
        covered += lo <= truth <= hi
    assert covered >= trials - 4  # nominal 95%, allow sampling slack


def test_mean_abs_bias_matches_shot_noise_prediction():
    """mean|C| on a true-zero baseline should sit near sigma*sqrt(2/pi)."""
    qc_off = build_two_chain_circuit(N_CHAIN, 1.0, 0.0, 6, 0.3)
    pairs = cross_chain_pairs(N_CHAIN)
    observed = [
        summarise_pairs(sample_counts(qc_off, 8192, seed=2000 + s),
                        N_TOTAL, pairs, n_boot=50)["mean_abs"]
        for s in range(30)
    ]
    predicted = summarise_pairs(
        sample_counts(qc_off, 8192, seed=1), N_TOTAL, pairs, n_boot=50
    )["mean_abs_noise_bias"]
    assert np.mean(observed) == pytest.approx(predicted, rel=0.4)


# ----------------------------------------------------------------------
# metric axioms
# ----------------------------------------------------------------------

def test_triangle_test_passes_on_a_genuine_metric():
    """Euclidean distances between random points must satisfy every triple."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(8, 3))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    res = check_triangle_inequality(d, tol=1e-9)
    assert res["violated"] == 0
    assert res["pct"] == pytest.approx(100.0)


def test_triangle_test_detects_a_planted_violation():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(6, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d[0, 5] = d[5, 0] = d.max() * 10  # break one edge
    res = check_triangle_inequality(d, tol=1e-9)
    assert res["violated"] > 0
    assert any(0 in v["path"] and 5 in v["path"] for v in res["violations"])


def test_triangle_test_can_fail_unlike_the_original():
    """The replaced test returned 100% for any input. This one does not."""
    qc = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    corr = connected_correlation_matrix(sample_counts(qc, 8192, seed=9), N_TOTAL)
    dp = discriminating_power(corr, kind="inverse", n_trials=60)
    assert dp["test_can_fail"]
    assert dp["scrambled_min_pct"] < 100.0


def test_original_coupling_space_test_is_a_tautology():
    """Documents the defect being fixed: |x_i - x_j| always satisfies it."""
    rng = np.random.default_rng(2)
    for _ in range(25):
        vals = rng.random(8)
        sat = tot = 0
        for a, b, c in itertools.combinations(range(8), 3):
            x, y, z = (abs(vals[a] - vals[b]), abs(vals[b] - vals[c]),
                       abs(vals[a] - vals[c]))
            tot += 1
            sat += (x + y >= z) and (x + z >= y) and (y + z >= x)
        assert sat == tot  # never fails, for any input


def test_inverse_distance_fails_identity_of_indiscernibles():
    qc = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    corr = connected_correlation_matrix(sample_counts(qc, 8192, seed=4), N_TOTAL)
    res = check_axioms(corr, kind="inverse")
    assert not res["identity_of_indiscernibles"]
    assert not res["is_metric"]


def test_measured_geometry_violates_some_triangles():
    """The headline '100% satisfaction' does not survive a real test."""
    qc = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    corr = connected_correlation_matrix(sample_counts(qc, 8192, seed=4), N_TOTAL)
    res = check_axioms(corr, kind="inverse")
    assert res["triangle"]["violated"] > 0
    assert res["triangle"]["pct"] < 100.0


def test_distance_matrix_is_symmetric_and_finite():
    qc = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    corr = connected_correlation_matrix(sample_counts(qc, 8192, seed=4), N_TOTAL)
    for kind in ("inverse", "log"):
        d = distance_matrix(corr, kind=kind)
        assert np.allclose(d, d.T)
        assert np.isfinite(d).all()
