#!/usr/bin/env python3
"""Tests for the corrected spacetime-tearing analysis."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))

from tearing_analysis import (  # noqa: E402
    measurable_reduction_ceiling,
    noise_only_mean_abs,
    residual_upper_bound,
    tearing_summary,
)
from two_chain_model import (  # noqa: E402
    bipartite_entropy,
    build_torn_circuit,
    build_two_chain_circuit,
    exact_correlation_matrix,
)

SHOTS = 8192
N_CHAIN = 4

# Published Torino 4+4 values.
COUPLED = [0.1538, 0.1618, 0.1265, 0.0905]
TORN = [0.0256, 0.0087, 0.0315, 0.0176]


def cross_mean(qc):
    c = exact_correlation_matrix(qc)
    return float(np.mean([abs(c[i, i + N_CHAIN]) for i in range(N_CHAIN)]))


# ----------------------------------------------------------------------
# the physics: local evolution cannot change entanglement
# ----------------------------------------------------------------------

def test_tearing_preserves_entanglement_exactly():
    """The core defect: the torn step is U_A (x) U_B, so S(A:B) is frozen."""
    torn = build_torn_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    stage = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 3, 0.3)
    assert bipartite_entropy(torn, N_CHAIN) == pytest.approx(
        bipartite_entropy(stage, N_CHAIN), abs=1e-10
    )


def test_tearing_collapses_the_correlator_anyway():
    """Same entanglement, very different ZZ correlator."""
    torn = build_torn_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    stage = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 3, 0.3)
    assert cross_mean(torn) < 0.25 * cross_mean(stage)


def test_entanglement_invariance_holds_for_more_local_steps():
    """Not a coincidence of step count: any amount of local evolution."""
    stage = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 3, 0.3)
    reference = bipartite_entropy(stage, N_CHAIN)
    for total in (6, 8, 12):
        torn = build_torn_circuit(N_CHAIN, 1.0, 1.0, total, 0.3)
        own_stage = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, total // 2, 0.3)
        assert bipartite_entropy(torn, N_CHAIN) == pytest.approx(
            bipartite_entropy(own_stage, N_CHAIN), abs=1e-10
        )
    assert reference > 0  # the coupling really did entangle the chains


def test_torn_state_is_not_less_entangled_than_its_comparison():
    """It was compared against a 6-step run, and carries more entanglement."""
    torn = build_torn_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    full = build_two_chain_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    assert bipartite_entropy(torn, N_CHAIN) > bipartite_entropy(full, N_CHAIN)


# ----------------------------------------------------------------------
# the statistics
# ----------------------------------------------------------------------

def test_reduction_cannot_reach_100_percent():
    """mean|C| is biased upward, so the reduction is capped below 100%."""
    ceiling = measurable_reduction_ceiling(0.1331, SHOTS, N_CHAIN)
    assert 85.0 < ceiling < 100.0


def test_ceiling_falls_as_the_coupled_signal_weakens():
    strong = measurable_reduction_ceiling(0.20, SHOTS, N_CHAIN)
    weak = measurable_reduction_ceiling(0.05, SHOTS, N_CHAIN)
    assert strong > weak


def test_ceiling_rises_with_more_shots():
    few = measurable_reduction_ceiling(0.13, 1024, N_CHAIN)
    many = measurable_reduction_ceiling(0.13, 131072, N_CHAIN)
    assert many > few


def test_a_truly_zero_residual_reports_less_than_full_reduction():
    """Simulated: true torn correlation exactly zero, measured at 8192 shots."""
    rng = np.random.default_rng(0)
    sigma = 1 / np.sqrt(SHOTS)
    reductions = []
    for _ in range(200):
        torn = rng.normal(0.0, sigma, N_CHAIN)  # true value is zero
        res = tearing_summary(COUPLED, torn, SHOTS, n_boot=200)
        reductions.append(res["reduction_pct"])
    assert np.mean(reductions) < 97.0
    assert np.mean(reductions) > 85.0


def test_published_reduction_carries_a_wide_interval():
    res = tearing_summary(COUPLED, TORN, SHOTS)
    lo, hi = res["reduction_ci"]
    assert hi - lo > 5.0          # the single number 83.4% hid this
    assert lo < res["reduction_pct"] < hi


def test_published_residual_is_a_real_non_zero():
    """Decoupling suppresses the correlator; it does not eliminate it."""
    res = residual_upper_bound(TORN, SHOTS)
    assert not res["consistent_with_zero"]
    assert res["p_value_vs_zero"] < 0.05
    assert res["observed_mean_abs"] > res["noise_only_expected"]


def test_measured_residual_agrees_with_simulation():
    torn = build_torn_circuit(N_CHAIN, 1.0, 1.0, 6, 0.3)
    measured = float(np.mean(np.abs(TORN)))
    assert cross_mean(torn) == pytest.approx(measured, abs=0.015)


def test_a_zero_residual_is_reported_as_a_bound_not_a_value():
    rng = np.random.default_rng(1)
    torn = rng.normal(0.0, 1 / np.sqrt(SHOTS), N_CHAIN)
    res = residual_upper_bound(torn, SHOTS)
    assert res["consistent_with_zero"]
    assert res["upper_bound"] > 0.0


def test_noise_only_distribution_matches_the_analytic_bias():
    null = noise_only_mean_abs(SHOTS, 1)
    predicted = (1 / np.sqrt(SHOTS)) * np.sqrt(2 / np.pi)
    assert null["expected"] == pytest.approx(predicted, rel=0.05)
