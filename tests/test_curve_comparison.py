#!/usr/bin/env python3
"""Tests for the corrected universality and gravity curve comparisons."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))

from curve_comparison import (  # noqa: E402
    jackknife_r,
    null_r_distribution,
    partial_correlation,
    pearson_ci,
    scaling_collapse,
    shot_noise_sigma,
)

LAMBDAS = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
SIGMA = shot_noise_sigma(8192, 4)

ISING = np.array([0.0056, 0.0269, 0.0579, 0.1005, 0.1602, 0.1523, 0.1361, 0.1150])
HEIS = np.array([0.0097, 0.0270, 0.0471, 0.0829, 0.0728, 0.0985, 0.1177, 0.0816])


def _unimodal(peak, width, amp=1.0, lam=LAMBDAS):
    return amp * np.exp(-((lam - peak) ** 2) / (2 * width ** 2))


# ----------------------------------------------------------------------
# what Pearson r is worth
# ----------------------------------------------------------------------

def test_published_universality_r_is_reproduced():
    assert pearson_ci(ISING, HEIS, SIGMA, n_boot=500)["r"] == pytest.approx(0.89, abs=0.01)


def test_r_interval_excludes_certainty():
    """Eight noisy points cannot pin r down; the old report quoted no interval."""
    res = pearson_ci(ISING, HEIS, SIGMA, n_boot=3000)
    lo, hi = res["ci"]
    assert hi - lo > 0.05
    assert lo < res["r"] < hi


def test_unrelated_unimodal_curves_also_correlate_strongly():
    """The defect being fixed: rise-then-fall shape alone produces high r."""
    null = null_r_distribution(ISING, LAMBDAS, SIGMA, n_trials=4000)
    assert null["median"] > 0.4
    assert float(np.mean(null["samples"] >= 0.89)) > 0.05


# ----------------------------------------------------------------------
# the collapse test: accepts real universality, rejects these curves
# ----------------------------------------------------------------------

def test_collapse_accepts_curves_that_share_a_shape():
    rng = np.random.default_rng(0)
    shape = _unimodal(0.8, 0.5)
    a = 0.17 * shape + rng.normal(0, SIGMA, LAMBDAS.size)
    b = 0.12 * shape + rng.normal(0, SIGMA, LAMBDAS.size)
    assert scaling_collapse(LAMBDAS, a, b, SIGMA)["collapses"]


def test_collapse_accepts_a_rescaled_coupling_axis():
    """Universality allows a non-universal coupling scale, not a new shape."""
    rng = np.random.default_rng(1)
    shape = _unimodal(0.6, 0.4)
    warped = np.interp(LAMBDAS / 1.5, LAMBDAS, shape, left=shape[0], right=shape[-1])
    a = 0.16 * shape + rng.normal(0, SIGMA, LAMBDAS.size)
    b = 0.11 * warped + rng.normal(0, SIGMA, LAMBDAS.size)
    res = scaling_collapse(LAMBDAS, a, b, SIGMA)
    assert res["collapses"]
    assert res["best_scale"] > 1.0


def test_collapse_rejects_genuinely_different_shapes():
    rng = np.random.default_rng(2)
    a = 0.16 * _unimodal(0.5, 0.25) + rng.normal(0, SIGMA, LAMBDAS.size)
    b = 0.16 * _unimodal(1.6, 0.9) + rng.normal(0, SIGMA, LAMBDAS.size)
    assert not scaling_collapse(LAMBDAS, a, b, SIGMA)["collapses"]


def test_collapse_rejects_the_measured_universality_claim():
    res = scaling_collapse(LAMBDAS, ISING, HEIS, SIGMA)
    assert res["chi2_per_dof"] > 2.0
    assert not res["collapses"]


def test_high_r_can_coexist_with_a_failed_collapse():
    """Exactly the situation in the data: r = 0.89 but chi2/dof = 9."""
    r = pearson_ci(ISING, HEIS, SIGMA, n_boot=300)["r"]
    collapse = scaling_collapse(LAMBDAS, ISING, HEIS, SIGMA)
    assert r > 0.85
    assert not collapse["collapses"]


# ----------------------------------------------------------------------
# the gravity correlation is definitional
# ----------------------------------------------------------------------

def test_gravity_correlation_is_one_with_no_measurement_at_all():
    """G = 1/(4*lambda*eta): constant eta still gives a perfect correlation."""
    lam = LAMBDAS[1:]
    g_a = 1.0 / (4 * lam * np.full_like(lam, 0.17))
    g_b = 1.0 / (4 * lam * np.full_like(lam, 0.09))
    assert np.corrcoef(g_a, g_b)[0, 1] == pytest.approx(1.0, abs=1e-12)


def test_published_gravity_r_is_below_the_definitional_ceiling():
    lam = LAMBDAS[1:]
    tr_a = np.array([0.0350, 0.0724, 0.1358, 0.1873, 0.2081, 0.1703, 0.1396])
    tr_b = np.array([0.0575, 0.0912, 0.1370, 0.0995, 0.1476, 0.1460, 0.0489])
    r_measured = np.corrcoef(1 / (4 * lam * tr_a), 1 / (4 * lam * tr_b))[0, 1]
    assert 0.95 < r_measured < 1.0  # high, yet below the r = 1 no-data ceiling


def test_partial_correlation_removes_a_shared_driver():
    rng = np.random.default_rng(3)
    z = np.linspace(1, 10, 40)
    x = z + rng.normal(0, 0.1, z.size)
    y = z + rng.normal(0, 0.1, z.size)
    pc = partial_correlation(x, y, z)
    assert pc["raw_r"] > 0.99          # driven entirely by the shared z
    assert abs(pc["partial_r"]) < 0.5  # nothing left once z is removed


def test_partial_correlation_keeps_genuine_shared_structure():
    rng = np.random.default_rng(4)
    z = np.linspace(1, 10, 40)
    common = rng.normal(0, 1, z.size)
    x = z + common
    y = z + common
    assert partial_correlation(x, y, z)["partial_r"] > 0.9


def test_jackknife_reports_leverage():
    x = np.arange(8, dtype=float)
    y = x.copy()
    y[0] = 100.0  # one point carrying the fit
    jk = jackknife_r(x, y)
    assert jk["most_influential_index"] in (0, 1)
    assert jk["swing"] != 0.0


def test_shot_noise_sigma_scales_correctly():
    assert shot_noise_sigma(8192, 4) == pytest.approx(1 / np.sqrt(8192) / 2)
    assert shot_noise_sigma(8192, 1) > shot_noise_sigma(8192, 4)
