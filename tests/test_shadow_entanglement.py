#!/usr/bin/env python3
"""Tests for the randomized-measurement entanglement protocol."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))

from qiskit.quantum_info import Statevector, partial_trace  # noqa: E402

from shadow_entanglement import (  # noqa: E402
    PAULI_ROTATIONS,
    build_witness,
    counts_to_snapshots,
    measure_witness,
    partial_transpose,
    ppt_certificate,
    pt_moment_p3,
    reduced_shadows,
    renyi2_entropy,
    snapshot_matrices,
    subsystem_purity,
    witness_validity_floor,
)
from two_chain_model import (  # noqa: E402
    build_torn_circuit,
    build_two_chain_circuit,
    simulate_randomized_measurements,
)


def bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def snaps_for(qc, n, seed=0, depolarizing=0.0):
    b, bits = simulate_randomized_measurements(qc, n, seed=seed,
                                               depolarizing=depolarizing)
    return snapshot_matrices(b, bits)


# ----------------------------------------------------------------------
# estimators against exactly known values
# ----------------------------------------------------------------------

def test_rotations_measure_the_intended_paulis():
    """Basis 1 must diagonalise X and basis 2 must diagonalise Y."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.diag([1, -1]).astype(complex)
    for basis, pauli in ((0, Z), (1, X), (2, Y)):
        u = PAULI_ROTATIONS[basis]
        assert np.allclose(u @ pauli @ u.conj().T, Z, atol=1e-12)


def test_shadow_recovers_a_pure_single_qubit_purity():
    qc = QuantumCircuit(1)
    assert subsystem_purity(snaps_for(qc, 4000, seed=1), [0]) == pytest.approx(
        1.0, abs=0.06)


def test_bell_reduced_purity_is_one_half():
    s = snaps_for(bell(), 6000, seed=2)
    assert subsystem_purity(s, [0]) == pytest.approx(0.5, abs=0.05)


def test_bell_renyi2_entropy_is_one_bit():
    s = snaps_for(bell(), 8000, seed=3)
    assert renyi2_entropy(subsystem_purity(s, [0])) == pytest.approx(1.0, abs=0.15)


def test_bell_pt_moments_match_the_analytic_values():
    """PT eigenvalues (1,1,1,-1)/2 give p2 = 1 and p3 = 1/4."""
    s = snaps_for(bell(), 20000, seed=4)
    assert subsystem_purity(s, [0, 1]) == pytest.approx(1.0, abs=0.08)
    assert pt_moment_p3(s, [0], 400_000) == pytest.approx(0.25, abs=0.05)


def test_product_state_has_unit_reduced_purity():
    qc = QuantumCircuit(2)
    qc.h(0)
    s = snaps_for(qc, 6000, seed=5)
    assert subsystem_purity(s, [0]) == pytest.approx(1.0, abs=0.08)


# ----------------------------------------------------------------------
# the certificate: fires on entangled, never on separable
# ----------------------------------------------------------------------

def test_certificate_fires_on_a_bell_pair():
    s = snaps_for(bell(), 20000, seed=6)
    cert = ppt_certificate(s, [0], n_triples=300_000, n_boot=40)
    assert cert["entangled_certified"]
    assert cert["gap"] < 0


def test_certificate_does_not_fire_on_a_product_state():
    """A false positive here would invalidate every hardware result."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ry(0.7, 1)
    s = snaps_for(qc, 20000, seed=7)
    cert = ppt_certificate(s, [0], n_triples=300_000, n_boot=40)
    assert not cert["entangled_certified"]


def test_certificate_does_not_fire_on_a_separable_mixed_state():
    """Mixedness alone must not be mistaken for entanglement."""
    qc = QuantumCircuit(2)
    qc.h(0)
    s = snaps_for(qc, 20000, seed=8, depolarizing=0.25)
    cert = ppt_certificate(s, [0], n_triples=300_000, n_boot=40)
    assert not cert["entangled_certified"]


def test_entropy_rises_with_noise_on_a_state_with_no_entanglement():
    """The reason S_2(A) alone is not a certificate."""
    qc = build_two_chain_circuit(2, 1.0, 0.0, 3, 0.3)   # decoupled: separable
    clean = renyi2_entropy(subsystem_purity(snaps_for(qc, 8000, seed=9), [0, 1]))
    noisy = renyi2_entropy(
        subsystem_purity(snaps_for(qc, 8000, seed=9, depolarizing=0.3), [0, 1]))
    assert noisy > clean + 0.3


# ----------------------------------------------------------------------
# the scientific claim: the torn state is still entangled
# ----------------------------------------------------------------------

KEEP = [0, 1, 4, 5]          # two qubits from each chain
LOCAL_A = [0, 1]
TRACE_OUT = [2, 3, 6, 7]


def ideal_reduced(qc):
    return partial_trace(Statevector.from_instruction(qc), TRACE_OUT).data


def witness_for(qc):
    return build_witness(ideal_reduced(qc), 4, LOCAL_A)[0]


def test_witness_is_valid_on_random_separable_states():
    """Tr(W sigma) >= 0 must hold for every separable sigma."""
    qc = build_two_chain_circuit(4, 1.0, 1.0, 6, 0.3)
    floor = witness_validity_floor(witness_for(qc), 4, LOCAL_A, n_trials=1500)
    assert floor >= 0.0


def test_witness_certifies_the_coupled_state():
    qc = build_two_chain_circuit(4, 1.0, 1.0, 6, 0.3)
    res = measure_witness(snaps_for(qc, 16000, seed=10), KEEP, witness_for(qc))
    assert res["entangled_certified"]
    assert res["ci"][1] < 0


def test_witness_does_not_certify_the_decoupled_state():
    """The decoupled chains are a product state. A hit here is a false positive."""
    qc = build_two_chain_circuit(4, 1.0, 0.0, 6, 0.3)
    res = measure_witness(snaps_for(qc, 16000, seed=11), KEEP, witness_for(qc))
    assert not res["entangled_certified"]


def test_torn_state_is_still_certified_entangled():
    """The scientific payoff: its correlator collapsed, its entanglement did not."""
    qc = build_torn_circuit(4, 1.0, 1.0, 6, 0.3)
    res = measure_witness(snaps_for(qc, 16000, seed=12), KEEP, witness_for(qc))
    assert res["entangled_certified"]


def test_witness_estimator_is_unbiased_on_a_separable_state():
    """Unlike a reconstructed negativity, which has a positive floor."""
    qc = build_two_chain_circuit(4, 1.0, 0.0, 6, 0.3)
    w = witness_for(qc)
    exact = float(np.real(np.trace(w @ ideal_reduced(qc))))
    vals = [measure_witness(snaps_for(qc, 8000, seed=200 + s), KEEP, w,
                            n_boot=50)["witness_value"] for s in range(6)]
    assert np.mean(vals) == pytest.approx(exact, abs=0.03)


def test_reconstructed_negativity_is_biased_on_a_separable_state():
    """Documents why the negativity route was rejected."""
    qc = build_two_chain_circuit(4, 1.0, 0.0, 6, 0.3)
    rho = reduced_shadows(snaps_for(qc, 16000, seed=13), KEEP).mean(axis=0)
    rho = (rho + rho.conj().T) / 2
    ev, vecs = np.linalg.eigh(rho)
    ev = np.maximum(ev, 0)
    ev /= ev.sum()
    rho = vecs @ np.diag(ev) @ vecs.conj().T
    pt = partial_transpose(rho, 4, LOCAL_A)
    neg = float((np.sum(np.abs(np.linalg.eigvalsh((pt + pt.conj().T) / 2))) - 1) / 2)
    assert neg > 0.01          # exact value is zero: this is the bias


def test_p3_condition_misses_an_entangled_sub_cut():
    """Documents why p3-PPT was rejected: sufficient, never necessary."""
    qc = build_two_chain_circuit(4, 1.0, 1.0, 6, 0.3)
    rho = ideal_reduced(qc)
    pt = partial_transpose(rho, 4, LOCAL_A)
    p2 = float(np.real(np.trace(pt @ pt)))
    p3 = float(np.real(np.trace(pt @ pt @ pt)))
    ev = np.linalg.eigvalsh((pt + pt.conj().T) / 2)
    negativity = float((np.sum(np.abs(ev)) - 1) / 2)
    assert negativity > 0.05   # genuinely entangled
    assert p3 - p2 ** 2 > 0    # yet the p3 test cannot see it


def test_ppt_certificate_still_works_where_it_applies():
    """It is kept in the module and is correct on a small cut."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    cert = ppt_certificate(snaps_for(qc, 20000, seed=14), [0],
                           n_triples=300_000, n_boot=30)
    assert cert["entangled_certified"]


# ----------------------------------------------------------------------
# hardware-path mechanics
# ----------------------------------------------------------------------

def test_same_setting_pairs_are_excluded():
    """Shots sharing a random basis are not independent snapshots."""
    s = snaps_for(bell(), 2000, seed=12)
    labels = np.zeros(2000, dtype=int)          # pretend all share one basis
    with pytest.raises(ValueError):
        subsystem_purity(s, [0], settings=labels, max_pairs=1000)


def test_settings_labels_change_the_estimate():
    s = snaps_for(bell(), 4000, seed=13)
    labels = np.arange(4000) // 50              # 80 settings of 50 shots
    with_labels = subsystem_purity(s, [0], settings=labels, max_pairs=200_000)
    assert with_labels == pytest.approx(0.5, abs=0.08)


def test_counts_to_snapshots_matches_direct_construction():
    records = [((0, 1), "00"), ((2, 0), "01"), ((1, 1), "11")]
    built = counts_to_snapshots(records, 2)
    bases = np.array([[0, 1], [2, 0], [1, 1]])
    bits = np.array([[0, 0], [1, 0], [1, 1]])   # little-endian
    assert np.allclose(built, snapshot_matrices(bases, bits))


def test_snapshot_average_reproduces_the_state():
    """The defining property: E[rho_hat] = rho."""
    qc = QuantumCircuit(1)
    qc.ry(0.9, 0)
    s = snaps_for(qc, 20000, seed=14)
    from qiskit.quantum_info import DensityMatrix, Statevector
    exact = DensityMatrix(Statevector.from_instruction(qc)).data
    assert np.allclose(s[:, 0].mean(axis=0), exact, atol=0.05)


def test_purity_is_memory_bounded_at_large_snapshot_counts():
    s = snaps_for(bell(), 60000, seed=15)
    assert subsystem_purity(s, [0]) == pytest.approx(0.5, abs=0.05)


def test_run_script_gates_match_the_analysis_convention():
    """A mismatch here would silently corrupt every hardware result.

    The experiment script emits gates; the analysis assumes rotation
    matrices. They must agree up to a global phase for each Pauli basis.
    """
    from qiskit.quantum_info import Operator

    for basis in (0, 1, 2):
        qc = QuantumCircuit(1)
        if basis == 1:
            qc.h(0)
        elif basis == 2:
            qc.sdg(0)
            qc.h(0)
        got = Operator(qc).data
        want = PAULI_ROTATIONS[basis]
        phase = np.vdot(want.ravel(), got.ravel())
        phase /= abs(phase)
        assert np.allclose(got, want * phase, atol=1e-12)


# ----------------------------------------------------------------------
# the hardware analysis path, exercised without hardware
# ----------------------------------------------------------------------

def _load_retrieve_module():
    """Import the retrieve script with the IBM runtime stubbed out."""
    import importlib.util
    import types

    if "qiskit_ibm_runtime" not in sys.modules:
        stub = types.ModuleType("qiskit_ibm_runtime")
        stub.QiskitRuntimeService = object
        sys.modules["qiskit_ibm_runtime"] = stub
    path = os.path.join(ROOT, "src", "analysis",
                        "09_entanglement_randomized_retrieve.py")
    spec = importlib.util.spec_from_file_location("retrieve_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_retrieve_path_reproduces_the_certificate_from_counts():
    """Full hardware path: settings + counts in, certificate out.

    Builds the same measurement records the device would return, feeds them
    through the retrieve script's own grouping and analysis, and checks the
    coupled state is certified and the decoupled one is not.
    """
    from two_chain_model import simulate_hardware_protocol

    mod = _load_retrieve_module()
    n_settings, shots = 400, 32

    for state_name, lam, expected in (("coupled", 1.0, True),
                                      ("decoupled", 0.0, False)):
        qc = build_two_chain_circuit(4, 1.0, lam, 6, 0.3)
        bases, bits, sid = simulate_hardware_protocol(qc, n_settings, shots,
                                                      seed=5)

        # Re-encode as Qiskit counts, exactly as the device would report them.
        settings_list = [bases[k * shots].tolist() for k in range(n_settings)]
        results = {}
        for k in range(n_settings):
            counts = {}
            for row in bits[k * shots:(k + 1) * shots]:
                key = "".join(str(int(row[q])) for q in range(7, -1, -1))
                counts[key] = counts.get(key, 0) + 1
            results[f"{state_name}_setting_{k}"] = counts

        meta = {"n_total_qubits": 8, "n_chain": 4,
                "settings": {state_name: settings_list}}
        grouped = mod.collect_records(meta, results)
        b, bt, s_ids = grouped[state_name]
        assert b.shape == (n_settings * shots, 8)

        rho_ideal = mod.ideal_reduced_state(state_name, 4, 6, 0.3)
        w, _ = build_witness(rho_ideal, 4, mod.LOCAL_A)
        res = mod.analyse(b, bt, s_ids, 4, 8, w)
        assert res["witness"]["entangled_certified"] is expected
        assert res["n_settings"] == n_settings
