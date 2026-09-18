#!/usr/bin/env python3
"""
Two coupled qubit chains: circuit, exact correlations, and sampled counts.

Mirrors the circuit in ``src/simulation/qiskit_experiment.py`` so that the
corrected statistics and metric-axiom tests can be exercised against real
correlation matrices rather than numbers pasted into a script.

This is a simulation. The archived hardware results in ``results/`` contain
only derived summaries, not raw bitstring counts, so hardware runs cannot
currently be re-analysed with corrected statistics. The loader below accepts
counts in Qiskit's format whenever those are added to the repository.

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

__all__ = [
    "build_two_chain_circuit",
    "exact_correlation_matrix",
    "sample_counts",
    "load_hardware_counts",
]


def build_two_chain_circuit(n_per_chain: int, j_intra: float, j_inter: float,
                            trotter_steps: int, dt: float) -> QuantumCircuit:
    """Trotterized evolution of two chains coupled by an inter-chain ZZ term.

    H = j_intra * sum (Z_i Z_{i+1} + X_i X_{i+1})   within each chain
      + j_inter * sum Z_iA Z_iB                      between chains
    """
    n_total = 2 * n_per_chain
    qc = QuantumCircuit(n_total)

    for i in range(n_per_chain):
        qc.h(i)

    for _ in range(trotter_steps):
        for base in (0, n_per_chain):
            for i in range(base, base + n_per_chain - 1):
                qc.cx(i, i + 1)
                qc.rz(2 * j_intra * dt, i + 1)
                qc.cx(i, i + 1)
                qc.h(i)
                qc.h(i + 1)
                qc.cx(i, i + 1)
                qc.rz(2 * j_intra * dt, i + 1)
                qc.cx(i, i + 1)
                qc.h(i)
                qc.h(i + 1)
        if j_inter > 0:
            for i in range(n_per_chain):
                j = i + n_per_chain
                qc.cx(i, j)
                qc.rz(2 * j_inter * dt, j)
                qc.cx(i, j)
    return qc


def _z_eigenvalues(n_qubits: int) -> np.ndarray:
    """z[s, i] = +/-1 for basis state s, qubit i (little-endian)."""
    states = np.arange(2 ** n_qubits)
    bits = (states[:, None] >> np.arange(n_qubits)[None, :]) & 1
    return 1.0 - 2.0 * bits


def exact_correlation_matrix(qc: QuantumCircuit) -> np.ndarray:
    """Connected ZZ correlation matrix from the exact statevector."""
    n = qc.num_qubits
    probs = np.abs(Statevector.from_instruction(qc).data) ** 2
    z = _z_eigenvalues(n)
    mean_z = probs @ z
    zz = (z * probs[:, None]).T @ z
    corr = zz - np.outer(mean_z, mean_z)
    np.fill_diagonal(corr, 1.0 - mean_z ** 2)
    return corr


def sample_counts(qc: QuantumCircuit, shots: int = 8192,
                  seed: int | None = 0) -> dict:
    """Sample Z-basis counts from the exact state, as a Qiskit counts dict."""
    n = qc.num_qubits
    probs = np.abs(Statevector.from_instruction(qc).data) ** 2
    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(shots, probs)
    return {
        format(s, f"0{n}b"): int(c)
        for s, c in enumerate(draws)
        if c > 0
    }


def load_hardware_counts(path: str, label: str) -> dict | None:
    """Load raw counts for one circuit label from a results JSON, if present.

    Returns ``None`` when the file stores only derived summaries, which is
    the case for every file currently in ``results/``.
    """
    with open(path) as fh:
        data = json.load(fh)
    labelled = data.get("labeled_results")
    if not isinstance(labelled, dict):
        return None
    counts = labelled.get(label)
    if not isinstance(counts, dict):
        return None
    return {k: int(v) for k, v in counts.items()}
