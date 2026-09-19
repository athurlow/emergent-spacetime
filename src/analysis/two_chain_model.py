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
from qiskit.quantum_info import Statevector, entropy, partial_trace

__all__ = [
    "build_two_chain_circuit",
    "build_torn_circuit",
    "bipartite_entropy",
    "exact_correlation_matrix",
    "sample_counts",
    "load_hardware_counts",
    "simulate_randomized_measurements",
    "simulate_hardware_protocol",
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

def _intra_chain_step(qc: QuantumCircuit, n_per_chain: int, j_intra: float,
                      dt: float) -> None:
    """One Trotter step of intra-chain ZZ + XX on both chains, no coupling."""
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


def build_torn_circuit(n_per_chain: int, j_intra: float, j_inter: float,
                       total_steps: int, dt: float) -> QuantumCircuit:
    """The "spacetime tearing" circuit, as built by the experiment scripts.

    Evolves with inter-chain coupling for half the Trotter steps, then
    continues with intra-chain evolution only. The second half therefore
    applies a product of local unitaries U_A (x) U_B, which cannot change
    the entanglement across the A:B cut no matter how many steps run.
    """
    half = total_steps // 2
    qc = build_two_chain_circuit(n_per_chain, j_intra, j_inter, half, dt)
    for _ in range(half):
        _intra_chain_step(qc, n_per_chain, j_intra, dt)
    return qc


def bipartite_entropy(qc: QuantumCircuit, n_per_chain: int) -> float:
    """Von Neumann entropy across the chain-A / chain-B cut, in bits."""
    sv = Statevector.from_instruction(qc)
    reduced = partial_trace(sv, list(range(n_per_chain, qc.num_qubits)))
    return float(entropy(reduced))


def simulate_randomized_measurements(qc: QuantumCircuit, n_shots: int,
                                     seed: int | None = 0,
                                     depolarizing: float = 0.0):
    """Simulate the randomized-measurement protocol on a circuit.

    One independent random Pauli basis per qubit per shot, then a
    computational-basis measurement. Returns ``(bases, bits)`` integer
    arrays of shape (n_shots, n_qubits) in qubit order, ready for
    ``shadow_entanglement.snapshot_matrices``.

    ``depolarizing`` applies a single-qubit depolarizing channel of that
    strength to each qubit immediately before measurement. Since that
    channel maps rho to (1-p) rho + p I/2, and every measurement here is of
    a single-qubit Pauli, it is implemented exactly by replacing each
    measured bit with a uniform random bit with probability p. Local
    channels cannot create entanglement, so a separable state stays
    separable under it, which is what makes it a fair noise control.
    """
    from shadow_entanglement import PAULI_ROTATIONS

    n = qc.num_qubits
    psi = Statevector.from_instruction(qc).data
    rng = np.random.default_rng(seed)

    bases = rng.integers(0, 3, size=(n_shots, n))
    bits = np.empty((n_shots, n), dtype=int)

    # Group shots by basis pattern so each rotation is built once.
    keys, inverse = np.unique(bases, axis=0, return_inverse=True)
    for k, pattern in enumerate(keys):
        op = np.array([1.0 + 0j])
        for q in range(n - 1, -1, -1):
            op = np.kron(op, PAULI_ROTATIONS[int(pattern[q])])
        probs = np.abs(op @ psi) ** 2
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()
        rows = np.flatnonzero(inverse == k)
        draws = rng.choice(len(probs), size=rows.size, p=probs)
        for q in range(n):
            bits[rows, q] = (draws >> q) & 1

    if depolarizing > 0.0:
        if not 0.0 <= depolarizing <= 1.0:
            raise ValueError("depolarizing must lie in [0, 1]")
        scrambled = rng.random((n_shots, n)) < depolarizing
        bits = np.where(scrambled, rng.integers(0, 2, size=(n_shots, n)), bits)
    return bases, bits


def simulate_hardware_protocol(qc: QuantumCircuit, n_settings: int,
                               shots_per_setting: int, seed: int | None = 0):
    """Mirror how the hardware run actually collects data.

    On a device one circuit carries one random basis and is repeated for
    many shots, so those shots share their unitary. Precision is therefore
    governed by the number of distinct bases, not by the total shot count.
    Returns ``(bases, bits, setting_ids)``.
    """
    from shadow_entanglement import PAULI_ROTATIONS

    n = qc.num_qubits
    psi = Statevector.from_instruction(qc).data
    rng = np.random.default_rng(seed)

    chosen = rng.integers(0, 3, size=(n_settings, n))
    bits = np.empty((n_settings * shots_per_setting, n), dtype=int)
    for k in range(n_settings):
        state = psi.reshape([2] * n)
        for q in range(n):
            b = int(chosen[k, q])
            if b == 0:
                continue
            axis = n - 1 - q
            state = np.moveaxis(
                np.tensordot(PAULI_ROTATIONS[b], state, axes=([1], [axis])),
                0, axis)
        probs = np.abs(state.reshape(-1)) ** 2
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()
        draws = rng.choice(len(probs), size=shots_per_setting, p=probs)
        lo = k * shots_per_setting
        hi = lo + shots_per_setting
        for q in range(n):
            bits[lo:hi, q] = (draws >> q) & 1

    bases = np.repeat(chosen, shots_per_setting, axis=0)
    setting_ids = np.repeat(np.arange(n_settings), shots_per_setting)
    return bases, bits, setting_ids
