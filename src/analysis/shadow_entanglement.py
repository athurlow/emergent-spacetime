#!/usr/bin/env python3
"""
Entanglement measurement on hardware via local randomized measurements.

The eight experiments in this repository measure a single-basis ZZ
correlator. That quantity cannot distinguish entanglement from classical
correlation, and the tearing analysis shows the gap concretely: the
correlator collapses by 88% while the entanglement across the chain cut is
provably unchanged. This module adds a measurement that does see
entanglement.

Protocol (Huang, Kueng and Preskill classical shadows; Brydges et al. and
Elben et al. for the entropy and PPT applications). Each shot applies an
independent random single-qubit rotation to every qubit, chosen uniformly
from the three Pauli bases, then measures in the computational basis. Each
outcome gives a classical snapshot

    rho_hat = tensor_i ( 3 U_i^dag |s_i><s_i| U_i  -  I )

whose average is the true state. Subsystem moments follow from U-statistics
over distinct snapshots, and because every snapshot is a product over
qubits, each moment factorises into 2x2 traces.

Two quantities are computed:

1. Renyi-2 entropy S_2(A) = -log2 Tr(rho_A^2). For a PURE global state this
   certifies entanglement. On hardware the global state is mixed, so a
   positive S_2(A) alone certifies nothing: a product state of two mixed
   chains has exactly the same signature. This is reported, but it is not
   the certificate.

2. An entanglement witness W, measured on a 2+2 reduced state. The
   certificate is Tr(W rho) < 0, which holds for no separable state, so it
   stays valid when the device state is mixed.

   Two approaches were tried and rejected before this one, both documented
   in 14_entanglement_validation.py so the reasoning is auditable:

   - The p3-PPT condition (p3 >= p2^2 for all separable states) is
     implemented below and is exact on small cuts, but the variance of
     shadow moment estimators grows exponentially with qubit count. On the
     full 4+4 cut the gap scatters by tens against an exact -0.84, so the
     spread dwarfs the quantity. On cuts small enough to converge, this state
     satisfies the condition despite being entangled, since p3-PPT is
     sufficient and never necessary.

   - Reconstructing the reduced state and computing its negativity is
     feasible at four qubits but biased: the positivity projection turns
     estimator noise into spurious negative eigenvalues, so a separable
     control reads 0.065 instead of zero. A certificate with false
     positives is worse than none.

   Tr(W rho) is linear in rho, so its shadow estimator is unbiased and
   carries no such floor. The separable control comes out at +0.005 +/-
   0.005, consistent with zero, while the coupled state gives -0.078 +/-
   0.006.

Both estimators are validated against exact values in
tests/test_shadow_entanglement.py and in 14_entanglement_validation.py.

Andrew Thurlow | 528 Labs
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PAULI_ROTATIONS",
    "partial_transpose",
    "reduced_shadows",
    "build_witness",
    "witness_validity_floor",
    "measure_witness",
    "snapshot_matrices",
    "pairwise_trace_tensor",
    "subsystem_purity",
    "renyi2_entropy",
    "pt_moment_p3",
    "ppt_certificate",
    "counts_to_snapshots",
]

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)

# Rotation applied BEFORE the computational-basis measurement, per basis
# index: 0 measures Z, 1 measures X, 2 measures Y.
PAULI_ROTATIONS = {
    0: np.eye(2, dtype=complex),
    1: _H,
    2: _H @ _SDG,
}

_EYE = np.eye(2, dtype=complex)


def snapshot_matrices(bases: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Per-qubit shadow matrices for a batch of shots.

    ``bases`` and ``bits`` are integer arrays of shape (n_shots, n_qubits).
    Returns an array of shape (n_shots, n_qubits, 2, 2).
    """
    bases = np.asarray(bases, dtype=int)
    bits = np.asarray(bits, dtype=int)
    if bases.shape != bits.shape:
        raise ValueError("bases and bits must have the same shape")

    n_shots, n_qubits = bases.shape
    out = np.empty((n_shots, n_qubits, 2, 2), dtype=complex)
    for b in (0, 1, 2):
        u = PAULI_ROTATIONS[b]
        for outcome in (0, 1):
            ket = np.zeros((2, 1), dtype=complex)
            ket[outcome] = 1.0
            mat = 3.0 * (u.conj().T @ (ket @ ket.conj().T) @ u) - _EYE
            mask = (bases == b) & (bits == outcome)
            out[mask] = mat
    return out


def pairwise_trace_tensor(snaps: np.ndarray) -> np.ndarray:
    """Tr[rho_i^q rho_j^q] for every snapshot pair and qubit.

    Returns shape (n_qubits, n_shots, n_shots). Materialises an
    n_qubits x n_shots x n_shots array, so it is only for small snapshot
    counts; ``subsystem_purity`` chunks instead and should be preferred.
    """
    n_shots, n_qubits = snaps.shape[0], snaps.shape[1]
    a = snaps.transpose(1, 0, 2, 3).reshape(n_qubits, n_shots, 4)
    b = snaps.transpose(1, 0, 3, 2).reshape(n_qubits, n_shots, 4)
    return np.einsum("qik,qjk->qij", a, b)


def subsystem_purity(snaps: np.ndarray, subsystem, max_pairs: int = 4_000_000,
                     seed: int | None = 0, chunk: int = 2048,
                     settings: np.ndarray | None = None) -> float:
    """Unbiased Tr(rho_S^2) from distinct snapshot pairs.

    Uses every distinct pair when the total fits within ``max_pairs``,
    accumulating in row chunks so the full M x M table is never held in
    memory. Above that it samples pairs uniformly at random, which stays
    unbiased and keeps the cost flat as the snapshot count grows.

    ``settings`` labels which random basis each snapshot came from. On
    hardware a single circuit carries one random basis and is repeated for
    many shots, so those shots share their unitary and are not independent
    draws of the shadow channel. Passing the labels excludes same-setting
    pairs, which is required for this estimator to stay unbiased. Leave it
    as None only when every snapshot has its own basis.
    """
    subsystem = list(subsystem)
    if not subsystem:
        raise ValueError("subsystem must be non-empty")
    n_shots = snaps.shape[0]
    if n_shots < 2:
        raise ValueError("need at least two snapshots")

    a_all = snaps[:, subsystem].transpose(1, 0, 2, 3)          # (|S|, M, 2, 2)
    a_flat = a_all.reshape(len(subsystem), n_shots, 4)
    b_flat = snaps[:, subsystem].transpose(1, 0, 3, 2).reshape(
        len(subsystem), n_shots, 4)

    if settings is not None:
        settings = np.asarray(settings)
        if settings.shape[0] != n_shots:
            raise ValueError("settings must have one label per snapshot")

    total_pairs = n_shots * (n_shots - 1) // 2
    if settings is None and total_pairs <= max_pairs:
        acc = 0.0
        count = 0
        for start in range(0, n_shots, chunk):
            stop = min(start + chunk, n_shots)
            block = np.einsum("qik,qjk->qij", a_flat[:, start:stop], b_flat)
            prod = np.ones_like(block[0])
            for k in range(len(subsystem)):
                prod = prod * block[k]
            rows = np.arange(start, stop)
            prod[np.arange(stop - start), rows] = 0.0   # drop i == j
            acc += float(np.real(prod.sum()))
            count += (stop - start) * n_shots - (stop - start)
        return acc / count

    # Sample pairs in blocks so the index and product arrays stay small.
    rng = np.random.default_rng(seed)
    block = 250_000
    acc = 0.0
    kept = 0
    remaining = max_pairs
    while remaining > 0:
        take = min(block, remaining)
        remaining -= take
        idx = rng.integers(0, n_shots, size=(take, 2))
        keep = idx[:, 0] != idx[:, 1]
        if settings is not None:
            keep &= settings[idx[:, 0]] != settings[idx[:, 1]]
        idx = idx[keep]
        if idx.size == 0:
            continue
        prod = np.ones(idx.shape[0], dtype=complex)
        for k in range(len(subsystem)):
            prod *= np.einsum("nk,nk->n", a_flat[k, idx[:, 0]],
                              b_flat[k, idx[:, 1]])
        acc += float(np.real(prod.sum()))
        kept += idx.shape[0]
    if kept == 0:
        raise ValueError("no valid snapshot pairs; check the settings labels")
    return acc / kept


def renyi2_entropy(purity: float) -> float:
    """S_2 = -log2 Tr(rho^2), in bits. Clipped at zero for noisy estimates."""
    return float(-np.log2(max(purity, 1e-12)))


def pt_moment_p3(snaps: np.ndarray, subsystem_a, n_triples: int = 400_000,
                 seed: int | None = 0,
                 settings: np.ndarray | None = None) -> float:
    """Unbiased Tr[(rho^{T_A})^3] from random distinct snapshot triples.

    The full U-statistic runs over M(M-1)(M-2) ordered triples, which is
    prohibitive, so triples are sampled uniformly at random without
    repetition inside a triple. The estimator stays unbiased.
    """
    subsystem_a = set(subsystem_a)
    n_shots, n_qubits = snaps.shape[0], snaps.shape[1]
    if n_shots < 3:
        raise ValueError("need at least three snapshots")

    if settings is not None:
        settings = np.asarray(settings)
        if settings.shape[0] != n_shots:
            raise ValueError("settings must have one label per snapshot")

    rng = np.random.default_rng(seed)
    block = 200_000
    acc = 0.0
    kept = 0
    remaining = int(n_triples)
    while remaining > 0:
        take = min(block, remaining)
        remaining -= take
        acc_b, kept_b = _p3_block(snaps, subsystem_a, rng, take, n_shots,
                                  n_qubits, settings)
        acc += acc_b
        kept += kept_b
    if kept == 0:
        raise ValueError("no distinct triples sampled; check the settings labels")
    return acc / kept


def _p3_block(snaps, subsystem_a, rng, take, n_shots, n_qubits, settings):
    idx = rng.integers(0, n_shots, size=(take, 3))
    distinct = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & \
               (idx[:, 0] != idx[:, 2])
    if settings is not None:
        distinct &= (settings[idx[:, 0]] != settings[idx[:, 1]]) & \
                    (settings[idx[:, 1]] != settings[idx[:, 2]]) & \
                    (settings[idx[:, 0]] != settings[idx[:, 2]])
    idx = idx[distinct]
    if idx.size == 0:
        return 0.0, 0

    total = np.ones(idx.shape[0], dtype=complex)
    for q in range(n_qubits):
        a = snaps[idx[:, 0], q]
        b = snaps[idx[:, 1], q]
        c = snaps[idx[:, 2], q]
        if q in subsystem_a:
            a, b, c = a.transpose(0, 2, 1), b.transpose(0, 2, 1), \
                      c.transpose(0, 2, 1)
        total *= np.einsum("nij,njk,nki->n", a, b, c)
    return float(np.real(total.sum())), int(idx.shape[0])


def ppt_certificate(snaps: np.ndarray, subsystem_a, n_triples: int = 400_000,
                    seed: int | None = 0, n_boot: int = 200,
                    settings: np.ndarray | None = None) -> dict:
    """Test the p3-PPT condition: separable states obey p3 >= p2^2.

    A bootstrap over snapshots gives an interval on p3 - p2^2. Entanglement
    is certified only when the whole interval lies below zero, which is a
    sufficient condition and never a necessary one: a PPT-entangled state
    passes this test while still being entangled.
    """
    all_qubits = list(range(snaps.shape[1]))
    p2 = subsystem_purity(snaps, all_qubits, settings=settings)
    p3 = pt_moment_p3(snaps, subsystem_a, n_triples, seed, settings=settings)
    gap = p3 - p2 ** 2

    # Subsampling bootstrap. These are U-statistics, so resampling with
    # replacement would place duplicate snapshots in the same pair or triple
    # and bias the estimate upward, the way including self-pairs would.
    # Subsampling without replacement avoids that, but the replicates then
    # describe a smaller dataset: their spread must be rescaled by
    # sqrt(subsample fraction) to describe the full-data estimator.
    rng = np.random.default_rng(seed)
    n_shots = snaps.shape[0]
    fraction = 0.5
    m = max(int(n_shots * fraction), 3)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(n_shots, size=m, replace=False)
        sub = snaps[pick]
        sub_settings = None if settings is None else np.asarray(settings)[pick]
        # Each replicate uses a smaller pair and triple budget than the
        # point estimate: replicate count matters more than the precision
        # of any one replicate, and the extra per-replicate noise only
        # widens the interval, which is the safe direction.
        p2_b = subsystem_purity(sub, all_qubits, settings=sub_settings,
                                max_pairs=200_000,
                                seed=int(rng.integers(1 << 30)))
        p3_b = pt_moment_p3(sub, subsystem_a, max(n_triples // 20, 20_000),
                            seed=int(rng.integers(1 << 30)),
                            settings=sub_settings)
        reps[i] = p3_b - p2_b ** 2

    sd_full = float(reps.std(ddof=1)) * np.sqrt(m / n_shots)
    lo, hi = gap - 1.96 * sd_full, gap + 1.96 * sd_full
    return {
        "p2": p2,
        "p3": p3,
        "gap": float(gap),
        "gap_se": sd_full,
        "gap_ci": (float(lo), float(hi)),
        "subsample_sd": float(reps.std(ddof=1)),
        "entangled_certified": bool(hi < 0.0),
        "n_snapshots": int(n_shots),
        "note": ("p3 < p2^2 certifies entanglement for mixed states. "
                 "Passing the test does not certify separability: a state "
                 "can be entangled and still satisfy it, which happens for "
                 "small sub-cuts of this circuit."),
    }


def counts_to_snapshots(records, n_qubits: int) -> np.ndarray:
    """Build snapshot matrices from (basis_string, bitstring) measurement records.

    ``records`` is an iterable of ``(bases, bitstring)`` pairs, where
    ``bases`` is a length-n_qubits sequence of 0/1/2 in qubit order and
    ``bitstring`` is the Qiskit little-endian outcome string.
    """
    bases, bits = [], []
    for b, s in records:
        s = s.replace(" ", "")
        if len(s) < n_qubits:
            raise ValueError(f"bitstring {s!r} shorter than {n_qubits} qubits")
        bases.append([int(x) for x in b])
        bits.append([int(s[len(s) - 1 - q]) for q in range(n_qubits)])
    return snapshot_matrices(np.array(bases), np.array(bits))

def partial_transpose(rho: np.ndarray, n_qubits: int, subsystem) -> np.ndarray:
    """Partial transpose of a density matrix over the given qubits."""
    t = rho.reshape([2] * (2 * n_qubits))
    perm = list(range(2 * n_qubits))
    for q in subsystem:
        axis = n_qubits - 1 - q
        perm[axis], perm[axis + n_qubits] = perm[axis + n_qubits], perm[axis]
    return t.transpose(perm).reshape(2 ** n_qubits, 2 ** n_qubits)


def reduced_shadows(snaps: np.ndarray, keep) -> np.ndarray:
    """Per-snapshot shadow density matrices on the kept qubits.

    Returns shape (n_shots, 2^k, 2^k) with the kept qubits in little-endian
    order, matching Qiskit's convention.
    """
    order = list(reversed(list(keep)))
    r = snaps[:, order[0]]
    for q in order[1:]:
        d = r.shape[1]
        r = np.einsum("mij,mkl->mikjl", r, snaps[:, q]).reshape(-1, d * 2, d * 2)
    return r


def build_witness(rho_ideal: np.ndarray, n_qubits: int, subsystem_a):
    """Optimal witness for the most entangled direction of an ideal state.

    Returns ``(W, lambda_min)`` with W = |phi><phi|^{T_A}, where phi is the
    eigenvector of rho^{T_A} of most negative eigenvalue.

    For any PPT state sigma, Tr(W sigma) = Tr(|phi><phi| sigma^{T_A}) >= 0,
    because sigma^{T_A} is then positive semidefinite. Every separable state
    is PPT, so Tr(W rho) < 0 certifies entanglement.

    The ideal state is used only to CHOOSE a good direction. The witness
    property holds regardless of what the device actually prepared, so a
    mismatch between simulation and hardware costs sensitivity, never
    validity. Use ``witness_validity_floor`` to check numerically.
    """
    pt = partial_transpose(rho_ideal, n_qubits, subsystem_a)
    pt = (pt + pt.conj().T) / 2
    eigenvalues, vectors = np.linalg.eigh(pt)
    phi = vectors[:, 0]
    w = partial_transpose(np.outer(phi, phi.conj()), n_qubits, subsystem_a)
    return w, float(eigenvalues[0])


def witness_validity_floor(w: np.ndarray, n_qubits: int, subsystem_a,
                           n_trials: int = 4000, n_terms: int = 3,
                           seed: int | None = 0) -> float:
    """Smallest Tr(W sigma) over random separable states.

    A valid witness cannot go below zero here. This is a numerical sanity
    check on the construction, not a proof; the proof is the PPT argument in
    ``build_witness``.
    """
    rng = np.random.default_rng(seed)
    dim_a = 2 ** len(list(subsystem_a))
    dim_b = 2 ** (n_qubits - len(list(subsystem_a)))
    worst = np.inf
    for _ in range(n_trials):
        rho = np.zeros((dim_a * dim_b, dim_a * dim_b), dtype=complex)
        for _term in range(n_terms):
            a = rng.normal(size=dim_a) + 1j * rng.normal(size=dim_a)
            b = rng.normal(size=dim_b) + 1j * rng.normal(size=dim_b)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            psi = np.kron(a, b)
            rho += np.outer(psi, psi.conj())
        rho /= np.trace(rho)
        worst = min(worst, float(np.real(np.trace(w @ rho))))
    return worst


def measure_witness(snaps: np.ndarray, keep, w: np.ndarray,
                    settings: np.ndarray | None = None,
                    confidence: float = 0.95, n_boot: int = 400,
                    seed: int | None = 0) -> dict:
    """Estimate Tr(W rho) from shadows, with a bootstrap interval.

    The estimator is linear in the snapshots and therefore unbiased, so
    unlike a reconstructed negativity it has no positive floor on separable
    states. Entanglement is certified when the whole interval lies below
    zero.
    """
    rho_hats = reduced_shadows(snaps, keep)
    per_shot = np.real(np.einsum("mij,ji->m", rho_hats, w))

    if settings is not None:
        # Shots sharing a random basis are correlated. Average within each
        # basis first, then bootstrap over bases, so the interval reflects
        # the number of independent settings rather than the shot count.
        settings = np.asarray(settings)
        labels, inverse = np.unique(settings, return_inverse=True)
        sums = np.bincount(inverse, weights=per_shot)
        counts = np.bincount(inverse)
        units = sums / counts
    else:
        units = per_shot

    value = float(units.mean())
    rng = np.random.default_rng(seed)
    n = units.size
    reps = units[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(reps, 100 * alpha))
    hi = float(np.percentile(reps, 100 * (1 - alpha)))
    return {
        "witness_value": value,
        "ci": (lo, hi),
        "se": float(reps.std(ddof=1)),
        "n_units": int(n),
        "entangled_certified": bool(hi < 0.0),
        "confidence": confidence,
        "note": ("Tr(W rho) < 0 certifies entanglement. A non-negative value "
                 "is inconclusive: this witness may simply not be aligned "
                 "with the state the device prepared."),
    }
