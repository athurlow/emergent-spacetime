#!/usr/bin/env python3
"""
ENTANGLEMENT MEASUREMENT VIA LOCAL RANDOMIZED MEASUREMENTS
Submits the randomized-measurement circuits to IBM hardware.
Andrew Thurlow | 528 Labs

WHY THIS EXPERIMENT EXISTS

Experiments 01-08 measure a single-basis ZZ correlator. That quantity
cannot tell entanglement from classical correlation. The tearing analysis
makes the gap concrete: the correlator falls 88% while the entanglement
across the chain cut is provably unchanged, because the decoupled stage is
a product of local unitaries.

This experiment measures entanglement itself. Every shot applies an
independent random Pauli-basis rotation to each qubit before the
computational-basis measurement. The resulting classical shadows give
Renyi-2 entropies and, more importantly, the partial-transpose moments
needed for the p3-PPT entanglement certificate, which remains valid when
the device state is mixed.

WHAT IS CERTIFIED

The certificate is an entanglement witness measured on a 2+2 reduced
state, two qubits from each chain. Entanglement in a reduced state implies
entanglement in the full state, since tracing out cannot create it.

Two simpler certificates were tried first and rejected, both recorded in
14_entanglement_validation.py:

  - The p3-PPT moment condition is exact but its shadow estimator has
    variance growing exponentially with qubit count. On the full 4+4 cut,
    10000 snapshots give a gap scattering by tens against an exact -0.84. On cuts small
    enough to converge, this state satisfies the condition while still
    being entangled, because p3-PPT is sufficient and never necessary.

  - Reconstructing the reduced state and taking its negativity converges at
    four qubits but is biased upward: projecting the noisy reconstruction
    back to a physical state manufactures negative eigenvalues, and a
    separable control reads 0.065 instead of zero.

The witness is linear, so its estimator is unbiased and has no such floor.
Run 14_entanglement_validation.py before booking hardware time; it prints
the data needed for a decision and verifies the witness never fires on a
separable state.

Run 14_entanglement_validation.py first: it reproduces this protocol in
simulation and reports the snapshot count needed before any hardware time
is spent.
"""

import json
from datetime import datetime

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# =============================================================================
# PASTE YOUR IBM API KEY BETWEEN THE QUOTES ON THE NEXT LINE
# =============================================================================
IBM_TOKEN = 'PASTE_YOUR_API_KEY_HERE'
# =============================================================================

BACKEND_NAME = 'ibm_torino'
N_CHAIN = 4
N_TOTAL = 2 * N_CHAIN
J_INTRA = 1.0
TROTTER_STEPS = 6
DT = 0.3

# Independent random bases, and shots per basis. Shots sharing a basis are
# correlated, so precision is governed by the basis count, not the product.
# Simulation of this exact protocol (Part 6 of 14_entanglement_validation.py)
# certifies the coupled state 4 times in 4 at 600 bases, but the torn state
# only 3 in 4; at 2000 bases both are reliable. The separable control is
# never certified at either.
N_SETTINGS = 2000
SHOTS_PER_SETTING = 64
SEED = 20260919

STATES = {
    'coupled': 1.0,      # inter-chain coupling on throughout
    'decoupled': 0.0,    # no inter-chain coupling at all
}


def build_two_chain(n_per_chain, j_intra, j_inter, steps, dt):
    """Same circuit as experiments 01-08."""
    qc = QuantumCircuit(2 * n_per_chain)
    for i in range(n_per_chain):
        qc.h(i)
    for _ in range(steps):
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


def build_torn(n_per_chain, j_intra, j_inter, total_steps, dt):
    """Couple for half the steps, then intra-chain evolution only."""
    half = total_steps // 2
    qc = build_two_chain(n_per_chain, j_intra, j_inter, half, dt)
    for _ in range(half):
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
    return qc


def apply_random_basis(qc, bases):
    """Rotate each qubit into its assigned Pauli basis, then measure Z.

    Basis 0 measures Z (no rotation), 1 measures X (H), 2 measures Y
    (S-dagger then H). This matches PAULI_ROTATIONS in
    src/analysis/shadow_entanglement.py.
    """
    out = qc.copy()
    for q, b in enumerate(bases):
        if b == 1:
            out.h(q)
        elif b == 2:
            out.sdg(q)
            out.h(q)
    out.measure_all()
    return out


def main():
    print("=" * 70)
    print("ENTANGLEMENT VIA LOCAL RANDOMIZED MEASUREMENTS")
    print("Andrew Thurlow | 528 Labs")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    service = QiskitRuntimeService(channel='ibm_quantum_platform',
                                   token=IBM_TOKEN)
    backend = service.backend(BACKEND_NAME)
    print(f"\nBackend: {backend.name}")

    base_circuits = {
        name: build_two_chain(N_CHAIN, J_INTRA, lam, TROTTER_STEPS, DT)
        for name, lam in STATES.items()
    }
    base_circuits['torn'] = build_torn(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)

    circuits, labels, settings = [], [], {}
    for name, base in base_circuits.items():
        bases = rng.integers(0, 3, size=(N_SETTINGS, N_TOTAL))
        settings[name] = bases.tolist()
        for k in range(N_SETTINGS):
            qc = apply_random_basis(base, bases[k])
            qc.name = f'{name}_setting_{k}'
            circuits.append(qc)
            labels.append(qc.name)

    print(f"\nStates: {list(base_circuits)}")
    print(f"Random bases per state: {N_SETTINGS}")
    print(f"Shots per basis: {SHOTS_PER_SETTING}")
    print(f"Total circuits: {len(circuits)}")
    print(f"Total shots: {len(circuits) * SHOTS_PER_SETTING}")

    print(f"\nTranspiling for {BACKEND_NAME} (optimization level 3)...")
    transpiled = transpile(circuits, backend=backend, optimization_level=3)
    depths = [qc.depth() for qc in transpiled]
    print(f"  depth: min {min(depths)}, max {max(depths)}")

    sampler = Sampler(backend)
    job_ids = {}
    batch_size = 100
    print(f"\nSubmitting in batches of {batch_size}...")
    for start in range(0, len(transpiled), batch_size):
        chunk = transpiled[start:start + batch_size]
        chunk_labels = labels[start:start + batch_size]
        job = sampler.run(chunk, shots=SHOTS_PER_SETTING)
        job_ids[f'batch_{start // batch_size}'] = {
            'job_id': job.job_id(),
            'labels': chunk_labels,
        }
        print(f"  batch {start // batch_size}: {job.job_id()} "
              f"({len(chunk)} circuits)")

    meta = {
        'timestamp': datetime.now().isoformat(),
        'backend': BACKEND_NAME,
        'experiment': 'Entanglement via local randomized measurements',
        'n_chain': N_CHAIN,
        'n_total_qubits': N_TOTAL,
        'trotter_steps': TROTTER_STEPS,
        'dt': DT,
        'n_settings': N_SETTINGS,
        'shots_per_setting': SHOTS_PER_SETTING,
        'seed': SEED,
        'basis_encoding': {'0': 'Z', '1': 'X', '2': 'Y'},
        'settings': settings,
        'job_ids': job_ids,
    }
    with open('entanglement_randomized_job_ids.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    print("\nSaved: entanglement_randomized_job_ids.json")
    print("Retrieve with: python src/analysis/09_entanglement_randomized_retrieve.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
