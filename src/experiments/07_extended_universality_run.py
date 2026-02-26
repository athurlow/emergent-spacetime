#!/usr/bin/env python3
"""
EMERGENT SPACETIME — EXTENDED UNIVERSALITY TEST
Four Hamiltonians: Ising (ZZ), Heisenberg (ZZ+XX), XY (XX+YY), Long-Range
4+4 Chain on IBM Torino | Lambda Sweep
Andrew Thurlow | 528 Labs | February 2026

PURPOSE: Extend universality test from 2 to 4 coupling types.
If all four produce the same emergent geometry curve shape,
universality is established beyond reasonable doubt.

COUPLING TYPES:
  1. Ising: ZZ only (already validated)
  2. Heisenberg: ZZ + XX (already validated)
  3. XY: XX + YY (new - tests non-diagonal coupling)
  4. Long-range: ZZ between ALL cross-chain pairs (new - tests spatial structure)
"""

import numpy as np
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# =============================================================================
IBM_TOKEN = 'PASTE_YOUR_API_KEY_HERE'
# =============================================================================

BACKEND_NAME = 'ibm_torino'
N_CHAIN = 4
N_QUBITS = 2 * N_CHAIN
TROTTER_STEPS = 6
DT = 0.3
J_INTRA = 1.0
SHOTS = 8192

LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

COUPLING_TYPES = ['xy', 'long_range']  # ising and heisenberg already done

print("=" * 70)
print("EMERGENT SPACETIME — EXTENDED UNIVERSALITY TEST")
print("XY + Long-Range Inter-Chain Coupling")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)
print(f"Backend: {BACKEND_NAME}")
print(f"Architecture: 4+4 chain, {TROTTER_STEPS} Trotter steps, dt={DT}")
print(f"Lambda values: {LAMBDAS}")
print(f"Coupling types: {COUPLING_TYPES}")
print(f"Shots: {SHOTS}")
print()

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def apply_intra_chain(qc, chain_start, n_chain, j, dt):
    """Heisenberg intra-chain coupling (always the same)."""
    for i in range(n_chain - 1):
        qi, qj = chain_start + i, chain_start + i + 1
        # ZZ
        qc.cx(qi, qj)
        qc.rz(2 * j * dt, qj)
        qc.cx(qi, qj)
        # XX
        qc.h(qi); qc.h(qj)
        qc.cx(qi, qj)
        qc.rz(2 * j * dt, qj)
        qc.cx(qi, qj)
        qc.h(qi); qc.h(qj)


def apply_inter_chain(qc, n_chain, lam, dt, coupling_type):
    """Apply inter-chain coupling of the specified type."""
    if lam <= 0:
        return

    if coupling_type == 'xy':
        # XX + YY coupling (no ZZ)
        for i in range(n_chain):
            qi, qj = i, n_chain + i
            # XX
            qc.h(qi); qc.h(qj)
            qc.cx(qi, qj)
            qc.rz(2 * lam * dt, qj)
            qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
            # YY: Ry(-pi/2) to rotate Y->Z, then ZZ, then Ry(pi/2) back
            qc.sdg(qi); qc.sdg(qj)
            qc.h(qi); qc.h(qj)
            qc.cx(qi, qj)
            qc.rz(2 * lam * dt, qj)
            qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
            qc.s(qi); qc.s(qj)

    elif coupling_type == 'long_range':
        # ZZ between ALL cross-chain pairs (not just corresponding sites)
        for i in range(n_chain):
            for j_idx in range(n_chain):
                qi, qj = i, n_chain + j_idx
                # Scale by 1/n_chain to keep total coupling comparable
                scaled_lam = lam / n_chain
                qc.cx(qi, qj)
                qc.rz(2 * scaled_lam * dt, qj)
                qc.cx(qi, qj)


def build_sweep_circuit(lam, coupling_type):
    """Build full 4+4 chain with specified coupling."""
    qc = QuantumCircuit(N_QUBITS)

    # Initialize chain A
    for i in range(N_CHAIN):
        qc.h(i)
    qc.barrier()

    for step in range(TROTTER_STEPS):
        # Intra-chain A
        apply_intra_chain(qc, 0, N_CHAIN, J_INTRA, DT)
        # Intra-chain B
        apply_intra_chain(qc, N_CHAIN, N_CHAIN, J_INTRA, DT)
        # Inter-chain
        apply_inter_chain(qc, N_CHAIN, lam, DT, coupling_type)
        qc.barrier()

    qc.measure_all()
    return qc


def build_tearing_circuit(coupling_type):
    """Coupled first half, uncoupled second half."""
    qc = QuantumCircuit(N_QUBITS)

    for i in range(N_CHAIN):
        qc.h(i)
    qc.barrier()

    half1 = TROTTER_STEPS // 2
    half2 = TROTTER_STEPS - half1

    # First half: coupled at lambda=1.0
    for step in range(half1):
        apply_intra_chain(qc, 0, N_CHAIN, J_INTRA, DT)
        apply_intra_chain(qc, N_CHAIN, N_CHAIN, J_INTRA, DT)
        apply_inter_chain(qc, N_CHAIN, 1.0, DT, coupling_type)
        qc.barrier()

    # Second half: uncoupled
    for step in range(half2):
        apply_intra_chain(qc, 0, N_CHAIN, J_INTRA, DT)
        apply_intra_chain(qc, N_CHAIN, N_CHAIN, J_INTRA, DT)
        qc.barrier()

    qc.measure_all()
    return qc


# =============================================================================
# CONNECT AND BUILD
# =============================================================================

print("Connecting to IBM Quantum...")
service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)
backend = service.backend(BACKEND_NAME)
print(f"Connected to {BACKEND_NAME}\n")

circuits = []
labels = []

for ctype in COUPLING_TYPES:
    print(f"Building {ctype.upper()} inter-chain circuits...")
    for lam in LAMBDAS:
        qc = build_sweep_circuit(lam, ctype)
        circuits.append(qc)
        labels.append(f'{ctype}_lam_{lam}')
        print(f"  lambda={lam}: depth={qc.depth()}")

    # Tearing circuit
    qc_torn = build_tearing_circuit(ctype)
    circuits.append(qc_torn)
    labels.append(f'{ctype}_torn')
    print(f"  tearing: depth={qc_torn.depth()}")
    print()

print(f"Total circuits: {len(circuits)}")

# Transpile
print(f"\nTranspiling for {BACKEND_NAME} (optimization level 3)...")
transpiled = transpile(circuits, backend=backend, optimization_level=3)

for ctype in COUPLING_TYPES:
    depths = [qc.depth() for qc, l in zip(transpiled, labels) if l.startswith(ctype)]
    print(f"  {ctype} max transpiled depth: {max(depths)}")

# Submit
print(f"\nSubmitting {len(transpiled)} circuits ({SHOTS} shots each)...")
sampler = Sampler(backend)
job_ids = {}

batch_size = 9
for start in range(0, len(transpiled), batch_size):
    end = min(start + batch_size, len(transpiled))
    batch_circuits = transpiled[start:end]
    batch_labels = labels[start:end]
    batch_name = f"batch_{start // batch_size}"

    print(f"  Submitting {batch_name} ({len(batch_circuits)} circuits)...", end=" ", flush=True)
    job = sampler.run(batch_circuits, shots=SHOTS)
    jid = job.job_id()
    job_ids[batch_name] = {
        'job_id': jid,
        'labels': batch_labels,
        'indices': list(range(start, end))
    }
    print(f"job_id = {jid}")

# Save
output = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'Extended universality test: XY + Long-range coupling',
    'backend': BACKEND_NAME,
    'n_chain': N_CHAIN,
    'trotter_steps': TROTTER_STEPS,
    'dt': DT,
    'lambdas': LAMBDAS,
    'shots': SHOTS,
    'coupling_types': COUPLING_TYPES,
    'labels': labels,
    'job_ids': job_ids,
}

with open('extended_universality_job_ids.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print("EXTENDED UNIVERSALITY TEST SUBMITTED")
print("=" * 70)
print(f"  {len(LAMBDAS)} lambda values x {len(COUPLING_TYPES)} coupling types = {len(LAMBDAS) * len(COUPLING_TYPES)} sweep circuits")
print(f"  + {len(COUPLING_TYPES)} tearing circuits")
print(f"  = {len(circuits)} total circuits")
print(f"\n  Coupling types tested:")
print(f"    XY (XX+YY):      Tests non-diagonal entangling interaction")
print(f"    Long-range (ZZ): Tests all-to-all vs nearest-neighbor coupling")
print(f"\n  Combined with previous results (Ising, Heisenberg),")
print(f"  this gives FOUR Hamiltonians testing universality.")
print(f"\n  Job IDs saved to extended_universality_job_ids.json")
print(f"  Monitor at: https://quantum.ibm.com/jobs")
print("=" * 70)
