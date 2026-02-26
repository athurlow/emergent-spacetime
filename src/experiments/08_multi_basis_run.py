#!/usr/bin/env python3
"""
EMERGENT SPACETIME — MULTI-BASIS CORRELATION ANALYSIS
Reanalyze all four Hamiltonians using ZZ, XX, and YY correlation metrics
Andrew Thurlow | 528 Labs | February 2026

KEY INSIGHT: If XY coupling produces geometry visible in XX/YY correlations
but not ZZ, then the geometry is ALWAYS present — the measurement basis
just needs to match the coupling basis. That's deeper universality.

NOTE: Standard measurement is in the Z basis. To extract XX correlations
from Z-basis measurements, we need to submit new circuits with basis
rotation gates (H before measurement for X basis, S†H for Y basis).
However, we CAN extract some information from the existing data.

This script:
1. Reanalyzes existing data with ZZ correlations (baseline)
2. Submits NEW circuits for X-basis and Y-basis measurement
   of XY and Ising Hamiltonians for direct comparison
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
N_QUBITS = 8
TROTTER_STEPS = 6
DT = 0.3
J_INTRA = 1.0
SHOTS = 8192
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

print("=" * 70)
print("EMERGENT SPACETIME — MULTI-BASIS MEASUREMENT")
print("Testing: Does XY geometry appear in XX correlations?")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def apply_intra_chain(qc, chain_start, n_chain, j, dt):
    """Heisenberg intra-chain (always the same)."""
    for i in range(n_chain - 1):
        qi, qj = chain_start + i, chain_start + i + 1
        qc.cx(qi, qj)
        qc.rz(2 * j * dt, qj)
        qc.cx(qi, qj)
        qc.h(qi); qc.h(qj)
        qc.cx(qi, qj)
        qc.rz(2 * j * dt, qj)
        qc.cx(qi, qj)
        qc.h(qi); qc.h(qj)


def apply_inter_xy(qc, n_chain, lam, dt):
    """XY inter-chain: XX + YY coupling."""
    if lam <= 0:
        return
    for i in range(n_chain):
        qi, qj = i, n_chain + i
        # XX
        qc.h(qi); qc.h(qj)
        qc.cx(qi, qj)
        qc.rz(2 * lam * dt, qj)
        qc.cx(qi, qj)
        qc.h(qi); qc.h(qj)
        # YY
        qc.sdg(qi); qc.sdg(qj)
        qc.h(qi); qc.h(qj)
        qc.cx(qi, qj)
        qc.rz(2 * lam * dt, qj)
        qc.cx(qi, qj)
        qc.h(qi); qc.h(qj)
        qc.s(qi); qc.s(qj)


def apply_inter_ising(qc, n_chain, lam, dt):
    """Ising inter-chain: ZZ only."""
    if lam <= 0:
        return
    for i in range(n_chain):
        qi, qj = i, n_chain + i
        qc.cx(qi, qj)
        qc.rz(2 * lam * dt, qj)
        qc.cx(qi, qj)


def build_circuit(lam, coupling_type, measurement_basis):
    """
    Build 4+4 chain with specified coupling and measurement basis.
    
    coupling_type: 'xy' or 'ising'
    measurement_basis: 'Z', 'X', or 'Y'
    """
    qc = QuantumCircuit(N_QUBITS)

    # Initialize chain A in superposition
    for i in range(N_CHAIN):
        qc.h(i)
    qc.barrier()

    for step in range(TROTTER_STEPS):
        apply_intra_chain(qc, 0, N_CHAIN, J_INTRA, DT)
        apply_intra_chain(qc, N_CHAIN, N_CHAIN, J_INTRA, DT)
        if coupling_type == 'xy':
            apply_inter_xy(qc, N_CHAIN, lam, DT)
        elif coupling_type == 'ising':
            apply_inter_ising(qc, N_CHAIN, lam, DT)
        qc.barrier()

    # Basis rotation before measurement
    if measurement_basis == 'X':
        # Rotate X -> Z: apply H to all qubits
        for i in range(N_QUBITS):
            qc.h(i)
    elif measurement_basis == 'Y':
        # Rotate Y -> Z: apply S†H to all qubits
        for i in range(N_QUBITS):
            qc.sdg(i)
            qc.h(i)
    # Z basis: no rotation needed

    qc.measure_all()
    return qc


# =============================================================================
# BUILD ALL CIRCUITS
# =============================================================================

print("\nBuilding circuits...")
circuits = []
labels = []

# XY coupling measured in Z, X, and Y bases
for basis in ['Z', 'X', 'Y']:
    for lam in LAMBDAS:
        qc = build_circuit(lam, 'xy', basis)
        circuits.append(qc)
        labels.append(f'xy_{basis}_lam_{lam}')
    print(f"  XY coupling, {basis}-basis: {len(LAMBDAS)} circuits, depth={circuits[-1].depth()}")

# Ising coupling measured in Z, X, and Y bases (for comparison)
for basis in ['Z', 'X', 'Y']:
    for lam in LAMBDAS:
        qc = build_circuit(lam, 'ising', basis)
        circuits.append(qc)
        labels.append(f'ising_{basis}_lam_{lam}')
    print(f"  Ising coupling, {basis}-basis: {len(LAMBDAS)} circuits, depth={circuits[-1].depth()}")

print(f"\nTotal circuits: {len(circuits)}")

# =============================================================================
# CONNECT AND SUBMIT
# =============================================================================

print(f"\nConnecting to IBM Quantum...")
service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)
backend = service.backend(BACKEND_NAME)
print(f"Connected to {BACKEND_NAME}")

print(f"\nTranspiling (optimization level 3)...")
transpiled = transpile(circuits, backend=backend, optimization_level=3)

max_depth = max(qc.depth() for qc in transpiled)
print(f"  Max transpiled depth: {max_depth}")

print(f"\nSubmitting {len(transpiled)} circuits ({SHOTS} shots each)...")
sampler = Sampler(backend)
job_ids = {}

batch_size = 8
for start in range(0, len(transpiled), batch_size):
    end = min(start + batch_size, len(transpiled))
    batch_circuits = transpiled[start:end]
    batch_labels = labels[start:end]
    batch_name = f"batch_{start // batch_size}"

    print(f"  {batch_name} ({len(batch_circuits)} circuits)...", end=" ", flush=True)
    job = sampler.run(batch_circuits, shots=SHOTS)
    jid = job.job_id()
    job_ids[batch_name] = {
        'job_id': jid,
        'labels': batch_labels,
    }
    print(f"job_id = {jid}")

# Save
output = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'Multi-basis measurement: XY and Ising in Z/X/Y bases',
    'backend': BACKEND_NAME,
    'n_chain': N_CHAIN,
    'trotter_steps': TROTTER_STEPS,
    'dt': DT,
    'lambdas': LAMBDAS,
    'shots': SHOTS,
    'labels': labels,
    'job_ids': job_ids,
}

with open('multi_basis_job_ids.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print("MULTI-BASIS MEASUREMENT SUBMITTED")
print("=" * 70)
print(f"  Circuits submitted: {len(circuits)}")
print(f"  XY coupling:    Z-basis, X-basis, Y-basis ({len(LAMBDAS)} λ values each)")
print(f"  Ising coupling:  Z-basis, X-basis, Y-basis ({len(LAMBDAS)} λ values each)")
print(f"")
print(f"  PREDICTION:")
print(f"    Ising (ZZ coupling):  geometry appears in Z-basis measurement")
print(f"    XY (XX+YY coupling):  geometry appears in X-basis measurement")
print(f"")
print(f"  If confirmed, the geometry is ALWAYS present — the measurement")
print(f"  basis just needs to match the coupling basis. That's basis-rotated")
print(f"  universality: deeper than Hamiltonian universality.")
print(f"")
print(f"  Job IDs saved to multi_basis_job_ids.json")
print("=" * 70)
