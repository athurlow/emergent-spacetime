#!/usr/bin/env python3
"""
EMERGENT SPACETIME — UNIVERSALITY TEST
Ising (ZZ-only) vs Heisenberg (ZZ+XX) Inter-Chain Coupling
4+4 Chain on IBM Torino | Lambda Sweep
Andrew Thurlow | 528 Labs | February 2026

PURPOSE: Test whether emergent geometry depends on the microscopic
Hamiltonian or is a universal property of coupled quantum fields.

If the lambda sweep curve is qualitatively the same for both Ising
and Heisenberg coupling, the geometric interpretation is robust.
If it fails, the geometry is an artifact of the specific Hamiltonian.

ARCHITECTURE:
  - Chain A (qubits 0-3): Heisenberg intra-chain (ZZ+XX)
  - Chain B (qubits 4-7): Heisenberg intra-chain (ZZ+XX)
  - Inter-chain coupling: ISING ONLY (ZZ) — this is the change
  - Lambda sweep: 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0
  - Also runs Heisenberg inter-chain at same lambdas for direct comparison
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

# Finer sweep including 0.1 to probe critical threshold
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

print("=" * 70)
print("EMERGENT SPACETIME — UNIVERSALITY TEST")
print("Ising vs Heisenberg Inter-Chain Coupling")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)
print(f"Backend: {BACKEND_NAME}")
print(f"Architecture: 4+4 chain, {TROTTER_STEPS} Trotter steps, dt={DT}")
print(f"Lambda values: {LAMBDAS}")
print(f"Shots: {SHOTS}")
print()

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_heisenberg_chain(lam, j_intra, steps, dt, n_chain, coupling_type='heisenberg'):
    """
    Build 4+4 chain circuit.
    Intra-chain: always Heisenberg (ZZ+XX)
    Inter-chain: 'heisenberg' (ZZ+XX) or 'ising' (ZZ only)
    """
    n_qubits = 2 * n_chain
    qc = QuantumCircuit(n_qubits)

    # Initialize chain A
    for i in range(n_chain):
        qc.h(i)
    qc.barrier()

    for step in range(steps):
        # Intra-chain A: ZZ + XX (always Heisenberg)
        for i in range(n_chain - 1):
            qi, qj = i, i + 1
            # ZZ
            qc.cx(qi, qj)
            qc.rz(2 * j_intra * dt, qj)
            qc.cx(qi, qj)
            # XX
            qc.h(qi); qc.h(qj)
            qc.cx(qi, qj)
            qc.rz(2 * j_intra * dt, qj)
            qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)

        # Intra-chain B: ZZ + XX (always Heisenberg)
        for i in range(n_chain - 1):
            qi, qj = n_chain + i, n_chain + i + 1
            # ZZ
            qc.cx(qi, qj)
            qc.rz(2 * j_intra * dt, qj)
            qc.cx(qi, qj)
            # XX
            qc.h(qi); qc.h(qj)
            qc.cx(qi, qj)
            qc.rz(2 * j_intra * dt, qj)
            qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)

        # Inter-chain coupling
        if lam > 0:
            for i in range(n_chain):
                qi, qj = i, n_chain + i
                # ZZ (both types have this)
                qc.cx(qi, qj)
                qc.rz(2 * lam * dt, qj)
                qc.cx(qi, qj)

                # XX (Heisenberg only)
                if coupling_type == 'heisenberg':
                    qc.h(qi); qc.h(qj)
                    qc.cx(qi, qj)
                    qc.rz(2 * lam * dt, qj)
                    qc.cx(qi, qj)
                    qc.h(qi); qc.h(qj)

        qc.barrier()

    qc.measure_all()
    return qc


def build_tearing(j_intra, steps, dt, n_chain, coupling_type='heisenberg'):
    """Coupled first half, uncoupled second half."""
    n_qubits = 2 * n_chain
    half1 = steps // 2
    half2 = steps - half1

    qc = QuantumCircuit(n_qubits)
    for i in range(n_chain):
        qc.h(i)
    qc.barrier()

    # First half: coupled at lambda=1.0
    for step in range(half1):
        for i in range(n_chain - 1):
            qi, qj = i, i + 1
            qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
            qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
        for i in range(n_chain - 1):
            qi, qj = n_chain+i, n_chain+i+1
            qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
            qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
        for i in range(n_chain):
            qi, qj = i, n_chain + i
            qc.cx(qi, qj); qc.rz(2*1.0*dt, qj); qc.cx(qi, qj)
            if coupling_type == 'heisenberg':
                qc.h(qi); qc.h(qj)
                qc.cx(qi, qj); qc.rz(2*1.0*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
        qc.barrier()

    # Second half: uncoupled
    for step in range(half2):
        for i in range(n_chain - 1):
            qi, qj = i, i + 1
            qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
            qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
        for i in range(n_chain - 1):
            qi, qj = n_chain+i, n_chain+i+1
            qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
            qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
            qc.h(qi); qc.h(qj)
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
print(f"Connected to {BACKEND_NAME}")
print()

circuits = []
labels = []

# Ising lambda sweep
print("Building ISING inter-chain circuits...")
for lam in LAMBDAS:
    qc = build_heisenberg_chain(lam, J_INTRA, TROTTER_STEPS, DT, N_CHAIN, coupling_type='ising')
    circuits.append(qc)
    labels.append(f'ising_lam_{lam}')
    print(f"  lambda={lam}: depth={qc.depth()}")

# Ising tearing
qc_ising_torn = build_tearing(J_INTRA, TROTTER_STEPS, DT, N_CHAIN, coupling_type='ising')
circuits.append(qc_ising_torn)
labels.append('ising_torn')
print(f"  tearing: depth={qc_ising_torn.depth()}")

# Heisenberg lambda sweep (for direct comparison on same job)
print("\nBuilding HEISENBERG inter-chain circuits...")
for lam in LAMBDAS:
    qc = build_heisenberg_chain(lam, J_INTRA, TROTTER_STEPS, DT, N_CHAIN, coupling_type='heisenberg')
    circuits.append(qc)
    labels.append(f'heis_lam_{lam}')
    print(f"  lambda={lam}: depth={qc.depth()}")

# Heisenberg tearing
qc_heis_torn = build_tearing(J_INTRA, TROTTER_STEPS, DT, N_CHAIN, coupling_type='heisenberg')
circuits.append(qc_heis_torn)
labels.append('heis_torn')
print(f"  tearing: depth={qc_heis_torn.depth()}")

print(f"\nTotal circuits: {len(circuits)}")

# Transpile
print(f"\nTranspiling for {BACKEND_NAME} (optimization level 3)...")
transpiled = transpile(circuits, backend=backend, optimization_level=3)

for i, (qc, label) in enumerate(zip(transpiled, labels)):
    ops = qc.count_ops()
    total = sum(ops.values())
    if 'lam_1.0' in label or 'torn' in label:
        print(f"  {label}: depth={qc.depth()}, gates={total}")

ising_depths = [qc.depth() for qc, l in zip(transpiled, labels) if l.startswith('ising')]
heis_depths = [qc.depth() for qc, l in zip(transpiled, labels) if l.startswith('heis')]
print(f"\n  Ising max depth: {max(ising_depths)}")
print(f"  Heisenberg max depth: {max(heis_depths)}")
print(f"  Ising should be SHALLOWER (fewer gates per step)")

# Submit
print(f"\nSubmitting {len(transpiled)} circuits ({SHOTS} shots each)...")
sampler = Sampler(backend)
job_ids = {}

# Submit in batches to avoid timeout
batch_size = 9
for start in range(0, len(transpiled), batch_size):
    end = min(start + batch_size, len(transpiled))
    batch_circuits = transpiled[start:end]
    batch_labels = labels[start:end]
    batch_name = f"batch_{start//batch_size}"

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
    'experiment': 'Universality test: Ising vs Heisenberg inter-chain coupling',
    'backend': BACKEND_NAME,
    'n_chain': N_CHAIN,
    'trotter_steps': TROTTER_STEPS,
    'dt': DT,
    'lambdas': LAMBDAS,
    'shots': SHOTS,
    'labels': labels,
    'job_ids': job_ids,
}

with open('universality_test_job_ids.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print("UNIVERSALITY TEST SUBMITTED")
print("=" * 70)
print(f"  {len(LAMBDAS)} lambda values × 2 coupling types = {len(LAMBDAS)*2} sweep circuits")
print(f"  + 2 tearing circuits (one per coupling type)")
print(f"  = {len(circuits)} total circuits")
print(f"  Job IDs saved to universality_test_job_ids.json")
print(f"  Monitor at: https://quantum.ibm.com/jobs")
print(f"\n  If the Ising curve matches the Heisenberg curve,")
print(f"  emergent geometry is UNIVERSAL — not Hamiltonian-specific.")
print("=" * 70)
