#!/usr/bin/env python3
"""
IBM QUANTUM HARDWARE SUBMISSION
Emergent Spacetime — Two Scalar Field Model
Andrew Thurlow | 528 Labs | February 2026

Submits 12 circuits (4 experiments x 3 measurement bases) to IBM quantum hardware.
Validated on IBM Torino (133-qubit Heron processor).

USAGE:
  1. Install: pip install qiskit qiskit-aer qiskit-ibm-runtime
  2. Save your IBM credentials:
     python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='YOUR_TOKEN', overwrite=True)"
  3. Run: python ibm_hardware_run.py
"""

import numpy as np
import json
import sys
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# =============================================================================
# CONFIGURATION
# =============================================================================

BACKEND_NAME = 'ibm_torino'  # Also validated on ibm_fez
N_CHAIN = 4                  # 4+4 = 8 qubits
TROTTER_STEPS = 6
DT = 0.3
J_INTRA = 1.0
SHOTS = 8192

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_two_chain(n_per_chain, j_intra, j_inter, steps, dt):
    """Build two coupled qubit chains with ZZ+XX intra-chain and ZZ inter-chain coupling."""
    n_total = 2 * n_per_chain
    qc = QuantumCircuit(n_total)
    for i in range(n_per_chain):
        qc.h(i)
    qc.barrier()
    for step in range(steps):
        for i in range(n_per_chain - 1):
            qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1)
            qc.h(i); qc.h(i+1); qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1); qc.h(i); qc.h(i+1)
        for i in range(n_per_chain, 2*n_per_chain - 1):
            qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1)
            qc.h(i); qc.h(i+1); qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1); qc.h(i); qc.h(i+1)
        if j_inter > 0:
            for i in range(n_per_chain):
                j = i + n_per_chain
                qc.cx(i, j); qc.rz(2*j_inter*dt, j); qc.cx(i, j)
        qc.barrier()
    return qc

def build_single_chain(n_qubits, j_intra, steps, dt):
    """Build single chain (null hypothesis control)."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits // 2):
        qc.h(i)
    qc.barrier()
    for step in range(steps):
        for i in range(n_qubits - 1):
            qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1)
            qc.h(i); qc.h(i+1); qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1); qc.h(i); qc.h(i+1)
        qc.barrier()
    return qc

def build_torn(n_per_chain, j_intra, j_inter, total_steps, dt):
    """Build spacetime tearing experiment: coupled for half, then uncoupled."""
    half = total_steps // 2
    qc = build_two_chain(n_per_chain, j_intra, j_inter, half, dt)
    for step in range(half):
        for i in range(n_per_chain - 1):
            qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1)
            qc.h(i); qc.h(i+1); qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1); qc.h(i); qc.h(i+1)
        for i in range(n_per_chain, 2*n_per_chain - 1):
            qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1)
            qc.h(i); qc.h(i+1); qc.cx(i, i+1); qc.rz(2*j_intra*dt, i+1); qc.cx(i, i+1); qc.h(i); qc.h(i+1)
        qc.barrier()
    return qc

def add_measurement(qc, basis='Z'):
    """Add measurement in specified Pauli basis."""
    qc_m = qc.copy()
    n = qc.num_qubits
    if basis == 'X':
        for i in range(n):
            qc_m.h(i)
    elif basis == 'Y':
        for i in range(n):
            qc_m.sdg(i)
            qc_m.h(i)
    qc_m.measure_all()
    return qc_m

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("EMERGENT SPACETIME — IBM HARDWARE SUBMISSION")
    print("Andrew Thurlow | 528 Labs")
    print("=" * 60)

    print(f"\nConnecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    print(f"Connected to {BACKEND_NAME}")
    print(f"Backend qubits: {backend.num_qubits}")

    print(f"\nBuilding circuits (8 qubits)...")
    n_total = 2 * N_CHAIN
    circuits = []
    labels = []

    qc_coupled = build_two_chain(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement(qc_coupled, basis)
        qc.name = f'coupled_1.0_{basis}'
        circuits.append(qc); labels.append(qc.name)

    qc_uncoupled = build_two_chain(N_CHAIN, J_INTRA, 0.0, TROTTER_STEPS, DT)
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement(qc_uncoupled, basis)
        qc.name = f'coupled_0.0_{basis}'
        circuits.append(qc); labels.append(qc.name)

    qc_single = build_single_chain(n_total, J_INTRA, TROTTER_STEPS, DT)
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement(qc_single, basis)
        qc.name = f'single_{basis}'
        circuits.append(qc); labels.append(qc.name)

    qc_torn = build_torn(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement(qc_torn, basis)
        qc.name = f'torn_{basis}'
        circuits.append(qc); labels.append(qc.name)

    print(f"Total circuits: {len(circuits)}")

    print(f"\nTranspiling for {BACKEND_NAME} (optimization level 3)...")
    transpiled = transpile(circuits, backend=backend, optimization_level=3)
    for i, qc in enumerate(transpiled):
        cx_count = qc.count_ops().get('cx', 0) + qc.count_ops().get('ecr', 0)
        print(f"  {labels[i]}: depth={qc.depth()}, two-qubit gates={cx_count}")

    print(f"\nSubmitting {len(transpiled)} circuits ({SHOTS} shots each)...")
    sampler = Sampler(backend)
    job_ids = {}
    for circuit, label in zip(transpiled, labels):
        print(f"  Submitting {label}...", end=" ", flush=True)
        job = sampler.run([circuit], shots=SHOTS)
        job_ids[label] = job.job_id()
        print(f"job_id = {job.job_id()}")

    output = {
        'timestamp': datetime.now().isoformat(),
        'backend': BACKEND_NAME,
        'shots': SHOTS,
        'n_chain': N_CHAIN,
        'job_ids': job_ids,
    }
    with open('ibm_job_ids.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ALL JOBS SUBMITTED SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"Job IDs saved to ibm_job_ids.json")
    print(f"Retrieve results with: python ibm_retrieve.py")
