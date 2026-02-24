#!/usr/bin/env python3
"""
IBM QUANTUM HARDWARE SUBMISSION — READY TO RUN
Emergent Spacetime — Two Scalar Field Model
Andrew Thurlow | 528 Labs

USAGE:
  1. In PowerShell, set your token:
     $env:IBM_QUANTUM_TOKEN = "your_token_here"

  2. Run:
     python ibm_submit_live.py
"""

import numpy as np
import json
import os
import sys
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# =============================================================================
# CONFIG
# =============================================================================

IBM_TOKEN = os.environ.get('IBM_QUANTUM_TOKEN', '')
BACKEND_NAME = 'ibm_brisbane'
N_CHAIN = 4  # 4+4 = 8 qubits
TROTTER_STEPS = 6
DT = 0.3
J_INTRA = 1.0
SHOTS = 8192

if not IBM_TOKEN:
    print("ERROR: No IBM token found.")
    print("Set it first in PowerShell:")
    print('  $env:IBM_QUANTUM_TOKEN = "your_token_here"')
    sys.exit(1)

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_two_chain(n_per_chain, j_intra, j_inter, steps, dt):
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
# CONNECT AND SUBMIT
# =============================================================================

print("=" * 60)
print("EMERGENT SPACETIME — IBM HARDWARE SUBMISSION")
print("Andrew Thurlow | 528 Labs")
print("=" * 60)

print(f"\nConnecting to IBM Quantum...")
service = QiskitRuntimeService(
    channel='ibm_quantum',
    token=IBM_TOKEN
)
backend = service.backend(BACKEND_NAME)
print(f"Connected to {BACKEND_NAME}")
print(f"Backend qubits: {backend.num_qubits}")
print(f"Pending jobs: {backend.status().pending_jobs}")

# Build all circuits
print(f"\nBuilding circuits (8 qubits)...")
n_total = 2 * N_CHAIN

circuits = []
labels = []

# Experiment 1: Coupled (lambda=1.0)
qc_coupled = build_two_chain(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)
for basis in ['Z', 'X', 'Y']:
    qc = add_measurement(qc_coupled, basis)
    qc.name = f'coupled_1.0_{basis}'
    circuits.append(qc)
    labels.append(qc.name)

# Experiment 1: Uncoupled (lambda=0.0)
qc_uncoupled = build_two_chain(N_CHAIN, J_INTRA, 0.0, TROTTER_STEPS, DT)
for basis in ['Z', 'X', 'Y']:
    qc = add_measurement(qc_uncoupled, basis)
    qc.name = f'coupled_0.0_{basis}'
    circuits.append(qc)
    labels.append(qc.name)

# Experiment 2: Single chain (null hypothesis)
qc_single = build_single_chain(n_total, J_INTRA, TROTTER_STEPS, DT)
for basis in ['Z', 'X', 'Y']:
    qc = add_measurement(qc_single, basis)
    qc.name = f'single_{basis}'
    circuits.append(qc)
    labels.append(qc.name)

# Experiment 3: Torn spacetime
qc_torn = build_torn(N_CHAIN, J_INTRA, 1.0, TROTTER_STEPS, DT)
for basis in ['Z', 'X', 'Y']:
    qc = add_measurement(qc_torn, basis)
    qc.name = f'torn_{basis}'
    circuits.append(qc)
    labels.append(qc.name)

print(f"Total circuits: {len(circuits)}")

# Transpile for hardware
print(f"\nTranspiling for {BACKEND_NAME} (optimization level 3)...")
transpiled = transpile(
    circuits,
    backend=backend,
    optimization_level=3,
)

for i, qc in enumerate(transpiled):
    cx_count = qc.count_ops().get('cx', 0) + qc.count_ops().get('ecr', 0)
    print(f"  {labels[i]}: depth={qc.depth()}, two-qubit gates={cx_count}")

# Submit jobs
print(f"\nSubmitting {len(transpiled)} circuits...")
print(f"Shots per circuit: {SHOTS}")

sampler = Sampler(backend)

job_ids = {}
for i, (circuit, label) in enumerate(zip(transpiled, labels)):
    print(f"  Submitting {label}...", end=" ", flush=True)
    job = sampler.run([circuit], shots=SHOTS)
    job_ids[label] = job.job_id()
    print(f"job_id = {job.job_id()}")

# Save job IDs
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
print(f"Jobs in queue: {len(job_ids)}")
print(f"\nTo retrieve results later, run:")
print(f"  python ibm_retrieve.py")
print(f"\nYou can close this window. Jobs will run on IBM's servers.")
print(f"{'=' * 60}")
