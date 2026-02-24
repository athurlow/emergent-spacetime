#!/usr/bin/env python3
"""
EMERGENT SPACETIME — 1D CHAIN (8+8)
16 Qubits on IBM Torino
Andrew Thurlow | 528 Labs | February 2026

8+8 chain: simplest topology, best hardware mapping.
Linear chains map well to Torino's physical qubit layout,
minimizing swap overhead and keeping circuit depth shallow.

This should produce the cleanest 16-qubit results.
"""

import numpy as np
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# =============================================================================
# PASTE YOUR IBM API KEY BETWEEN THE QUOTES ON THE NEXT LINE
# =============================================================================
IBM_TOKEN = 'PASTE_YOUR_API_KEY_HERE'
# =============================================================================

BACKEND_NAME = 'ibm_torino'
N_CHAIN = 8
N_TOTAL = 2 * N_CHAIN  # 16
TROTTER_STEPS = 6
DT = 0.3
J_INTRA = 1.0
SHOTS = 8192

LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

print("=" * 70)
print("EMERGENT SPACETIME — 1D CHAIN (8+8)")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)
print(f"Backend: {BACKEND_NAME}")
print(f"Chain: {N_CHAIN}+{N_CHAIN} = {N_TOTAL} qubits")
print(f"Trotter steps: {TROTTER_STEPS}, dt = {DT}")
print(f"Lambda sweep: {LAMBDA_VALUES}")
print(f"Shots: {SHOTS}")
print()

# =============================================================================
# TOPOLOGY
#
#  Chain A: q0 -- q1 -- q2 -- q3 -- q4 -- q5 -- q6 -- q7
#           |     |     |     |     |     |     |     |
#  Chain B: q8 -- q9 -- q10-- q11-- q12-- q13-- q14-- q15
#
# =============================================================================

def get_intra_pairs(n):
    return [(i, i+1) for i in range(n-1)]

def get_inter_pairs(n):
    return [(i, i+n) for i in range(n)]

intra = get_intra_pairs(N_CHAIN)
inter = get_inter_pairs(N_CHAIN)
print(f"Intra-chain pairs per chain: {len(intra)} (linear)")
print(f"Inter-chain pairs: {len(inter)}")
print()

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_coupled(j_intra, j_inter, steps, dt):
    qc = QuantumCircuit(N_TOTAL)
    for i in range(N_CHAIN):
        qc.h(i)
    qc.barrier()

    intra_A = get_intra_pairs(N_CHAIN)
    intra_B = [(a + N_CHAIN, b + N_CHAIN) for a, b in intra_A]
    inter = get_inter_pairs(N_CHAIN)

    for step in range(steps):
        # Intra A: ZZ + XX
        for i, j in intra_A:
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
        # Intra B: ZZ + XX
        for i, j in intra_B:
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
        # Inter: ZZ
        if j_inter > 0:
            for i, j in inter:
                qc.cx(i, j); qc.rz(2*j_inter*dt, j); qc.cx(i, j)
        qc.barrier()
    return qc

def build_single(j_intra, steps, dt):
    """Single chain of 16 qubits (null hypothesis)."""
    qc = QuantumCircuit(N_TOTAL)
    for i in range(N_TOTAL // 2):
        qc.h(i)
    qc.barrier()
    pairs = [(i, i+1) for i in range(N_TOTAL - 1)]
    for step in range(steps):
        for i, j in pairs:
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
        qc.barrier()
    return qc

def build_torn(j_intra, j_inter, steps, dt):
    """Coupled first half, uncoupled second half."""
    half1 = steps // 2
    half2 = steps - half1

    intra_A = get_intra_pairs(N_CHAIN)
    intra_B = [(a + N_CHAIN, b + N_CHAIN) for a, b in intra_A]

    qc = build_coupled(j_intra, j_inter, half1, dt)

    for step in range(half2):
        for i, j in intra_A:
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
        for i, j in intra_B:
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
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
# CONNECT AND BUILD
# =============================================================================

print("Connecting to IBM Quantum...")
service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)
backend = service.backend(BACKEND_NAME)
print(f"Connected to {BACKEND_NAME} ({backend.num_qubits} qubits)")
print()

circuits = []
labels = []

# Lambda sweep
print("Building lambda sweep circuits...")
for lam in LAMBDA_VALUES:
    qc = build_coupled(J_INTRA, lam, TROTTER_STEPS, DT)
    qc_z = add_measurement(qc, 'Z')
    qc_z.name = f'coupled_{lam}_Z'
    circuits.append(qc_z)
    labels.append(qc_z.name)
    print(f"  lambda={lam}: depth={qc.depth()}")

# Null hypothesis
print("Building null hypothesis...")
qc_single = build_single(J_INTRA, TROTTER_STEPS, DT)
qc_single_z = add_measurement(qc_single, 'Z')
qc_single_z.name = 'single_Z'
circuits.append(qc_single_z)
labels.append(qc_single_z.name)
print(f"  single: depth={qc_single.depth()}")

# Tearing
print("Building tearing circuit...")
qc_torn = build_torn(J_INTRA, 1.0, TROTTER_STEPS, DT)
qc_torn_z = add_measurement(qc_torn, 'Z')
qc_torn_z.name = 'torn_Z'
circuits.append(qc_torn_z)
labels.append(qc_torn_z.name)
print(f"  torn: depth={qc_torn.depth()}")

print(f"\nTotal circuits: {len(circuits)}")

# Transpile
print(f"\nTranspiling for {BACKEND_NAME} (optimization level 3)...")
transpiled = transpile(circuits, backend=backend, optimization_level=3)
for i, qc in enumerate(transpiled):
    ops = qc.count_ops()
    cx_count = ops.get('cx', 0) + ops.get('ecr', 0)
    print(f"  {labels[i]}: depth={qc.depth()}, two-qubit gates={cx_count}")

max_depth = max(qc.depth() for qc in transpiled)
avg_depth = np.mean([qc.depth() for qc in transpiled])
print(f"\n  Max depth: {max_depth}")
print(f"  Avg depth: {avg_depth:.0f}")
if max_depth < 500:
    print(f"  EXCELLENT: In the sweet spot!")
elif max_depth < 700:
    print(f"  GOOD: Should work")
else:
    print(f"  CAUTION: Getting deep")

# Submit
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
    'experiment': '1D 8+8 chain with lambda sweep',
    'backend': BACKEND_NAME,
    'n_chain': N_CHAIN,
    'n_total_qubits': N_TOTAL,
    'trotter_steps': TROTTER_STEPS,
    'dt': DT,
    'lambda_values': LAMBDA_VALUES,
    'shots': SHOTS,
    'job_ids': job_ids,
}

with open('ibm_1d_8x8_job_ids.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print(f"ALL 8+8 CHAIN JOBS SUBMITTED ({N_TOTAL} qubits)")
print(f"{'=' * 70}")
print(f"Job IDs saved to ibm_1d_8x8_job_ids.json")
print(f"Monitor at: https://quantum.ibm.com/jobs")
print(f"\nThis is the cleanest 16-qubit experiment:")
print(f"  - Linear chains map well to hardware topology")
print(f"  - Minimal swap overhead")
print(f"  - Lambda sweep at double the original scale")
print(f"  - 8 cross-chain pairs for rich distance structure")
print(f"{'=' * 70}")
