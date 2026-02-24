#!/usr/bin/env python3
"""
EMERGENT SPACETIME — FULL-CHIP PARALLEL EXPERIMENT
16 Independent 4+4 Experiments Across IBM Torino (128/133 qubits)
Andrew Thurlow | 528 Labs | February 2026

Architecture: 16 independent 8-qubit experiments running simultaneously.
Each uses the proven 4+4 chain that produced 95.7x on Torino.
Different lambda values assigned to different regions for a full
sweep in a single circuit submission.

Region Map:
  Region 0-1:   λ=0.0  (uncoupled controls, 2 replicates)
  Region 2-3:   λ=0.25 (2 replicates)
  Region 4-5:   λ=0.5  (2 replicates)
  Region 6-7:   λ=0.75 (2 replicates)
  Region 8-10:  λ=1.0  (3 replicates)
  Region 11-12: λ=1.5  (2 replicates)
  Region 13-14: λ=2.0  (2 replicates)
  Region 15:    λ=1.0  (extra replicate)
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
N_CHAIN = 4
N_PER_EXP = 8
N_PARALLEL = 16
N_TOTAL = N_PARALLEL * N_PER_EXP  # 128
TROTTER_STEPS = 4
DT = 0.3
J_INTRA = 1.0
SHOTS = 8192

REGION_LAMBDAS = [
    0.0,   0.0,          # Regions 0-1:   uncoupled controls
    0.25,  0.25,         # Regions 2-3:   weak coupling
    0.5,   0.5,          # Regions 4-5:   medium coupling
    0.75,  0.75,         # Regions 6-7:   medium-strong
    1.0,   1.0,   1.0,   # Regions 8-10:  standard coupling (3x)
    1.5,   1.5,          # Regions 11-12: strong coupling
    2.0,   2.0,          # Regions 13-14: very strong
    1.0,                  # Region 15:    extra λ=1.0 replicate
]

print("=" * 70)
print("EMERGENT SPACETIME — FULL-CHIP PARALLEL EXPERIMENT")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)
print(f"Backend: {BACKEND_NAME}")
print(f"Architecture: {N_PARALLEL} x {N_PER_EXP}-qubit experiments in parallel")
print(f"Total qubits: {N_TOTAL} / 133")
print(f"Trotter steps: {TROTTER_STEPS}, dt = {DT}")
print(f"Shots: {SHOTS}")
print()
print("Region assignments:")
for i, lam in enumerate(REGION_LAMBDAS):
    q_start = i * N_PER_EXP
    q_end = q_start + N_PER_EXP - 1
    print(f"  Region {i:2d}: qubits {q_start:3d}-{q_end:3d}  lambda = {lam}")
print()

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_parallel_sweep(region_lambdas, j_intra, steps, dt, n_chain):
    """16 independent 4+4 experiments with different lambdas, one circuit."""
    n_per = 2 * n_chain
    n_regions = len(region_lambdas)
    n_total = n_regions * n_per

    qc = QuantumCircuit(n_total)

    # Initialize chain A of each region
    for reg in range(n_regions):
        offset = reg * n_per
        for i in range(n_chain):
            qc.h(offset + i)
    qc.barrier()

    for step in range(steps):
        for reg in range(n_regions):
            offset = reg * n_per
            lam = region_lambdas[reg]

            # Intra-chain A: ZZ + XX
            for i in range(n_chain - 1):
                qi, qj = offset + i, offset + i + 1
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)

            # Intra-chain B: ZZ + XX
            for i in range(n_chain - 1):
                qi, qj = offset + n_chain + i, offset + n_chain + i + 1
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)

            # Inter-chain: ZZ
            if lam > 0:
                for i in range(n_chain):
                    qi, qj = offset + i, offset + n_chain + i
                    qc.cx(qi, qj); qc.rz(2*lam*dt, qj); qc.cx(qi, qj)

        qc.barrier()

    qc.measure_all()
    return qc


def build_parallel_torn(j_intra, steps, dt, n_chain, n_regions):
    """All regions at lambda=1.0 first half, uncoupled second half."""
    n_per = 2 * n_chain
    n_total = n_regions * n_per
    half1 = steps // 2
    half2 = steps - half1

    qc = QuantumCircuit(n_total)

    for reg in range(n_regions):
        offset = reg * n_per
        for i in range(n_chain):
            qc.h(offset + i)
    qc.barrier()

    # First half: coupled at lambda=1.0
    for step in range(half1):
        for reg in range(n_regions):
            offset = reg * n_per
            for i in range(n_chain - 1):
                qi, qj = offset + i, offset + i + 1
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
            for i in range(n_chain - 1):
                qi, qj = offset + n_chain + i, offset + n_chain + i + 1
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
            for i in range(n_chain):
                qi, qj = offset + i, offset + n_chain + i
                qc.cx(qi, qj); qc.rz(2*1.0*dt, qj); qc.cx(qi, qj)
        qc.barrier()

    # Second half: uncoupled
    for step in range(half2):
        for reg in range(n_regions):
            offset = reg * n_per
            for i in range(n_chain - 1):
                qi, qj = offset + i, offset + i + 1
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
                qc.cx(qi, qj); qc.rz(2*j_intra*dt, qj); qc.cx(qi, qj)
                qc.h(qi); qc.h(qj)
            for i in range(n_chain - 1):
                qi, qj = offset + n_chain + i, offset + n_chain + i + 1
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
print(f"Connected to {BACKEND_NAME} ({backend.num_qubits} qubits)")
print()

print("Building parallel lambda sweep (128 qubits)...")
qc_sweep = build_parallel_sweep(REGION_LAMBDAS, J_INTRA, TROTTER_STEPS, DT, N_CHAIN)
print(f"  Sweep: {qc_sweep.num_qubits} qubits, pre-transpile depth = {qc_sweep.depth()}")

print("Building parallel tearing (128 qubits)...")
qc_torn = build_parallel_torn(J_INTRA, TROTTER_STEPS, DT, N_CHAIN, N_PARALLEL)
print(f"  Torn:  {qc_torn.num_qubits} qubits, pre-transpile depth = {qc_torn.depth()}")

circuits = [qc_sweep, qc_torn]
labels = ['parallel_sweep', 'parallel_torn']

print(f"\nTranspiling for {BACKEND_NAME} (optimization level 3)...")
print("  128-qubit transpilation may take a minute...")
transpiled = transpile(circuits, backend=backend, optimization_level=3)

for i, qc in enumerate(transpiled):
    ops = qc.count_ops()
    total_gates = sum(ops.values())
    print(f"  {labels[i]}: depth={qc.depth()}, total gates={total_gates}")

max_depth = max(qc.depth() for qc in transpiled)
print(f"\n  Max depth: {max_depth}")
if max_depth < 600:
    print(f"  EXCELLENT: Parallel architecture keeping depth manageable!")
elif max_depth < 1000:
    print(f"  GOOD: Should produce usable signals per region")
else:
    print(f"  CAUTION: Transpiler may be routing across regions")

# Submit
print(f"\nSubmitting {len(transpiled)} circuits ({SHOTS} shots each)...")
sampler = Sampler(backend)
job_ids = {}

for circuit, label in zip(transpiled, labels):
    print(f"  Submitting {label} ({circuit.num_qubits} qubits)...", end=" ", flush=True)
    job = sampler.run([circuit], shots=SHOTS)
    job_ids[label] = job.job_id()
    print(f"job_id = {job.job_id()}")

output = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'Full-chip parallel: 16x 4+4 chain across 128 qubits',
    'backend': BACKEND_NAME,
    'architecture': '16 independent 8-qubit experiments in parallel',
    'n_chain': N_CHAIN,
    'n_per_experiment': N_PER_EXP,
    'n_parallel': N_PARALLEL,
    'n_total_qubits': N_TOTAL,
    'trotter_steps': TROTTER_STEPS,
    'dt': DT,
    'shots': SHOTS,
    'region_lambdas': REGION_LAMBDAS,
    'region_map': {
        f'region_{i}': {
            'qubits': f'{i*N_PER_EXP}-{i*N_PER_EXP+N_PER_EXP-1}',
            'lambda': REGION_LAMBDAS[i]
        } for i in range(N_PARALLEL)
    },
    'job_ids': job_ids,
}

with open('ibm_fullchip_parallel_job_ids.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print(f"FULL-CHIP EXPERIMENT SUBMITTED")
print(f"{'=' * 70}")
print(f"  {N_TOTAL} qubits across {N_PARALLEL} independent regions")
print(f"  Lambda sweep + replicates in a single circuit")
print(f"  Tearing across all 16 regions simultaneously")
print(f"  Job IDs saved to ibm_fullchip_parallel_job_ids.json")
print(f"  Monitor at: https://quantum.ibm.com/jobs")
print(f"{'=' * 70}")
print(f"\nThis is the first full-chip emergent spacetime experiment.")
print(f"16 independent replications of the proven 4+4 framework")
print(f"running simultaneously across IBM Torino.")
print(f"{'=' * 70}")
