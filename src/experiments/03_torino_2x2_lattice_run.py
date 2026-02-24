#!/usr/bin/env python3
"""
EMERGENT SPACETIME — MINIMAL 2D LATTICE (2x2)
Two Coupled 2x2 Scalar Field Grids on IBM Torino
Andrew Thurlow | 528 Labs | February 2026

Minimal 2D topology: 2x2 + 2x2 = 8 qubits total.
Same qubit count as successful 1D experiment, but with genuine 2D connectivity.
Each site has 2 intra-field neighbors (vs 1-2 in 1D chain).
Targets circuit depth < 300 for clean hardware signals.

Key 2D features measurable even at 2x2:
  - Corner vs edge correlation differences (2D geometric effect)
  - 2D emergent distance (4 cross-field site pairs)
  - Tearing across a 2D surface
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
GRID_SIZE = 2
N_PER_FIELD = GRID_SIZE * GRID_SIZE  # 4
N_TOTAL = 2 * N_PER_FIELD            # 8
TROTTER_STEPS = 6     # Can afford more steps at this size
DT = 0.3
J_INTRA = 1.0
SHOTS = 8192

# Also run with varied coupling strengths for geometry mapping
LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

print("=" * 70)
print("EMERGENT SPACETIME — MINIMAL 2D LATTICE (2x2)")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)
print(f"Backend: {BACKEND_NAME}")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE} per field = {N_TOTAL} qubits")
print(f"Trotter steps: {TROTTER_STEPS}, dt = {DT}")
print(f"Lambda sweep: {LAMBDA_VALUES}")
print(f"Shots: {SHOTS}")
print()

# =============================================================================
# QUBIT LAYOUT
#
#  Field A (qubits 0-3):    Field B (qubits 4-7):
#    q0 -- q1                  q4 -- q5
#    |     |                   |     |
#    q2 -- q3                  q6 -- q7
#
#  Intra-field pairs: (0,1), (2,3), (0,2), (1,3)  [horizontal + vertical]
#  Inter-field pairs: (0,4), (1,5), (2,6), (3,7)  [corresponding sites]
# =============================================================================

def get_intra_pairs_2x2():
    """Nearest-neighbor pairs within a 2x2 grid."""
    return [(0, 1), (2, 3), (0, 2), (1, 3)]

def get_inter_pairs_2x2(n_per):
    """Corresponding site pairs between fields."""
    return [(i, i + n_per) for i in range(n_per)]

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_2x2_coupled(j_intra, j_inter, steps, dt):
    """Two coupled 2x2 grids."""
    qc = QuantumCircuit(N_TOTAL)
    # Initialize field A in superposition
    for i in range(N_PER_FIELD):
        qc.h(i)
    qc.barrier()

    intra_A = get_intra_pairs_2x2()
    intra_B = [(a + N_PER_FIELD, b + N_PER_FIELD) for a, b in intra_A]
    inter = get_inter_pairs_2x2(N_PER_FIELD)

    for step in range(steps):
        # Intra-field A: ZZ + XX
        for i, j in intra_A:
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
        # Intra-field B: ZZ + XX
        for i, j in intra_B:
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
        # Inter-field: ZZ
        if j_inter > 0:
            for i, j in inter:
                qc.cx(i, j); qc.rz(2*j_inter*dt, j); qc.cx(i, j)
        qc.barrier()
    return qc

def build_2x2_single(j_intra, steps, dt):
    """Single 2x4 lattice (null hypothesis)."""
    qc = QuantumCircuit(N_TOTAL)
    for i in range(N_TOTAL // 2):
        qc.h(i)
    qc.barrier()
    # Treat as 2x4 grid
    pairs = [(0,1),(2,3),(4,5),(6,7),  # horizontal
             (0,2),(1,3),(2,4),(3,5),(4,6),(5,7)]  # vertical
    for step in range(steps):
        for i, j in pairs:
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
            qc.cx(i, j); qc.rz(2*j_intra*dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j)
        qc.barrier()
    return qc

def build_2x2_torn(j_intra, j_inter, total_steps, dt):
    """Coupled for half, then uncoupled."""
    half = total_steps // 2
    intra_A = get_intra_pairs_2x2()
    intra_B = [(a + N_PER_FIELD, b + N_PER_FIELD) for a, b in intra_A]

    # First half: coupled
    qc = build_2x2_coupled(j_intra, j_inter, half, dt)

    # Second half: uncoupled
    for step in range(half):
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

# --- Core experiments (Z-basis only for speed, add X/Y if depth allows) ---

# Experiment 1: Lambda sweep (coupling strength variation)
print("Building lambda sweep circuits...")
for lam in LAMBDA_VALUES:
    qc = build_2x2_coupled(J_INTRA, lam, TROTTER_STEPS, DT)
    qc_z = add_measurement(qc, 'Z')
    qc_z.name = f'coupled_{lam}_Z'
    circuits.append(qc_z)
    labels.append(qc_z.name)
    print(f"  lambda={lam}: depth={qc.depth()}")

# Experiment 2: Single lattice null hypothesis
print("Building null hypothesis circuit...")
qc_single = build_2x2_single(J_INTRA, TROTTER_STEPS, DT)
qc_single_z = add_measurement(qc_single, 'Z')
qc_single_z.name = 'single_Z'
circuits.append(qc_single_z)
labels.append(qc_single_z.name)
print(f"  single: depth={qc_single.depth()}")

# Experiment 3: Spacetime tearing
print("Building tearing circuit...")
qc_torn = build_2x2_torn(J_INTRA, 1.0, TROTTER_STEPS, DT)
qc_torn_z = add_measurement(qc_torn, 'Z')
qc_torn_z.name = 'torn_Z'
circuits.append(qc_torn_z)
labels.append(qc_torn_z.name)
print(f"  torn: depth={qc_torn.depth()}")

# Experiment 4: Full Pauli tomography for coupled and uncoupled
print("Building tomography circuits...")
for lam_val, lam_name in [(1.0, 'coupled_1.0'), (0.0, 'coupled_0.0')]:
    qc_base = build_2x2_coupled(J_INTRA, lam_val, TROTTER_STEPS, DT)
    for basis in ['X', 'Y']:
        qc_m = add_measurement(qc_base, basis)
        qc_m.name = f'{lam_name}_{basis}'
        circuits.append(qc_m)
        labels.append(qc_m.name)

print(f"\nTotal circuits: {len(circuits)}")

# Transpile
print(f"\nTranspiling for {BACKEND_NAME} (optimization level 3)...")
transpiled = transpile(circuits, backend=backend, optimization_level=3)
for i, qc in enumerate(transpiled):
    ops = qc.count_ops()
    cx_count = ops.get('cx', 0) + ops.get('ecr', 0)
    print(f"  {labels[i]}: depth={qc.depth()}, two-qubit gates={cx_count}")

max_depth = max(qc.depth() for qc in transpiled)
print(f"\n  Max depth: {max_depth}")
if max_depth < 500:
    print(f"  EXCELLENT: Well within hardware sweet spot")
elif max_depth < 800:
    print(f"  GOOD: Should produce clean signals")
else:
    print(f"  WARNING: May be too deep")

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
    'experiment': '2D minimal 2x2 lattice with lambda sweep',
    'backend': BACKEND_NAME,
    'grid_size': GRID_SIZE,
    'n_total_qubits': N_TOTAL,
    'trotter_steps': TROTTER_STEPS,
    'dt': DT,
    'lambda_values': LAMBDA_VALUES,
    'shots': SHOTS,
    'job_ids': job_ids,
}

with open('ibm_2d_2x2_job_ids.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print(f"ALL 2x2 LATTICE JOBS SUBMITTED")
print(f"{'=' * 70}")
print(f"Job IDs saved to ibm_2d_2x2_job_ids.json")
print(f"Monitor at: https://quantum.ibm.com/jobs")
print(f"\nNew in this experiment:")
print(f"  - Lambda sweep: geometry strength vs coupling (7 values)")
print(f"  - 2D topology at shallow depth (should match 1D signal quality)")
print(f"  - Genuine 2D geometric effects at hardware-friendly scale")
print(f"{'=' * 70}")
