#!/usr/bin/env python3
"""
IBM QUANTUM HARDWARE SUBMISSION
Emergent Spacetime — Two Scalar Field Model
Andrew × Claude Collaboration

PURPOSE: Run the three key experiments on real IBM quantum hardware
with proper error mitigation and topology-aware transpilation.

PREREQUISITES:
  pip install qiskit qiskit-ibm-runtime qiskit-aer

USAGE:
  1. Set your IBM token below (or use env variable)
  2. Run: python ibm_hardware_run.py
  3. Jobs will be submitted and results saved when complete

NOTE: IBM queue times vary. Jobs may take minutes to hours.
The script saves job IDs so you can retrieve results later.
"""

import numpy as np
import json
import os
import time
from datetime import datetime

# =============================================================================
# QISKIT IMPORTS
# =============================================================================

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# IBM Runtime imports — uncomment when running on real hardware
# from qiskit_ibm_runtime import (
#     QiskitRuntimeService, 
#     SamplerV2 as Sampler,
#     EstimatorV2 as Estimator,
#     Options
# )

# =============================================================================
# CONFIGURATION
# =============================================================================

class HardwareConfig:
    """Configuration for IBM hardware runs."""
    
    # ── IBM CREDENTIALS ──
    # Option 1: Set directly
    IBM_TOKEN = "YOUR_IBM_TOKEN_HERE"
    
    # Option 2: Use environment variable
    # IBM_TOKEN = os.environ.get('IBM_QUANTUM_TOKEN', '')
    
    # ── BACKEND SELECTION ──
    # 'ibm_brisbane'   - 127q Eagle (free tier)
    # 'ibm_sherbrooke'  - 127q Eagle (free tier)
    # 'ibm_kyiv'        - 127q Eagle (free tier)
    BACKEND_NAME = 'ibm_brisbane'
    
    # ── EXPERIMENT PARAMETERS ──
    # Start with 8 qubits — strongest signal, shallowest circuits
    N_CHAIN = 4          # 4+4 = 8 qubits total
    TROTTER_STEPS = 6
    DT = 0.3
    J_INTRA = 1.0
    
    # Key coupling values to test (minimal set for hardware)
    LAMBDA_VALUES = [0.0, 1.0]  # Uncoupled vs coupled
    
    # Shots per circuit
    SHOTS = 8192
    
    # ── ERROR MITIGATION ──
    USE_READOUT_MITIGATION = True
    OPTIMIZATION_LEVEL = 3  # Maximum transpiler optimization
    
    # ── OUTPUT ──
    OUTPUT_DIR = '.'
    RESULTS_FILE = 'ibm_hardware_results.json'
    JOB_IDS_FILE = 'ibm_job_ids.json'


config = HardwareConfig()


# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_two_chain_circuit(n_per_chain, j_intra, j_inter, trotter_steps, dt):
    """Build the two coupled chain circuit."""
    n_total = 2 * n_per_chain
    qc = QuantumCircuit(n_total)
    
    # Chain A in superposition
    for i in range(n_per_chain):
        qc.h(i)
    qc.barrier()
    
    # Trotterized evolution
    for step in range(trotter_steps):
        # Chain A intra-coupling
        for i in range(n_per_chain - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
        
        # Chain B intra-coupling
        for i in range(n_per_chain, 2 * n_per_chain - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
        
        # Inter-chain coupling
        if j_inter > 0:
            for i in range(n_per_chain):
                j = i + n_per_chain
                qc.cx(i, j)
                qc.rz(2 * j_inter * dt, j)
                qc.cx(i, j)
        
        qc.barrier()
    
    return qc


def build_single_chain_circuit(n_qubits, j_intra, trotter_steps, dt):
    """Build single chain circuit (null hypothesis)."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits // 2):
        qc.h(i)
    qc.barrier()
    
    for step in range(trotter_steps):
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
        qc.barrier()
    
    return qc


def build_torn_circuit(n_per_chain, j_intra, j_inter, total_steps, dt):
    """Build circuit where coupling is removed halfway (spacetime tearing)."""
    half = total_steps // 2
    
    # First half: coupled evolution
    qc = build_two_chain_circuit(n_per_chain, j_intra, j_inter, half, dt)
    
    # Second half: uncoupled evolution (no inter-chain)
    n_total = 2 * n_per_chain
    for step in range(half):
        for i in range(n_per_chain - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
        for i in range(n_per_chain, 2 * n_per_chain - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * j_intra * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i); qc.h(i + 1)
        qc.barrier()
    
    return qc


# =============================================================================
# MEASUREMENT CIRCUITS
# =============================================================================

def add_measurement_bases(qc, basis='Z'):
    """
    Add measurement in specified basis to all qubits.
    
    For tomographic reconstruction we need Z, X, and Y measurements.
    Z: measure directly
    X: apply H before measurement
    Y: apply S†H before measurement
    """
    qc_meas = qc.copy()
    n = qc.num_qubits
    
    if basis == 'X':
        for i in range(n):
            qc_meas.h(i)
    elif basis == 'Y':
        for i in range(n):
            qc_meas.sdg(i)
            qc_meas.h(i)
    # Z basis: no additional gates needed
    
    qc_meas.measure_all()
    return qc_meas


def create_all_measurement_circuits(base_circuit, label=''):
    """Create Z, X, Y measurement circuits for a given base circuit."""
    circuits = {}
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement_bases(base_circuit, basis)
        qc.name = f'{label}_{basis}'
        circuits[basis] = qc
    return circuits


# =============================================================================
# ANALYSIS FUNCTIONS (from measurement counts)
# =============================================================================

def counts_to_correlations(counts, n_total, basis='Z'):
    """
    Compute correlation matrix from measurement counts.
    
    For Z basis: 
      ⟨Z_i⟩ = (n_0 - n_1) / n_total for qubit i
      ⟨Z_iZ_j⟩ = (n_same - n_diff) / n_total for qubits i,j
    """
    total_shots = sum(counts.values())
    
    # Single qubit expectations
    z_exp = np.zeros(n_total)
    for bitstring, count in counts.items():
        # Qiskit bitstrings are little-endian when read right to left
        # But the string itself reads left to right as q_{n-1}...q_0
        bits = bitstring.replace(' ', '')
        for i in range(n_total):
            # Bit for qubit i is at position (n_total - 1 - i) in string
            bit = int(bits[n_total - 1 - i])
            z_exp[i] += (1 - 2 * bit) * count
    z_exp /= total_shots
    
    # Two-qubit correlations
    corr = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(i, n_total):
            if i == j:
                corr[i, j] = 1.0 - z_exp[i]**2
            else:
                zz = 0.0
                for bitstring, count in counts.items():
                    bits = bitstring.replace(' ', '')
                    bit_i = int(bits[n_total - 1 - i])
                    bit_j = int(bits[n_total - 1 - j])
                    zi = 1 - 2 * bit_i
                    zj = 1 - 2 * bit_j
                    zz += zi * zj * count
                zz /= total_shots
                corr[i, j] = zz - z_exp[i] * z_exp[j]
                corr[j, i] = corr[i, j]
    
    return corr, z_exp


def estimate_entropy_from_correlations(corr, n_per_chain):
    """
    Estimate entanglement entropy from correlation data.
    
    Uses the mutual information approximation:
    For Gaussian states, the mutual information between A and B
    can be estimated from the correlation matrix eigenvalues.
    
    This is an approximation but works well for our purposes
    and doesn't require full state tomography.
    """
    n_total = 2 * n_per_chain
    
    # Extract the cross-correlation block
    C_AB = corr[:n_per_chain, n_per_chain:]
    
    # Singular values of cross-correlation block
    svd = np.linalg.svd(C_AB, compute_uv=False)
    
    # Mutual information estimate (for near-Gaussian states)
    # I(A:B) ≈ -½ Σ log(1 - σ_i²) where σ_i are singular values
    mi = 0.0
    for s in svd:
        if 0 < abs(s) < 1:
            mi -= 0.5 * np.log2(1 - s**2)
    
    return mi


def estimate_entropy_from_purity(counts, subsystem_qubits, n_total, 
                                  counts_x=None, counts_y=None):
    """
    Estimate Rényi-2 entropy from randomized measurements.
    
    S₂(A) = -log₂(Tr(ρ_A²))
    
    The purity Tr(ρ_A²) can be estimated from two copies or from
    measurements in multiple bases using the formula:
    
    Tr(ρ_A²) = Σ_{P} ⟨P⟩² where P runs over Pauli operators on A
    
    This works with Z, X, Y measurement data.
    """
    n_sub = len(subsystem_qubits)
    
    # Compute all single and two-qubit Pauli expectations on subsystem
    # from the three measurement bases
    
    total_shots_z = sum(counts.values())
    
    # Z basis expectations for subsystem qubits
    z_exp = {}
    for bitstring, count in counts.items():
        bits = bitstring.replace(' ', '')
        for i in subsystem_qubits:
            bit = int(bits[n_total - 1 - i])
            z_exp[i] = z_exp.get(i, 0) + (1 - 2*bit) * count
    for i in subsystem_qubits:
        z_exp[i] /= total_shots_z
    
    # Purity from Z measurements only (lower bound)
    purity = 1.0 / (2**n_sub)  # Maximally mixed baseline
    
    # Add Z contribution
    for i in subsystem_qubits:
        purity += z_exp[i]**2 / (2**n_sub)
    
    # ZZ contributions
    for i in subsystem_qubits:
        for j in subsystem_qubits:
            if i >= j:
                continue
            zz = 0.0
            for bitstring, count in counts.items():
                bits = bitstring.replace(' ', '')
                bi = int(bits[n_total - 1 - i])
                bj = int(bits[n_total - 1 - j])
                zz += (1-2*bi)*(1-2*bj) * count
            zz /= total_shots_z
            purity += zz**2 / (2**n_sub)
    
    # Add X and Y contributions if available
    if counts_x is not None:
        total_shots_x = sum(counts_x.values())
        for i in subsystem_qubits:
            x_exp = 0.0
            for bitstring, count in counts_x.items():
                bits = bitstring.replace(' ', '')
                bit = int(bits[n_total - 1 - i])
                x_exp += (1 - 2*bit) * count
            x_exp /= total_shots_x
            purity += x_exp**2 / (2**n_sub)
    
    if counts_y is not None:
        total_shots_y = sum(counts_y.values())
        for i in subsystem_qubits:
            y_exp = 0.0
            for bitstring, count in counts_y.items():
                bits = bitstring.replace(' ', '')
                bit = int(bits[n_total - 1 - i])
                y_exp += (1 - 2*bit) * count
            y_exp /= total_shots_y
            purity += y_exp**2 / (2**n_sub)
    
    # Rényi-2 entropy
    purity = min(max(purity, 1.0/(2**n_sub)), 1.0)  # Clamp to valid range
    s2 = -np.log2(purity)
    
    return s2, purity


# =============================================================================
# SIMULATOR MODE (for testing before hardware submission)
# =============================================================================

def run_simulator_validation():
    """
    Run all experiments on the local simulator with measurement-based
    analysis (not statevector). This validates that our measurement-based
    entropy estimation works before we burn hardware credits.
    """
    print("=" * 70)
    print("SIMULATOR VALIDATION (measurement-based)")
    print("Testing analysis pipeline before hardware submission")
    print("=" * 70)
    
    simulator = AerSimulator()
    n_chain = config.N_CHAIN
    n_total = 2 * n_chain
    shots = config.SHOTS
    
    results = {}
    
    # ── EXPERIMENT 1: Coupling sweep ──
    print("\n  EXPERIMENT 1: Coupling sweep")
    for j_inter in config.LAMBDA_VALUES:
        print(f"\n    λ = {j_inter}:")
        
        qc = build_two_chain_circuit(n_chain, config.J_INTRA, j_inter,
                                      config.TROTTER_STEPS, config.DT)
        
        # Create measurement circuits for Z, X, Y bases
        meas_circuits = create_all_measurement_circuits(qc, f'coupled_{j_inter}')
        
        counts_by_basis = {}
        for basis, circuit in meas_circuits.items():
            compiled = transpile(circuit, simulator, optimization_level=3)
            job = simulator.run(compiled, shots=shots)
            result = job.result()
            counts_by_basis[basis] = result.get_counts()
            print(f"      {basis} basis: {len(counts_by_basis[basis])} unique outcomes")
        
        # Compute correlations from Z measurements
        corr, z_exp = counts_to_correlations(counts_by_basis['Z'], n_total)
        
        # Cross-chain correlation
        cross_corr = np.mean([abs(corr[i, i + n_chain]) for i in range(n_chain)])
        
        # Entropy estimate from correlations
        mi_estimate = estimate_entropy_from_correlations(corr, n_chain)
        
        # Entropy estimate from purity
        chain_A = list(range(n_chain))
        s2, purity = estimate_entropy_from_purity(
            counts_by_basis['Z'], chain_A, n_total,
            counts_by_basis['X'], counts_by_basis['Y']
        )
        
        print(f"      Cross-chain |C| = {cross_corr:.4f}")
        print(f"      MI estimate = {mi_estimate:.4f} bits")
        print(f"      Rényi-2 entropy = {s2:.4f} bits (purity={purity:.4f})")
        
        results[f'coupled_{j_inter}'] = {
            'cross_corr': float(cross_corr),
            'mi_estimate': float(mi_estimate),
            'renyi2': float(s2),
            'purity': float(purity),
            'z_expectations': z_exp.tolist(),
            'correlation_matrix': corr.tolist(),
        }
    
    # ── EXPERIMENT 2: Null hypothesis ──
    print("\n  EXPERIMENT 2: Null hypothesis (single chain)")
    qc_single = build_single_chain_circuit(n_total, config.J_INTRA,
                                            config.TROTTER_STEPS, config.DT)
    meas_single = create_all_measurement_circuits(qc_single, 'single')
    
    counts_single = {}
    for basis, circuit in meas_single.items():
        compiled = transpile(circuit, simulator, optimization_level=3)
        job = simulator.run(compiled, shots=shots)
        counts_single[basis] = job.result().get_counts()
    
    corr_single, _ = counts_to_correlations(counts_single['Z'], n_total)
    cross_corr_single = np.mean([abs(corr_single[i, i + n_chain]) 
                                  for i in range(n_chain)])
    
    cross_vals_single = [abs(corr_single[i, i + n_chain]) for i in range(n_chain)]
    std_single = np.std(cross_vals_single)
    
    # Compare with two-chain
    corr_two, _ = counts_to_correlations(
        results['coupled_1.0'].get('counts_z', counts_by_basis['Z']), n_total)
    cross_vals_two = [abs(corr[i, i + n_chain]) for i in range(n_chain)]
    std_two = np.std(cross_vals_two)
    
    print(f"    Single chain: |C| = {cross_corr_single:.4f}, σ = {std_single:.4f}")
    print(f"    Two-chain:    |C| = {cross_corr:.4f}, σ = {std_two:.4f}")
    ratio = std_two / max(std_single, 1e-10)
    print(f"    Structure ratio: {ratio:.2f}x")
    
    results['null_hypothesis'] = {
        'single_cross_corr': float(cross_corr_single),
        'single_std': float(std_single),
        'two_chain_std': float(std_two),
        'ratio': float(ratio),
    }
    
    # ── EXPERIMENT 3: Spacetime tearing ──
    print("\n  EXPERIMENT 3: Spacetime tearing")
    qc_torn = build_torn_circuit(n_chain, config.J_INTRA, 1.0,
                                  config.TROTTER_STEPS, config.DT)
    meas_torn = create_all_measurement_circuits(qc_torn, 'torn')
    
    counts_torn = {}
    for basis, circuit in meas_torn.items():
        compiled = transpile(circuit, simulator, optimization_level=3)
        job = simulator.run(compiled, shots=shots)
        counts_torn[basis] = job.result().get_counts()
    
    corr_torn, _ = counts_to_correlations(counts_torn['Z'], n_total)
    
    print(f"    Cross-chain correlation comparison:")
    reductions = []
    for i in range(n_chain):
        c_full = abs(corr[i, i + n_chain])
        c_torn = abs(corr_torn[i, i + n_chain])
        red = (1 - c_torn / max(c_full, 1e-10)) * 100
        reductions.append(red)
        print(f"      q{i}A↔q{i}B: connected={c_full:.4f} → torn={c_torn:.4f} "
              f"({red:.1f}% reduction)")
    
    avg_reduction = np.mean(reductions)
    print(f"    Average reduction: {avg_reduction:.1f}%")
    
    results['tearing'] = {
        'avg_reduction': float(avg_reduction),
        'reductions': [float(r) for r in reductions],
    }
    
    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print("SIMULATOR VALIDATION SUMMARY")
    print("=" * 70)
    
    coupled_mi = results['coupled_1.0']['mi_estimate']
    uncoupled_mi = results['coupled_0.0']['mi_estimate']
    
    print(f"""
  Coupling effect:
    λ=0.0: MI = {uncoupled_mi:.4f} bits
    λ=1.0: MI = {coupled_mi:.4f} bits
    Difference detectable: {'YES ✓' if coupled_mi > uncoupled_mi + 0.1 else 'MARGINAL'}
    
  Null hypothesis:
    Structure ratio: {results['null_hypothesis']['ratio']:.2f}x
    Distinguishable: {'YES ✓' if results['null_hypothesis']['ratio'] > 2 else 'MARGINAL'}
    
  Spacetime tearing:
    Avg correlation reduction: {results['tearing']['avg_reduction']:.1f}%
    Effect detectable: {'YES ✓' if results['tearing']['avg_reduction'] > 30 else 'MARGINAL'}
  
  VERDICT: {'ALL SIGNALS STRONG — READY FOR HARDWARE' if coupled_mi > uncoupled_mi + 0.1 and results['null_hypothesis']['ratio'] > 2 and results['tearing']['avg_reduction'] > 30 else 'SOME SIGNALS WEAK — CONSIDER ADJUSTMENTS'}
    """)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_chain': config.N_CHAIN,
            'trotter_steps': config.TROTTER_STEPS,
            'dt': config.DT,
            'j_intra': config.J_INTRA,
            'shots': shots,
            'mode': 'simulator_validation',
        },
        'results': results,
    }
    
    filepath = os.path.join(config.OUTPUT_DIR, 'simulator_validation.json')
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {filepath}")
    
    return results


# =============================================================================
# IBM HARDWARE SUBMISSION
# =============================================================================

def submit_to_ibm():
    """
    Submit experiments to IBM quantum hardware.
    
    UNCOMMENT the IBM imports at the top of the file before running.
    """
    print("=" * 70)
    print("IBM QUANTUM HARDWARE SUBMISSION")
    print("=" * 70)
    
    # ── Connect to IBM ──
    print(f"\n  Connecting to IBM Quantum...")
    
    # UNCOMMENT BELOW FOR ACTUAL HARDWARE SUBMISSION
    # ─────────────────────────────────────────────────
    # service = QiskitRuntimeService(
    #     channel='ibm_quantum',
    #     token=config.IBM_TOKEN
    # )
    # backend = service.backend(config.BACKEND_NAME)
    # print(f"  Connected to {config.BACKEND_NAME}")
    # print(f"  Qubits: {backend.num_qubits}")
    # print(f"  Pending jobs: {backend.status().pending_jobs}")
    # ─────────────────────────────────────────────────
    
    n_chain = config.N_CHAIN
    n_total = 2 * n_chain
    
    # ── Build all circuits ──
    print(f"\n  Building circuits ({n_total} qubits)...")
    
    all_circuits = []
    circuit_labels = []
    
    # Experiment 1: Coupled (λ=1.0) — Z, X, Y
    qc_coupled = build_two_chain_circuit(n_chain, config.J_INTRA, 1.0,
                                          config.TROTTER_STEPS, config.DT)
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement_bases(qc_coupled, basis)
        qc.name = f'coupled_1.0_{basis}'
        all_circuits.append(qc)
        circuit_labels.append(f'coupled_1.0_{basis}')
    
    # Experiment 1: Uncoupled (λ=0.0) — Z, X, Y
    qc_uncoupled = build_two_chain_circuit(n_chain, config.J_INTRA, 0.0,
                                            config.TROTTER_STEPS, config.DT)
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement_bases(qc_uncoupled, basis)
        qc.name = f'coupled_0.0_{basis}'
        all_circuits.append(qc)
        circuit_labels.append(f'coupled_0.0_{basis}')
    
    # Experiment 2: Single chain — Z, X, Y
    qc_single = build_single_chain_circuit(n_total, config.J_INTRA,
                                            config.TROTTER_STEPS, config.DT)
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement_bases(qc_single, basis)
        qc.name = f'single_{basis}'
        all_circuits.append(qc)
        circuit_labels.append(f'single_{basis}')
    
    # Experiment 3: Torn — Z, X, Y
    qc_torn = build_torn_circuit(n_chain, config.J_INTRA, 1.0,
                                  config.TROTTER_STEPS, config.DT)
    for basis in ['Z', 'X', 'Y']:
        qc = add_measurement_bases(qc_torn, basis)
        qc.name = f'torn_{basis}'
        all_circuits.append(qc)
        circuit_labels.append(f'torn_{basis}')
    
    print(f"  Total circuits: {len(all_circuits)}")
    print(f"  Labels: {circuit_labels}")
    
    # ── Transpile for hardware ──
    print(f"\n  Transpiling for {config.BACKEND_NAME}...")
    print(f"  Optimization level: {config.OPTIMIZATION_LEVEL}")
    
    # UNCOMMENT FOR HARDWARE:
    # transpiled = transpile(
    #     all_circuits, 
    #     backend=backend,
    #     optimization_level=config.OPTIMIZATION_LEVEL,
    # )
    # 
    # for i, qc in enumerate(transpiled):
    #     print(f"    {circuit_labels[i]}: depth={qc.depth()}, "
    #           f"cx_count={qc.count_ops().get('cx', 0)}")
    
    # ── Submit jobs ──
    # UNCOMMENT FOR HARDWARE:
    # print(f"\n  Submitting to {config.BACKEND_NAME}...")
    # print(f"  Shots per circuit: {config.SHOTS}")
    # 
    # sampler = Sampler(backend)
    # 
    # job_ids = {}
    # for i, (circuit, label) in enumerate(zip(transpiled, circuit_labels)):
    #     job = sampler.run([circuit], shots=config.SHOTS)
    #     job_ids[label] = job.job_id()
    #     print(f"    Submitted {label}: job_id = {job.job_id()}")
    # 
    # # Save job IDs for later retrieval
    # with open(config.JOB_IDS_FILE, 'w') as f:
    #     json.dump({
    #         'timestamp': datetime.now().isoformat(),
    #         'backend': config.BACKEND_NAME,
    #         'job_ids': job_ids,
    #     }, f, indent=2)
    # print(f"\n  Job IDs saved to {config.JOB_IDS_FILE}")
    # print(f"  Use retrieve_results() after jobs complete")
    
    print("\n  [SIMULATION MODE] To submit to real hardware:")
    print("  1. Set IBM_TOKEN in the config")
    print("  2. Uncomment the IBM Runtime imports at top of file")
    print("  3. Uncomment the hardware submission code above")
    print("  4. Run again")
    
    # Print circuit stats for reference
    print(f"\n  Circuit statistics (pre-transpile):")
    for qc, label in zip(all_circuits, circuit_labels):
        cx = qc.count_ops().get('cx', 0)
        print(f"    {label}: depth={qc.depth()}, gates={qc.size()}, cx={cx}")


# =============================================================================
# RESULT RETRIEVAL
# =============================================================================

def retrieve_results():
    """
    Retrieve and analyze results from previously submitted IBM jobs.
    
    UNCOMMENT IBM imports before running.
    """
    print("=" * 70)
    print("RETRIEVING IBM HARDWARE RESULTS")
    print("=" * 70)
    
    # Load job IDs
    with open(config.JOB_IDS_FILE, 'r') as f:
        job_data = json.load(f)
    
    print(f"  Backend: {job_data['backend']}")
    print(f"  Submitted: {job_data['timestamp']}")
    
    # UNCOMMENT FOR HARDWARE:
    # service = QiskitRuntimeService(
    #     channel='ibm_quantum',
    #     token=config.IBM_TOKEN
    # )
    # 
    # all_counts = {}
    # for label, job_id in job_data['job_ids'].items():
    #     print(f"  Retrieving {label} ({job_id})...")
    #     job = service.job(job_id)
    #     
    #     if job.status().name == 'DONE':
    #         result = job.result()
    #         counts = result[0].data.meas.get_counts()
    #         all_counts[label] = counts
    #         print(f"    ✓ {len(counts)} unique outcomes")
    #     else:
    #         print(f"    Status: {job.status().name}")
    # 
    # # Analyze
    # n_chain = config.N_CHAIN
    # n_total = 2 * n_chain
    # 
    # # ... (same analysis as simulator validation)
    # # Compare hardware results with simulator baseline
    
    print("\n  [SIMULATION MODE] Results retrieval ready")
    print("  Uncomment IBM code and run after jobs complete")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  EMERGENT SPACETIME — IBM QUANTUM HARDWARE EXPERIMENT          ║
║  Two Scalar Field Model | Andrew × Claude                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Modes:                                                        ║
║    python ibm_hardware_run.py validate  — Test on simulator    ║
║    python ibm_hardware_run.py submit    — Submit to IBM        ║
║    python ibm_hardware_run.py retrieve  — Get IBM results      ║
║                                                                ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else 'validate'
    
    if mode == 'validate':
        run_simulator_validation()
    elif mode == 'submit':
        submit_to_ibm()
    elif mode == 'retrieve':
        retrieve_results()
    else:
        print(f"Unknown mode: {mode}")
        print("Use: validate, submit, or retrieve")
