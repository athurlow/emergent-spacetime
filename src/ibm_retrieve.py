#!/usr/bin/env python3
"""
IBM QUANTUM HARDWARE — RETRIEVE AND ANALYZE RESULTS
Emergent Spacetime — Two Scalar Field Model
Andrew Thurlow | 528 Labs | February 2026

Retrieves completed jobs from IBM Quantum and computes correlation analysis.

USAGE:
  python ibm_retrieve.py
"""

import numpy as np
import json
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

# =============================================================================
# CONFIGURATION
# =============================================================================

N_CHAIN = 4
N_TOTAL = 8

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def counts_to_correlations(counts, n_total):
    """Compute Z-basis correlations from measurement counts."""
    total_shots = sum(counts.values())
    z_exp = np.zeros(n_total)
    for bitstring, count in counts.items():
        bits = bitstring.replace(' ', '')
        for i in range(n_total):
            bit = int(bits[n_total - 1 - i])
            z_exp[i] += (1 - 2 * bit) * count
    z_exp /= total_shots
    
    corr = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(i, n_total):
            if i == j:
                corr[i, j] = 1.0 - z_exp[i]**2
            else:
                zz = 0.0
                for bitstring, count in counts.items():
                    bits = bitstring.replace(' ', '')
                    bi = int(bits[n_total - 1 - i])
                    bj = int(bits[n_total - 1 - j])
                    zz += (1 - 2*bi) * (1 - 2*bj) * count
                zz /= total_shots
                corr[i, j] = zz - z_exp[i] * z_exp[j]
                corr[j, i] = corr[i, j]
    return corr, z_exp

# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("EMERGENT SPACETIME — IBM HARDWARE RESULTS")
print("Andrew Thurlow | 528 Labs")
print("=" * 60)

service = QiskitRuntimeService()

# Load job IDs
try:
    with open('ibm_job_ids.json', 'r') as f:
        job_data = json.load(f)
    label_to_id = job_data.get('job_ids', {})
    id_to_label = {v: k for k, v in label_to_id.items()}
    print(f"\nLoaded {len(label_to_id)} job IDs from ibm_job_ids.json")
except FileNotFoundError:
    print("\nibm_job_ids.json not found. Fetching recent jobs...")
    label_to_id = {}
    id_to_label = {}

# Retrieve jobs
print("\nRetrieving jobs...")
jobs = service.jobs(limit=24)

results = {}
labeled_results = {}

for job in jobs:
    jid = job.job_id()
    status = str(job.status())
    label = id_to_label.get(jid, jid)
    print(f"  {label}: {status}")
    
    if 'DONE' in status:
        try:
            result = job.result()
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()
            labeled_results[label] = counts
            print(f"    -> {len(counts)} unique outcomes, {sum(counts.values())} shots")
        except Exception as e:
            print(f"    -> Error: {e}")

if not labeled_results:
    print("\nNo completed results found. Jobs may still be running.")
    print("Check status at: https://quantum.ibm.com/jobs")
    exit()

# =============================================================================
# CROSS-CHAIN CORRELATIONS
# =============================================================================

print("\n" + "=" * 60)
print("CROSS-CHAIN CORRELATIONS (Z-basis)")
print("=" * 60)

for label in ['coupled_1.0_Z', 'coupled_0.0_Z', 'single_Z', 'torn_Z']:
    if label in labeled_results:
        corr, z_exp = counts_to_correlations(labeled_results[label], N_TOTAL)
        cross = [abs(corr[i, i + N_CHAIN]) for i in range(N_CHAIN)]
        avg_cross = np.mean(cross)
        std_cross = np.std(cross)
        print(f"\n  {label}:")
        print(f"    Cross-chain correlations: {[f'{c:.4f}' for c in cross]}")
        print(f"    Average |C|: {avg_cross:.4f}")
        print(f"    Std dev (structure): {std_cross:.4f}")

# =============================================================================
# KEY COMPARISONS
# =============================================================================

print("\n" + "=" * 60)
print("KEY COMPARISONS")
print("=" * 60)

if 'coupled_1.0_Z' in labeled_results and 'coupled_0.0_Z' in labeled_results:
    corr_c, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_u, _ = counts_to_correlations(labeled_results['coupled_0.0_Z'], N_TOTAL)
    cross_coupled = np.mean([abs(corr_c[i, i+N_CHAIN]) for i in range(N_CHAIN)])
    cross_uncoupled = np.mean([abs(corr_u[i, i+N_CHAIN]) for i in range(N_CHAIN)])
    ratio = cross_coupled / max(cross_uncoupled, 1e-10)
    print(f"\n  COUPLING EFFECT:")
    print(f"    Coupled (lambda=1.0):   |C| = {cross_coupled:.4f}")
    print(f"    Uncoupled (lambda=0.0): |C| = {cross_uncoupled:.4f}")
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    Signal: {'STRONG' if ratio > 10 else 'DETECTED' if ratio > 2 else 'MARGINAL'}")

if 'coupled_1.0_Z' in labeled_results and 'single_Z' in labeled_results:
    corr_two, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_one, _ = counts_to_correlations(labeled_results['single_Z'], N_TOTAL)
    std_two = np.std([abs(corr_two[i, i+N_CHAIN]) for i in range(N_CHAIN)])
    std_one = np.std([abs(corr_one[i, i+N_CHAIN]) for i in range(N_CHAIN)])
    ratio = std_two / max(std_one, 1e-10)
    print(f"\n  NULL HYPOTHESIS:")
    print(f"    Two-field structure: {std_two:.4f}")
    print(f"    Single chain:       {std_one:.4f}")
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    Signal: {'STRONG' if ratio > 5 else 'DETECTED' if ratio > 2 else 'MARGINAL'}")

if 'coupled_1.0_Z' in labeled_results and 'torn_Z' in labeled_results:
    corr_full, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_torn, _ = counts_to_correlations(labeled_results['torn_Z'], N_TOTAL)
    print(f"\n  SPACETIME TEARING:")
    reductions = []
    for i in range(N_CHAIN):
        c_f = abs(corr_full[i, i+N_CHAIN])
        c_t = abs(corr_torn[i, i+N_CHAIN])
        red = (1 - c_t/max(c_f, 1e-10)) * 100
        reductions.append(red)
        print(f"    q{i}A<->q{i}B: {c_f:.4f} -> {c_t:.4f} ({red:.1f}% reduction)")
    avg_red = np.mean(reductions)
    print(f"    Average reduction: {avg_red:.1f}%")
    print(f"    Signal: {'STRONG' if avg_red > 50 else 'DETECTED' if avg_red > 20 else 'MARGINAL'}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output = {
    'timestamp': datetime.now().isoformat(),
    'labeled_results': {k: {bk: int(bv) for bk, bv in v.items()} 
                        for k, v in labeled_results.items()},
}

fname = f'ibm_hardware_results.json'
with open(fname, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {fname}")
print("=" * 60)
