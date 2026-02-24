#!/usr/bin/env python3
"""
IBM QUANTUM HARDWARE — RETRIEVE AND ANALYZE RESULTS
Emergent Spacetime — Two Scalar Field Model
Andrew Thurlow | 528 Labs
"""

import numpy as np
import json
from datetime import datetime
from qiskit_ibm_runtime import QiskitRuntimeService

# =============================================================================
# PASTE YOUR IBM API KEY BETWEEN THE QUOTES ON THE NEXT LINE
# =============================================================================
IBM_TOKEN = 'PASTE_YOUR_API_KEY_HERE'
# =============================================================================

print("=" * 60)
print("EMERGENT SPACETIME — IBM HARDWARE RESULTS")
print("Andrew Thurlow | 528 Labs")
print("=" * 60)

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)

# Get recent jobs
print("\nRetrieving jobs...")
jobs = service.jobs(limit=24)

# Sort into experiment groups
results = {}
for job in jobs:
    jid = job.job_id()
    status = str(job.status())
    print(f"  {jid}: {status}")
    
    if 'DONE' in status:
        try:
            result = job.result()
            # Get counts from the result
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()
            results[jid] = counts
            print(f"    -> {len(counts)} unique outcomes")
        except Exception as e:
            print(f"    -> Error retrieving: {e}")

if not results:
    print("\nNo completed results found yet. Jobs may still be running.")
    exit()

# =============================================================================
# ANALYSIS
# =============================================================================

N_CHAIN = 4
N_TOTAL = 8

def counts_to_correlations(counts, n_total):
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

# Try to load job IDs to match labels
try:
    with open('ibm_job_ids.json', 'r') as f:
        job_data = json.load(f)
    label_to_id = job_data.get('job_ids', {})
    id_to_label = {v: k for k, v in label_to_id.items()}
except:
    id_to_label = {}

print("\n" + "=" * 60)
print("HARDWARE RESULTS ANALYSIS")
print("=" * 60)

# Organize results by label
labeled_results = {}
for jid, counts in results.items():
    label = id_to_label.get(jid, jid)
    labeled_results[label] = counts
    
    # Basic stats
    total = sum(counts.values())
    n_unique = len(counts)
    print(f"\n  {label}:")
    print(f"    Total shots: {total}")
    print(f"    Unique outcomes: {n_unique}")

# Analyze Z-basis results
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

# Key comparisons
print("\n" + "=" * 60)
print("KEY COMPARISONS")
print("=" * 60)

if 'coupled_1.0_Z' in labeled_results and 'coupled_0.0_Z' in labeled_results:
    corr_c, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_u, _ = counts_to_correlations(labeled_results['coupled_0.0_Z'], N_TOTAL)
    
    cross_coupled = np.mean([abs(corr_c[i, i+N_CHAIN]) for i in range(N_CHAIN)])
    cross_uncoupled = np.mean([abs(corr_u[i, i+N_CHAIN]) for i in range(N_CHAIN)])
    
    print(f"\n  COUPLING EFFECT:")
    print(f"    Coupled (lambda=1.0):   |C| = {cross_coupled:.4f}")
    print(f"    Uncoupled (lambda=0.0): |C| = {cross_uncoupled:.4f}")
    print(f"    Ratio: {cross_coupled/max(cross_uncoupled, 1e-10):.2f}x")
    print(f"    Signal detected: {'YES' if cross_coupled > cross_uncoupled * 1.5 else 'MARGINAL'}")

if 'coupled_1.0_Z' in labeled_results and 'single_Z' in labeled_results:
    corr_two, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_one, _ = counts_to_correlations(labeled_results['single_Z'], N_TOTAL)
    
    std_two = np.std([abs(corr_two[i, i+N_CHAIN]) for i in range(N_CHAIN)])
    std_one = np.std([abs(corr_one[i, i+N_CHAIN]) for i in range(N_CHAIN)])
    
    print(f"\n  NULL HYPOTHESIS:")
    print(f"    Two-field structure: {std_two:.4f}")
    print(f"    Single chain:       {std_one:.4f}")
    ratio = std_two / max(std_one, 1e-10)
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    Two-field structure distinct: {'YES' if ratio > 2 else 'MARGINAL'}")

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
    print(f"    Tearing detected: {'YES' if avg_red > 30 else 'MARGINAL'}")

# Compare with simulator
print(f"\n" + "=" * 60)
print("HARDWARE vs SIMULATOR COMPARISON")
print("=" * 60)
print(f"""
  Simulator results (8 qubits, lambda=1.0):
    Cross-chain |C|: 0.2393
    Null hypothesis ratio: 10.33x
    Tearing reduction: 87.2%
""")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'backend': 'ibm_torino',
    'labeled_results': {k: {bk: int(bv) for bk, bv in v.items()} 
                        for k, v in labeled_results.items()},
}

with open('ibm_torino_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Results saved to ibm_torino_results.json")
print("=" * 60)
