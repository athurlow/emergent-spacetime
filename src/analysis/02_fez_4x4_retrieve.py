#!/usr/bin/env python3
"""
EMERGENT SPACETIME — IBM FEZ HARDWARE RESULTS
1D Chain Experiment
Andrew Thurlow | 528 Labs | February 2026
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

N_CHAIN = 4
N_TOTAL = 8

print("=" * 70)
print("EMERGENT SPACETIME — IBM FEZ HARDWARE RESULTS")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)

# Fetch all recent jobs and find the Fez ones
print("\nRetrieving jobs...")
jobs = service.jobs(limit=50)

labeled_results = {}
for job in jobs:
    jid = job.job_id()
    status = str(job.status())
    backend = str(job.backend())
    
    # Only grab fez jobs
    if 'fez' not in backend.lower():
        continue
    
    print(f"  {jid}: {backend} - {status}")
    if 'DONE' in status:
        try:
            result = job.result()
            counts = result[0].data.meas.get_counts()
            # Try to identify by circuit name from job metadata
            circuits = job.inputs.get('pubs', [])
            label = jid  # fallback
            print(f"    -> {len(counts)} unique outcomes, {sum(counts.values())} shots")
            labeled_results[jid] = counts
        except Exception as e:
            print(f"    -> Error: {e}")

if not labeled_results:
    print("\nNo completed Fez results found.")
    exit()

# =============================================================================
# ANALYSIS
# =============================================================================

def counts_to_correlations(counts, n_total):
    total_shots = sum(counts.values())
    sample_bits = list(counts.keys())[0].replace(' ', '')
    n_bits = len(sample_bits)
    n = min(n_total, n_bits)
    
    z_exp = np.zeros(n)
    for bitstring, count in counts.items():
        bits = bitstring.replace(' ', '')
        for i in range(n):
            bit = int(bits[n_bits - 1 - i])
            z_exp[i] += (1 - 2 * bit) * count
    z_exp /= total_shots

    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr[i, j] = 1.0 - z_exp[i]**2
            else:
                zz = 0.0
                for bitstring, count in counts.items():
                    bits = bitstring.replace(' ', '')
                    bi = int(bits[n_bits - 1 - i])
                    bj = int(bits[n_bits - 1 - j])
                    zz += (1 - 2*bi) * (1 - 2*bj) * count
                zz /= total_shots
                corr[i, j] = zz - z_exp[i] * z_exp[j]
                corr[j, i] = corr[i, j]
    return corr, z_exp

# =============================================================================
# RESULTS
# =============================================================================

print(f"\n{'=' * 70}")
print(f"IBM FEZ RESULTS — {len(labeled_results)} completed jobs")
print(f"{'=' * 70}")

all_cross = []
for jid, counts in labeled_results.items():
    sample_bits = list(counts.keys())[0].replace(' ', '')
    n_bits = len(sample_bits)
    
    if n_bits < N_TOTAL:
        print(f"\n  {jid}: only {n_bits} bits, skipping")
        continue
    
    corr, z_exp = counts_to_correlations(counts, N_TOTAL)
    cross = [abs(corr[i, i + N_CHAIN]) for i in range(N_CHAIN)]
    avg_cross = np.mean(cross)
    std_cross = np.std(cross)
    
    print(f"\n  Job {jid}:")
    print(f"    Bitstring length: {n_bits}")
    print(f"    Cross-chain correlations: {[f'{c:.4f}' for c in cross]}")
    print(f"    Average |C|: {avg_cross:.4f}")
    print(f"    Std dev: {std_cross:.4f}")
    all_cross.append({'jid': jid, 'cross': cross, 'avg': avg_cross})

# =============================================================================
# COMPARISON TO TORINO
# =============================================================================

if all_cross:
    print(f"\n{'=' * 70}")
    print(f"FEZ vs TORINO COMPARISON")
    print(f"{'=' * 70}")
    print(f"""
  IBM TORINO (1D, 8 qubits):
    Coupled avg |C|:    0.1332
    Uncoupled avg |C|:  0.0014
    Coupling ratio:     95.7x
    Tearing reduction:  83.4%

  IBM FEZ (1D, 8 qubits):""")
    for entry in all_cross:
        print(f"    Job {entry['jid'][:12]}...: avg |C| = {entry['avg']:.4f}")

# Save
output = {
    'timestamp': datetime.now().isoformat(),
    'backend': 'ibm_fez',
    'results': {jid: {k: int(v) for k, v in counts.items()} 
                for jid, counts in labeled_results.items()},
}
with open('ibm_fez_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to ibm_fez_results.json")
print("=" * 70)
