#!/usr/bin/env python3
"""
EMERGENT SPACETIME — UNIVERSALITY TEST RESULTS
Ising vs Heisenberg Inter-Chain Coupling
Andrew Thurlow | 528 Labs | February 2026
"""

import numpy as np
import json
from qiskit_ibm_runtime import QiskitRuntimeService

# =============================================================================
IBM_TOKEN = 'PASTE_YOUR_API_KEY_HERE'
# =============================================================================

N_CHAIN = 4
N_QUBITS = 8
LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

print("=" * 70)
print("EMERGENT SPACETIME — UNIVERSALITY TEST RESULTS")
print("Ising vs Heisenberg Inter-Chain Coupling")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)

with open('universality_test_job_ids.json', 'r') as f:
    job_data = json.load(f)

# =============================================================================
# RETRIEVE ALL JOBS
# =============================================================================

all_labels = job_data['labels']
all_counts = {}

print("\nRetrieving jobs...")
for batch_name, batch_info in job_data['job_ids'].items():
    jid = batch_info['job_id']
    batch_labels = batch_info['labels']
    print(f"  {batch_name} ({jid})...", end=" ", flush=True)

    job = service.job(jid)
    status = str(job.status())
    print(status)

    if 'DONE' in status:
        result = job.result()
        for i, label in enumerate(batch_labels):
            try:
                counts = result[i].data.meas.get_counts()
                all_counts[label] = counts
                print(f"    {label}: {len(counts)} outcomes")
            except Exception as e:
                print(f"    {label}: ERROR - {e}")

if not all_counts:
    print("No results ready yet.")
    exit()

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def counts_to_zz_correlations(counts, n_qubits):
    """Compute ZZ correlation matrix."""
    total = sum(counts.values())
    sample = list(counts.keys())[0].replace(' ', '')
    n_bits = len(sample)
    n = min(n_qubits, n_bits)

    z_exp = np.zeros(n)
    for bitstring, count in counts.items():
        bits = bitstring.replace(' ', '')
        for i in range(n):
            bit = int(bits[n_bits - 1 - i])
            z_exp[i] += (1 - 2 * bit) * count
    z_exp /= total

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
                zz /= total
                corr[i, j] = zz - z_exp[i] * z_exp[j]
                corr[j, i] = corr[i, j]
    return corr

def cross_field_corr(corr, n_chain):
    """Cross-chain |C| for corresponding site pairs."""
    return [abs(corr[i, i + n_chain]) for i in range(n_chain)]

# =============================================================================
# ISING ANALYSIS
# =============================================================================

print(f"\n{'=' * 70}")
print("ISING INTER-CHAIN COUPLING (ZZ only)")
print("=" * 70)

ising_sweep = {}
print(f"\n  {'Lambda':<10} {'q0':<10} {'q1':<10} {'q2':<10} {'q3':<10} {'Avg |C|':<12}")
print(f"  {'-' * 60}")

for lam in LAMBDAS:
    key = f'ising_lam_{lam}'
    if key not in all_counts:
        continue
    corr = counts_to_zz_correlations(all_counts[key], N_QUBITS)
    cross = cross_field_corr(corr, N_CHAIN)
    avg = np.mean(cross)
    ising_sweep[lam] = {'cross': cross, 'avg': avg}
    print(f"  {lam:<10} {cross[0]:<10.4f} {cross[1]:<10.4f} {cross[2]:<10.4f} {cross[3]:<10.4f} {avg:<12.4f}")

# Ising tearing
if 'ising_torn' in all_counts and 1.0 in ising_sweep:
    corr_torn = counts_to_zz_correlations(all_counts['ising_torn'], N_QUBITS)
    cross_torn = cross_field_corr(corr_torn, N_CHAIN)
    avg_torn = np.mean(cross_torn)
    avg_coupled = ising_sweep[1.0]['avg']
    ising_reduction = (1 - avg_torn / max(avg_coupled, 1e-10)) * 100
    print(f"\n  Ising Tearing:")
    print(f"    Coupled:  {avg_coupled:.4f}")
    print(f"    Torn:     {avg_torn:.4f}")
    print(f"    Reduction: {ising_reduction:.1f}%")

# =============================================================================
# HEISENBERG ANALYSIS
# =============================================================================

print(f"\n{'=' * 70}")
print("HEISENBERG INTER-CHAIN COUPLING (ZZ + XX)")
print("=" * 70)

heis_sweep = {}
print(f"\n  {'Lambda':<10} {'q0':<10} {'q1':<10} {'q2':<10} {'q3':<10} {'Avg |C|':<12}")
print(f"  {'-' * 60}")

for lam in LAMBDAS:
    key = f'heis_lam_{lam}'
    if key not in all_counts:
        continue
    corr = counts_to_zz_correlations(all_counts[key], N_QUBITS)
    cross = cross_field_corr(corr, N_CHAIN)
    avg = np.mean(cross)
    heis_sweep[lam] = {'cross': cross, 'avg': avg}
    print(f"  {lam:<10} {cross[0]:<10.4f} {cross[1]:<10.4f} {cross[2]:<10.4f} {cross[3]:<10.4f} {avg:<12.4f}")

# Heisenberg tearing
if 'heis_torn' in all_counts and 1.0 in heis_sweep:
    corr_torn = counts_to_zz_correlations(all_counts['heis_torn'], N_QUBITS)
    cross_torn = cross_field_corr(corr_torn, N_CHAIN)
    avg_torn = np.mean(cross_torn)
    avg_coupled = heis_sweep[1.0]['avg']
    heis_reduction = (1 - avg_torn / max(avg_coupled, 1e-10)) * 100
    print(f"\n  Heisenberg Tearing:")
    print(f"    Coupled:  {avg_coupled:.4f}")
    print(f"    Torn:     {avg_torn:.4f}")
    print(f"    Reduction: {heis_reduction:.1f}%")

# =============================================================================
# HEAD-TO-HEAD COMPARISON
# =============================================================================

print(f"\n{'=' * 70}")
print("HEAD-TO-HEAD: ISING vs HEISENBERG")
print("=" * 70)

print(f"\n  {'Lambda':<10} {'Ising |C|':<14} {'Heis |C|':<14} {'Ratio I/H':<12}")
print(f"  {'-' * 52}")

common_lambdas = sorted(set(ising_sweep.keys()) & set(heis_sweep.keys()))
for lam in common_lambdas:
    i_avg = ising_sweep[lam]['avg']
    h_avg = heis_sweep[lam]['avg']
    ratio = i_avg / max(h_avg, 1e-10)
    print(f"  {lam:<10} {i_avg:<14.4f} {h_avg:<14.4f} {ratio:<12.2f}")

# Monotonicity check
for name, sweep in [('Ising', ising_sweep), ('Heisenberg', heis_sweep)]:
    avgs = [sweep[lam]['avg'] for lam in sorted(sweep.keys())]
    mono = sum(1 for i in range(len(avgs)-1) if avgs[i+1] > avgs[i])
    total = len(avgs) - 1
    max_val = max(avgs)
    min_val = avgs[0] if avgs else 0
    dyn_range = max_val / max(min_val, 1e-10)
    peak_lam = sorted(sweep.keys())[np.argmax(avgs)]
    print(f"\n  {name}:")
    print(f"    Monotonicity: {mono}/{total} steps increasing")
    print(f"    Dynamic range: {dyn_range:.2f}x")
    print(f"    Peak at lambda = {peak_lam}")

# Coupling ratios
print(f"\n  COUPLING RATIOS:")
for name, sweep in [('Ising', ising_sweep), ('Heisenberg', heis_sweep)]:
    if 1.0 in sweep and 0.0 in sweep:
        coupled = sweep[1.0]['avg']
        uncoupled = sweep[0.0]['avg']
        ratio = coupled / max(uncoupled, 1e-10)
        print(f"    {name}: {ratio:.2f}x  (coupled={coupled:.4f}, uncoupled={uncoupled:.4f})")

# =============================================================================
# UNIVERSALITY VERDICT
# =============================================================================

print(f"\n{'=' * 70}")
print("UNIVERSALITY VERDICT")
print("=" * 70)

if ising_sweep and heis_sweep:
    # Check if both are monotonic
    i_avgs = [ising_sweep[l]['avg'] for l in sorted(ising_sweep.keys())]
    h_avgs = [heis_sweep[l]['avg'] for l in sorted(heis_sweep.keys())]

    i_mono = sum(1 for j in range(len(i_avgs)-1) if i_avgs[j+1] > i_avgs[j])
    h_mono = sum(1 for j in range(len(h_avgs)-1) if h_avgs[j+1] > h_avgs[j])

    # Correlation between the two curves
    if len(i_avgs) == len(h_avgs):
        correlation = np.corrcoef(i_avgs, h_avgs)[0, 1]
        print(f"\n  Curve correlation (Ising vs Heisenberg): {correlation:.4f}")

        if correlation > 0.9:
            print(f"  STRONG UNIVERSALITY: Both Hamiltonians produce the same")
            print(f"  emergent geometry curve. The geometric signal is NOT an")
            print(f"  artifact of the specific Hamiltonian.")
        elif correlation > 0.7:
            print(f"  MODERATE UNIVERSALITY: Curves are similar but not identical.")
            print(f"  Geometry partially depends on coupling type.")
        elif correlation > 0.4:
            print(f"  WEAK UNIVERSALITY: Some shared structure, but significant")
            print(f"  differences. Hamiltonian matters more than expected.")
        else:
            print(f"  NO UNIVERSALITY: Different Hamiltonians produce different")
            print(f"  correlation structures. Geometric interpretation may be")
            print(f"  specific to Heisenberg coupling.")

    # Both monotonic?
    i_total = len(i_avgs) - 1
    h_total = len(h_avgs) - 1
    print(f"\n  Ising monotonic: {i_mono}/{i_total}")
    print(f"  Heisenberg monotonic: {h_mono}/{h_total}")

    if i_mono >= i_total - 1 and h_mono >= h_total - 1:
        print(f"\n  BOTH curves show monotonic coupling-geometry scaling.")
        print(f"  Emergent geometry is a UNIVERSAL property of coupled fields.")
    elif i_mono >= i_total - 2:
        print(f"\n  Ising shows clear monotonic trend.")
        print(f"  Geometry survives Hamiltonian change.")

    # Both show tearing?
    if 'ising_torn' in all_counts and 'heis_torn' in all_counts:
        print(f"\n  Ising tearing:     {ising_reduction:.1f}%")
        print(f"  Heisenberg tearing: {heis_reduction:.1f}%")
        if ising_reduction > 20 and heis_reduction > 20:
            print(f"  BOTH show spacetime disconnection upon decoupling.")

print(f"\n{'=' * 70}")

# Save results
output = {
    'timestamp': __import__('datetime').datetime.now().isoformat(),
    'experiment': 'Universality test: Ising vs Heisenberg',
    'backend': 'ibm_torino',
    'ising_sweep': {str(k): v['avg'] for k, v in ising_sweep.items()},
    'heisenberg_sweep': {str(k): v['avg'] for k, v in heis_sweep.items()},
}
if 'ising_reduction' in dir():
    output['ising_tearing_pct'] = ising_reduction
if 'heis_reduction' in dir():
    output['heisenberg_tearing_pct'] = heis_reduction
if 'correlation' in dir():
    output['curve_correlation'] = correlation

with open('universality_test_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to universality_test_results.json")
print("=" * 70)
