#!/usr/bin/env python3
"""
EMERGENT SPACETIME — 2x2 LATTICE RESULTS
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

GRID_SIZE = 2
N_PER_FIELD = GRID_SIZE * GRID_SIZE  # 4
N_TOTAL = 2 * N_PER_FIELD            # 8
LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

print("=" * 70)
print("EMERGENT SPACETIME — 2x2 LATTICE HARDWARE RESULTS")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)

# Load job IDs
with open('ibm_2d_2x2_job_ids.json', 'r') as f:
    job_data = json.load(f)
label_to_id = job_data.get('job_ids', {})
id_to_label = {v: k for k, v in label_to_id.items()}
print(f"Loaded {len(label_to_id)} job IDs")

# Retrieve
print("\nRetrieving jobs...")
jobs = service.jobs(limit=30)

labeled_results = {}
for job in jobs:
    jid = job.job_id()
    label = id_to_label.get(jid, None)
    if label is None:
        continue
    status = str(job.status())
    print(f"  {label}: {status}")
    if 'DONE' in status:
        try:
            result = job.result()
            counts = result[0].data.meas.get_counts()
            labeled_results[label] = counts
            print(f"    -> {len(counts)} outcomes, {sum(counts.values())} shots")
        except Exception as e:
            print(f"    -> Error: {e}")

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

def cross_field_correlations(corr, n_per):
    """Get cross-field |C| for each corresponding site pair."""
    cross = [abs(corr[i, i + n_per]) for i in range(n_per)]
    return cross

# =============================================================================
# LAMBDA SWEEP — THE KEY RESULT
# =============================================================================

print(f"\n{'=' * 70}")
print("LAMBDA SWEEP — EMERGENT GEOMETRY vs COUPLING STRENGTH")
print("=" * 70)
print(f"\n  {'Lambda':<10} {'Avg |C|':<12} {'q0A-q0B':<12} {'q1A-q1B':<12} {'q2A-q2B':<12} {'q3A-q3B':<12}")
print(f"  {'-'*70}")

lambda_avgs = []
lambda_crosses = []

for lam in LAMBDA_VALUES:
    label = f'coupled_{lam}_Z'
    if label in labeled_results:
        corr, _ = counts_to_correlations(labeled_results[label], N_TOTAL)
        cross = cross_field_correlations(corr, N_PER_FIELD)
        avg = np.mean(cross)
        lambda_avgs.append((lam, avg))
        lambda_crosses.append((lam, cross))
        print(f"  {lam:<10} {avg:<12.4f} {cross[0]:<12.4f} {cross[1]:<12.4f} {cross[2]:<12.4f} {cross[3]:<12.4f}")
    else:
        print(f"  {lam:<10} MISSING")

# Check monotonicity
if len(lambda_avgs) >= 2:
    avgs_only = [a for _, a in lambda_avgs]
    monotonic_steps = sum(1 for i in range(len(avgs_only)-1) if avgs_only[i+1] > avgs_only[i])
    total_steps = len(avgs_only) - 1
    print(f"\n  Monotonicity: {monotonic_steps}/{total_steps} steps increasing")
    if monotonic_steps >= total_steps - 1:
        print(f"  STRONG: Geometry strength scales with coupling")
    elif monotonic_steps >= total_steps // 2:
        print(f"  DETECTED: General upward trend with some noise")
    else:
        print(f"  MARGINAL: No clear monotonic relationship")

# Ratio analysis
if len(lambda_avgs) >= 2:
    min_lam, min_avg = lambda_avgs[0]
    max_lam, max_avg = max(lambda_avgs, key=lambda x: x[1])
    ratio = max_avg / max(min_avg, 1e-10)
    print(f"\n  Weakest signal (lambda={min_lam}): |C| = {min_avg:.4f}")
    print(f"  Strongest signal (lambda={max_lam}): |C| = {max_avg:.4f}")
    print(f"  Dynamic range: {ratio:.2f}x")

# =============================================================================
# CORE COMPARISONS
# =============================================================================

print(f"\n{'=' * 70}")
print("CORE COMPARISONS")
print("=" * 70)

# Coupling effect (lambda=1.0 vs lambda=0.0)
if 'coupled_1.0_Z' in labeled_results and 'coupled_0.0_Z' in labeled_results:
    corr_c, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_u, _ = counts_to_correlations(labeled_results['coupled_0.0_Z'], N_TOTAL)
    cross_c = cross_field_correlations(corr_c, N_PER_FIELD)
    cross_u = cross_field_correlations(corr_u, N_PER_FIELD)
    avg_c = np.mean(cross_c)
    avg_u = np.mean(cross_u)
    ratio = avg_c / max(avg_u, 1e-10)
    print(f"\n  COUPLING EFFECT (2D, 2x2):")
    print(f"    Coupled (lambda=1.0):   avg |C| = {avg_c:.4f}")
    print(f"    Uncoupled (lambda=0.0): avg |C| = {avg_u:.4f}")
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    Signal: {'STRONG' if ratio > 10 else 'DETECTED' if ratio > 2 else 'MARGINAL'}")
    print(f"    Per-site coupled:   {[f'{c:.4f}' for c in cross_c]}")
    print(f"    Per-site uncoupled: {[f'{c:.4f}' for c in cross_u]}")

# Null hypothesis
if 'coupled_1.0_Z' in labeled_results and 'single_Z' in labeled_results:
    corr_two, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_one, _ = counts_to_correlations(labeled_results['single_Z'], N_TOTAL)
    cross_two = cross_field_correlations(corr_two, N_PER_FIELD)
    cross_one = cross_field_correlations(corr_one, N_PER_FIELD)
    std_two = np.std(cross_two)
    std_one = np.std(cross_one)
    ratio = std_two / max(std_one, 1e-10)
    print(f"\n  NULL HYPOTHESIS (2D, 2x2):")
    print(f"    Two-field sigma: {std_two:.4f}")
    print(f"    Single lattice sigma: {std_one:.4f}")
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    Signal: {'STRONG' if ratio > 5 else 'DETECTED' if ratio > 2 else 'MARGINAL'}")

# Spacetime tearing
if 'coupled_1.0_Z' in labeled_results and 'torn_Z' in labeled_results:
    corr_full, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_torn, _ = counts_to_correlations(labeled_results['torn_Z'], N_TOTAL)
    cross_full = cross_field_correlations(corr_full, N_PER_FIELD)
    cross_torn = cross_field_correlations(corr_torn, N_PER_FIELD)

    print(f"\n  SPACETIME TEARING (2D, 2x2):")
    reductions = []
    site_labels = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    for i in range(N_PER_FIELD):
        c_f = cross_full[i]
        c_t = cross_torn[i]
        red = (1 - c_t / max(c_f, 1e-10)) * 100
        reductions.append(red)
        print(f"    {site_labels[i]}: {c_f:.4f} -> {c_t:.4f}  ({red:.1f}% reduction)")
    avg_red = np.mean(reductions)
    print(f"    Average reduction: {avg_red:.1f}%")
    print(f"    Signal: {'STRONG' if avg_red > 50 else 'DETECTED' if avg_red > 20 else 'MARGINAL'}")

# =============================================================================
# 2D TOPOLOGY CHECK
# =============================================================================

print(f"\n{'=' * 70}")
print("2D TOPOLOGY ANALYSIS")
print("=" * 70)

# In 2x2, all sites are corners (each has 2 neighbors)
# But we can look at diagonal vs adjacent correlations
if 'coupled_1.0_Z' in labeled_results:
    corr_c, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)

    # Cross-field (corresponding sites)
    cross = cross_field_correlations(corr_c, N_PER_FIELD)

    # Intra-field A correlations (2D structure)
    adj_corr = []  # Adjacent (share an edge)
    diag_corr = [] # Diagonal (don't share an edge)

    # Adjacent pairs in 2x2: (0,1), (0,2), (1,3), (2,3)
    for i, j in [(0,1), (0,2), (1,3), (2,3)]:
        adj_corr.append(abs(corr_c[i, j]))

    # Diagonal pair: (0,3), (1,2)
    for i, j in [(0,3), (1,2)]:
        diag_corr.append(abs(corr_c[i, j]))

    print(f"\n  INTRA-FIELD A CORRELATIONS (2D geometry check):")
    print(f"    Adjacent (share edge):  avg = {np.mean(adj_corr):.4f}  {[f'{c:.4f}' for c in adj_corr]}")
    print(f"    Diagonal (no edge):     avg = {np.mean(diag_corr):.4f}  {[f'{c:.4f}' for c in diag_corr]}")
    if np.mean(adj_corr) > np.mean(diag_corr):
        ratio = np.mean(adj_corr) / max(np.mean(diag_corr), 1e-10)
        print(f"    Adjacent/Diagonal ratio: {ratio:.2f}x")
        print(f"    -> Adjacent sites more correlated than diagonal = 2D distance structure")
    else:
        print(f"    -> No clear 2D distance structure in intra-field correlations")

    # Non-corresponding cross-field correlations (should be weaker)
    corresponding = np.mean(cross)
    non_corresponding = []
    for i in range(N_PER_FIELD):
        for j in range(N_PER_FIELD):
            if i != j:
                non_corresponding.append(abs(corr_c[i, j + N_PER_FIELD]))
    avg_non = np.mean(non_corresponding)
    ratio = corresponding / max(avg_non, 1e-10)
    print(f"\n  CROSS-FIELD SITE SPECIFICITY:")
    print(f"    Corresponding sites avg |C|:     {corresponding:.4f}")
    print(f"    Non-corresponding sites avg |C|: {avg_non:.4f}")
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    -> {'STRONG' if ratio > 2 else 'DETECTED' if ratio > 1.3 else 'MARGINAL'} site-specific geometry")

# =============================================================================
# COMPARISON TABLE
# =============================================================================

print(f"\n{'=' * 70}")
print("FULL COMPARISON TABLE")
print("=" * 70)
print(f"""
  Experiment           | Coupling Ratio | Tearing  | Notes
  ---------------------|---------------|----------|------------------
  1D Simulator (8q)    | ~456x         | 87.2%    | Baseline
  1D Torino (8q)       | 95.7x         | 83.4%    | First hardware
  1D Fez (8q)          | 13.7x         | -        | Cross-validation
  2D 2x2 Torino (8q)   | see above     | see above| This experiment
""")

# Save
output = {
    'timestamp': datetime.now().isoformat(),
    'experiment': '2D minimal 2x2 lattice',
    'backend': 'ibm_torino',
    'grid_size': GRID_SIZE,
    'n_total_qubits': N_TOTAL,
    'lambda_sweep': [{'lambda': lam, 'avg_C': avg} for lam, avg in lambda_avgs],
    'lambda_cross_details': [{'lambda': lam, 'cross': cross} for lam, cross in lambda_crosses],
}
with open('ibm_2d_2x2_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to ibm_2d_2x2_results.json")
print("=" * 70)
