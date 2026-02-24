#!/usr/bin/env python3
"""
EMERGENT SPACETIME — 1D CHAIN (8+8) RESULTS
Andrew Thurlow | 528 Labs | February 2026
"""

import numpy as np
import json
from qiskit_ibm_runtime import QiskitRuntimeService

# =============================================================================
# PASTE YOUR IBM API KEY BETWEEN THE QUOTES ON THE NEXT LINE
# =============================================================================
IBM_TOKEN = 'PASTE_YOUR_API_KEY_HERE'
# =============================================================================

N_CHAIN = 8
N_TOTAL = 2 * N_CHAIN  # 16
LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

print("=" * 70)
print("EMERGENT SPACETIME — 1D CHAIN (8+8) RESULTS")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)

with open('ibm_1d_8x8_job_ids.json', 'r') as f:
    job_data = json.load(f)
label_to_id = job_data.get('job_ids', {})
id_to_label = {v: k for k, v in label_to_id.items()}
print(f"Loaded {len(label_to_id)} job IDs")

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

if not labeled_results:
    print("No results found.")
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

def cross_chain(corr, n_chain):
    return [abs(corr[i, i + n_chain]) for i in range(n_chain)]

# =============================================================================
# LAMBDA SWEEP
# =============================================================================

print(f"\n{'=' * 70}")
print("LAMBDA SWEEP — 16 QUBITS (8+8 CHAIN)")
print("=" * 70)

header = f"  {'Lambda':<8}"
for i in range(N_CHAIN):
    header += f"  q{i}A-q{i}B "
header += f"  {'Avg |C|':>8}"
print(header)
print(f"  {'-' * 100}")

lambda_avgs = []
for lam in LAMBDA_VALUES:
    label = f'coupled_{lam}_Z'
    if label in labeled_results:
        corr, _ = counts_to_correlations(labeled_results[label], N_TOTAL)
        cross = cross_chain(corr, N_CHAIN)
        avg = np.mean(cross)
        lambda_avgs.append((lam, avg, cross))
        row = f"  {lam:<8}"
        for c in cross:
            row += f"  {c:.4f}  "
        row += f"  {avg:>8.4f}"
        print(row)

if len(lambda_avgs) >= 2:
    avgs = [a for _, a, _ in lambda_avgs]
    mono = sum(1 for i in range(len(avgs)-1) if avgs[i+1] > avgs[i])
    total = len(avgs) - 1
    print(f"\n  Monotonicity: {mono}/{total} steps increasing")
    min_l, min_a = lambda_avgs[0][0], lambda_avgs[0][1]
    max_entry = max(lambda_avgs, key=lambda x: x[1])
    max_l, max_a = max_entry[0], max_entry[1]
    ratio = max_a / max(min_a, 1e-10)
    print(f"  Weakest (lambda={min_l}): |C| = {min_a:.4f}")
    print(f"  Strongest (lambda={max_l}): |C| = {max_a:.4f}")
    print(f"  Dynamic range: {ratio:.2f}x")

# =============================================================================
# CORE COMPARISONS
# =============================================================================

print(f"\n{'=' * 70}")
print("CORE COMPARISONS")
print("=" * 70)

cross_c = None
if 'coupled_1.0_Z' in labeled_results and 'coupled_0.0_Z' in labeled_results:
    corr_c, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    corr_u, _ = counts_to_correlations(labeled_results['coupled_0.0_Z'], N_TOTAL)
    cross_c = cross_chain(corr_c, N_CHAIN)
    cross_u = cross_chain(corr_u, N_CHAIN)
    avg_c = np.mean(cross_c)
    avg_u = np.mean(cross_u)
    ratio = avg_c / max(avg_u, 1e-10)
    print(f"\n  COUPLING EFFECT (8+8 chain, 16q):")
    print(f"    Coupled (lambda=1.0):   avg |C| = {avg_c:.4f}")
    print(f"    Uncoupled (lambda=0.0): avg |C| = {avg_u:.4f}")
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    Signal: {'STRONG' if ratio > 10 else 'DETECTED' if ratio > 2 else 'MARGINAL'}")
    print(f"    Per-site coupled:   {[f'{c:.4f}' for c in cross_c]}")
    print(f"    Per-site uncoupled: {[f'{c:.4f}' for c in cross_u]}")

if 'coupled_1.0_Z' in labeled_results and 'single_Z' in labeled_results:
    corr_s, _ = counts_to_correlations(labeled_results['single_Z'], N_TOTAL)
    cross_s = cross_chain(corr_s, N_CHAIN)
    avg_s = np.mean(cross_s)
    print(f"\n  NULL HYPOTHESIS (8+8 chain, 16q):")
    print(f"    Two-chain avg |C|:    {avg_c:.4f}")
    print(f"    Single chain avg |C|: {avg_s:.4f}")
    print(f"    Ratio: {avg_c / max(avg_s, 1e-10):.2f}x")

if 'coupled_1.0_Z' in labeled_results and 'torn_Z' in labeled_results:
    corr_t, _ = counts_to_correlations(labeled_results['torn_Z'], N_TOTAL)
    cross_t = cross_chain(corr_t, N_CHAIN)
    print(f"\n  ENTANGLEMENT REDUCTION (8+8 chain, 16q):")
    reductions = []
    for i in range(N_CHAIN):
        cf = cross_c[i]
        ct = cross_t[i]
        red = (1 - ct / max(cf, 1e-10)) * 100
        reductions.append(red)
        print(f"    q{i}A-q{i}B: {cf:.4f} -> {ct:.4f}  ({red:.1f}%)")
    pos_red = [r for r in reductions if r > 0]
    print(f"    Average (all): {np.mean(reductions):.1f}%")
    if pos_red:
        print(f"    Average (positive): {np.mean(pos_red):.1f}%")

# =============================================================================
# DISTANCE STRUCTURE
# =============================================================================

if 'coupled_1.0_Z' in labeled_results:
    print(f"\n{'=' * 70}")
    print("INTRA-CHAIN DISTANCE STRUCTURE")
    print("=" * 70)

    corr_c_full, _ = counts_to_correlations(labeled_results['coupled_1.0_Z'], N_TOTAL)
    dist_bins = {}
    for i in range(N_CHAIN):
        for j in range(i+1, N_CHAIN):
            d = abs(i - j)
            c_val = abs(corr_c_full[i, j])
            if d not in dist_bins:
                dist_bins[d] = []
            dist_bins[d].append(c_val)

    for d in sorted(dist_bins.keys()):
        vals = dist_bins[d]
        avg = np.mean(vals)
        print(f"  Distance {d}: avg |C| = {avg:.4f} ({len(vals)} pairs)")

    dists = sorted(dist_bins.keys())
    avgs_d = [np.mean(dist_bins[d]) for d in dists]
    decreasing = sum(1 for i in range(len(avgs_d)-1) if avgs_d[i+1] < avgs_d[i])
    print(f"  Decreasing: {decreasing}/{len(dists)-1} steps")
    print(f"  -> {'STRONG emergent metric' if decreasing >= len(dists)-2 else 'DETECTED' if decreasing >= len(dists)//2 else 'MARGINAL'}")

    # End-to-end vs neighbors
    print(f"\n  CHAIN POSITION ANALYSIS:")
    print(f"    End sites (q0, q7):    avg |C| = {np.mean([cross_c[0], cross_c[7]]):.4f}")
    print(f"    Middle sites (q3, q4): avg |C| = {np.mean([cross_c[3], cross_c[4]]):.4f}")
    print(f"    All interior (q1-q6):  avg |C| = {np.mean(cross_c[1:7]):.4f}")

# =============================================================================
# FULL COMPARISON
# =============================================================================

print(f"\n{'=' * 70}")
print("FULL EXPERIMENT COMPARISON")
print("=" * 70)
print(f"""
  Experiment             | Qubits | Depth | Coupling | Reduction
  -----------------------|--------|-------|----------|----------
  1D 4+4 (Torino)        |   8    | ~410  |  95.7x   |   83.4%
  1D 4+4 (Fez)           |   8    | ~410  |  13.7x   |    --
  2D 2x2 (Torino)        |   8    | ~450  |  11.6x   |   91.6%
  2D 2x4 (Torino)        |  16    | ~500  |   2.6x   |   82.8%*
  1D 8+8 (Torino)        |  16    | ~780  | see above| see above
  
  * positive sites only
""")

# Save
output = {
    'timestamp': __import__('datetime').datetime.now().isoformat(),
    'experiment': '1D 8+8 chain',
    'backend': 'ibm_torino',
    'n_chain': N_CHAIN,
    'n_total': N_TOTAL,
    'lambda_sweep': [{'lambda': l, 'avg_C': a, 'per_site': c} for l, a, c in lambda_avgs],
}
with open('ibm_1d_8x8_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to ibm_1d_8x8_results.json")
print("=" * 70)
