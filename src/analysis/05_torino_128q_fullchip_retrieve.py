#!/usr/bin/env python3
"""
EMERGENT SPACETIME — FULL-CHIP PARALLEL RESULTS
16 Independent 4+4 Experiments Across IBM Torino (128 qubits)
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

N_CHAIN = 4
N_PER_EXP = 8
N_PARALLEL = 16
N_TOTAL = N_PARALLEL * N_PER_EXP  # 128

REGION_LAMBDAS = [
    0.0, 0.0,
    0.25, 0.25,
    0.5, 0.5,
    0.75, 0.75,
    1.0, 1.0, 1.0,
    1.5, 1.5,
    2.0, 2.0,
    1.0,
]

print("=" * 70)
print("EMERGENT SPACETIME — FULL-CHIP PARALLEL RESULTS")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)

with open('ibm_fullchip_parallel_job_ids.json', 'r') as f:
    job_data = json.load(f)
label_to_id = job_data.get('job_ids', {})
id_to_label = {v: k for k, v in label_to_id.items()}
print(f"Loaded {len(label_to_id)} job IDs")

print("\nRetrieving jobs...")
jobs = service.jobs(limit=10)

results = {}
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
            results[label] = counts
            print(f"    -> {len(counts)} outcomes, {sum(counts.values())} shots")
        except Exception as e:
            print(f"    -> Error: {e}")

if not results:
    print("No results found.")
    exit()

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def extract_region_counts(full_counts, region_idx, n_per_exp, n_total_bits):
    """Extract measurement outcomes for a specific region from the full bitstring."""
    region_counts = {}
    start_bit = region_idx * n_per_exp
    end_bit = start_bit + n_per_exp

    for bitstring, count in full_counts.items():
        bits = bitstring.replace(' ', '')
        n_bits = len(bits)
        # Extract region bits (qiskit is little-endian: bit 0 is rightmost)
        region_bits = ''
        for b in range(start_bit, end_bit):
            if b < n_bits:
                region_bits = bits[n_bits - 1 - b] + region_bits
            else:
                region_bits = '0' + region_bits

        if region_bits in region_counts:
            region_counts[region_bits] += count
        else:
            region_counts[region_bits] = count

    return region_counts

def counts_to_correlations(counts, n_qubits):
    """Compute ZZ correlation matrix from counts."""
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
    """Get cross-chain |C| for corresponding site pairs."""
    return [abs(corr[i, i + n_chain]) for i in range(n_chain)]

# =============================================================================
# EXTRACT AND ANALYZE EACH REGION
# =============================================================================

if 'parallel_sweep' in results:
    full_counts = results['parallel_sweep']
    n_total_bits = max(len(k.replace(' ', '')) for k in full_counts.keys())

    print(f"\n{'=' * 70}")
    print("FULL-CHIP LAMBDA SWEEP — 16 REGIONS")
    print("=" * 70)
    print(f"  {'Region':<8} {'Qubits':<12} {'Lambda':<8} {'q0':<8} {'q1':<8} {'q2':<8} {'q3':<8} {'Avg |C|':<10}")
    print(f"  {'-' * 75}")

    # Collect per-lambda results
    lambda_results = {}

    for reg in range(N_PARALLEL):
        lam = REGION_LAMBDAS[reg]
        q_start = reg * N_PER_EXP
        q_end = q_start + N_PER_EXP - 1

        region_counts = extract_region_counts(full_counts, reg, N_PER_EXP, n_total_bits)
        corr = counts_to_correlations(region_counts, N_PER_EXP)
        cross = cross_field_corr(corr, N_CHAIN)
        avg = np.mean(cross)

        if lam not in lambda_results:
            lambda_results[lam] = []
        lambda_results[lam].append({'region': reg, 'cross': cross, 'avg': avg})

        print(f"  R{reg:<6} {q_start:3d}-{q_end:3d}    {lam:<8} {cross[0]:<8.4f} {cross[1]:<8.4f} {cross[2]:<8.4f} {cross[3]:<8.4f} {avg:<10.4f}")

    # Aggregate by lambda
    print(f"\n{'=' * 70}")
    print("AGGREGATED LAMBDA SWEEP (averaged across replicates)")
    print("=" * 70)
    print(f"  {'Lambda':<10} {'Avg |C|':<12} {'Std':<10} {'N replicates':<14}")
    print(f"  {'-' * 50}")

    sorted_lambdas = sorted(lambda_results.keys())
    agg_avgs = []
    for lam in sorted_lambdas:
        entries = lambda_results[lam]
        avgs = [e['avg'] for e in entries]
        mean_avg = np.mean(avgs)
        std_avg = np.std(avgs)
        agg_avgs.append((lam, mean_avg, std_avg, len(entries)))
        print(f"  {lam:<10} {mean_avg:<12.4f} {std_avg:<10.4f} {len(entries):<14}")

    # Monotonicity
    means = [a[1] for a in agg_avgs]
    mono = sum(1 for i in range(len(means)-1) if means[i+1] > means[i])
    total_steps = len(means) - 1
    print(f"\n  Monotonicity: {mono}/{total_steps} steps increasing")

    # Dynamic range
    min_lam = agg_avgs[0]
    max_entry = max(agg_avgs, key=lambda x: x[1])
    ratio = max_entry[1] / max(min_lam[1], 1e-10)
    print(f"  Weakest (lambda={min_lam[0]}): |C| = {min_lam[1]:.4f}")
    print(f"  Strongest (lambda={max_entry[0]}): |C| = {max_entry[1]:.4f}")
    print(f"  Dynamic range: {ratio:.2f}x")

    # Coupling ratio (lambda=1.0 vs lambda=0.0)
    if 1.0 in lambda_results and 0.0 in lambda_results:
        coupled_avgs = [e['avg'] for e in lambda_results[1.0]]
        uncoupled_avgs = [e['avg'] for e in lambda_results[0.0]]
        coupled_mean = np.mean(coupled_avgs)
        uncoupled_mean = np.mean(uncoupled_avgs)
        coupling_ratio = coupled_mean / max(uncoupled_mean, 1e-10)
        print(f"\n  COUPLING EFFECT (128 qubits, 16 regions):")
        print(f"    Coupled (lambda=1.0, {len(coupled_avgs)} regions):   avg |C| = {coupled_mean:.4f}")
        print(f"    Uncoupled (lambda=0.0, {len(uncoupled_avgs)} regions): avg |C| = {uncoupled_mean:.4f}")
        print(f"    Ratio: {coupling_ratio:.2f}x")
        print(f"    Signal: {'STRONG' if coupling_ratio > 10 else 'DETECTED' if coupling_ratio > 2 else 'MARGINAL'}")

    # Consistency across regions at lambda=1.0
    if 1.0 in lambda_results:
        print(f"\n  CONSISTENCY CHECK (lambda=1.0 across chip):")
        for entry in lambda_results[1.0]:
            print(f"    Region {entry['region']:2d} (qubits {entry['region']*8:3d}-{entry['region']*8+7:3d}): avg |C| = {entry['avg']:.4f}  per-site: {[f'{c:.4f}' for c in entry['cross']]}")
        avgs_10 = [e['avg'] for e in lambda_results[1.0]]
        print(f"    Mean: {np.mean(avgs_10):.4f}, Std: {np.std(avgs_10):.4f}, CV: {np.std(avgs_10)/max(np.mean(avgs_10),1e-10)*100:.1f}%")

# =============================================================================
# TEARING ANALYSIS
# =============================================================================

if 'parallel_torn' in results and 'parallel_sweep' in results:
    torn_counts = results['parallel_torn']
    sweep_counts = results['parallel_sweep']

    print(f"\n{'=' * 70}")
    print("FULL-CHIP TEARING — ALL 16 REGIONS")
    print("=" * 70)

    # For tearing, compare lambda=1.0 regions in sweep vs torn
    # But all torn regions were at lambda=1.0
    print(f"  {'Region':<8} {'Qubits':<12} {'Coupled |C|':<14} {'Torn |C|':<12} {'Reduction':<12}")
    print(f"  {'-' * 60}")

    all_reductions = []
    for reg in range(N_PARALLEL):
        q_start = reg * N_PER_EXP

        torn_region = extract_region_counts(torn_counts, reg, N_PER_EXP,
                        max(len(k.replace(' ', '')) for k in torn_counts.keys()))
        corr_torn = counts_to_correlations(torn_region, N_PER_EXP)
        cross_torn = cross_field_corr(corr_torn, N_CHAIN)
        avg_torn = np.mean(cross_torn)

        # For coupled reference, use region's own lambda from sweep
        sweep_region = extract_region_counts(sweep_counts, reg, N_PER_EXP, n_total_bits)
        corr_sweep = counts_to_correlations(sweep_region, N_PER_EXP)
        cross_sweep = cross_field_corr(corr_sweep, N_CHAIN)
        avg_sweep = np.mean(cross_sweep)

        # Torn circuit was all lambda=1.0, so compare torn vs torn's own coupled signal
        # We need to compare against a lambda=1.0 coupled reference
        # Use the average of lambda=1.0 regions from the sweep as reference
        if 1.0 in lambda_results:
            coupled_ref = np.mean([e['avg'] for e in lambda_results[1.0]])
        else:
            coupled_ref = avg_sweep

        red = (1 - avg_torn / max(coupled_ref, 1e-10)) * 100
        all_reductions.append(red)
        print(f"  R{reg:<6} {q_start:3d}-{q_start+7:3d}    {coupled_ref:<14.4f} {avg_torn:<12.4f} {red:<12.1f}%")

    pos_red = [r for r in all_reductions if r > 0]
    print(f"\n  Average reduction (all): {np.mean(all_reductions):.1f}%")
    if pos_red:
        print(f"  Average reduction (positive): {np.mean(pos_red):.1f}%")
        print(f"  Positive regions: {len(pos_red)}/{N_PARALLEL}")

# =============================================================================
# CHIP QUALITY MAP
# =============================================================================

if 'parallel_sweep' in results and 1.0 in lambda_results:
    print(f"\n{'=' * 70}")
    print("CHIP QUALITY MAP — SIGNAL STRENGTH BY REGION")
    print("=" * 70)

    all_regions = []
    for reg in range(N_PARALLEL):
        lam = REGION_LAMBDAS[reg]
        entries = [e for e in lambda_results.get(lam, []) if e['region'] == reg]
        if entries:
            all_regions.append((reg, lam, entries[0]['avg']))

    # Sort by signal strength
    sorted_regions = sorted(all_regions, key=lambda x: x[2], reverse=True)
    print(f"\n  Ranked by signal strength:")
    for reg, lam, avg in sorted_regions:
        q_start = reg * N_PER_EXP
        bar = '#' * int(avg * 500)
        print(f"    R{reg:2d} (q{q_start:3d}-{q_start+7:3d}) lambda={lam}: |C|={avg:.4f} {bar}")

# =============================================================================
# FULL COMPARISON TABLE
# =============================================================================

print(f"\n{'=' * 70}")
print("COMPLETE EXPERIMENT HISTORY")
print("=" * 70)
print(f"""
  Experiment               | Qubits | Coupling | Tearing  | Lambda
  -------------------------|--------|----------|----------|--------
  1D 4+4 (Torino)          |   8    |  95.7x   |  83.4%   |   --
  1D 4+4 (Fez)             |   8    |  13.7x   |   --     |   --
  2D 2x2 (Torino)          |   8    |  11.6x   |  91.6%   | 5/6 mono
  1D 8+8 (Torino)          |  16    |  15.1x   |  85.3%   | 5/6 mono
  Full-chip parallel        | 128    | see above| see above| see above
""")

# Save
output = {
    'timestamp': __import__('datetime').datetime.now().isoformat(),
    'experiment': 'Full-chip parallel 128 qubits',
    'backend': 'ibm_torino',
    'n_total_qubits': N_TOTAL,
    'n_parallel': N_PARALLEL,
    'aggregated_lambda_sweep': [
        {'lambda': lam, 'mean_C': mean, 'std_C': std, 'n_replicates': n}
        for lam, mean, std, n in agg_avgs
    ] if 'agg_avgs' in dir() else [],
}
with open('ibm_fullchip_parallel_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to ibm_fullchip_parallel_results.json")
print("=" * 70)
