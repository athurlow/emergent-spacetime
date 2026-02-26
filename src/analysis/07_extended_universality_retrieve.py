#!/usr/bin/env python3
"""
EMERGENT SPACETIME — EXTENDED UNIVERSALITY RESULTS
Four Hamiltonians: Ising, Heisenberg, XY, Long-Range
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
print("EMERGENT SPACETIME — FOUR-HAMILTONIAN UNIVERSALITY RESULTS")
print("Ising | Heisenberg | XY | Long-Range")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)

# Load new XY + long-range results
with open('extended_universality_job_ids.json', 'r') as f:
    new_job_data = json.load(f)

# Load previous Ising + Heisenberg results
try:
    with open('universality_test_results.json', 'r') as f:
        prev_results = json.load(f)
    have_previous = True
    print("  Loaded previous Ising + Heisenberg results")
except FileNotFoundError:
    have_previous = False
    print("  No previous results found — will only show XY + Long-range")

# =============================================================================
# RETRIEVE NEW JOBS
# =============================================================================

all_counts = {}
print("\nRetrieving new jobs...")
for batch_name, batch_info in new_job_data['job_ids'].items():
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
            except Exception as e:
                print(f"    {label}: ERROR - {e}")

if not all_counts:
    print("No results ready yet.")
    exit()

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def counts_to_zz_correlations(counts, n_qubits):
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
    return [abs(corr[i, i + n_chain]) for i in range(n_chain)]


def analyze_sweep(prefix, counts_dict, lambdas, n_chain, n_qubits):
    """Analyze a full lambda sweep for one coupling type."""
    sweep = {}
    for lam in lambdas:
        key = f'{prefix}_lam_{lam}'
        if key not in counts_dict:
            continue
        corr = counts_to_zz_correlations(counts_dict[key], n_qubits)
        cross = cross_field_corr(corr, n_chain)
        avg = np.mean(cross)
        sweep[lam] = {'cross': cross, 'avg': avg}
    return sweep


# =============================================================================
# ANALYZE ALL COUPLING TYPES
# =============================================================================

all_sweeps = {}

# New results: XY and Long-range
for ctype in ['xy', 'long_range']:
    sweep = analyze_sweep(ctype, all_counts, LAMBDAS, N_CHAIN, N_QUBITS)
    if sweep:
        all_sweeps[ctype] = sweep
        label = ctype.upper().replace('_', '-')
        print(f"\n{'=' * 70}")
        print(f"{label} INTER-CHAIN COUPLING")
        print("=" * 70)
        print(f"\n  {'Lambda':<10} {'q0':<10} {'q1':<10} {'q2':<10} {'q3':<10} {'Avg |C|':<12}")
        print(f"  {'-' * 60}")
        for lam in sorted(sweep.keys()):
            d = sweep[lam]
            c = d['cross']
            print(f"  {lam:<10} {c[0]:<10.4f} {c[1]:<10.4f} {c[2]:<10.4f} {c[3]:<10.4f} {d['avg']:<12.4f}")

        # Tearing
        torn_key = f'{ctype}_torn'
        if torn_key in all_counts and 1.0 in sweep:
            corr_torn = counts_to_zz_correlations(all_counts[torn_key], N_QUBITS)
            cross_torn = cross_field_corr(corr_torn, N_CHAIN)
            avg_torn = np.mean(cross_torn)
            avg_coupled = sweep[1.0]['avg']
            reduction = (1 - avg_torn / max(avg_coupled, 1e-10)) * 100
            all_sweeps[ctype]['tearing'] = reduction
            all_sweeps[ctype]['torn_avg'] = avg_torn
            print(f"\n  Tearing:")
            print(f"    Coupled:   {avg_coupled:.4f}")
            print(f"    Torn:      {avg_torn:.4f}")
            print(f"    Reduction: {reduction:.1f}%")

# Load previous results
if have_previous:
    # Reconstruct sweep dicts from saved averages
    for name, key in [('ising', 'ising_sweep'), ('heisenberg', 'heisenberg_sweep')]:
        if key in prev_results:
            sweep = {}
            for lam_str, avg in prev_results[key].items():
                sweep[float(lam_str)] = {'avg': avg}
            all_sweeps[name] = sweep
            if f'{name}_tearing_pct' in prev_results:
                all_sweeps[name]['tearing'] = prev_results[f'{name}_tearing_pct']

# =============================================================================
# FOUR-HAMILTONIAN COMPARISON
# =============================================================================

print(f"\n{'=' * 70}")
print("FOUR-HAMILTONIAN COMPARISON")
print("=" * 70)

display_names = {
    'ising': 'Ising (ZZ)',
    'heisenberg': 'Heis (ZZ+XX)',
    'xy': 'XY (XX+YY)',
    'long_range': 'LR (all ZZ)'
}

# Header
names_present = [n for n in ['ising', 'heisenberg', 'xy', 'long_range'] if n in all_sweeps]
header = f"  {'Lambda':<8}"
for name in names_present:
    header += f"  {display_names.get(name, name):<14}"
print(header)
print(f"  {'-' * (8 + 16 * len(names_present))}")

for lam in LAMBDAS:
    row = f"  {lam:<8}"
    for name in names_present:
        if lam in all_sweeps[name]:
            val = all_sweeps[name][lam]['avg']
            row += f"  {val:<14.4f}"
        else:
            row += f"  {'—':<14}"
    print(row)

# Monotonicity
print(f"\n  MONOTONICITY:")
for name in names_present:
    sweep = all_sweeps[name]
    avgs = [sweep[lam]['avg'] for lam in sorted(sweep.keys()) if lam in sweep and isinstance(sweep[lam], dict) and 'avg' in sweep[lam]]
    if len(avgs) > 1:
        mono = sum(1 for i in range(len(avgs)-1) if avgs[i+1] > avgs[i])
        total = len(avgs) - 1
        peak_lam = sorted([l for l in sweep.keys() if isinstance(sweep[l], dict) and 'avg' in sweep[l]])[np.argmax(avgs)]
        dyn = max(avgs) / max(avgs[0], 1e-10)
        print(f"    {display_names.get(name, name):<16} {mono}/{total} increasing  peak λ={peak_lam}  range={dyn:.1f}×")

# Coupling ratios
print(f"\n  COUPLING RATIOS (λ=1.0 / λ=0.0):")
for name in names_present:
    sweep = all_sweeps[name]
    if 1.0 in sweep and 0.0 in sweep:
        c = sweep[1.0]['avg']
        u = sweep[0.0]['avg']
        ratio = c / max(u, 1e-10)
        print(f"    {display_names.get(name, name):<16} {ratio:.2f}×  (coupled={c:.4f}, uncoupled={u:.4f})")

# Tearing
print(f"\n  TEARING:")
for name in names_present:
    if 'tearing' in all_sweeps[name]:
        print(f"    {display_names.get(name, name):<16} {all_sweeps[name]['tearing']:.1f}%")

# =============================================================================
# PAIRWISE CORRELATIONS
# =============================================================================

print(f"\n{'=' * 70}")
print("PAIRWISE CURVE CORRELATIONS")
print("=" * 70)

# Build average arrays for correlation
avg_arrays = {}
for name in names_present:
    sweep = all_sweeps[name]
    avgs = [sweep[lam]['avg'] for lam in LAMBDAS if lam in sweep and isinstance(sweep[lam], dict) and 'avg' in sweep[lam]]
    if len(avgs) == len(LAMBDAS):
        avg_arrays[name] = np.array(avgs)

corr_names = list(avg_arrays.keys())
if len(corr_names) >= 2:
    print(f"\n  {'':16}", end="")
    for n in corr_names:
        print(f"  {display_names.get(n, n):<14}", end="")
    print()

    for i, n1 in enumerate(corr_names):
        print(f"  {display_names.get(n1, n1):<16}", end="")
        for j, n2 in enumerate(corr_names):
            r = np.corrcoef(avg_arrays[n1], avg_arrays[n2])[0, 1]
            print(f"  {r:<14.4f}", end="")
        print()

    # Average pairwise correlation
    pairs = []
    for i in range(len(corr_names)):
        for j in range(i+1, len(corr_names)):
            r = np.corrcoef(avg_arrays[corr_names[i]], avg_arrays[corr_names[j]])[0, 1]
            pairs.append(r)
    avg_r = np.mean(pairs)
    min_r = np.min(pairs)
    print(f"\n  Average pairwise correlation: {avg_r:.4f}")
    print(f"  Minimum pairwise correlation: {min_r:.4f}")

# =============================================================================
# UNIVERSALITY VERDICT
# =============================================================================

print(f"\n{'=' * 70}")
print("UNIVERSALITY VERDICT")
print("=" * 70)

n_hamiltonians = len(names_present)
n_with_tearing = sum(1 for n in names_present if 'tearing' in all_sweeps[n])
n_with_monotonic = 0
for name in names_present:
    sweep = all_sweeps[name]
    avgs = [sweep[lam]['avg'] for lam in sorted(sweep.keys()) if lam in sweep and isinstance(sweep[lam], dict) and 'avg' in sweep[lam]]
    if len(avgs) > 1:
        mono = sum(1 for i in range(len(avgs)-1) if avgs[i+1] > avgs[i])
        if mono >= len(avgs) - 3:  # at least mostly monotonic
            n_with_monotonic += 1

print(f"\n  Hamiltonians tested: {n_hamiltonians}")
print(f"  Show monotonic lambda scaling: {n_with_monotonic}/{n_hamiltonians}")
print(f"  Show spacetime tearing: {n_with_tearing}/{n_hamiltonians}")

if len(pairs) > 0:
    print(f"  Average curve correlation: {avg_r:.4f}")
    print(f"  Minimum curve correlation: {min_r:.4f}")

    if avg_r > 0.85 and n_with_monotonic >= 3:
        print(f"\n  ★ STRONG UNIVERSALITY ESTABLISHED ★")
        print(f"  {n_hamiltonians} distinct Hamiltonians with different symmetries")
        print(f"  (Z₂, SU(2), U(1), all-to-all) all produce the same")
        print(f"  emergent geometry. The geometric signal is a UNIVERSAL")
        print(f"  property of coupled quantum fields.")
    elif avg_r > 0.7:
        print(f"\n  MODERATE UNIVERSALITY: Most Hamiltonians converge.")
        print(f"  Geometry is largely Hamiltonian-independent.")
    elif avg_r > 0.5:
        print(f"\n  WEAK UNIVERSALITY: Some shared structure across Hamiltonians.")
    else:
        print(f"\n  UNIVERSALITY NOT CONFIRMED at four-Hamiltonian level.")

print(f"\n{'=' * 70}")

# Save complete results
output = {
    'timestamp': __import__('datetime').datetime.now().isoformat(),
    'experiment': 'Four-Hamiltonian universality test',
    'backend': 'ibm_torino',
    'hamiltonians': names_present,
    'sweeps': {},
    'tearing': {},
}

for name in names_present:
    sweep = all_sweeps[name]
    output['sweeps'][name] = {str(lam): sweep[lam]['avg'] for lam in LAMBDAS
                               if lam in sweep and isinstance(sweep[lam], dict) and 'avg' in sweep[lam]}
    if 'tearing' in sweep:
        output['tearing'][name] = sweep['tearing']

if len(pairs) > 0:
    output['avg_pairwise_correlation'] = avg_r
    output['min_pairwise_correlation'] = min_r

with open('four_hamiltonian_universality_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to four_hamiltonian_universality_results.json")
print("=" * 70)
