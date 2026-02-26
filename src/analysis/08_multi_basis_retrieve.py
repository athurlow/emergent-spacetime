#!/usr/bin/env python3
"""
EMERGENT SPACETIME — MULTI-BASIS MEASUREMENT RESULTS
Does geometry rotate with the Hamiltonian?
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
print("EMERGENT SPACETIME — MULTI-BASIS RESULTS")
print("Does XY geometry appear in X-basis measurements?")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=IBM_TOKEN
)

with open('multi_basis_job_ids.json', 'r') as f:
    job_data = json.load(f)

# Retrieve all
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
                all_counts[label] = result[i].data.meas.get_counts()
            except Exception as e:
                print(f"    {label}: ERROR - {e}")

if not all_counts:
    print("No results ready yet.")
    exit()

# =============================================================================
# After basis rotation, all measurements are in Z basis
# So we always compute ZZ correlations on the rotated measurements
# =============================================================================

def compute_correlations(counts, n_qubits):
    """Compute connected correlation matrix from Z-basis counts."""
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


def cross_field_avg(counts, n_chain, n_qubits):
    """Average |C| for corresponding cross-chain pairs."""
    corr = compute_correlations(counts, n_qubits)
    return np.mean([abs(corr[i, i + n_chain]) for i in range(n_chain)])


# =============================================================================
# ANALYZE EACH COUPLING × BASIS COMBINATION
# =============================================================================

results = {}

for coupling in ['xy', 'ising']:
    for basis in ['Z', 'X', 'Y']:
        key = f'{coupling}_{basis}'
        avgs = []
        for lam in LAMBDAS:
            label = f'{coupling}_{basis}_lam_{lam}'
            if label in all_counts:
                avg = cross_field_avg(all_counts[label], N_CHAIN, N_QUBITS)
                avgs.append(avg)
            else:
                avgs.append(None)
        results[key] = avgs

# =============================================================================
# PRINT RESULTS
# =============================================================================

for coupling, coupling_name in [('xy', 'XY COUPLING (XX+YY)'), ('ising', 'ISING COUPLING (ZZ)')]:
    print(f"\n{'=' * 70}")
    print(f"{coupling_name}")
    print("=" * 70)

    header = f"  {'Lambda':<8}"
    for basis in ['Z', 'X', 'Y']:
        header += f"  {basis+'-basis':<14}"
    header += f"  {'Best basis':<14}"
    print(header)
    print(f"  {'-' * 62}")

    for i, lam in enumerate(LAMBDAS):
        row = f"  {lam:<8}"
        vals = {}
        for basis in ['Z', 'X', 'Y']:
            key = f'{coupling}_{basis}'
            val = results[key][i]
            if val is not None:
                row += f"  {val:<14.4f}"
                vals[basis] = val
            else:
                row += f"  {'—':<14}"
        if vals:
            best = max(vals, key=vals.get)
            row += f"  {best:<14}"
        print(row)

    # Monotonicity per basis
    print(f"\n  Monotonicity:")
    for basis in ['Z', 'X', 'Y']:
        key = f'{coupling}_{basis}'
        avgs = [v for v in results[key] if v is not None]
        if len(avgs) > 1:
            mono = sum(1 for j in range(len(avgs)-1) if avgs[j+1] > avgs[j])
            dyn = max(avgs) / max(avgs[0], 1e-10)
            peak_idx = np.argmax(avgs)
            print(f"    {basis}-basis: {mono}/{len(avgs)-1} increasing  range={dyn:.1f}×  peak λ={LAMBDAS[peak_idx]}")

    # Coupling ratio per basis
    print(f"\n  Coupling ratios (λ=1.0 / λ=0.0):")
    for basis in ['Z', 'X', 'Y']:
        key = f'{coupling}_{basis}'
        avgs = results[key]
        if avgs[0] is not None and avgs[5] is not None:
            ratio = avgs[5] / max(avgs[0], 1e-10)
            print(f"    {basis}-basis: {ratio:.2f}×")

# =============================================================================
# HEAD-TO-HEAD: BEST BASIS PER HAMILTONIAN
# =============================================================================

print(f"\n{'=' * 70}")
print("HEAD-TO-HEAD: BEST BASIS PER HAMILTONIAN")
print("=" * 70)

print(f"\n  {'Lambda':<8}  {'Ising Z':<12}  {'Ising X':<12}  {'XY Z':<12}  {'XY X':<12}")
print(f"  {'-' * 56}")

for i, lam in enumerate(LAMBDAS):
    iz = results['ising_Z'][i]
    ix = results['ising_X'][i]
    xz = results['xy_Z'][i]
    xx = results['xy_X'][i]
    row = f"  {lam:<8}"
    for v in [iz, ix, xz, xx]:
        if v is not None:
            row += f"  {v:<12.4f}"
        else:
            row += f"  {'—':<12}"
    print(row)

# Correlations
print(f"\n  Curve correlations:")
pairs_to_check = [
    ('ising_Z', 'xy_X', 'Ising-Z vs XY-X (predicted match)'),
    ('ising_Z', 'xy_Z', 'Ising-Z vs XY-Z (known mismatch)'),
    ('ising_Z', 'ising_X', 'Ising-Z vs Ising-X'),
    ('xy_X', 'xy_Z', 'XY-X vs XY-Z'),
    ('xy_X', 'xy_Y', 'XY-X vs XY-Y'),
]

for key1, key2, label in pairs_to_check:
    a1 = np.array([v for v in results[key1] if v is not None])
    a2 = np.array([v for v in results[key2] if v is not None])
    if len(a1) == len(a2) and len(a1) > 1:
        r = np.corrcoef(a1, a2)[0, 1]
        print(f"    {label:<42} r = {r:.4f}")

# =============================================================================
# VERDICT
# =============================================================================

print(f"\n{'=' * 70}")
print("BASIS-ROTATED UNIVERSALITY VERDICT")
print("=" * 70)

# Check if XY in X-basis matches Ising in Z-basis
ising_z = np.array([v for v in results['ising_Z'] if v is not None])
xy_x = np.array([v for v in results['xy_X'] if v is not None])
xy_z = np.array([v for v in results['xy_Z'] if v is not None])

if len(ising_z) == len(xy_x):
    r_match = np.corrcoef(ising_z, xy_x)[0, 1]
    r_mismatch = np.corrcoef(ising_z, xy_z)[0, 1]

    # Coupling ratios
    ising_z_ratio = ising_z[5] / max(ising_z[0], 1e-10)
    xy_x_ratio = xy_x[5] / max(xy_x[0], 1e-10)
    xy_z_ratio = xy_z[5] / max(xy_z[0], 1e-10)

    print(f"\n  Ising in Z-basis coupling ratio:    {ising_z_ratio:.2f}×")
    print(f"  XY in X-basis coupling ratio:       {xy_x_ratio:.2f}×")
    print(f"  XY in Z-basis coupling ratio:       {xy_z_ratio:.2f}× (mismatch)")
    print(f"\n  Ising-Z vs XY-X correlation:        {r_match:.4f}")
    print(f"  Ising-Z vs XY-Z correlation:        {r_mismatch:.4f}")

    if r_match > 0.8 and xy_x_ratio > 5:
        print(f"\n  ★ BASIS-ROTATED UNIVERSALITY CONFIRMED ★")
        print(f"  The geometry is ALWAYS present. The measurement basis")
        print(f"  must match the coupling basis to detect it.")
        print(f"  ZZ coupling → Z-basis geometry")
        print(f"  XX+YY coupling → X-basis geometry")
        print(f"  This is deeper than Hamiltonian universality.")
    elif r_match > r_mismatch and xy_x_ratio > xy_z_ratio:
        print(f"\n  PARTIAL BASIS-ROTATION EFFECT DETECTED")
        print(f"  XY geometry is stronger in X-basis than Z-basis,")
        print(f"  consistent with basis-rotated universality.")
        print(f"  Signal may be too weak for definitive claim.")
    else:
        print(f"\n  BASIS ROTATION NOT CONFIRMED")
        print(f"  XY geometry does not clearly appear in X-basis.")
        print(f"  The coupling type may matter beyond basis rotation.")

print(f"\n{'=' * 70}")

# Save
output = {
    'timestamp': __import__('datetime').datetime.now().isoformat(),
    'experiment': 'Multi-basis measurement',
    'results': {k: [float(v) if v is not None else None for v in vals]
                for k, vals in results.items()},
}
with open('multi_basis_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Results saved to multi_basis_results.json")
