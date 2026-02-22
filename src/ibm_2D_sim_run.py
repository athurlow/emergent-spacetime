
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

GRID_SIZE = 4          # 4x4 lattice per field
N_PER_FIELD = GRID_SIZE * GRID_SIZE  # 16
N_TOTAL = 2 * N_PER_FIELD            # 32
TROTTER_STEPS = 6
DT = 0.3
J_INTRA = 1.0
SHOTS = 8192

print("=" * 70)
print("EMERGENT SPACETIME — 2D LATTICE EXPERIMENT")
print("Andrew Thurlow | 528 Labs")
print("=" * 70)
print(f"Grid: {GRID_SIZE}x{GRID_SIZE} per field")
print(f"Total qubits: {N_TOTAL}")
print(f"Trotter steps: {TROTTER_STEPS}, dt = {DT}")
print()

# =============================================================================
# QUBIT INDEXING
# =============================================================================

def qubit_index_A(row, col):
    """Map 2D grid position to qubit index for field A."""
    return row * GRID_SIZE + col

def qubit_index_B(row, col):
    """Map 2D grid position to qubit index for field B."""
    return N_PER_FIELD + row * GRID_SIZE + col

def get_intra_pairs(n_grid):
    """Get all nearest-neighbor pairs within a 2D grid."""
    pairs = []
    for r in range(n_grid):
        for c in range(n_grid):
            # Horizontal neighbor
            if c < n_grid - 1:
                pairs.append((r * n_grid + c, r * n_grid + c + 1))
            # Vertical neighbor
            if r < n_grid - 1:
                pairs.append((r * n_grid + c, (r + 1) * n_grid + c))
    return pairs

def get_inter_pairs(n_grid):
    """Get all corresponding site pairs between fields A and B."""
    pairs = []
    n = n_grid * n_grid
    for i in range(n):
        pairs.append((i, i + n))
    return pairs

# =============================================================================
# CIRCUIT BUILDERS
# =============================================================================

def build_2d_coupled(grid_size, j_intra, j_inter, steps, dt):
    """Build two coupled 2D lattices."""
    n_per = grid_size * grid_size
    n_total = 2 * n_per
    qc = QuantumCircuit(n_total)
    
    # Initialize field A in superposition, field B in |0>
    for i in range(n_per):
        qc.h(i)
    qc.barrier()
    
    intra_pairs_A = get_intra_pairs(grid_size)
    intra_pairs_B = [(a + n_per, b + n_per) for a, b in intra_pairs_A]
    inter_pairs = get_inter_pairs(grid_size)
    
    for step in range(steps):
        # Intra-field A: ZZ + XX on all nearest neighbors
        for i, j in intra_pairs_A:
            # ZZ
            qc.cx(i, j)
            qc.rz(2 * j_intra * dt, j)
            qc.cx(i, j)
            # XX
            qc.h(i); qc.h(j)
            qc.cx(i, j)
            qc.rz(2 * j_intra * dt, j)
            qc.cx(i, j)
            qc.h(i); qc.h(j)
        
        # Intra-field B: ZZ + XX on all nearest neighbors
        for i, j in intra_pairs_B:
            # ZZ
            qc.cx(i, j)
            qc.rz(2 * j_intra * dt, j)
            qc.cx(i, j)
            # XX
            qc.h(i); qc.h(j)
            qc.cx(i, j)
            qc.rz(2 * j_intra * dt, j)
            qc.cx(i, j)
            qc.h(i); qc.h(j)
        
        # Inter-field coupling: ZZ between corresponding sites
        if j_inter > 0:
            for i, j in inter_pairs:
                qc.cx(i, j)
                qc.rz(2 * j_inter * dt, j)
                qc.cx(i, j)
        
        qc.barrier()
    
    return qc

def build_2d_single(grid_size, j_intra, steps, dt):
    """Build single 2D lattice (null hypothesis)."""
    n_total = 2 * grid_size * grid_size
    qc = QuantumCircuit(n_total)
    
    # Initialize half in superposition
    for i in range(n_total // 2):
        qc.h(i)
    qc.barrier()
    
    # All-to-all nearest neighbor on full grid treated as one system
    # Use an 8x4 or equivalent rectangular grid
    full_cols = grid_size
    full_rows = 2 * grid_size  # Stack the two fields into one rectangle
    
    pairs = []
    for r in range(full_rows):
        for c in range(full_cols):
            idx = r * full_cols + c
            if c < full_cols - 1:
                pairs.append((idx, idx + 1))
            if r < full_rows - 1:
                pairs.append((idx, idx + full_cols))
    
    for step in range(steps):
        for i, j in pairs:
            if i < n_total and j < n_total:
                qc.cx(i, j)
                qc.rz(2 * j_intra * dt, j)
                qc.cx(i, j)
                qc.h(i); qc.h(j)
                qc.cx(i, j)
                qc.rz(2 * j_intra * dt, j)
                qc.cx(i, j)
                qc.h(i); qc.h(j)
        qc.barrier()
    
    return qc

def build_2d_torn(grid_size, j_intra, j_inter, total_steps, dt):
    """Build 2D spacetime tearing: coupled then uncoupled."""
    half = total_steps // 2
    n_per = grid_size * grid_size
    n_total = 2 * n_per
    
    intra_pairs_A = get_intra_pairs(grid_size)
    intra_pairs_B = [(a + n_per, b + n_per) for a, b in intra_pairs_A]
    inter_pairs = get_inter_pairs(grid_size)
    
    # Start with coupled evolution
    qc = build_2d_coupled(grid_size, j_intra, j_inter, half, dt)
    
    # Continue with uncoupled evolution (intra only)
    for step in range(half):
        for i, j in intra_pairs_A:
            qc.cx(i, j); qc.rz(2 * j_intra * dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j); qc.cx(i, j); qc.rz(2 * j_intra * dt, j); qc.cx(i, j); qc.h(i); qc.h(j)
        for i, j in intra_pairs_B:
            qc.cx(i, j); qc.rz(2 * j_intra * dt, j); qc.cx(i, j)
            qc.h(i); qc.h(j); qc.cx(i, j); qc.rz(2 * j_intra * dt, j); qc.cx(i, j); qc.h(i); qc.h(j)
        qc.barrier()
    
    return qc

def add_measurement(qc, basis='Z'):
    """Add measurement in specified Pauli basis."""
    qc_m = qc.copy()
    n = qc.num_qubits
    if basis == 'X':
        for i in range(n):
            qc_m.h(i)
    elif basis == 'Y':
        for i in range(n):
            qc_m.sdg(i)
            qc_m.h(i)
    qc_m.measure_all()
    return qc_m

# =============================================================================
# ANALYSIS
# =============================================================================

def counts_to_correlations(counts, n_total):
    """Compute correlation matrix from measurement counts."""
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

def cross_field_correlation_matrix(corr, grid_size):
    """Extract the cross-field correlation as a 2D grid."""
    n_per = grid_size * grid_size
    cross = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            i_a = r * grid_size + c
            i_b = i_a + n_per
            cross[r, c] = abs(corr[i_a, i_b])
    return cross

def emergent_distance_matrix(corr, grid_size):
    """Compute emergent 2D distance matrix from cross-correlations."""
    n_per = grid_size * grid_size
    dist = np.zeros((n_per, n_per))
    for i in range(n_per):
        for j in range(n_per):
            c_ij = abs(corr[i, j + n_per])
            if c_ij > 1e-6:
                dist[i, j] = 1.0 / c_ij
            else:
                dist[i, j] = 1e6  # effectively infinite
    return dist

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

sim = AerSimulator(method='statevector')

# --- Experiment 1: Coupled 2D lattices ---
print("Experiment 1: Coupled 2D lattices (lambda=1.0)...")
qc_coupled = build_2d_coupled(GRID_SIZE, J_INTRA, 1.0, TROTTER_STEPS, DT)
print(f"  Circuit: {qc_coupled.num_qubits} qubits, depth={qc_coupled.depth()}")
qc_coupled_z = add_measurement(qc_coupled, 'Z')
result_coupled = sim.run(qc_coupled_z, shots=SHOTS).result()
counts_coupled = result_coupled.get_counts()
corr_coupled, _ = counts_to_correlations(counts_coupled, N_TOTAL)
cross_coupled = cross_field_correlation_matrix(corr_coupled, GRID_SIZE)
print(f"  Cross-field |C| grid:")
print(f"  {cross_coupled}")
print(f"  Average |C|: {np.mean(cross_coupled):.4f}")
print(f"  Std dev: {np.std(cross_coupled):.4f}")

# --- Experiment 2: Uncoupled 2D lattices ---
print("\nExperiment 2: Uncoupled 2D lattices (lambda=0.0)...")
qc_uncoupled = build_2d_coupled(GRID_SIZE, J_INTRA, 0.0, TROTTER_STEPS, DT)
qc_uncoupled_z = add_measurement(qc_uncoupled, 'Z')
result_uncoupled = sim.run(qc_uncoupled_z, shots=SHOTS).result()
counts_uncoupled = result_uncoupled.get_counts()
corr_uncoupled, _ = counts_to_correlations(counts_uncoupled, N_TOTAL)
cross_uncoupled = cross_field_correlation_matrix(corr_uncoupled, GRID_SIZE)
print(f"  Average |C|: {np.mean(cross_uncoupled):.4f}")

# --- Experiment 3: Single lattice (null hypothesis) ---
print("\nExperiment 3: Single 2D lattice - null hypothesis...")
qc_single = build_2d_single(GRID_SIZE, J_INTRA, TROTTER_STEPS, DT)
print(f"  Circuit: {qc_single.num_qubits} qubits, depth={qc_single.depth()}")
qc_single_z = add_measurement(qc_single, 'Z')
result_single = sim.run(qc_single_z, shots=SHOTS).result()
counts_single = result_single.get_counts()
corr_single, _ = counts_to_correlations(counts_single, N_TOTAL)
cross_single = cross_field_correlation_matrix(corr_single, GRID_SIZE)
print(f"  Average |C|: {np.mean(cross_single):.4f}")
print(f"  Std dev: {np.std(cross_single):.4f}")

# --- Experiment 4: Spacetime tearing ---
print("\nExperiment 4: Spacetime tearing (2D)...")
qc_torn = build_2d_torn(GRID_SIZE, J_INTRA, 1.0, TROTTER_STEPS, DT)
print(f"  Circuit: {qc_torn.num_qubits} qubits, depth={qc_torn.depth()}")
qc_torn_z = add_measurement(qc_torn, 'Z')
result_torn = sim.run(qc_torn_z, shots=SHOTS).result()
counts_torn = result_torn.get_counts()
corr_torn, _ = counts_to_correlations(counts_torn, N_TOTAL)
cross_torn = cross_field_correlation_matrix(corr_torn, GRID_SIZE)
print(f"  Average |C|: {np.mean(cross_torn):.4f}")

# =============================================================================
# KEY COMPARISONS
# =============================================================================

print("\n" + "=" * 70)
print("2D LATTICE RESULTS SUMMARY")
print("=" * 70)

avg_coupled = np.mean(cross_coupled)
avg_uncoupled = np.mean(cross_uncoupled)
avg_single = np.mean(cross_single)
avg_torn = np.mean(cross_torn)

coupling_ratio = avg_coupled / max(avg_uncoupled, 1e-10)
null_ratio = np.std(cross_coupled) / max(np.std(cross_single), 1e-10)
tearing_reduction = (1 - avg_torn / max(avg_coupled, 1e-10)) * 100

print(f"\n  COUPLING EFFECT:")
print(f"    Coupled (lambda=1.0):   avg |C| = {avg_coupled:.4f}")
print(f"    Uncoupled (lambda=0.0): avg |C| = {avg_uncoupled:.4f}")
print(f"    Ratio: {coupling_ratio:.2f}x")
print(f"    Signal: {'STRONG' if coupling_ratio > 10 else 'DETECTED' if coupling_ratio > 2 else 'MARGINAL'}")

print(f"\n  NULL HYPOTHESIS (2D):")
print(f"    Two-field sigma: {np.std(cross_coupled):.4f}")
print(f"    Single lattice sigma: {np.std(cross_single):.4f}")
print(f"    Ratio: {null_ratio:.2f}x")
print(f"    Signal: {'STRONG' if null_ratio > 5 else 'DETECTED' if null_ratio > 2 else 'MARGINAL'}")

print(f"\n  SPACETIME TEARING (2D):")
print(f"    Connected avg |C|: {avg_coupled:.4f}")
print(f"    Torn avg |C|:      {avg_torn:.4f}")
print(f"    Reduction: {tearing_reduction:.1f}%")
print(f"    Signal: {'STRONG' if tearing_reduction > 50 else 'DETECTED' if tearing_reduction > 20 else 'MARGINAL'}")

# Per-site tearing
print(f"\n  PER-SITE TEARING DETAIL:")
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        c_conn = cross_coupled[r, c]
        c_torn = cross_torn[r, c]
        red = (1 - c_torn / max(c_conn, 1e-10)) * 100
        print(f"    ({r},{c}): {c_conn:.4f} -> {c_torn:.4f}  ({red:.1f}% reduction)")

# Comparison to 1D
print(f"\n  COMPARISON TO 1D:")
print(f"    1D coupling ratio:  95.7x (hardware) / ~456x (simulator)")
print(f"    2D coupling ratio:  {coupling_ratio:.1f}x (simulator)")
print(f"    1D tearing:         83.4% (hardware) / 87.2% (simulator)")
print(f"    2D tearing:         {tearing_reduction:.1f}% (simulator)")
print(f"    1D null hypothesis: 10.3x (simulator)")
print(f"    2D null hypothesis: {null_ratio:.1f}x (simulator)")

# =============================================================================
# 2D-SPECIFIC: EMERGENT DISTANCE AND GEOMETRY
# =============================================================================

print(f"\n" + "=" * 70)
print("2D EMERGENT GEOMETRY ANALYSIS")
print("=" * 70)

# Emergent distance matrix
dist_matrix = emergent_distance_matrix(corr_coupled, GRID_SIZE)

# Check triangle inequality in 2D
violations = 0
total_checks = 0
for i in range(N_PER_FIELD):
    for j in range(i + 1, N_PER_FIELD):
        for k in range(j + 1, N_PER_FIELD):
            dij = dist_matrix[i, j]
            djk = dist_matrix[j, k]
            dik = dist_matrix[i, k]
            if dij < 1e5 and djk < 1e5 and dik < 1e5:
                total_checks += 1
                if dij > djk + dik or djk > dij + dik or dik > dij + djk:
                    violations += 1

if total_checks > 0:
    satisfaction_rate = (1 - violations / total_checks) * 100
    print(f"\n  Triangle inequality satisfaction: {satisfaction_rate:.1f}%")
    print(f"    ({total_checks - violations}/{total_checks} triplets satisfied)")
else:
    print("\n  Triangle inequality: insufficient data")

# Emergent area: entanglement entropy of 2D boundary regions
print(f"\n  CROSS-FIELD CORRELATION MAP (emergent geometry heatmap):")
print(f"  Rows=grid rows, Cols=grid cols, Values=|C(A(r,c), B(r,c))|")
for r in range(GRID_SIZE):
    row_str = "    "
    for c in range(GRID_SIZE):
        row_str += f"{cross_coupled[r, c]:.4f}  "
    print(row_str)

# Check for boundary effects (edge vs interior)
interior = []
edge = []
corner = []
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        neighbors = 0
        if r > 0: neighbors += 1
        if r < GRID_SIZE - 1: neighbors += 1
        if c > 0: neighbors += 1
        if c < GRID_SIZE - 1: neighbors += 1
        
        val = cross_coupled[r, c]
        if neighbors == 4:
            interior.append(val)
        elif neighbors == 3:
            edge.append(val)
        else:
            corner.append(val)

print(f"\n  BOUNDARY vs INTERIOR CORRELATIONS:")
if corner:
    print(f"    Corner sites (2 neighbors):   avg |C| = {np.mean(corner):.4f}")
if edge:
    print(f"    Edge sites (3 neighbors):     avg |C| = {np.mean(edge):.4f}")
if interior:
    print(f"    Interior sites (4 neighbors): avg |C| = {np.mean(interior):.4f}")
print(f"    Ratio interior/corner: {np.mean(interior)/max(np.mean(corner), 1e-10):.2f}x")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Emergent Spacetime — 2D Lattice Results\nAndrew Thurlow | 528 Labs', 
             fontsize=16, fontweight='bold')

# 1. Cross-field correlation heatmap (coupled)
im1 = axes[0, 0].imshow(cross_coupled, cmap='viridis', interpolation='nearest')
axes[0, 0].set_title('Coupled: Cross-Field |C|', fontweight='bold')
axes[0, 0].set_xlabel('Column'); axes[0, 0].set_ylabel('Row')
plt.colorbar(im1, ax=axes[0, 0])
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        axes[0, 0].text(c, r, f'{cross_coupled[r,c]:.3f}', ha='center', va='center', 
                        color='white', fontsize=9)

# 2. Cross-field correlation heatmap (torn)
im2 = axes[0, 1].imshow(cross_torn, cmap='viridis', interpolation='nearest',
                          vmin=0, vmax=np.max(cross_coupled))
axes[0, 1].set_title('Torn: Cross-Field |C|', fontweight='bold')
axes[0, 1].set_xlabel('Column'); axes[0, 1].set_ylabel('Row')
plt.colorbar(im2, ax=axes[0, 1])
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        axes[0, 1].text(c, r, f'{cross_torn[r,c]:.3f}', ha='center', va='center', 
                        color='white', fontsize=9)

# 3. Tearing reduction map
tearing_map = np.zeros((GRID_SIZE, GRID_SIZE))
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if cross_coupled[r, c] > 1e-6:
            tearing_map[r, c] = (1 - cross_torn[r, c] / cross_coupled[r, c]) * 100
im3 = axes[0, 2].imshow(tearing_map, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=100)
axes[0, 2].set_title('Tearing Reduction (%)', fontweight='bold')
axes[0, 2].set_xlabel('Column'); axes[0, 2].set_ylabel('Row')
plt.colorbar(im3, ax=axes[0, 2])
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        axes[0, 2].text(c, r, f'{tearing_map[r,c]:.0f}%', ha='center', va='center', 
                        color='black', fontsize=9, fontweight='bold')

# 4. Comparison bar chart
labels = ['Coupled', 'Uncoupled', 'Single', 'Torn']
avgs = [avg_coupled, avg_uncoupled, avg_single, avg_torn]
bar_colors = ['#4fc3f7', '#ff7043', '#ffa726', '#ef5350']
axes[1, 0].bar(labels, avgs, color=bar_colors)
axes[1, 0].set_title('Average Cross-Field |C|', fontweight='bold')
axes[1, 0].set_ylabel('|C|')
for i, v in enumerate(avgs):
    axes[1, 0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10)

# 5. Emergent distance visualization
# Show distances from center site
center = (GRID_SIZE // 2) * GRID_SIZE + (GRID_SIZE // 2)
dist_from_center = np.zeros((GRID_SIZE, GRID_SIZE))
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        idx = r * GRID_SIZE + c
        d = dist_matrix[center, idx]
        dist_from_center[r, c] = min(d, 50)  # cap for visualization

im5 = axes[1, 1].imshow(dist_from_center, cmap='plasma', interpolation='nearest')
axes[1, 1].set_title(f'Emergent Distance from Center ({GRID_SIZE//2},{GRID_SIZE//2})', fontweight='bold')
axes[1, 1].set_xlabel('Column'); axes[1, 1].set_ylabel('Row')
plt.colorbar(im5, ax=axes[1, 1], label='d = 1/|C|')

# 6. Boundary vs interior analysis
categories = []
values_list = []
if corner: 
    categories.append(f'Corner\n(n=2)')
    values_list.append(corner)
if edge:
    categories.append(f'Edge\n(n=3)')
    values_list.append(edge)
if interior:
    categories.append(f'Interior\n(n=4)')
    values_list.append(interior)

bp = axes[1, 2].boxplot(values_list, labels=categories, patch_artist=True,
                         boxprops=dict(facecolor='#4fc3f7', alpha=0.6))
axes[1, 2].set_title('Correlation by Site Type', fontweight='bold')
axes[1, 2].set_ylabel('|C|')

plt.tight_layout()
plt.savefig('2d_lattice_results.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to 2d_lattice_results.png")

# =============================================================================
# ENTANGLEMENT ENTROPY (via statevector)
# =============================================================================

print(f"\n" + "=" * 70)
print("ENTANGLEMENT ENTROPY (2D)")
print("=" * 70)

# Run without measurement to get statevector
qc_sv = build_2d_coupled(GRID_SIZE, J_INTRA, 1.0, TROTTER_STEPS, DT)
qc_sv.save_statevector()
result_sv = sim.run(qc_sv).result()
sv = np.array(result_sv.get_statevector())

# Compute S(A:B) via SVD
sv_matrix = sv.reshape(2**N_PER_FIELD, 2**N_PER_FIELD)
u, s_vals, vh = np.linalg.svd(sv_matrix, full_matrices=False)
s_vals = s_vals[s_vals > 1e-12]
probs = s_vals**2
entropy = -np.sum(probs * np.log2(probs + 1e-15))
entropy_per_site = entropy / N_PER_FIELD

print(f"  S(A:B) = {entropy:.4f} bits")
print(f"  S per site = {entropy_per_site:.4f} bits/site")
print(f"  Max possible S = {N_PER_FIELD:.1f} bits")
print(f"  Filling fraction = {entropy / N_PER_FIELD * 100:.1f}%")

# Compare to 1D
print(f"\n  COMPARISON TO 1D:")
print(f"    1D (4+4):   S = 2.001 bits, S/site = 0.500 bits/site")
print(f"    1D (6+6):   S = 3.325 bits, S/site = 0.554 bits/site")
print(f"    1D (8+8):   S = 4.629 bits, S/site = 0.579 bits/site")
print(f"    2D (4x4+4x4): S = {entropy:.3f} bits, S/site = {entropy_per_site:.3f} bits/site")

# Area-law check: entropy of boundary subregions
print(f"\n  2D AREA-LAW CHECK:")
print(f"  (Entropy of rectangular subregions of field A)")

for sub_size in range(1, GRID_SIZE):
    # Take sub_size rows of field A
    n_sub = sub_size * GRID_SIZE
    sv_reshaped = sv.reshape(2**n_sub, 2**(N_TOTAL - n_sub))
    _, s_sub, _ = np.linalg.svd(sv_reshaped, full_matrices=False)
    s_sub = s_sub[s_sub > 1e-12]
    p_sub = s_sub**2
    ent_sub = -np.sum(p_sub * np.log2(p_sub + 1e-15))
    boundary = GRID_SIZE  # boundary length = width of grid
    print(f"    {sub_size} rows ({n_sub} qubits), boundary={boundary}: S = {ent_sub:.4f} bits, S/boundary = {ent_sub/boundary:.4f}")

print(f"\n{'=' * 70}")
print(f"2D LATTICE EXPERIMENT COMPLETE")
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"{'=' * 70}")