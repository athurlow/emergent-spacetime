# Emergent Spacetime from Quantum Entanglement

**Testing whether geometric structure emerges from entanglement between coupled quantum fields on IBM quantum hardware.**

Andrew Thurlow | [528 Labs](https://528labs.org) | February 2026

---

## Overview

This repository contains a computational framework for studying emergent spacetime geometry from quantum entanglement. The core idea: two coupled qubit chains act as a discrete analog of two pre-geometric scalar fields. When the fields are entangled, geometric structure emerges in the correlation pattern. When entanglement is removed, the geometry vanishes.

The framework is validated across **8 hardware experiments** on IBM quantum processors (Torino and Fez), spanning 8 to 128 qubits, three Hamiltonians, three measurement bases, and multiple topologies.

## Key Results

**1. Coupling creates geometry.** Cross-field correlations scale smoothly with coupling strength λ, producing coupling ratios from 8× to 27× above the uncoupled baseline. Reproduced at 8, 16, and 128 qubits.

**2. Removing coupling destroys geometry.** "Spacetime tearing" — 62–92% correlation collapse upon decoupling — confirmed across all experiments, consistent with Van Raamsdonk's disconnection prediction.

**3. Universality across Hamiltonians.** Ising (ZZ) and Heisenberg (ZZ+XX) inter-chain couplings produce the same emergent geometry curve (Pearson r = 0.89), establishing that the geometric signal is a property of coupled fields, not a specific interaction.

**4. Geometry has tensor structure.** Multi-basis measurement reveals that emergent geometry has directional structure inherited from the Hamiltonian symmetry:
- Ising (ZZ) coupling → geometry visible in Z-basis (27.9×), invisible in X-basis (2.15×)
- XY (XX+YY) coupling → geometry visible in X-basis (9.93×), invisible in Z-basis (5.66×)
- Magnitude from coupling strength. Shape from Hamiltonian symmetry.

**5. The metric is mathematically valid.** The emergent correlation tensor satisfies:
- Positive definiteness at all coupling strengths
- Triangle inequality (100% satisfaction)
- Ricci scalar analog showing a geometric phase transition at λ ≈ 0.31
- Eigenvalue spectrum evolving from sphere (flatness 0.84) to needle (flatness 0.07)

## Hardware Experiments

| # | Experiment | Backend | Qubits | Coupling Ratio | Tearing |
|---|-----------|---------|--------|---------------|---------|
| 1 | 4+4 chain | IBM Torino | 8 | 95.7× | 83.4% |
| 2 | 4+4 chain | IBM Fez | 8 | 13.7× | — |
| 3 | 2×2 lattice | IBM Torino | 8 | 11.6× | 91.6% |
| 4 | 8+8 chain | IBM Torino | 16 | 15.1× | 85.3% |
| 5 | Full-chip parallel | IBM Torino | 128 | 8.56× | 62.2% |
| 6 | Universality (Ising vs Heisenberg) | IBM Torino | 8 | 27.1× / 10.1× | 85.7% / 82.8% |
| 7 | Extended universality (XY, long-range) | IBM Torino | 8 | 9.93× (X-basis) | 12.1% |
| 8 | Multi-basis measurement (Z, X, Y) | IBM Torino | 8 | 27.9× (matched) | — |

## Repository Structure

```
emergent-spacetime/
├── docs/
│   ├── Thurlow_2026_Emergent_Spacetime.pdf    # Full paper (v9, 12 pages)
│   └── Thurlow_2026_Emergent_Spacetime.docx
├── figures/
│   ├── fig1_correlation_matrix.png             # Coupled vs uncoupled structure
│   ├── fig2_lambda_sweep.png                   # Coupling strength sweep
│   ├── fig3_tearing.png                        # Spacetime disconnection
│   ├── fig4_null_hypothesis.png                # Two-field vs single-field
│   ├── fig5_hardware_torino.png                # IBM Torino validation
│   ├── fig6_hardware_fez.png                   # IBM Fez confirmation
│   ├── fig7_2d_lattice.png                     # 2D topology
│   ├── fig8_16qubit.png                        # 16-qubit scaling
│   ├── fig9_fullchip_128q.png                  # 128-qubit parallel
│   ├── fig10_universality.png                  # Ising vs Heisenberg
│   ├── fig11_multi_basis.png                   # Basis-dependent geometry
│   ├── fig12_metric_tensor.png                 # Metric tensor components
│   └── fig13_metric_properties.png             # Eigenvalue evolution & Ricci analog
├── results/
│   ├── universality_test_results.json
│   ├── multi_basis_results.json
│   ├── metric_tensor_analysis.json
│   ├── metric_properties_results.json
│   └── ...                                     # All raw hardware data
├── src/
│   ├── simulation/
│   │   └── emergent_spacetime_sim.py           # Full simulator framework
│   ├── experiments/
│   │   ├── 01_torino_4x4_run.py                # Original Torino experiment
│   │   ├── 02_fez_4x4_run.py                   # Fez replication
│   │   ├── 03_2d_lattice_run.py                # 2D lattice
│   │   ├── 04_16qubit_run.py                   # 8+8 chain
│   │   ├── 05_fullchip_parallel_run.py         # 128-qubit parallel
│   │   ├── 06_universality_test_run.py         # Ising vs Heisenberg
│   │   ├── 07_extended_universality_run.py     # XY + long-range
│   │   └── 08_multi_basis_run.py               # Three-basis measurement
│   └── analysis/
│       ├── 01_torino_4x4_retrieve.py
│       ├── 02_fez_4x4_retrieve.py
│       ├── 03_2d_lattice_retrieve.py
│       ├── 04_16qubit_retrieve.py
│       ├── 05_fullchip_parallel_retrieve.py
│       ├── 06_universality_test_retrieve.py
│       ├── 07_extended_universality_retrieve.py
│       ├── 08_multi_basis_retrieve.py
│       └── 09_metric_properties_analysis.py    # Tensor validation
└── README.md
```

## How to Reproduce

### Requirements
- Python 3.10+
- Qiskit 1.x
- qiskit-ibm-runtime
- numpy, matplotlib
- Free IBM Quantum account ([quantum.ibm.com](https://quantum.ibm.com))

### Quick Start

```bash
git clone https://github.com/athurlow/emergent-spacetime.git
cd emergent-spacetime
pip install qiskit qiskit-ibm-runtime numpy matplotlib
```

1. Get a free IBM Quantum API key from [quantum.ibm.com](https://quantum.ibm.com)
2. Paste your key into any experiment script
3. Run the experiment: `python src/experiments/01_torino_4x4_run.py`
4. Wait for job completion (~2-5 minutes)
5. Retrieve results: `python src/analysis/01_torino_4x4_retrieve.py`

### Run Without Hardware

The simulator framework requires no IBM account:

```bash
python src/simulation/emergent_spacetime_sim.py
```

This produces the full lambda sweep, tearing experiment, null hypothesis comparison, and correlation matrices using statevector simulation.

## Theoretical Framework

The framework adapts three foundational results:

- **Van Raamsdonk (2010)**: Spacetime connectivity arises from quantum entanglement. Removing entanglement disconnects spacetime.
- **Jacobson (1995)**: Einstein's field equations emerge from entanglement thermodynamics, yielding G_eff = 1/(4λη).
- **Ryu-Takayanagi (2006)**: Entanglement entropy is proportional to geometric area in holographic systems.

The two-chain architecture provides a minimal testbed: chain A and chain B represent two pre-geometric scalar fields. Their inter-chain coupling λ controls the entanglement. Cross-field correlations C(i, j) define an emergent distance metric d(i, j) = 1/|C(i, j)|.

## Key Findings in Detail

### Lambda Sweep — The Coupling-Geometry Relationship
The cross-field correlation |C| scales smoothly with λ across 8 values (0.0 to 2.0). The curve is monotonic in the rising phase, peaks between λ = 0.5 and 1.5 depending on the Hamiltonian, and turns over at strong coupling. This shape reproduces at 8, 16, and 128 qubits — the first experimental measurement of the coupling-geometry relationship on quantum hardware.

### Universality — Same Geometry, Different Hamiltonians
Ising (ZZ-only) and Heisenberg (ZZ+XX) inter-chain couplings produce lambda sweep curves with Pearson correlation r = 0.89. The simpler Ising Hamiltonian produces the stronger signal (27.1× vs 10.1×), with the peak shifting from λ = 0.75 (Ising) to λ = 1.5 (Heisenberg). Different microscopic physics, same emergent geometry.

### Multi-Basis Discovery — Geometry Has Coordinates
XY (XX+YY) coupling appeared to produce no geometry when measured in the Z-basis. Rotating the measurement to the X-basis revealed strong geometry (9.93×). Conversely, Ising geometry vanishes in the X-basis (2.15×). The emergent metric tensor has directional structure: ZZ coupling → Z-basis geometry, XX+YY coupling → X-basis geometry. The measurement basis must match the coupling basis.

### Metric Tensor Validation
The emergent correlation structure satisfies the mathematical requirements of a metric tensor:
- **Positive definite** at all λ for both Hamiltonians
- **Triangle inequality** satisfied for 100% of testable triangles
- **Eigenvalue spectrum** evolves from sphere (flatness 0.84) at λ = 0 to needle (flatness 0.07) at strong coupling
- **Ricci scalar analog** shows geometric phase transition at λ ≈ 0.31: accelerating geometry below, decelerating above
- **Magnitude from coupling, shape from symmetry**: λ controls the total geometric content, the Hamiltonian symmetry controls the directional distribution

## Related Work

- Vishveshwara et al. (arXiv:2602.15524) — Emergent curved spacetime from engineered spatially varying couplings on IBM hardware (80 qubits). Complementary approach: they engineer geometry by design; this framework lets it emerge from the coupling.
- Jafferis et al. (Nature, 2022) — Traversable wormhole dynamics on Google Sycamore.
- Swingle (2012) — Tensor network / entanglement renormalization connection to holographic spacetime.

## Citation

Paper and data are open. If you use this framework:

```
Thurlow, A. (2026). Emergent Spacetime Geometry from Quantum Entanglement:
A Computational Framework with Hardware Validation. 528 Labs.
https://github.com/athurlow/emergent-spacetime
```

## Contact

Andrew Thurlow  
528 Labs  
andythurlow15@gmail.com  
[GitHub](https://github.com/athurlow) | [LinkedIn](https://linkedin.com/in/andrew-thurlow)

Feedback, criticism, and collaboration inquiries welcome.

*Seeking arXiv endorsement for quant-ph or gr-qc (username: athurlow).*

## License

MIT License — use freely with attribution.
