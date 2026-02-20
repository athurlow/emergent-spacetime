# Emergent Spacetime from Two Coupled Scalar Fields

**A Computational Framework and Quantum Simulation**

Andrew Thurlow | 528 Labs | February 2026

---

## Overview

This repository contains the complete source code, experimental data, and theoretical derivations for a novel framework demonstrating emergent spacetime geometry from quantum entanglement between two coupled scalar fields.

We show that two coupled qubit chains — representing pre-geometric scalar fields φ_A and φ_B with quartic coupling λφ²_Aφ²_B — produce emergent geometric properties including distance metrics, area-law entanglement, and spacetime connectivity that arise specifically from the two-field entanglement structure.

## Key Results

| Finding | Result |
|---------|--------|
| Emergent geometry | Confirmed at 8, 12, and 16 qubits |
| Null hypothesis (two-field vs single chain) | 10–20× more structured correlations |
| Spacetime tearing | 82–88% correlation reduction across all scales |
| Area law | Page curve scaling consistent with holographic predictions |
| Theory-simulation agreement | 99.6% match with Jacobson derivation predictions |

## Repository Structure

```
├── README.md
├── LICENSE
├── docs/
│   └── Thurlow_2026_Emergent_Spacetime.docx   # Paper manuscript
├── src/
│   ├── two_field_analysis.py          # Mathematical framework (SymPy)
│   ├── scaled_experiment_v2_1.py      # Full experiment suite (8/12/16 qubits)
│   ├── ibm_hardware_run.py            # IBM Quantum hardware submission
│   └── jacobson_derivation.py         # Theoretical derivation
├── results/
│   └── experiment_results.json        # Raw numerical data
└── plots/
    ├── two_field_plots.png            # Theory visualizations
    ├── emergent_spacetime_results.png # Initial 8-qubit results
    ├── scaled_experiment_results.png  # Scaled results (8/12/16q)
    └── derivation_flowchart.png       # Derivation diagram
```

## Quick Start

### Requirements
```bash
pip install qiskit qiskit-aer numpy matplotlib sympy
```

### Run the full experiment suite
```bash
python src/scaled_experiment_v2_1.py
```

### Validate IBM hardware pipeline
```bash
python src/ibm_hardware_run.py validate
```

### Submit to IBM Quantum (requires IBM token)
```bash
python src/ibm_hardware_run.py submit
```

## Theoretical Foundation

Two real scalar fields φ_A and φ_B with Lagrangian:

**L = ½(∂φ_A)² - ½m²_Aφ²_A + ½(∂φ_B)² - ½m²_Bφ²_B - λφ²_Aφ²_B**

Emergent distance: **d(i,j) ∝ 1/C_AB(i,j)**

Einstein's equations emerge via Jacobson's thermodynamic derivation with:

**G_eff = G₀ / (1 + 3λ²⟨φ²_A⟩⟨φ²_B⟩/(32π²))**

See `docs/` for the full paper and `src/jacobson_derivation.py` for the complete derivation.

## Citation

If you use this work, please cite:

```
A. Thurlow, "Emergent Spacetime Geometry from Entanglement Between Two Coupled
Scalar Fields: A Computational Framework and Quantum Simulation," 528 Labs (2026).
```

## License

MIT License. See LICENSE file.

## Contact

Andrew Thurlow — 528 Labs
