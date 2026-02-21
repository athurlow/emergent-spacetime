# Emergent Spacetime from Two Coupled Scalar Fields

**A Computational Framework with Quantum Hardware Validation**

Andrew Thurlow | 528 Labs | February 2026

---

## Overview

This repository contains the complete source code, experimental data, theoretical derivations, and IBM quantum hardware results for a novel framework demonstrating emergent spacetime geometry from quantum entanglement between two coupled scalar fields.

We show that two coupled qubit chains — representing pre-geometric scalar fields φ_A and φ_B with quartic coupling λφ²_Aφ²_B — produce emergent geometric properties including distance metrics, area-law entanglement, and spacetime connectivity that arise specifically from the two-field entanglement structure. These predictions are validated on IBM Torino (133-qubit Heron processor).

## Key Results

### Simulator (8, 12, 16 qubits)
| Finding | Result |
|---------|--------|
| Emergent geometry | Confirmed at all three scales |
| Null hypothesis (two-field vs single chain) | 10–20× more structured correlations |
| Spacetime tearing | 82–88% correlation reduction |
| Area law | Page curve scaling confirmed |
| Theory-simulation agreement | 99.6% match |

### IBM Torino Hardware (8 qubits, 8192 shots)
| Finding | Result |
|---------|--------|
| Coupling effect | **95.7× signal ratio** |
| Spacetime tearing | **83.4% reduction** (simulator: 87.2%) |
| Null hypothesis | 1.63× (marginal on noisy hardware) |

## Repository Structure

```
├── README.md
├── LICENSE
├── docs/
│   ├── Thurlow_2026_Emergent_Spacetime.pdf    # Paper (submission-ready)
│   └── Thurlow_2026_Emergent_Spacetime.docx   # Paper (editable)
├── src/
│   ├── two_field_analysis.py          # Mathematical framework (SymPy)
│   ├── scaled_experiment_v2_1.py      # Full experiment suite (8/12/16 qubits)
│   ├── ibm_hardware_run.py            # IBM Quantum hardware submission
│   ├── ibm_retrieve.py                # Hardware results retrieval & analysis
│   ├── jacobson_derivation.py         # Theoretical derivation
│   └── initial_8qubit_experiment.py   # Original proof of concept
├── results/
│   └── ibm_torino_results.json        # Hardware results data
└── plots/
    ├── two_field_plots.png
    ├── emergent_spacetime_results.png
    └── derivation_flowchart.png
```

## Quick Start

### Requirements
```bash
pip install qiskit qiskit-aer numpy matplotlib sympy
```

### Run the full simulation suite
```bash
python src/scaled_experiment_v2_1.py
```

### Run on IBM quantum hardware
```bash
# Install runtime
pip install qiskit-ibm-runtime

# Save your IBM credentials (one time)
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \
  QiskitRuntimeService.save_account(channel='ibm_quantum_platform', \
  token='YOUR_TOKEN', overwrite=True)"

# Submit experiments
python src/ibm_hardware_run.py

# Retrieve results
python src/ibm_retrieve.py
```

## Theoretical Foundation

Two real scalar fields φ_A and φ_B with Lagrangian:

**ℒ = ½(∂φ_A)² − ½m²_Aφ²_A + ½(∂φ_B)² − ½m²_Bφ²_B − λφ²_Aφ²_B**

Emergent distance: **d(i,j) ∝ 1/C_AB(i,j)**

Einstein's equations emerge via adapted Jacobson thermodynamic derivation with:

**G_eff = G₀ / (1 + 3λ²⟨φ²_A⟩⟨φ²_B⟩/(32π²))**

See `docs/` for the full paper and `src/jacobson_derivation.py` for the complete derivation.

## Acknowledgments

Computational tools including Claude (Anthropic) were used as aids in developing simulations and preparing the manuscript. IBM Quantum provided access to the IBM Torino processor. All scientific conclusions, interpretations, and any errors are solely the responsibility of the author.

## Citation

```
A. Thurlow, "Emergent Spacetime Geometry from Entanglement Between Two Coupled
Scalar Fields: A Computational Framework with Quantum Hardware Validation,"
528 Labs (2026). https://github.com/athurlow/emergent-spacetime
```

## License

MIT License. See LICENSE file.

## Contact

Andrew Thurlow — 528 Labs — andythurlow15@gmail.com
