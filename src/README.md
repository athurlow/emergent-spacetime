# Source Code

All experiment source code for the emergent spacetime project.

## Directory Structure

```
src/
├── simulation/          # Local simulation (no hardware required)
│   ├── qiskit_experiment.py       # Qiskit statevector simulation of two coupled chains
│   └── two_field_analysis.py      # Mathematical analysis and equation balancing
│
├── experiments/         # IBM Quantum hardware submission scripts
│   ├── 01_torino_4x4_run.py       # 1D 4+4 chain on IBM Torino (8 qubits)
│   ├── 03_torino_2x2_lattice_run.py  # 2D 2×2 lattice on IBM Torino (8 qubits)
│   ├── 04_torino_8x8_chain_run.py    # 1D 8+8 chain on IBM Torino (16 qubits)
│   └── 05_torino_128q_fullchip_run.py # Full-chip parallel on IBM Torino (128 qubits)
│
└── analysis/            # Results retrieval and analysis scripts
    ├── 01_torino_4x4_retrieve.py      # Retrieve and analyze 4+4 Torino results
    ├── 02_fez_4x4_retrieve.py         # Retrieve and analyze 4+4 Fez results
    ├── 03_torino_2x2_lattice_retrieve.py  # Retrieve and analyze 2×2 lattice results
    ├── 04_torino_8x8_chain_retrieve.py    # Retrieve and analyze 8+8 chain results
    └── 05_torino_128q_fullchip_retrieve.py # Retrieve and analyze 128-qubit results
```

## Experiment Summary

| # | Experiment | Qubits | Backend | Coupling | Tearing | Lambda Sweep |
|---|-----------|--------|---------|----------|---------|-------------|
| 1 | 1D 4+4 chain | 8 | IBM Torino | 95.7× | 83.4% | — |
| 2 | 1D 4+4 chain | 8 | IBM Fez | 13.7× | — | — |
| 3 | 2D 2×2 lattice | 8 | IBM Torino | 11.6× | 91.6% | 5/6 monotonic |
| 4 | 1D 8+8 chain | 16 | IBM Torino | 15.1× | 85.3% | 5/6 monotonic |
| 5 | Full-chip parallel | 128 | IBM Torino | 8.56× | 62.2% | 5/6 monotonic |

## Running the Experiments

### Prerequisites

```bash
pip install qiskit qiskit-ibm-runtime numpy
```

### Setup

1. Create a free IBM Quantum account at https://quantum.ibm.com
2. Copy your API token from your account settings
3. Paste the token into the `IBM_TOKEN` variable in any experiment script

### Simulation (no IBM account needed)

```bash
python src/simulation/qiskit_experiment.py
```

### Hardware Experiments

Each experiment has a run script and a corresponding retrieve/analysis script.

```bash
# Step 1: Submit to hardware (returns job IDs)
python src/experiments/01_torino_4x4_run.py

# Step 2: Wait for completion, then retrieve results
python src/analysis/01_torino_4x4_retrieve.py
```

### Notes

- The Fez experiment (02) was a cross-validation run on a different IBM backend.
  The submission used the same script as experiment 01, only changing the backend name.
  The retrieve script handles the Fez-specific job IDs.
- Experiment 05 (full-chip) runs 16 independent 4+4 experiments simultaneously
  across 128 of 133 qubits on IBM Torino, with different lambda values per region.
- All API keys have been replaced with `'PASTE_YOUR_API_KEY_HERE'` placeholders.
