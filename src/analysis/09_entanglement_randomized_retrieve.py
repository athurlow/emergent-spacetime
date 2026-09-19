#!/usr/bin/env python3
"""
ENTANGLEMENT VIA LOCAL RANDOMIZED MEASUREMENTS — RETRIEVE AND ANALYSE
Andrew Thurlow | 528 Labs

Reads the jobs submitted by src/experiments/09_entanglement_randomized_run.py
and reports, for each prepared state:

  - Renyi-2 entropy of chain A, S_2(A) = -log2 Tr(rho_A^2)
  - Renyi-2 mutual information between the chains
  - the p3-PPT entanglement certificate on a 2+2 sub-cut

Read the caveat printed at the end before quoting S_2(A) as evidence of
entanglement. On a noisy device it is not.

Unlike the earlier retrieve scripts, this one saves the raw per-shot
records, so the analysis can be rerun and audited without re-running the
hardware.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shadow_entanglement import (  # noqa: E402
    build_witness,
    measure_witness,
    renyi2_entropy,
    snapshot_matrices,
    subsystem_purity,
    witness_validity_floor,
)
from two_chain_model import (  # noqa: E402
    build_torn_circuit,
    build_two_chain_circuit,
)
from qiskit.quantum_info import Statevector, partial_trace  # noqa: E402

# =============================================================================
IBM_TOKEN = 'PASTE_YOUR_API_KEY_HERE'
# =============================================================================

META_FILE = 'entanglement_randomized_job_ids.json'
RAW_FILE = 'results/entanglement_randomized_raw.json'
OUT_FILE = 'results/entanglement_randomized_results.json'

# The witness is measured on a 2+2 reduced state: two qubits from each
# chain. Entanglement in a reduced state implies entanglement in the full
# state, since tracing out cannot create it. Four qubits is also where the
# estimators are reliable; see 14_entanglement_validation.py for the cut
# scan and for the two certificates that were rejected.
KEEP = [0, 1, 4, 5]
LOCAL_A = [0, 1]
TRACE_OUT = [2, 3, 6, 7]


def collect_records(meta, results_by_label):
    """Group shots into (bases, bits, setting_id) arrays per prepared state."""
    n_qubits = meta['n_total_qubits']
    per_state = {}
    for state, settings in meta['settings'].items():
        bases, bits, setting_ids = [], [], []
        for k, basis in enumerate(settings):
            counts = results_by_label.get(f'{state}_setting_{k}')
            if not counts:
                continue
            for bitstring, n in counts.items():
                s = bitstring.replace(' ', '')
                row = [int(s[len(s) - 1 - q]) for q in range(n_qubits)]
                for _ in range(int(n)):
                    bases.append(basis)
                    bits.append(row)
                    setting_ids.append(k)
        if bases:
            per_state[state] = (np.array(bases), np.array(bits),
                                np.array(setting_ids))
    return per_state


def ideal_reduced_state(state_name, n_chain, steps, dt):
    """Ideal reduced state used only to choose the witness direction."""
    if state_name == 'torn':
        qc = build_torn_circuit(n_chain, 1.0, 1.0, steps, dt)
    else:
        lam = 0.0 if state_name == 'decoupled' else 1.0
        qc = build_two_chain_circuit(n_chain, 1.0, lam, steps, dt)
    return partial_trace(Statevector.from_instruction(qc), TRACE_OUT).data


def analyse(bases, bits, setting_ids, n_chain, n_qubits, witness):
    snaps = snapshot_matrices(bases, bits)
    chain_a = list(range(n_chain))
    chain_b = list(range(n_chain, n_qubits))

    pur_a = subsystem_purity(snaps, chain_a, settings=setting_ids)
    pur_b = subsystem_purity(snaps, chain_b, settings=setting_ids)
    pur_ab = subsystem_purity(snaps, chain_a + chain_b, settings=setting_ids)

    s2_a, s2_b, s2_ab = (renyi2_entropy(p) for p in (pur_a, pur_b, pur_ab))

    cert = measure_witness(snaps, KEEP, witness, settings=setting_ids)
    return {
        'n_shots': int(snaps.shape[0]),
        'n_settings': int(len(np.unique(setting_ids))),
        'purity_A': pur_a,
        'purity_B': pur_b,
        'purity_AB': pur_ab,
        'renyi2_A': s2_a,
        'renyi2_B': s2_b,
        'renyi2_AB': s2_ab,
        'renyi2_mutual_information': s2_a + s2_b - s2_ab,
        'witness': {'keep': KEEP, 'local_A': LOCAL_A, **cert},
    }


def main():
    print('=' * 70)
    print('ENTANGLEMENT VIA RANDOMIZED MEASUREMENTS — RESULTS')
    print('Andrew Thurlow | 528 Labs')
    print('=' * 70)

    with open(META_FILE) as fh:
        meta = json.load(fh)

    service = QiskitRuntimeService(channel='ibm_quantum_platform',
                                   token=IBM_TOKEN)

    results_by_label = {}
    print('\nRetrieving jobs...')
    for batch, info in meta['job_ids'].items():
        job = service.job(info['job_id'])
        status = str(job.status())
        print(f'  {batch} ({info["job_id"]}): {status}')
        if 'DONE' not in status:
            continue
        result = job.result()
        for i, label in enumerate(info['labels']):
            try:
                results_by_label[label] = result[i].data.meas.get_counts()
            except Exception as exc:
                print(f'    {label}: ERROR {exc}')

    if not results_by_label:
        print('\nNo completed results yet.')
        return

    per_state = collect_records(meta, results_by_label)
    n_chain, n_qubits = meta['n_chain'], meta['n_total_qubits']

    # Archive the raw records. Earlier experiments in this repository stored
    # only derived summaries, which made them impossible to re-analyse.
    os.makedirs('results', exist_ok=True)
    with open(RAW_FILE, 'w') as fh:
        json.dump({
            'meta': {k: v for k, v in meta.items() if k != 'settings'},
            'records': {
                state: {'bases': b.tolist(), 'bits': bits.tolist(),
                        'setting_ids': sid.tolist()}
                for state, (b, bits, sid) in per_state.items()
            },
        }, fh)
    print(f'\nRaw per-shot records saved to {RAW_FILE}')

    out = {'timestamp': datetime.now().isoformat(),
           'backend': meta['backend'], 'states': {}}

    for state, (b, bits, sid) in per_state.items():
        print(f'\n{"=" * 70}\n{state.upper()}\n{"=" * 70}')
        rho_ideal = ideal_reduced_state(state, n_chain, meta['trotter_steps'],
                                        meta['dt'])
        witness, lam = build_witness(rho_ideal, len(KEEP), LOCAL_A)
        floor = witness_validity_floor(witness, len(KEEP), LOCAL_A,
                                       n_trials=2000)
        res = analyse(b, bits, sid, n_chain, n_qubits, witness)
        res['witness_design'] = {'ideal_min_pt_eigenvalue': lam,
                                 'validity_floor': floor}
        out['states'][state] = res
        print(f'  shots {res["n_shots"]}, independent bases {res["n_settings"]}')
        print(f'  S_2(A)  = {res["renyi2_A"]:.3f} bits')
        print(f'  S_2(B)  = {res["renyi2_B"]:.3f} bits')
        print(f'  S_2(AB) = {res["renyi2_AB"]:.3f} bits')
        print(f'  Renyi-2 mutual information = '
              f'{res["renyi2_mutual_information"]:.3f} bits')
        cert = res['witness']
        print(f'\n  Entanglement witness on qubits {KEEP} '
              f'(cut {LOCAL_A} | rest):')
        print(f'    validity floor over random separable states: '
              f'{res["witness_design"]["validity_floor"]:+.5f}  '
              f'(must be >= 0)')
        print(f'    Tr(W rho) = {cert["witness_value"]:+.4f}  '
              f'[{cert["ci"][0]:+.4f}, {cert["ci"][1]:+.4f}]  '
              f'over {cert["n_units"]} bases')
        print(f'    ENTANGLEMENT CERTIFIED: {cert["entangled_certified"]}')

    with open(OUT_FILE, 'w') as fh:
        json.dump(out, fh, indent=2, default=float)

    print(f'\n{"=" * 70}')
    print('HOW TO READ THIS')
    print('=' * 70)
    print("""
  S_2(A) alone does NOT certify entanglement on hardware. The device state
  is mixed, and two independently prepared mixed chains with no
  entanglement between them give a positive S_2(A) too. Quote it as a
  subsystem entropy, not as evidence of entanglement.

  Tr(W rho) is the certificate. No separable state can make it negative,
  so an interval lying entirely below zero certifies entanglement and stays
  valid when the device state is mixed. The converse does not hold: a
  non-negative value may mean this witness is not aligned with the state
  the device actually prepared, so it is inconclusive rather than evidence
  of separability.

  The witness direction is chosen from simulation. That choice cannot
  create a false positive, because Tr(W sigma) >= 0 holds for every
  separable sigma whatever the simulation said; the printed validity floor
  checks this numerically on each run.
""")
    print(f'Saved: {OUT_FILE}')
    print('=' * 70)


if __name__ == '__main__':
    main()
