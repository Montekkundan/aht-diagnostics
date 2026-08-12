# AHT Diagnostics

Code for the controlled Overcooked causal-localization experiment on frozen JaxAHT policies.

## Repository layout

- `scripts/run_localization_gate.py`: runs the planted-defect localization gate.
- `src/aht_diagnostics/`: semantic event extraction, defect wrappers, exact replay, and counterfactual repair.
- `tests/`: deterministic replay, semantic labeling, and planted-defect tests.
- `data/checkpoint_manifest.json`: pinned Hugging Face checkpoint revision and paths.
- `results/localization_gate_exhaustive_pilot/`: exhaustive matched-BR pilot outputs.

## Run

Use the `jax-aht` environment and a local JaxAHT checkout. Checkpoints are downloaded under the ignored `data/checkpoints/` directory.

```bash
MPLCONFIGDIR=/tmp/aht-mpl \
PYTHONPYCACHEPREFIX=/tmp/aht-pycache \
PYTHONPATH=src \
conda run -n jax-aht python scripts/run_localization_gate.py \
  --jaxaht /path/to/jax-aht \
  --checkpoint-root data/checkpoints/hf \
  --output results/localization_gate_exhaustive_pilot \
  --mode pilot \
  --ego-mode heuristic \
  --reference-mode matched \
  --candidate-mode exhaustive
```

## Test

```bash
MPLCONFIGDIR=/tmp/aht-mpl \
PYTHONPYCACHEPREFIX=/tmp/aht-pycache \
PYTHONPATH=src \
JAXAHT_PATH=/path/to/jax-aht \
conda run -n jax-aht python -m unittest discover -s tests -v
```

## Result

Exhaustive replay corrected the earlier candidate-selection failure: semantic localization reached `0.933`, localization AUROC reached `0.820`, and every planted failure had an outcome-improving repair. The pilot remains unsuccessful because repair gain did not beat the strongest event-rule baseline by the predefined margin. See [`summary.json`](results/localization_gate_exhaustive_pilot/summary.json) and [`localization_gate.png`](results/localization_gate_exhaustive_pilot/localization_gate.png).
