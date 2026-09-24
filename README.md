# Fixed-ego interaction diagnosis in Overcooked

Code for the Overcooked-v1 Cramped Room experiment. It trains ROTATE teammates
against a fixed learned ego, records factual and matched-reference episodes,
selects interaction contexts from factual trajectories, and replays short ego
action sequences with matched controls. The ego is never retrained. The two
detectors and evaluation settings are in
[`configs/confirmation/diagnostic_baseline_confirmation_protocol.json`](configs/confirmation/diagnostic_baseline_confirmation_protocol.json).

The [experiment report](https://docs.google.com/document/d/1eyxhWgk50M0YzFTQiTCQ8FHbCkegiEw_NHVL4WTUGj4/edit?tab=t.wlx5uxnecf9)
contains the method, results, and limitations. Generated results, model weights,
review material, and draft notes are not included in this repository.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required. The completed
study used JaxAHT commit `e834de05898e8dfc82518abd9fd2d3fcd595edeb`.
Use a separate checkout at that revision; another JaxAHT branch may change
environment or policy behavior. From the directory containing this repository:

```bash
git clone https://github.com/LARG/jax-aht.git
cd jax-aht
git checkout e834de05898e8dfc82518abd9fd2d3fcd595edeb
cd ../aht-diagnostics
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-frozen.txt
uv pip install --no-deps -e ../jax-aht -e .
export JAXAHT_PATH="$(cd ../jax-aht && pwd)"
```

The [dependency lock](requirements-frozen.txt) was resolved from that JaxAHT
revision for Python 3.11 on macOS arm64. It records a later verification
environment, not the original training environment. Other platforms may need
compatible wheels or a fresh resolution.

The fixed ego and evaluation best-response checkpoints are indexed in
[`data/checkpoints.json`](data/checkpoints.json). To retrieve the pinned
[JaxAHT checkpoint dataset](https://hf.co/datasets/jaxaht/eval-teammates-br)
into the path used by the scripts, run:

```bash
hf download jaxaht/eval-teammates-br --repo-type dataset \
  --revision 6e4debde5359fff91c767e0625f78ddebca71014 \
  --include "overcooked_v1_cramped_room/**" \
  --local-dir data/checkpoints/hf
```

The download is multi-gigabyte. Generated teammates and their learned reference
policies are separate training outputs and are not bundled here.

## Code path

1. `scripts/run_fixed_ego_teamgen.py` trains one generated teammate and its
   matched learned reference. `scripts/run_confirmation_training.py` enforces
   the frozen 12-seed training configuration.
2. `scripts/export_diagnostic_baseline_rows.py` evaluates every policy/start
   pair and records factual events and reference returns.
3. `scripts/analyze_soup_carrier_swap_lock.py` and
   `scripts/analyze_initial_completed_soup_delivery.py` replay the two
   predefined interaction contexts.
4. `scripts/summarize_diagnostic_suite.py` applies matched-control decision
   rules. `scripts/compare_diagnostic_baselines.py` compares factual case
   selection at a shared review budget.
5. `scripts/plot_diagnostic_confirmation.py` plots saved comparisons. Each
   script has `--help` for its input and output paths.

The frozen protocol records the original source revision and file digests.
This release commit has a different Git revision: do not silently relabel a
new run as the original confirmation. Freeze a new protocol before using this
code for a new study. The code does not validate semantic diagnoses against
independent human labels.

## Tests

```bash
PYTHONPATH=.:src python -m pytest -q
```
