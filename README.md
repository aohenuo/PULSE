# PULSE

Clean experiment code for PULSE-guided in-context exemplar selection.

This repository keeps only the public-facing core:

- `scripts/layer_a/run_selection_accuracy.py`: Layer A candidate-set scoring.
- `scripts/layer_b/accuracy_eval.py`: Layer B retrieval accuracy.
- `scripts/layer_b/generative_eval.py`: generative-task utility and downstream generation evaluation.
- Core method: `pulse`, with `softd2_fpw` retrieval enabled by default.

Experiment outputs, logs, paper drafts, caches, launch queues, and archived
ablation scripts are intentionally excluded.

## Setup

```bash
cd PULSE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `PULSE_HF_CACHE_DIR` only if you want model and SAE loading to use a
pre-existing cache.

## Data

All datasets are loaded from local files. Place files under `data/`, pass
`--data_root data` for classification scripts, or pass `--data-root data` for
generative scripts.

Expected local filenames are documented in `data/README.md`.

## Layer A

Layer A scores sampled k-shot candidate sets and selects the top set under
PULSE.

```bash
python scripts/layer_a/run_selection_accuracy.py \
  --datasets agnews \
  --layer 12 \
  --disc-queries 16 \
  --eval-queries 16 \
  --pool-size 200 \
  --n-cand 16 \
  --n-shot 4 \
  --results_dir experiments/layer_a
```

## Layer B

Layer B retrieves k exemplars from a larger pool with PULSE. By default it uses
FPW scores and softD2 quotas.

```bash
python scripts/layer_b/accuracy_eval.py \
  --datasets agnews \
  --layer 12 \
  --disc-queries 16 \
  --eval-queries 16 \
  --pool-size 200 \
  --n-cand 16 \
  --n-shot 4 \
  --results_dir experiments/layer_b
```

Use larger values for full experiments, for example `--eval-queries 512`,
`--pool-size 2000`, and `--disc-queries 64` or higher. Pass
`--retrieval-score blend` to explicitly use the `blend_03` greedy selector.

## Generative Tasks

The generative entry point learns PULSE weights from teacher-forced utility,
then evaluates candidate-set ranking, retrieval utility, and optional
downstream generation.

```bash
python scripts/layer_b/generative_eval.py \
  --task common_gen \
  --layer 12 \
  --disc-queries 2 \
  --eval-queries 3 \
  --candidate-pool-size 6 \
  --pool-size 48 \
  --n-shot 2 \
  --data-root data \
  --output experiments/generative/common_gen_smoke.json
```

Supported tasks are `agnews_headline`, `common_gen`, and `gsm8k`. They expect
local files under `data/` as documented in `data/README.md`.

## Configuration

`configs/layer_a.yaml` and `configs/layer_b.yaml` are lightweight examples for
recording experiment settings. The executable entry points are the scripts
above.

## Environment Variables

- `PULSE_HF_CACHE_DIR`: Hugging Face hub cache path.
- `PULSE_MODEL_REVISION`: optional model revision for local/offline loading.
- `PULSE_DATA_SPLIT_MODE`: `full_test` or `paper_eval`.

## Anonymous Release

The repository intentionally omits experiment logs, local caches, private data,
paper drafts, identifying metadata, and machine-specific paths. Initialize a fresh
git history before publishing if you need an anonymous repository.

## License

MIT.
