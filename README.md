# PULSE User Guide

PULSE uses sparse autoencoder (SAE) features to select few-shot demonstrations from training examples for a language model prompt.

The package provides two commands:

- `pulse-select`: reads precomputed SAE features and returns the selected example indices.
- `pulse-eval`: reads raw text, loads a language model and an SAE, extracts features, selects demonstrations, and generates predictions.

## 1. Installation

Requires Python 3.10+ and PyTorch 2.1+. Run the following from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`. If you already have a CUDA-enabled PyTorch environment, you can install the package in that environment.

For inference with a real model, also install:

```bash
python -m pip install -e '.[eval]'
```

## 2. Quick start: select demonstrations

The following commands do not download a model or require a GPU:

```bash
pulse-select examples/tiny.json --n-shot 2 --output outputs/selection.json
python examples/select_context.py
```

The output contains:

- `indices`: selected candidate-pool row indices, starting at 0 and ordered by greedy selection.
- `selected_ids`: corresponding IDs when the input provides `pool_ids`.
- `pulse_weight` and `active_features`: retrieval weights and the number of nonzero features.
- `config`, `method`, and `input_sha256`: configuration, method, and input-file hash.

## 3. Use your own features

### JSON input

Save the following as `input.json`:

```json
{
  "discovery": [
    {
      "utilities": [0.0, 0.5, 1.0],
      "activations": [[0, 2], [1, 1], [2, 0]]
    }
  ],
  "topk_each_sign": 1,
  "query_sae": [2, 0],
  "pool_sae": [[2, 0], [0, 2], [1, 1]],
  "pool_ids": ["train-0", "train-1", "train-2"]
}
```

Run:

```bash
pulse-select input.json --n-shot 2 --output outputs/selection.json
```

Each `discovery` entry represents one training query. `utilities` contains the utilities of its candidate contexts, and `activations` contains their SAE features in the same order. For classification, utility is the score difference between the correct label and the highest-scoring incorrect label. For generation, utility is the length-normalized conditional log likelihood of the reference answer.

If weights are already available, replace `discovery` and `topk_each_sign` with `"pulse_weight": [0.8, -0.3]`. Do not provide `pulse_weight` and `discovery` together.

| Field | Shape | Meaning |
| --- | --- | --- |
| `query_sae` | `[D]` | SAE features of the current query |
| `pool_sae` | `[P, D]` | SAE features of P candidate examples |
| `pulse_weight` | `[D]` | Learned sparse weights |
| `discovery[q].utilities` | `[N_q]` | Candidate-context utilities for training query q |
| `discovery[q].activations` | `[N_q, D]` | Features of those candidate contexts |
| `pool_ids` | `[P]` | Optional unique candidate-example IDs |

All features must use the same model, SAE, layer, feature order, and pooling method. Retrieval encodings for the query and candidate examples must exclude their respective answers. Each discovery query needs at least two candidate contexts, with at least two unordered candidate pairs in total. All numeric inputs must be finite, and the retrieval weights must contain at least one nonzero value.

### Python API

```python
import torch
from pceu_rtr import RetrievalConfig, learn_pulse_weight, select_pulse_context

utilities = [torch.tensor([0.0, 0.5, 1.0])]
activations = [torch.tensor([[0., 2.], [1., 1.], [2., 0.]])]
query_sae = torch.tensor([2., 0.])
pool_sae = torch.tensor([[2., 0.], [0., 2.], [1., 1.]])

weight = learn_pulse_weight(utilities, activations, topk_each_sign=1)
indices = select_pulse_context(
    query_sae,
    pool_sae,
    weight,
    config=RetrievalConfig(n_shot=2, shortlist=50, beta=0.3, lambda_r=0.3),
)
print(indices)
```

Use `indices` to retrieve demonstrations from your training-example list. The list order must match the row order of `pool_sae`. The number of returned examples is at most `min(n_shot, shortlist, pool_size)`. For large feature tensors, use the Python tensor API directly.

### Optional semantic fusion

The default `--method pulse` uses SAE retrieval. Fusion methods also require `query_sbert` (`[H]`) and `pool_sbert` (`[P, H]`) in the JSON input:

```bash
pulse-select input-with-semantic.json --method classification-fusion --alpha 0.5
pulse-select input-with-semantic.json --method generation-fusion --alpha 0.5
```

## 4. Run predictions from raw data

### Prepare the model

Default settings:

- Language model: `google/gemma-2-2b`
- SAE release: `gemma-scope-2b-pt-res-canonical`
- SAE ID: `layer_12/width_16k/canonical`
- Device and precision: `cuda` and `bfloat16`

Obtain access to the model and authenticate with Hugging Face on your machine, or prepare a local model cache. Use `--local-files-only` to load only from the cache. GPU memory requirements depend on the model, prompt length, and number of demonstrations.

### Prepare training and evaluation data

Use a separate JSONL training file with one example per line:

```json
{"id":"train-0","input":"A team wins the match.","target":"Sports"}
```

| Task | `input` | `target` |
| --- | --- | --- |
| `agnews` | News text | `World`, `Sports`, `Business`, `Sci/Tech` |
| `rest14`, `lap14` | Text to classify, including aspect information | `Positive`, `Negative`, `Neutral` |
| `emoc` | Dialogue text | `angry`, `happy`, `others`, `sad` |
| `commongen` | Comma-separated concepts | Reference sentence |
| `gsm8k` | Math question | Worked solution ending in `#### <number>` |

Evaluation data may use the same JSONL format or the TSV format in the project's `dataset/` directory:

- Classification tasks: `index`, `text`, `gold`.
- CommonGen: `index`, `concepts`, `reference`.
- GSM8K: `index`, `question`, `answer` (or `target`; do not provide both target columns).

With `--task` specified, `pulse-eval` reads `dataset/<task>.tsv` by default. After installation, it can also read the copy installed with the package. Use `--eval-data` to supply your own TSV or JSONL file. Training data is not bundled; you must provide a separate labeled file with `--train-data`.

Training examples must have answers. GSM8K evaluation examples may omit answers, in which case the program generates predictions without scoring them. The program removes train/evaluation overlaps after case-insensitive whitespace normalization and deduplicates training inputs. At least `pool_size + discovery_queries` training examples must remain so it can sample disjoint candidate-pool and feature-discovery rows.

### Run one task

```bash
pulse-eval \
  --task agnews \
  --train-data /path/to/train/agnews.jsonl \
  --output outputs/agnews-run \
  --limit 8 \
  --pool-size 32 \
  --discovery-queries 2 \
  --candidate-sets 3 \
  --topk-each-sign 64 \
  --n-shot 4 \
  --max-input-tokens 2048
```

Replace the training path with the path to your file. `--limit 0` processes every evaluation example. Use a new or empty output directory for each run.

### Run all tasks

After preparing a training file for each task, run:

```bash
for task in agnews rest14 lap14 emoc commongen gsm8k; do
  pulse-eval \
    --task "$task" \
    --train-data "/path/to/train/$task.jsonl" \
    --output "outputs/$task-run" \
    --limit 0 --pool-size 32 --discovery-queries 2 \
    --candidate-sets 3 --topk-each-sign 64 \
    --max-input-tokens 2048 --max-new-tokens 256 || break
done
```

### Common options

| Option | Default | Description |
| --- | --- | --- |
| `--n-shot` | 4 | Number of demonstrations to select |
| `--shortlist` | 50 | Number of candidates retained before greedy selection |
| `--beta` | 0.3 | Mixture weight for full-SAE similarity |
| `--lambda-r` | 0.3 | Demonstration redundancy penalty |
| `--pool-size` | 2000 | Training candidate-pool size for evaluation |
| `--discovery-queries` | 64 | Number of training queries used to learn feature weights |
| `--candidate-sets` | 32 | Number of demonstration sets tried per training query |
| `--topk-each-sign` | 512 | Maximum number of positive and negative features retained per sign |
| `--limit` | 0 | Number of evaluation examples; 0 means all |
| `--seed` | 42 | Sampling seed |
| `--max-input-tokens` | 2048 | Input token limit; longer inputs raise an error |
| `--max-new-tokens` | 128 | Generated token limit |

The first four options apply to both `pulse-select` and `pulse-eval`. The other options in the table apply to `pulse-eval`. For `pulse-select`, `topk_each_sign` comes from the input JSON. To see all available options, run:

```bash
pulse-select --help
pulse-eval --help
```

### Inspect the output

- `predictions.jsonl`: records each input, reference answer, prediction, selected training-example IDs, prompt hash, and correctness when it can be computed.
- `summary.json`: records the configuration, model information, data hashes, sample split, runtime, and metrics after successful completion.
- `discovery.pt`: stores the learned weights, utilities, and SAE activation tensors.

Classification tasks select answers using conditional label log probabilities and report accuracy. CommonGen saves generated sentences and references; compute BLEU separately. GSM8K extracts the number after `####` for matching. A generation without that marker is counted as incorrect, while an example without a parseable reference answer is not scored.

If `predictions.jsonl` is only partially written and there is no complete `summary.json`, the run did not finish successfully. Overlong prompts raise an error; adjust the input budget or the number of demonstrations, then use a new output directory.

## 5. Run tests and build the package

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m build
```
