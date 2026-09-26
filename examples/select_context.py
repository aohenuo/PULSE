"""Synthetic discovery-to-selection example; no downloads or model needed."""
import json
from pathlib import Path

import torch

from pceu_rtr import RetrievalConfig, learn_pulse_weight, select_pulse_context


def main():
    payload = json.loads(Path(__file__).with_name("tiny.json").read_text())
    utilities = [torch.tensor(row["utilities"]) for row in payload["discovery"]]
    activations = [torch.tensor(row["activations"]) for row in payload["discovery"]]
    weight = learn_pulse_weight(utilities, activations, topk_each_sign=1)
    indices = select_pulse_context(
        torch.tensor(payload["query_sae"]), torch.tensor(payload["pool_sae"]), weight,
        config=RetrievalConfig(n_shot=2),
    )
    # These texts and features are invented separately to illustrate row mapping.
    # They are not features extracted from these texts by a real SAE.
    training_examples = [
        {"input": "A spacecraft launches.", "target": "science"},
        {"input": "A team wins the match.", "target": "sports"},
        {"input": "An athlete tests a new sensor.", "target": "sports"},
        {"input": "A telescope detects a planet.", "target": "science"},
    ]
    print(json.dumps({
        "synthetic": True, "indices": indices, "weight": weight.tolist(),
        "demonstrations": [training_examples[i] for i in indices],
    }, indent=2))


if __name__ == "__main__":
    main()
