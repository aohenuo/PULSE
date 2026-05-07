"""PULSE core library.

Core modules:
  data              — dataset loading
  modeling          — model/SAE loading, activation capture, region segmentation
  metrics           — label scoring, margin computation
  stats             — online statistics, bootstrap CI, McNemar
  exemplar_utility  — PULSE weight learning
  intervention      — hooks and weight reconstruction helpers
  w_cosine          — PULSE discovery implementation
  scoring           — canonical blend_03 scoring and retrieval

Clean experiment scripts live under scripts/layer_a/ and scripts/layer_b/.
"""
from .data import prepare_dataset, select_eval_examples
from .pulse_scaling import ScalingConfig
from .w_cosine import WCosineConfig, discover_pulse_weight

__all__ = [
    # data
    "prepare_dataset",
    "select_eval_examples",
    # config
    "ScalingConfig",
    # discovery
    "WCosineConfig",
    "discover_pulse_weight",
]
