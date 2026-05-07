from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

try:
    from sae_lens import SAE, HookedSAETransformer
except Exception as exc:  # pragma: no cover
    raise ImportError(f"Missing sae_lens: {exc}")

REGIONS = ["ex_input", "ex_label", "query", "all"]

# Per-model default layers (empirically tuned)
DEFAULT_LAYER = {"agnews": 12, "rest14": 12, "lap14": 12, "emoc": 12, "tweetemotion": 12, "sst2": 12, "trec": 12, "dbpedia": 12}
DEFAULT_LAYER_LLAMA = {"agnews": 16, "rest14": 26, "lap14": 27, "emoc": 29, "tweetemotion": 29, "sst2": 20, "trec": 22, "dbpedia": 16}


def _resolve_default_layer(dataset: str, model_name: str = "google/gemma-2-2b") -> int:
    """Return the default layer for a dataset+model combination."""
    if "llama" in model_name.lower():
        return DEFAULT_LAYER_LLAMA.get(dataset, 16)
    return DEFAULT_LAYER.get(dataset, 12)


def _cached_hf_model_dir(model_name: str, hf_cache_dir: str | None) -> Path | None:
    if not hf_cache_dir:
        return None
    root = Path(hf_cache_dir)
    model_dir = root / f"models--{model_name.replace('/', '--')}"
    if not model_dir.exists():
        return None
    snapshots = model_dir / "snapshots"
    refs = model_dir / "refs"
    if snapshots.exists() and any(snapshots.iterdir()):
        return model_dir
    if refs.exists() and any(refs.iterdir()):
        return model_dir
    return None


def _cached_hf_snapshot_dir(model_name: str, hf_cache_dir: str | None) -> Path | None:
    model_dir = _cached_hf_model_dir(model_name, hf_cache_dir)
    if model_dir is None:
        return None
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    ref_main = model_dir / "refs" / "main"
    if ref_main.exists():
        revision = ref_main.read_text().strip()
        snap = snapshots / revision
        if snap.exists():
            return snap
    for snap in snapshots.iterdir():
        if snap.is_dir():
            return snap
    return None


def _sae_cache_lookup_names(sae_release: str) -> list[str]:
    names = [sae_release]
    if sae_release.endswith("-canonical"):
        names.append(sae_release[: -len("-canonical")])
    return names


@contextmanager
def _hf_cache_env(hf_cache_dir: str | None):
    if not hf_cache_dir:
        yield
        return
    old_hub_cache = os.environ.get("HF_HUB_CACHE")
    old_home = os.environ.get("HF_HOME")
    old_pulse_cache = os.environ.get("PULSE_HF_CACHE_DIR")
    try:
        os.environ["HF_HUB_CACHE"] = hf_cache_dir
        os.environ["PULSE_HF_CACHE_DIR"] = hf_cache_dir
        if hf_cache_dir.endswith("/hub"):
            os.environ["HF_HOME"] = hf_cache_dir[: -len("/hub")]
        yield
    finally:
        if old_hub_cache is None:
            os.environ.pop("HF_HUB_CACHE", None)
        else:
            os.environ["HF_HUB_CACHE"] = old_hub_cache
        if old_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = old_home
        if old_pulse_cache is None:
            os.environ.pop("PULSE_HF_CACHE_DIR", None)
        else:
            os.environ["PULSE_HF_CACHE_DIR"] = old_pulse_cache


class TokenLengthCache:
    def __init__(self, model: HookedSAETransformer):
        self.model = model
        self.cache: Dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if text not in self.cache:
            self.cache[text] = int(self.model.to_tokens(text).shape[1])
        return self.cache[text]


class FlatFeatureFormatter:
    def __init__(self, feature_dim: int, regions: Sequence[str] | None = None):
        self.regions = list(regions) if regions is not None else list(REGIONS)
        self.feature_dim = feature_dim
        self.flat_dim = feature_dim * len(self.regions)

    def flatten(self, pooled_by_region: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([pooled_by_region[region].reshape(-1) for region in self.regions], dim=0)


def load_model(model_name: str, hf_cache_dir: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    from_pretrained_kwargs: Dict[str, object] = {}
    offline_mode = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    local_snapshot_dir = _cached_hf_snapshot_dir(model_name, hf_cache_dir)
    local_cache_available = local_snapshot_dir is not None
    model_source = str(local_snapshot_dir) if local_snapshot_dir is not None else model_name
    model_revision = os.getenv("PULSE_MODEL_REVISION")
    tokenizer = None
    old_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    with _hf_cache_env(hf_cache_dir):
        try:
            tokenizer_source = model_source
            hf_model = None
            if local_cache_available and not offline_mode:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            if model_revision or offline_mode or local_cache_available:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                tokenizer_kwargs: Dict[str, object] = {
                    "cache_dir": hf_cache_dir,
                    "local_files_only": True,
                }
                if model_revision:
                    tokenizer_kwargs["revision"] = model_revision
                    from_pretrained_kwargs["revision"] = model_revision
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
                if local_cache_available:
                    hf_model_kwargs: Dict[str, object] = {
                        "cache_dir": hf_cache_dir,
                        "local_files_only": True,
                        "torch_dtype": dtype,
                    }
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        **hf_model_kwargs,
                    )
            if offline_mode or local_cache_available:
                from_pretrained_kwargs["local_files_only"] = True
            if local_cache_available and hf_model is not None:
                from transformer_lens import loading_from_pretrained as loading

                official_model_name = loading.get_official_model_name(model_name)
                cfg = loading.get_pretrained_model_config(
                    official_model_name,
                    hf_cfg=hf_model.config.to_dict(),
                    device=device,
                    dtype=dtype,
                    **from_pretrained_kwargs,
                )
                cfg.tokenizer_name = tokenizer_source
                state_dict = loading.get_pretrained_state_dict(
                    official_model_name,
                    cfg,
                    hf_model=hf_model,
                    dtype=dtype,
                    **from_pretrained_kwargs,
                )
                model = HookedSAETransformer(
                    cfg,
                    tokenizer,
                    move_to_device=False,
                )
                model.load_and_process_state_dict(
                    state_dict,
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    fold_value_biases=False,
                    refactor_factored_attn_matrices=False,
                )
                model.move_model_modules_to_device()
                print(f"Loaded pretrained model {model_name} into HookedTransformer")
            else:
                model = HookedSAETransformer.from_pretrained(
                    model_name,
                    device=device,
                    cache_dir=hf_cache_dir,
                    tokenizer=tokenizer,
                    dtype=dtype,
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    fold_value_biases=False,
                    refactor_factored_attn_matrices=False,
                    **from_pretrained_kwargs,
                )
        finally:
            if old_hf_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf_hub_offline
            if old_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline
    return model


def _resolve_sae_id(sae_release: str, layer: int) -> str:
    """Build the SAE ID string for a given release and layer.

    gemma-scope:  "layer_{layer}/width_16k/canonical"
    llama-scope:  "l{layer}r_32x"  (residual, 131k width)
                  "l{layer}r_8x"   (residual, 32k width)
    """
    release_lower = sae_release.lower()
    if "llama_scope" in release_lower or "llama-scope" in release_lower:
        if "32x" in sae_release:
            suffix = "32x"
        elif "8x" in sae_release:
            suffix = "8x"
        else:
            suffix = "32x"
        if "lxm" in release_lower:
            component = "m"
        elif "lxa" in release_lower:
            component = "a"
        else:
            component = "r"
        return f"l{layer}{component}_{suffix}"
    # gemma-scope and other standard releases
    return f"layer_{layer}/width_16k/canonical"


def load_sae_for_layer(sae_release: str, layer: int, device: str | None = None, hf_cache_dir: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = (
        hf_cache_dir
        or os.getenv("PULSE_HF_CACHE_DIR")
        or os.getenv("HF_HUB_CACHE")
    )
    sae_id = _resolve_sae_id(sae_release, layer)
    local_cache_available = any(
        _cached_hf_model_dir(name, cache_dir) is not None
        for name in _sae_cache_lookup_names(sae_release)
    )
    old_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    old_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    with _hf_cache_env(cache_dir):
        try:
            if local_cache_available:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            result = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )
        finally:
            if old_hf_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_hf_hub_offline
            if old_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_transformers_offline
    sae = result[0] if isinstance(result, tuple) else result
    hook_name = getattr(sae.cfg, 'hook_name', None) or sae.cfg.metadata.hook_name
    sae_acts_name = f"{hook_name}.hook_sae_acts_post"
    return sae, hook_name, sae_acts_name


def load_model_and_sae(model_name: str, sae_release: str, layer: int, hf_cache_dir: str, device: str | None = None):
    model = load_model(model_name, hf_cache_dir, device)
    sae, hook_name, sae_acts_name = load_sae_for_layer(sae_release, layer, device, hf_cache_dir)
    return model, sae, hook_name, sae_acts_name


def capture_sae_acts_post_single(model: HookedSAETransformer, sae: SAE, sae_acts_name: str, prompt: str) -> torch.Tensor:
    saved: Dict[str, torch.Tensor] = {}
    def _capture(acts: torch.Tensor, hook):
        saved['acts'] = acts
        return acts
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[(sae_acts_name, _capture)])
    return saved['acts'][0].detach()


def capture_sae_acts_post_batch(
    model: HookedSAETransformer,
    sae: SAE,
    sae_acts_name: str,
    prompts: List[str],
    batch_size: int = 8,
) -> List[torch.Tensor]:
    """Batched SAE activation capture. Returns list of per-prompt activations."""
    all_acts: List[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        # Tokenize with left-padding for batched inference
        tok = model.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        token_lists = [tok.encode(p, add_special_tokens=True) for p in batch_prompts]
        max_len = max(len(t) for t in token_lists)
        padded = []
        for toks in token_lists:
            pad_len = max_len - len(toks)
            padded.append([pad_id] * pad_len + toks)
        batch_tokens = torch.tensor(padded, dtype=torch.long, device=next(model.parameters()).device)

        saved: Dict[str, torch.Tensor] = {}
        def _capture(acts: torch.Tensor, hook):
            saved['acts'] = acts
            return acts
        with torch.no_grad():
            model.run_with_hooks_with_saes(batch_tokens, saes=[sae], fwd_hooks=[(sae_acts_name, _capture)])
        batch_acts = saved['acts'].detach()  # (B, max_len, d_sae)

        # Un-pad: extract only the non-padded positions for each prompt
        for i, toks in enumerate(token_lists):
            seq_len = len(toks)
            pad_len = max_len - seq_len
            all_acts.append(batch_acts[i, pad_len:, :].cpu())
    return all_acts


def pooled_regions_from_acts(
    acts: torch.Tensor,
    region_spans: Dict[str, List[Tuple[int, int]]],
    pooling: str = 'mean',
) -> Dict[str, torch.Tensor]:
    """Pool SAE activations by region from pre-computed activations."""
    acts_f = acts.float()
    pooled: Dict[str, torch.Tensor] = {}
    for region in REGIONS:
        positions = [
            pos for start, end in region_spans.get(region, [])
            for pos in range(max(0, start), min(end, acts_f.shape[0]))
        ]
        if not positions:
            pooled[region] = torch.zeros(acts_f.shape[-1], dtype=acts_f.dtype)
        elif pooling == 'max':
            pooled[region] = acts_f[positions].max(dim=0).values
        elif pooling == 'hybrid':
            mean_pool = acts_f[positions].mean(dim=0)
            max_pool = acts_f[positions].max(dim=0).values
            pooled[region] = 0.5 * mean_pool + 0.5 * max_pool
        else:
            pooled[region] = acts_f[positions].mean(dim=0)
    return pooled


def build_prompt_with_segments(instruction: str, exemplars: Sequence[Tuple[str, str, str]], query_text: str, template_id: str) -> Tuple[str, List[Tuple[str, str]]]:
    # Keep the argument for call-site compatibility; prompt composition is ICE + query only.
    _ = instruction
    if template_id == 'default':
        segments: List[Tuple[str, str]] = []
        for text, _gold, label_word in exemplars:
            segments.extend([('none', 'Input: '), ('ex_input', text), ('none', '\nLabel: '), ('ex_label', label_word), ('none', '\n\n')])
        segments.extend([('none', 'Input: '), ('query', query_text), ('none', '\nLabel:')])
        return ''.join(text for _, text in segments), segments
    if template_id == 'qa':
        segments = []
        for text, _gold, label_word in exemplars:
            segments.extend([('none', 'Text: '), ('ex_input', text), ('none', '\nAnswer: '), ('ex_label', label_word), ('none', '\n\n')])
        segments.extend([('none', 'Text: '), ('query', query_text), ('none', '\nAnswer:')])
        return ''.join(text for _, text in segments), segments
    if template_id == 'block':
        segments = []
        for idx, (text, _gold, label_word) in enumerate(exemplars, start=1):
            segments.extend([('none', f'[Example {idx}]\nContent: '), ('ex_input', text), ('none', '\nCategory: '), ('ex_label', label_word), ('none', '\n---\n')])
        segments.extend([('none', '[Query]\nContent: '), ('query', query_text), ('none', '\nCategory:')])
        return ''.join(text for _, text in segments), segments
    raise ValueError(template_id)


def region_spans_from_segments(segments: Sequence[Tuple[str, str]], token_len: TokenLengthCache) -> Tuple[Dict[str, List[Tuple[int, int]]], int]:
    spans: Dict[str, List[Tuple[int, int]]] = {region: [] for region in REGIONS}
    prefix = ''
    before = token_len(prefix)
    for region, text in segments:
        prefix += text
        after = token_len(prefix)
        if region != 'none' and after > before:
            spans[region].append((before, after))
        before = after
    total = before
    all_start = 1 if total > 1 else 0
    spans['all'] = [(all_start, total)] if total > all_start else []
    return spans, total


def capture_mean_pooled_embedding(model: HookedSAETransformer, text: str, layer: int | None = None) -> torch.Tensor:
    """Capture mean-pooled hidden state from a specific layer (default: last)."""
    tokens = model.to_tokens(text)
    target_layer = layer if layer is not None else model.cfg.n_layers - 1
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    saved: Dict[str, torch.Tensor] = {}
    def _capture(resid: torch.Tensor, hook):
        saved['resid'] = resid[0].float().cpu()
        return resid
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, _capture)])
    return saved['resid'].mean(dim=0)


def pooled_regions_from_prompt(model: HookedSAETransformer, sae: SAE, sae_acts_name: str, prompt: str, region_spans: Dict[str, List[Tuple[int, int]]], pooling: str = 'mean') -> Dict[str, torch.Tensor]:
    acts = capture_sae_acts_post_single(model, sae, sae_acts_name, prompt).float().cpu()
    pooled: Dict[str, torch.Tensor] = {}
    for region in REGIONS:
        positions = [pos for start, end in region_spans.get(region, []) for pos in range(max(0, start), min(end, acts.shape[0]))]
        if not positions:
            pooled[region] = torch.zeros(acts.shape[-1], dtype=acts.dtype)
        elif pooling == 'max':
            pooled[region] = acts[positions].max(dim=0).values
        elif pooling == 'hybrid':
            mean_pool = acts[positions].mean(dim=0)
            max_pool = acts[positions].max(dim=0).values
            pooled[region] = 0.5 * mean_pool + 0.5 * max_pool
        else:
            pooled[region] = acts[positions].mean(dim=0)
    return pooled
