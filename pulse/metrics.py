from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from sae_lens import HookedSAETransformer, SAE
except Exception:  # pragma: no cover
    HookedSAETransformer = object  # type: ignore
    SAE = object  # type: ignore


def label_token_sequences(model: HookedSAETransformer, label_words: Mapping[str, str]) -> Dict[str, List[int]]:
    tok = model.tokenizer
    out: Dict[str, List[int]] = {}
    for gold, word in label_words.items():
        token_ids = tok.encode(" " + str(word), add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Empty token sequence for label word: {word!r}")
        out[str(gold)] = [int(token_id) for token_id in token_ids]
    return out


def _build_candidate_sequence_batch(prompt_tokens: torch.Tensor, label_token_seqs: Mapping[str, Sequence[int]], pad_id: int) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, int]:
    labels = list(label_token_seqs.keys())
    device = prompt_tokens.device
    prompt_row = prompt_tokens[0]
    prompt_len = int(prompt_row.shape[0])
    max_answer_len = max(len(label_token_seqs[label]) for label in labels)
    full_rows: List[torch.Tensor] = []
    answer_tokens = torch.full((len(labels), max_answer_len), pad_id, dtype=torch.long, device=device)
    answer_mask = torch.zeros((len(labels), max_answer_len), dtype=torch.bool, device=device)
    for row_idx, label in enumerate(labels):
        seq = list(label_token_seqs[label])
        seq_t = torch.tensor(seq, dtype=torch.long, device=device)
        answer_len = int(seq_t.shape[0])
        answer_tokens[row_idx, :answer_len] = seq_t
        answer_mask[row_idx, :answer_len] = True
        if answer_len < max_answer_len:
            pad = torch.full((max_answer_len - answer_len,), pad_id, dtype=torch.long, device=device)
            seq_t = torch.cat([seq_t, pad], dim=0)
        full_rows.append(torch.cat([prompt_row, seq_t], dim=0))
    full_tokens = torch.stack(full_rows, dim=0)
    return labels, full_tokens, answer_tokens, answer_mask, prompt_len


def _sequence_scores_from_logits(logits: torch.Tensor, prompt_len: int, answer_tokens: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    answer_len = int(answer_tokens.shape[1])
    logits_pred = logits[:, prompt_len - 1 : prompt_len - 1 + answer_len, :].float()
    logprobs = F.log_softmax(logits_pred, dim=-1)
    token_lp = logprobs.gather(-1, answer_tokens.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * answer_mask.to(dtype=token_lp.dtype)
    return token_lp.sum(dim=-1)


def score_candidate_label_sequences(model: HookedSAETransformer, prompt: str, label_words: Mapping[str, str], *, sae: SAE | None = None, fwd_hooks: list[tuple[str, object]] | None = None, plain_hooks: list[tuple[str, object]] | None = None) -> Dict[str, float]:
    prompt_tokens = model.to_tokens(prompt)
    pad_id = model.tokenizer.pad_token_id if getattr(model.tokenizer, "pad_token_id", None) is not None else 0
    label_token_seqs = label_token_sequences(model, label_words)
    labels, full_tokens, answer_tokens, answer_mask, prompt_len = _build_candidate_sequence_batch(prompt_tokens, label_token_seqs, pad_id)
    with torch.no_grad():
        if sae is not None:
            logits = model.run_with_hooks_with_saes(full_tokens, saes=[sae], fwd_hooks=fwd_hooks or [])
        elif plain_hooks:
            logits = model.run_with_hooks(full_tokens, fwd_hooks=plain_hooks)
        else:
            logits = model(full_tokens)
    scores = _sequence_scores_from_logits(logits, prompt_len, answer_tokens, answer_mask)
    return {label: float(scores[idx].item()) for idx, label in enumerate(labels)}


def score_candidate_label_sequences_batch(
    model: HookedSAETransformer,
    prompts: list[str],
    label_words: Mapping[str, str],
    batch_size: int = 4,
) -> list[Dict[str, float]]:
    """Batched label scoring: score multiple prompts against label words.

    Groups prompts into batches, pads to uniform length within each batch,
    and scores all (prompt × label) combinations in a single forward pass
    per batch.  Returns one {label: score} dict per prompt.
    """
    if not prompts:
        return []

    tok = model.tokenizer
    pad_id = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else 0
    label_token_seqs = label_token_sequences(model, label_words)
    labels = list(label_token_seqs.keys())
    n_labels = len(labels)
    max_answer_len = max(len(label_token_seqs[lab]) for lab in labels)

    # Pre-tokenize all prompts
    all_prompt_tokens = [tok.encode(p, add_special_tokens=True) for p in prompts]
    device = next(model.parameters()).device

    all_results: list[Dict[str, float]] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompt_tokens = all_prompt_tokens[start:start + batch_size]
        bs = len(batch_prompt_tokens)

        # For each prompt in this batch, build (prompt + label_i) for all labels
        # Result: bs * n_labels sequences
        max_prompt_len = max(len(t) for t in batch_prompt_tokens)
        max_total_len = max_prompt_len + max_answer_len

        full_tokens_list = []
        prompt_lens = []
        answer_tokens_list = []
        answer_mask_list = []

        for prompt_toks in batch_prompt_tokens:
            plen = len(prompt_toks)
            prompt_lens.append(plen)
            for lab in labels:
                ans_seq = list(label_token_seqs[lab])
                ans_len = len(ans_seq)
                # left-pad prompt, then append answer, then right-pad answer
                left_pad = max_prompt_len - plen
                seq = [pad_id] * left_pad + prompt_toks + ans_seq
                if len(seq) < max_total_len:
                    seq = seq + [pad_id] * (max_total_len - len(seq))
                full_tokens_list.append(seq[:max_total_len])

                ans_t = [0] * max_answer_len
                ans_m = [0] * max_answer_len
                for k in range(ans_len):
                    ans_t[k] = ans_seq[k]
                    ans_m[k] = 1
                answer_tokens_list.append(ans_t)
                answer_mask_list.append(ans_m)

        full_tokens = torch.tensor(full_tokens_list, dtype=torch.long, device=device)
        answer_tokens = torch.tensor(answer_tokens_list, dtype=torch.long, device=device)
        answer_mask = torch.tensor(answer_mask_list, dtype=torch.bool, device=device)

        with torch.no_grad():
            logits = model(full_tokens)  # (bs*n_labels, max_total_len, vocab)

        # Extract scores for each prompt
        for i in range(bs):
            plen = prompt_lens[i]
            # Account for left-padding: actual prompt starts at (max_prompt_len - plen)
            effective_prompt_end = max_prompt_len  # all prompts end at same position due to left-pad
            row_start = i * n_labels
            row_end = row_start + n_labels
            row_logits = logits[row_start:row_end]  # (n_labels, max_total_len, vocab)
            row_ans = answer_tokens[row_start:row_end]
            row_mask = answer_mask[row_start:row_end]
            scores = _sequence_scores_from_logits(row_logits, effective_prompt_end, row_ans, row_mask)
            all_results.append({lab: float(scores[j].item()) for j, lab in enumerate(labels)})

    return all_results


def margin_for_gold_label_scores(label_scores: Mapping[str, float], gold_label: str) -> float:
    gold_score = float(label_scores[gold_label])
    other_scores = [float(score) for label, score in label_scores.items() if label != gold_label]
    return gold_score - max(other_scores) if other_scores else float("inf")


def predict_label_and_confidence_gap_from_scores(label_scores: Mapping[str, float]) -> Tuple[str, float]:
    items = list(label_scores.items())
    labels = [label for label, _ in items]
    scores = torch.tensor([float(score) for _, score in items], dtype=torch.float32)
    top_vals, top_pos = torch.topk(scores, k=min(2, int(scores.numel())))
    pred = labels[int(top_pos[0].item())]
    if int(top_vals.numel()) < 2:
        return pred, float("inf")
    return pred, float((top_vals[0] - top_vals[1]).item())
