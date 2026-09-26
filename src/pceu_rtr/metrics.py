"""Reference-bound metrics used by the released evaluation runner."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
import re


_BLEU_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\w\s]", re.UNICODE)


def _ngrams(tokens: Sequence[str], width: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index:index + width])
        for index in range(len(tokens) - width + 1)
    )


def corpus_bleu4(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Single-reference, unsmoothed corpus BLEU-4 on a 0–100 scale.

    Matches the CommonGen paper-analysis metric: lowercase ASCII alphanumeric
    tokens and separate punctuation, corpus-level clipping and brevity penalty.
    Empty corpora and any missing n-gram precision return zero.
    """
    if len(predictions) != len(references):
        raise ValueError("Corpus BLEU inputs must be aligned")
    clipped = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    hypothesis_length = 0
    reference_length = 0
    for prediction, reference in zip(predictions, references):
        prediction_tokens = _BLEU_TOKEN_RE.findall(str(prediction).lower())
        reference_tokens = _BLEU_TOKEN_RE.findall(str(reference).lower())
        hypothesis_length += len(prediction_tokens)
        reference_length += len(reference_tokens)
        for width in range(1, 5):
            prediction_ngrams = _ngrams(prediction_tokens, width)
            reference_ngrams = _ngrams(reference_tokens, width)
            clipped[width - 1] += sum(
                min(count, reference_ngrams.get(ngram, 0))
                for ngram, count in prediction_ngrams.items()
            )
            totals[width - 1] += max(len(prediction_tokens) - width + 1, 0)
    if hypothesis_length == 0:
        return 0.0
    precisions = []
    for matches, total in zip(clipped, totals):
        if matches == 0 or total == 0:
            return 0.0
        precisions.append(matches / total)
    brevity_penalty = (
        1.0 if hypothesis_length > reference_length
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    return float(100.0 * brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / 4.0
    ))
