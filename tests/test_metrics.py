import math

import pytest

from pceu_rtr.metrics import corpus_bleu4


def test_exact_match_and_case_punctuation_tokenization():
    assert corpus_bleu4(["A cat sits there."], ["A cat sits there."]) == 100.0
    # Commas and periods are separate tokens, independent of adjacent spaces.
    assert corpus_bleu4(["A,B C.D!"], ["a , b c . d !"]) == 100.0


def test_repeated_ngrams_are_clipped_to_reference_counts():
    # Predicted n-gram counts: 5,4,3,2; clipped counts: 4,3,2,1.
    expected = 100.0 * ((4 / 5) * (3 / 4) * (2 / 3) * (1 / 2)) ** 0.25
    assert corpus_bleu4(["a a a a a"], ["a a a a"]) == pytest.approx(expected)


def test_brevity_penalty_uses_total_token_lengths():
    assert corpus_bleu4(["a b c d"], ["a b c d e f"]) == pytest.approx(100 * math.exp(-0.5))
    # The second pair contributes lengths but no 4-grams. This is corpus BLEU,
    # not an average of sentence BLEU scores.
    expected = 100 * math.exp(1 - 8 / 7)
    assert corpus_bleu4(["a b c d", "x y z"], ["a b c d e", "x y z"]) == pytest.approx(expected)


def test_clipping_is_per_sentence_before_corpus_aggregation():
    assert corpus_bleu4(["a b c d", "w x y z"], ["w x y z", "a b c d"]) == 0.0


@pytest.mark.parametrize("predictions,references", [
    ([], []), ([""], ["a b c d"]), ([""], [""]),
    (["a b c"], ["a b c"]),  # No smoothing of missing 4-grams.
    (["a b c d"], ["a b c x"]),  # Nonzero lower orders, zero 4-gram match.
    (["a b c d"], [""]),
])
def test_empty_and_zero_precision(predictions, references):
    assert corpus_bleu4(predictions, references) == 0.0


@pytest.mark.parametrize("predictions,references", [([], ["x"]), (["x"], []), (["x", "y"], ["x"])])
def test_misaligned_corpora_fail(predictions, references):
    with pytest.raises(ValueError, match="aligned"):
        corpus_bleu4(predictions, references)
