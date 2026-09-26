import json
from pathlib import Path

import pytest

from pceu_rtr.data import Example, load_examples, normalize_input


def write(tmp_path, content, suffix="tsv"):
    path = tmp_path / f"examples.{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("task,count", [
    ("agnews", 512), ("rest14", 512), ("lap14", 463),
    ("emoc", 512), ("commongen", 512), ("gsm8k", 512),
])
def test_bundled_clean_datasets(task, count):
    path = Path(__file__).resolve().parents[1] / "dataset" / f"{task}.tsv"
    examples = load_examples(path, task)
    assert len(examples) == count
    assert len({e.id for e in examples}) == count
    assert all(e.input for e in examples)
    assert all(e.target and e.target.strip() for e in examples)
    if task == "gsm8k":
        assert all("####" in e.target for e in examples)


def test_tsv_ignores_predictions_and_preserves_source_text(tmp_path):
    path = write(tmp_path, 'index\ttext\tgold\tn_wrong_of_7\tpaper_pulse\n'
                 '007\t"  first\tline\nsecond  "\tWorld\t3\twrong\n')
    assert load_examples(path, "agnews") == [Example("007", "  first\tline\nsecond  ", "World")]


def test_jsonl_training(tmp_path):
    path = write(tmp_path, json.dumps({"id": 7, "input": " Q ", "target": " A ",
                                      "paper_pulse": "wrong"}) + "\n", "jsonl")
    assert load_examples(path, "gsm8k", require_targets=True) == [Example("7", " Q ", " A ")]


@pytest.mark.parametrize("column", ["answer", "target"])
def test_gsm8k_optional_answer_columns(tmp_path, column):
    path = write(tmp_path, f"index\tquestion\t{column}\n0\t1+1?\t2\n")
    assert load_examples(path, "gsm8k", require_targets=True)[0].target == "2"


def test_unlabeled_gsm8k_cannot_train(tmp_path):
    path = write(tmp_path, "index\tquestion\n0\t1+1?\n")
    assert load_examples(path, "gsm8k") == [Example("0", "1+1?", None)]
    with pytest.raises(ValueError, match="missing TSV columns"):
        load_examples(path, "gsm8k", require_targets=True)


@pytest.mark.parametrize("content,message", [
    ("index\ttext\n0\tq\n", "missing TSV columns"),
    ("index\ttext\tgold\n0\tq\ta\n0\tr\tb\n", "duplicate id"),
    ("index\ttext\tgold\n0\t \ta\n", "input must"),
    ("index\ttext\tgold\n0\tq\t\n", "target must"),
    ("index\ttext\tgold\n \tq\ta\n", "id must"),
    ("index\ttext\tgold\n0\tq\n", "wrong number"),
    ("index\ttext\tgold\n0\tq\ta\textra\n", "wrong number"),
    ("index\ttext\tgold\tgold\n0\tq\ta\tb\n", "duplicate or empty"),
    ("index\ttext\tgold\n", "no examples"),
    ('index\ttext\tgold\n0\t"unclosed\ta\n', "malformed TSV"),
])
def test_invalid_tsv(tmp_path, content, message):
    with pytest.raises(ValueError, match=message):
        load_examples(write(tmp_path, content), "agnews")


@pytest.mark.parametrize("content,message", [
    ('{"id":"0","input":"q"}\n', "target is required"),
    ('{"id":true,"input":"q","target":"a"}\n', "id must"),
    ('{"id":"0","input":5,"target":"a"}\n', "input must"),
    ('{"id":"0","input":"q","target":5}\n', "target must"),
    ('{"id":"0","input":"q","target":" "}\n', "target must"),
    ('{"input":"q","target":"a"}\n', "id must"),
    ('[]\n', "must be an object"),
    ('{bad json}\n', "malformed JSON"),
    ('\n', "empty JSONL"),
    ('', "no examples"),
    ('{"id":0,"input":"q","target":"a"}\n'
     '{"id":"0","input":"r","target":"b"}\n', "duplicate id"),
])
def test_invalid_jsonl(tmp_path, content, message):
    with pytest.raises(ValueError, match=message):
        load_examples(write(tmp_path, content, "jsonl"), "agnews")


def test_jsonl_unlabeled_gsm8k(tmp_path):
    path = write(tmp_path, '{"id":"0","input":"q"}\n', "jsonl")
    assert load_examples(path, "gsm8k")[0].target is None
    with pytest.raises(ValueError, match="target is required"):
        load_examples(path, "gsm8k", require_targets=True)


def test_reject_unknown_task_and_format(tmp_path):
    with pytest.raises(ValueError, match="Unknown task"):
        load_examples(tmp_path / "x.tsv", "unknown")
    with pytest.raises(ValueError, match="expected a .tsv or .jsonl"):
        load_examples(tmp_path / "x.csv", "agnews")


def test_normalization_is_separate():
    assert normalize_input("  A\n\tB  Straße ") == "a b strasse"
