from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _candidate_data_roots(data_root: str | Path | None) -> list[Path]:
    roots: list[Path] = []
    if data_root is not None:
        roots.append(Path(data_root).expanduser())
    pkg_root = Path(__file__).resolve().parents[1]
    roots.append(pkg_root / "data")
    dedup: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            dedup.append(root)
    return dedup


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_split_mode(split_mode: str | None) -> str:
    raw = (
        split_mode
        or os.environ.get("PULSE_DATA_SPLIT_MODE")
        or "full_test"
    ).strip().lower()
    aliases = {
        "full": "full_test",
        "all": "full_test",
        "full_test": "full_test",
        "paper": "paper_eval",
        "paper_eval": "paper_eval",
        "legacy": "paper_eval",
    }
    mode = aliases.get(raw)
    if mode is None:
        raise ValueError(
            f"Unsupported split mode: {raw!r}. "
            "Expected one of: full_test, paper_eval."
        )
    return mode


def load_agnews(split: str = "test", data_root: str | Path | None = None) -> list[tuple[str, str]]:
    roots = _candidate_data_roots(data_root)
    local = _first_existing([root / f"agnews_{split}.csv" for root in roots])
    if local is not None:
        return _load_csv_pairs(
            local,
            "text",
            "label",
            {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
        )
    raise FileNotFoundError(
        f"AGNews split '{split}' not found. Expected agnews_{split}.csv in data roots."
    )


def _load_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append((str(row["text"]).strip(), str(row["label"]).strip()))
    return rows


def _load_frozen_split(dataset: str, data_root: str | Path | None = None) -> tuple[list[tuple[str, str]] | None, list[tuple[str, str]] | None]:
    """Load frozen seed=42 80/20 split if files exist in data/. Returns (train, test)."""
    roots = _candidate_data_roots(data_root)
    pkg_root = Path(__file__).resolve().parents[1]
    search_dirs = [pkg_root / "data", *roots]
    train_path = _first_existing([d / f"{dataset}_train_frozen.tsv" for d in search_dirs])
    test_path = _first_existing([d / f"{dataset}_test_frozen.tsv" for d in search_dirs])
    if train_path is None or test_path is None:
        return None, None
    train = _load_tsv_pairs(train_path)
    test = _load_tsv_pairs(test_path)
    print(f"[data] using frozen split for {dataset}: train={len(train)} ({train_path.name}), test={len(test)} ({test_path.name})", flush=True)
    return train, test


def _load_eval_tsv(dataset: str, data_root: str | Path | None = None) -> list[tuple[str, str]] | None:
    """Load fixed evaluation set from eval_data_{dataset}.tsv if it exists."""
    roots = _candidate_data_roots(data_root)
    # Also check the project root (parent of data/)
    pkg_root = Path(__file__).resolve().parents[1]
    search = [pkg_root / f"eval_data_{dataset}.tsv"]
    search += [root / f"eval_data_{dataset}.tsv" for root in roots]
    path = _first_existing(search)
    if path is None:
        return None
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            text = str(row["text"]).strip()
            label = str(row["label"]).strip()
            rows.append((text, label))
    return rows if rows else None


def _load_csv_pairs(path: Path, text_key: str, label_key: str, label_map: Dict[str, str] | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = str(row[text_key]).strip()
            label = str(row[label_key]).strip()
            if label_map is not None:
                label = label_map.get(label, label)
            rows.append((text, label))
    return rows


def _normalize_emoc_label(raw_label: object) -> str | None:
    if raw_label is None:
        return None
    raw = str(raw_label).strip()
    if not raw:
        return None
    mapping = {
        "anger": "angry",
        "angry": "angry",
        "happy": "happy",
        "joy": "happy",
        "joyful": "happy",
        "happiness": "happy",
        "other": "others",
        "others": "others",
        "sad": "sad",
        "sadness": "sad",
    }
    return mapping.get(raw.lower(), raw.lower())


def try_load_semeval14(dataset_name: str, data_root: str | Path | None = None) -> list[tuple[str, str]]:
    roots = _candidate_data_roots(data_root)
    filenames = {
        "rest14": ["rest14.csv", "SemEval14_Restaurants.csv"],
        "lap14": ["lap14.csv", "SemEval14_Laptops.csv"],
    }[dataset_name]
    for root in roots:
        path = _first_existing([root / name for name in filenames])
        if path is not None:
            label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral", "Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"}
            for text_key in ["text", "sentence", "content"]:
                for label_key in ["label", "sentiment", "polarity"]:
                    try:
                        rows = _load_csv_pairs(path, text_key, label_key, label_map)
                        if rows:
                            return rows
                    except Exception:
                        continue
    raise FileNotFoundError(f"Could not find local SemEval14 data for {dataset_name}")


def load_emoc(split: str = "test", data_root: str | Path | None = None) -> list[tuple[str, str]]:
    roots = _candidate_data_roots(data_root)

    split_locals = []
    split_aliases = {
        "train": ["emoc_train.csv", "emocontext_train.csv"],
        "test": ["emoc_test.csv", "emocontext_test.csv"],
        "validation": ["emoc_validation.csv", "emocontext_validation.csv"],
        "valid": ["emoc_validation.csv", "emocontext_validation.csv"],
        "dev": ["emoc_validation.csv", "emocontext_validation.csv"],
    }
    for root in roots:
        for name in split_aliases.get(split, [f"emoc_{split}.csv", f"emocontext_{split}.csv"]):
            split_locals.append(root / name)

    local_split = _first_existing(split_locals)
    if local_split is not None:
        for text_key in ["text", "utterance", "content"]:
            for label_key in ["label", "emotion"]:
                try:
                    rows = _load_csv_pairs(local_split, text_key, label_key)
                    if rows:
                        return [(text, _normalize_emoc_label(label) or str(label)) for text, label in rows]
                except Exception:
                    continue

    local = _first_existing([root / "emoc.csv" for root in roots])
    if local is not None:
        for text_key in ["text", "utterance", "content"]:
            for label_key in ["label", "emotion"]:
                try:
                    rows = _load_csv_pairs(local, text_key, label_key)
                    if rows:
                        return [(text, _normalize_emoc_label(label) or str(label)) for text, label in rows]
                except Exception:
                    continue
    raise FileNotFoundError(
        "EmoC not found in local split files or generic local data roots"
    )


def load_sst2(split: str = "test", data_root: str | Path | None = None) -> list[tuple[str, str]]:
    roots = _candidate_data_roots(data_root)
    candidates = [f"sst2_{split}.csv"]
    if split == "test":
        candidates.append("sst2_validation.csv")
    local = _first_existing([root / name for root in roots for name in candidates])
    if local is None:
        raise FileNotFoundError(
            f"SST-2 split '{split}' not found. Expected one of {candidates} in data roots."
        )
    label_map = {0: "Negative", 1: "Positive"}
    for text_key in ["sentence", "text"]:
        for label_key in ["label", "sentiment"]:
            try:
                rows = _load_csv_pairs(
                    local,
                    text_key,
                    label_key,
                    {str(key): value for key, value in label_map.items()},
                )
                if rows:
                    return rows
            except Exception:
                continue
    raise ValueError(f"Could not parse SST-2 file: {local}")


def load_trec(split: str = "test", data_root: str | Path | None = None) -> list[tuple[str, str]]:
    roots = _candidate_data_roots(data_root)
    candidates = [f"trec_{split}.csv", f"trec_qc_{split}.csv", f"TREC-QC_{split}.csv"]
    local = _first_existing([root / name for root in roots for name in candidates])
    if local is None:
        raise FileNotFoundError(
            f"TREC split '{split}' not found. Expected one of {candidates} in data roots."
        )
    label_map = {0: "Description", 1: "Entity", 2: "Abbreviation",
                 3: "Human", 4: "Number", 5: "Location"}
    for text_key in ["text", "question"]:
        for label_key in ["label", "label_coarse"]:
            try:
                rows = _load_csv_pairs(
                    local,
                    text_key,
                    label_key,
                    {str(key): value for key, value in label_map.items()},
                )
                if rows:
                    return rows
            except Exception:
                continue
    raise ValueError(f"Could not parse TREC file: {local}")


def load_dbpedia(split: str = "test", data_root: str | Path | None = None) -> list[tuple[str, str]]:
    roots = _candidate_data_roots(data_root)
    candidates = [f"dbpedia_{split}.csv", f"dbpedia14_{split}.csv", f"dbpedia_14_{split}.csv"]
    local = _first_existing([root / name for root in roots for name in candidates])
    if local is None:
        raise FileNotFoundError(
            f"DBPedia-14 split '{split}' not found. Expected one of {candidates} in data roots."
        )
    label_map = {0: "Company", 1: "School", 2: "Artist", 3: "Athlete",
                 4: "Politics", 5: "Transportation", 6: "Building",
                 7: "Nature", 8: "Village", 9: "Animal",
                 10: "Plant", 11: "Album", 12: "Film", 13: "Writing"}
    results: list[tuple[str, str]] = []
    with local.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = str(row.get("text") or "").strip()
            if not text:
                title = str(row.get("title") or "").strip()
                content = str(row.get("content") or "").strip()
                text = f"{title}. {content[:200]}" if content else title
            raw_label = str(row.get("label") or "").strip()
            label = label_map.get(int(raw_label), raw_label) if raw_label.isdigit() else raw_label
            if text and label:
                results.append((text, label))
    if not results:
        raise ValueError(f"Could not parse DBPedia-14 file: {local}")
    return results


def load_tweetemotion(split: str = "test", data_root: str | Path | None = None) -> list[tuple[str, str]]:
    roots = _candidate_data_roots(data_root)
    filename = {
        "train": "tweetemotion_train.csv",
        "test": "tweetemotion_test.csv",
        "validation": "tweetemotion_validation.csv",
        "valid": "tweetemotion_validation.csv",
        "dev": "tweetemotion_validation.csv",
    }.get(split, f"tweetemotion_{split}.csv")
    local = _first_existing([root / filename for root in roots])
    if local is not None:
        rows = _load_csv_pairs(local, "text", "label")
        if rows:
            return rows
    raise FileNotFoundError(f"TweetEmotion split '{split}' not found in local data roots")


def prepare_dataset(
    dataset: str,
    seed: int,
    data_root: str | Path | None = None,
    split_mode: str | None = None,
):
    mode = _resolve_split_mode(split_mode)
    if dataset == "agnews":
        train = load_agnews("train", data_root=data_root)
        if mode == "paper_eval":
            tsv_test = _load_eval_tsv("agnews", data_root=data_root)
            test = tsv_test if tsv_test is not None else load_agnews("test", data_root=data_root)
        else:
            test = load_agnews("test", data_root=data_root)
        label_words = {"World": "World", "Sports": "Sports", "Business": "Business", "Sci/Tech": "Sci/Tech"}
        instruction = "Classify the topic of the following news article."
    elif dataset in {"rest14", "lap14"}:
        frozen_train, frozen_test = _load_frozen_split(dataset, data_root=data_root)
        if mode == "paper_eval":
            full = try_load_semeval14(dataset, data_root=data_root)
            tsv_test = _load_eval_tsv(dataset, data_root=data_root)
            if tsv_test is not None:
                test = tsv_test
                train = full
            elif frozen_train is not None and frozen_test is not None:
                train, test = frozen_train, frozen_test
            else:
                full2 = full[:]
                random.Random(seed).shuffle(full2)
                split = int(0.8 * len(full2))
                train, test = full2[:split], full2[split:]
        elif frozen_train is not None and frozen_test is not None:
            train, test = frozen_train, frozen_test
        else:
            full = try_load_semeval14(dataset, data_root=data_root)
            full2 = full[:]
            random.Random(seed).shuffle(full2)
            split = int(0.8 * len(full2))
            train, test = full2[:split], full2[split:]
        label_words = {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"}
        instruction = "Analyze the sentiment of the following aspect."
    elif dataset == "emoc":
        if mode == "paper_eval":
            tsv_test = _load_eval_tsv("emoc", data_root=data_root)
            if tsv_test is not None:
                train = load_emoc("train", data_root=data_root)
                test = tsv_test
            else:
                train = load_emoc("train", data_root=data_root)
                test = load_emoc("test", data_root=data_root)
        else:
            train = load_emoc("train", data_root=data_root)
            test = load_emoc("test", data_root=data_root)
            if train == test:
                raise RuntimeError(
                    "emoc full_test mode requires an actual train/test split; "
                    "only a single unsplit local CSV was found."
                )
        labels = sorted({label for _text, label in train + test})
        label_words = {label: label for label in labels}
        instruction = "Identify the core emotion of the following dialogue."
    elif dataset == "sst2":
        train = load_sst2("train", data_root=data_root)
        test = load_sst2("test", data_root=data_root)
        label_words = {"Positive": "Positive", "Negative": "Negative"}
        instruction = "Classify the sentiment of the following sentence."
    elif dataset == "trec":
        train = load_trec("train", data_root=data_root)
        test = load_trec("test", data_root=data_root)
        label_words = {"Description": "Description", "Entity": "Entity",
                       "Abbreviation": "Abbreviation", "Human": "Human",
                       "Number": "Number", "Location": "Location"}
        instruction = "Classify the type of the following question."
    elif dataset == "dbpedia":
        train = load_dbpedia("train", data_root=data_root)
        test = load_dbpedia("test", data_root=data_root)
        label_words = {"Company": "Company", "School": "School", "Artist": "Artist",
                       "Athlete": "Athlete", "Politics": "Politics",
                       "Transportation": "Transportation", "Building": "Building",
                       "Nature": "Nature", "Village": "Village", "Animal": "Animal",
                       "Plant": "Plant", "Album": "Album", "Film": "Film",
                       "Writing": "Writing"}
        instruction = "Classify the topic of the following text."
    elif dataset == "tweetemotion":
        train = load_tweetemotion("train", data_root=data_root)
        test = load_tweetemotion("test", data_root=data_root)
        labels = ["anger", "joy", "optimism", "sadness"]
        label_words = {label: label for label in labels}
        instruction = "Identify the core emotion expressed in the following tweet."
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train, test, label_words, instruction


def select_eval_examples(test_examples: Sequence[Tuple[str, str]], seed: int, max_queries: int) -> List[Tuple[str, str]]:
    shuffled = list(test_examples)
    random.Random(seed).shuffle(shuffled)
    if max_queries <= 0:
        return shuffled
    return shuffled[: min(max_queries, len(shuffled))]
