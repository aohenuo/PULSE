# Data Layout

The code uses local dataset files only. It does not download data from dataset
hubs.

Place files in this directory. Classification scripts also accept
`--data_root data`; the generative script accepts `--data-root data`.

## Supported Local Files

AGNews:

- `agnews_train.csv`
- `agnews_test.csv`

Required columns: `text,label`.

SemEval14:

- `rest14_train_frozen.tsv`
- `rest14_test_frozen.tsv`
- `lap14_train_frozen.tsv`
- `lap14_test_frozen.tsv`

Required columns: `text<TAB>label`.

Alternative SemEval14 CSV filenames are also supported:

- `rest14.csv` or `SemEval14_Restaurants.csv`
- `lap14.csv` or `SemEval14_Laptops.csv`

Supported text columns: `text`, `sentence`, or `content`.
Supported label columns: `label`, `sentiment`, or `polarity`.

EmoC:

- `emoc_train.csv`
- `emoc_test.csv`
- `emocontext_train.csv`
- `emocontext_test.csv`

Required columns: `text,label`, `utterance,label`, or `content,emotion`.

Optional classification datasets:

- `sst2_train.csv`, `sst2_test.csv`, or `sst2_validation.csv`
- `trec_train.csv`, `trec_test.csv`, `trec_qc_train.csv`, or `trec_qc_test.csv`
- `dbpedia_train.csv`, `dbpedia_test.csv`, `dbpedia14_train.csv`, or `dbpedia14_test.csv`

SST-2 columns: `sentence,label` or `text,sentiment`.
TREC columns: `text,label`, `question,label`, or `text,label_coarse`.
DBPedia columns: either `text,label`, or `title,content,label`.

Fixed paper-eval TSV files can also be placed at the repository root or in
`data/` as `eval_data_<dataset>.tsv`.

## Generative Tasks

CommonGen uses local JSONL or CSV files:

- `common_gen_train.jsonl`
- `common_gen_validation.jsonl`
- `commongen_train.jsonl`
- `commongen_validation.jsonl`
- `common_gen_train.csv`
- `common_gen_validation.csv`
- `commongen_train.csv`
- `commongen_validation.csv`

JSONL fields: `concepts` and `target`, or `source` and `sentence`.
CSV columns: `concepts,target` or `source,sentence`.

GSM8K uses local JSONL files:

- `gsm8k_train.jsonl`
- `gsm8k_test.jsonl`

Required fields are `question` and `answer`. The answer string should include
the standard final-answer marker `####`.
