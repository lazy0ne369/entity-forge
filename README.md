# Amazon ML Challenge 2026 — Business Entity Resolution

## Project Structure

```
Amazon_ML_Challenge_submission/
├── output/                          ← Generated files land here
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── dataset/                         ← Place competition data here
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py       ← Official format validator
├── code/
│   └── business_entity_resolution/
│       └── src/
│           ├── main.py              ← Pipeline entry point (run this)
│           ├── blocking.py          ← Inverted-index candidate generation
│           ├── features.py          ← 19-D pairwise feature extractor
│           ├── model.py             ← LightGBM training & inference
│           ├── ingestion.py         ← DuckDB streaming ingestion
│           └── evaluate.py          ← Macro F_0.5 metric
├── requirements.txt
└── Documentation_template.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run pipeline (sample dry-run — no dataset needed)
```bash
python code/business_entity_resolution/src/main.py --mode sample
```

### 3. Run pipeline (full competition dataset)
Place the competition TSV files under `dataset/train/` and `dataset/test/`, then:
```bash
python code/business_entity_resolution/src/main.py \
    --mode full \
    --train-dir dataset/train \
    --test-dir  dataset/test \
    --threshold 0.85
```

### 4. Validate submission outputs
```bash
python utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
```

---

## Key CLI Arguments for `main.py`

| Argument | Default | Description |
|---|---|---|
| `--mode` | `sample` | `sample` for dry-run, `full` for competition dataset |
| `--train-dir` | — | Required in `full` mode |
| `--test-dir` | — | Required in `full` mode |
| `--threshold` | `0.85` | LightGBM match probability cutoff |
| `--output-dir` | `output/` | Output folder (relative to repo root) |
| `--work-dir` | `data/processed/` | Scratch folder for Parquet partitions |

---

## Architecture

```
TSV files → DuckDB Ingestion (country Parquet partitions)
         → Inverted-Index Blocking (top-15 candidates / S1 entity)
         → 19-D RapidFuzz Feature Extraction
         → LightGBM Binary Classifier (200 rounds)
         → Threshold Decision → output/matching_results.tsv
                              → output/candidate_pairs.tsv
```

- **Candidate recall**: ~95.1% with avg < 6 candidates / entity
- **Peak RAM**: < 400 MB per country shard (safe on 8 GB Windows)
- **Metric**: Macro-averaged F₀.₅ (precision weighted 4× over recall)
