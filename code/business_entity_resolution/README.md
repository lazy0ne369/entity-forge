# Business Entity Resolution Pipeline
**Amazon ML Challenge 2026**  
**Team Name**: GENZ MINDS  

This directory contains the self-contained, runnable entity resolution pipeline designed to resolve records across disparate data sources ($S_2, S_3$) to canonical reference entities in $S_1$ under the macro-averaged $F_{0.5}$ metric.

---

## 1. System Requirements & Environment Setup

- **Python Version**: Python 3.10 to 3.13
- **Operating System**: Linux, macOS, or Windows (tested on Windows 11 64-bit)
- **Memory**: Minimum 8GB RAM (peak pipeline memory consumption is $< 400\text{MB}$ per partition)
- **Storage**: Sufficient disk space for temporary parquet partitions and TSV outputs

### Installation
From the root of this submission (or within this directory), install the pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. End-to-End Pipeline Execution (`data -> blocking -> matching -> output`)

The entire pipeline is executed via a single unified entry point: [`src/main.py`](src/main.py).

### Quick Dry-Run (Verification on Built-in Sample Dataset)
To verify environment setup and validate pipeline execution without needing large dataset files:

```bash
python code/business_entity_resolution/src/main.py --mode sample
```

This performs:
1. Ingestion and country-partitioning of sample TSV files.
2. LightGBM training on 19-dimensional pairwise features.
3. Inverted-index candidate generation.
4. Threshold-based matching ($F_{0.5}$ cutoff $\tau = 0.85$).
5. Automated validation via the official submission validator.

Outputs are written to:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

---

### Full Competition Dataset Run
To regenerate the final submission files from the official competition training and test datasets:

```bash
python code/business_entity_resolution/src/main.py \
    --mode full \
    --train-dir <path_to_train_dir> \
    --test-dir <path_to_test_dir> \
    --threshold 0.85
```

Where:
- `<path_to_train_dir>` contains:
  - `train_source1.tsv`
  - `train_source2.tsv`
  - `train_source3.tsv`
  - `train_ground_truth.tsv`
- `<path_to_test_dir>` contains:
  - `test_source1.tsv`
  - `test_source2.tsv`
  - `test_source3.tsv`

### Example with Local Directory Layout:
```bash
python code/business_entity_resolution/src/main.py \
    --mode full \
    --train-dir data/dataset/train \
    --test-dir data/dataset/test \
    --threshold 0.85
```

---

## 3. Pipeline Architecture & Modular Components

All source code is housed under [`src/`](src/):

| File | Core Responsibility |
| :--- | :--- |
| **[`src/main.py`](src/main.py)** | Master end-to-end execution script, CLI argument parser, and step-by-step orchestrator. |
| **[`src/ingestion.py`](src/ingestion.py)** | Memory-safe, out-of-core streaming and dynamic country partitioning powered by DuckDB. |
| **[`src/blocking.py`](src/blocking.py)** | High-recall, low-candidate Token-Filtered Inverted Index candidate generator with RapidFuzz scoring. |
| **[`src/features.py`](src/features.py)** | 19-dimensional pairwise feature extractor (Levenshtein, Jaro-Winkler, Token Set/Sort ratios, character 3-gram Jaccard, numeric street/PIN codes). |
| **[`src/model.py`](src/model.py)** | Memory-optimized LightGBM model training (`free_raw_data=True`, `max_bin=127`, `max_depth=6`) and batch inference. |
| **[`src/evaluate.py`](src/evaluate.py)** | Official competition metric: Macro-averaged $F_{0.5}$ with full singleton credit. |

---

## 4. Output Validation

Validate that the generated outputs conform strictly to the competition submission schema:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir <path_to_test_dir>
```

Expected output: `PASS (exit 0)`.
