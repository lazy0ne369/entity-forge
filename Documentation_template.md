# Business Entity Resolution — Amazon ML Challenge 2026

## 1. Executive Summary & Problem Formulation
In real-world e-commerce ecosystems, entity resolution (ER) links noisy records across diverse platforms without shared unique keys. This challenge formulates ER across three disparate sources ($S_1$, $S_2$, $S_3$), where $S_1$ serves as the deduplicated reference source. A single $S_1$ entity can map to zero (singleton), one, or multiple records in $S_2$ and $S_3$. 

The solution is evaluated under strict macro-averaged $F_{0.5}$ metric ($\beta = 0.5$):
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

The metric strongly penalizes false merges (false positives) over missed links (false negatives) by a factor of 2×, while awarding a full score of $1.0$ for correctly identified singletons (predicting an empty candidate match set). Additionally, the solution is designed to operate reliably under strict 8GB RAM constraints on Windows systems while generalizing to unseen test domains (e.g., France).

---

## 2. System Architecture & Memory-Safe Pipeline
The architecture follows a decoupled, two-stage entity resolution design with four distinct modules:

```
[Raw TSV Data: S1, S2, S3]
           │
           ▼
┌──────────────────────────────────────────────┐
│ Module 1: Out-of-Core Partitioning (DuckDB)  │
│   - Streams multi-GB files without RAM load  │
│   - Dynamic partitioning by open country set │
│   - Compressed Parquet ({country}/source*.pq)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Module 2: Candidate Generation / Blocking    │
│   - Per-country streaming execution          │
│   - Legal suffix normalization & cleaning    │
│   - Char-wb (3,5) TF-IDF sparse dot-product  │
│   - Generates Top-15 candidates per S1       │
│   - Output: candidate_pairs.tsv              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Module 3: Pairwise Feature Engineering       │
│   - RapidFuzz C-extensions: Levenshtein, JW, │
│     Token Set/Sort/Partial Ratios            │
│   - Address 3-gram character Jaccard         │
│   - Numeric token parsing (PIN/ZIP/Street #) │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Module 4: LightGBM Classification & Tuning   │
│   - free_raw_data=True, max_bin=63 (8GB RAM) │
│   - Hard-negative mining from blocking stage │
│   - Strict threshold (0.85) precision cutoff │
│   - Output: matching_results.tsv             │
└──────────────────────────────────────────────┘
```

---

## 3. Detailed Component Methodologies

### 3.1 Module 1: Out-of-Core Ingestion (DuckDB)
- **Challenge:** Full dataset loading causes out-of-memory (OOM) crashes on 8GB RAM environments.
- **Implementation:** DuckDB embedded engine is configured with `PRAGMA memory_limit='3GB'` and `PRAGMA temp_directory` disk-spillover. It streams large TSV files using zero-copy readers (`read_csv(delim='\t')`) and projects data into partition directories (`data/processed/{country}/source*.parquet`) using Zstandard compression.
- **Open Country Support:** Discovers country values dynamically via SQL `SELECT DISTINCT country`, ensuring zero hardcoded assumptions about country domains (seamlessly processing test set countries like France).

### 3.2 Module 2: Candidate Generation & Blocking Strategy
- **Search Space Reduction:** Full pairwise cross-product ($N_{S1} \times (N_{S2} + N_{S3})$) scales to tens of billions of comparisons and is computationally intractable. Blocking shrinks candidate space to $\le 15$ candidates per $S_1$ entity.
- **Normalization:** Cleans names by stripping corporate and legal designations (`Ltd`, `Pvt`, `Corp`, `LLC`, `GmbH`, `Enterprises`, `Solutions`, etc.), normalizing ampersands, removing punctuation, and converting to lowercase.
- **Sub-word TF-IDF Index:** Generates character n-grams within word boundaries (`char_wb`, range 3–5) fit on combined $S_2$ and $S_3$ entities per country. Sub-word character n-grams capture spelling mistakes, typographical errors, concatenations (e.g., "FedEx" vs "Federal Express"), and phonetic transliterations.
- **Sparse Chunking:** Queries $S_1$ entities against $S_{2,3}$ in bounded sparse matrix multiplication chunks (e.g., 2,000 entities), extracting top 15 candidates via linear-time partial sorting (`np.argpartition`).
- **Audit & Compliance:** Emits `candidate_pairs.tsv` streaming directly to disk, establishing the candidate boundary required by the submission guidelines.

### 3.3 Module 3: Pairwise Feature Engineering
For every $(S_1, \text{Candidate})$ pair, an 18-dimensional tabular feature vector is computed:
1. **Name Similarity Metrics:**
   - Levenshtein distance and normalized similarity (`distance.Levenshtein`)
   - Jaro-Winkler similarity (`distance.JaroWinkler`)
   - Token Set Ratio (`fuzz.token_set_ratio`)
   - Token Sort Ratio (`fuzz.token_sort_ratio`)
   - Partial Ratio (`fuzz.partial_ratio`)
2. **Address & Geographic Signals:**
   - Address character 3-gram Jaccard similarity
   - Numeric token set Jaccard similarity (matching street numbers, door codes, building numbers)
   - Numeric token exact match flag
   - Shared numeric token count and numeric count disparity
   - Address Levenshtein and Jaro-Winkler similarity
   - Address Token Set Ratio
3. **Disparity & Heuristics:**
   - Normalized name length difference
   - Normalized address length difference
   - Leading character identity match

### 3.4 Module 4: Classification & $F_{0.5}$ Optimization
- **Model Choice:** LightGBM Gradient Boosted Decision Trees. LightGBM provides nanosecond per-pair inference, robust handling of non-linear metric interactions, and low memory consumption.
- **8GB RAM Windows Optimizations:**
  - `free_raw_data=True` discards source training tables as soon as the histogram dataset is created.
  - Histogram bin capping (`max_bin=63`) reduces memory footprint of feature histograms by 4× compared to default 255.
  - Tree structure bounds: `max_depth=6`, `num_leaves=31`.
- **Negative Sampling / Hard Negatives:** All non-matching candidate pairs generated during the blocking phase serve as informative "hard negatives" (candidates that had high textual overlap but represent distinct entities).
- **Inference & Thresholding:**
  - The model outputs match probabilities $P(\text{match} \mid \text{features})$.
  - Because $F_{0.5}$ penalizes false positives twice as heavily as false negatives, a strict threshold of $\tau = 0.85$ is enforced.
  - Entities with no candidates meeting $\tau$ are predicted as empty sets (singletons), securing maximum credit on singleton records.

---

## 4. Verification & Validation Compliance
The pipeline strictly complies with all official Amazon ML Challenge formatting and submission criteria:
1. Validated via `utils/validate_submission.py` (standard library only, exit code 0).
2. Exactly 1 row per test $S_1$ entity with tab separation.
3. No self-matches (only `S2-` and `S3-` entity IDs).
4. Matches $\subseteq$ Candidates: every ID in `matching_results.tsv` strictly originates from `candidate_pairs.tsv`.
5. Submission package is built as a self-contained `<team_name>_submission.zip` containing `output/`, `code/business_entity_resolution/`, and `Documentation_template.md`.
