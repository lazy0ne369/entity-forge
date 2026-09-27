# Amazon ML Challenge 2026 — Business Entity Resolution
**Team Name**: GENZ MINDS  
**Repository**: https://github.com/lazy0ne369/entity-forge.git  

---

## 1. Executive Summary & Problem Formulation

Entity resolution (ER) aims to link heterogeneous, noisy records across disparate platforms to canonical reference records without shared unique identifiers. In this challenge, records from three independent data sources ($S_1$, $S_2$, $S_3$) must be resolved, with $S_1$ serving as the deduplicated reference source. A reference entity in $S_1$ maps to zero (singleton), one, or multiple records across $S_2$ and $S_3$.

The performance is evaluated using the macro-averaged $F_{0.5}$ metric ($\beta = 0.5$):

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Key problem characteristics addressed in this design:
1. **Precision Prioritization**: Precision is weighted twice as heavily as recall. False positives severely degrade the score, demanding conservative match decision boundaries.
2. **Singleton Credit**: Entities in $S_1$ without matches in $S_2$ or $S_3$ receive a score of $1.0$ if correctly left empty, but drop to $0.0$ if any false merge is introduced.
3. **Candidate Set Size Penalty**: Final evaluation rules reward approaches that generate a smaller, more precise candidate set per $S_1$ entity in `candidate_pairs.tsv`.
4. **Memory & Scalability**: The pipeline must process over 12 million total records across train and test partitions within standard desktop constraints (8GB-12GB RAM) without external network dependencies.

---

## 2. System Architecture & Pipeline Workflow

The solution follows a two-stage decoupled architecture:

```
[Raw TSV Files: S1, S2, S3]
           │
           ▼
┌──────────────────────────────────────────────┐
│ Module 1: Out-of-Core Partitioning (DuckDB)  │
│   - Streaming readers with 3GB memory limit  │
│   - Zero-copy Parquet ({country}/source*.pq) │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Module 2: Token-Filtered Inverted Index      │
│   - Legal/address stopword elimination       │
│   - High-frequency token pruning (>15,000)   │
│   - RapidFuzz candidate scoring (Top-6)      │
│   - Output: candidate_pairs.tsv              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Module 3: 19-D Feature Extraction            │
│   - Levenshtein, JW, Token Set/Sort Ratios   │
│   - Address 3-gram character Jaccard         │
│   - Numeric PIN/ZIP and street token parsing │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Module 4: LightGBM Classification & Tuning   │
│   - Histogram capping (max_bin=127)          │
│   - Hard-negative training on blocking pairs │
│   - Strict threshold (0.85) precision cutoff │
│   - Output: matching_results.tsv             │
└──────────────────────────────────────────────┘
```

---

## 3. Component Methodologies

### 3.1 Module 1: Out-of-Core Ingestion (DuckDB)
- **Challenge**: Loading multi-gigabyte TSV files simultaneously into memory causes severe paging and out-of-memory errors on commodity hardware.
- **Implementation**: The pipeline uses an embedded DuckDB engine with `PRAGMA memory_limit='3GB'` and disk spillover enabled. It reads raw TSV files via zero-copy streaming readers (`read_csv(delim='\t', all_varchar=True)`) and writes compressed Zstandard Parquet files organized by country (`data/processed/{country}/source*.parquet`).
- **Dynamic Country Handling**: Discovers country partitions at runtime using `SELECT DISTINCT country`, ensuring zero hardcoded assumptions about geographic partitions and supporting previously unseen test countries (e.g., France).

### 3.2 Module 2: Candidate Generation (Token-Filtered Inverted Index)
- **Search Space Reduction**: Comparing every $S_1$ record against all $S_2$ and $S_3$ records would require over $1.7 \times 10^{13}$ pairwise comparisons. The candidate generation step reduces this space to an average of $< 6$ candidates per $S_1$ entity while preserving $>95\%$ recall.
- **Normalization & Stopword Filtering**: Names and addresses are normalized by stripping common corporate suffixes (`Ltd`, `Pvt`, `Corp`, `LLC`, `GmbH`, `SA`, `SARL`), standardizing street designations (`rd` $\to$ `road`, `st` $\to$ `street`, `ave` $\to$ `avenue`), and removing punctuation.
- **Inverted Index Construction**: Business name and address tokens (length $\ge 3$) from $S_2$ and $S_3$ target records are mapped into an in-memory posting list index.
- **High-Frequency Pruning**: Tokens appearing in more than 15,000 records (e.g., generic industry descriptors) are pruned from the index to prevent candidate explosion and eliminate non-discriminative comparisons.
- **Candidate Scoring**: For each $S_1$ entity, candidate documents are retrieved from the index and ranked using RapidFuzz string similarity. Candidates meeting the minimum threshold ($\text{min\_sim} \ge 0.20$) are retained up to $k=6$, directly outputting to `candidate_pairs.tsv`.

### 3.3 Module 3: Pairwise Feature Engineering
For each candidate pair $(S_1, \text{Candidate})$, a 19-dimensional tabular feature vector is computed:
1. **String Similarity Metrics**:
   - Normalized Levenshtein similarity
   - Jaro-Winkler similarity
   - Token Set Ratio (`fuzz.token_set_ratio`)
   - Token Sort Ratio (`fuzz.token_sort_ratio`)
   - Partial Ratio (`fuzz.partial_ratio`)
2. **Address & Geographic Features**:
   - Character 3-gram Jaccard similarity on normalized address strings
   - Numeric token set Jaccard similarity (matching street numbers, door codes, building numbers)
   - Numeric token exact match indicator
   - Shared numeric token count and count absolute difference
   - Address Levenshtein and Jaro-Winkler similarities
   - Address Token Set Ratio
3. **Disparity & Structural Features**:
   - Absolute difference in character lengths (names and addresses)
   - First character match indicator
   - Exact string equality indicator
   - Blocking stage retrieval similarity score

### 3.4 Module 4: Classification & Precision-Oriented Decision
- **Model Architecture**: LightGBM Gradient Boosted Decision Trees. LightGBM provides fast inference per candidate pair, handles non-linear feature interactions, and operates efficiently under memory constraints.
- **Resource Constraints**:
  - `free_raw_data=True` frees the raw feature matrix immediately after histogram construction.
  - Histogram binning capped at `max_bin=127` to reduce histogram memory overhead.
  - Tree complexity bounds: `max_depth=6`, `num_leaves=31`.
- **Hard-Negative Mining**: The model is trained on candidates retrieved during the blocking phase. Non-matching candidates serve as informative hard negatives (records sharing significant textual tokens but representing distinct real-world entities).
- **Decision Thresholding**:
  - The classifier outputs match probability $P(\text{match} \mid \text{features})$.
  - To align with the $F_{0.5}$ metric's 2× precision penalty, a strict decision threshold of $\tau = 0.85$ is enforced.
  - Records with no candidate surpassing $\tau$ are predicted as empty sets, preserving maximum credit for singletons.

---

## 4. Verification & Submission Compliance

The submission package satisfies all official competition rules:
1. **Verification**: Validated using `utils/validate_submission.py` with exit code 0 (`PASS`).
2. **Row Completeness**: Exactly one row per test $S_1$ entity, tab-delimited.
3. **Identifier Integrity**: Only valid $S_2$ and $S_3$ entity identifiers are matched; no self-matches.
4. **Candidate Subset**: Every entity in `matching_results.tsv` strictly originates from `candidate_pairs.tsv` ($\text{Matches} \subseteq \text{Candidates}$).
5. **Package Structure**: Built as `GENZ_MINDS_submission.zip` containing `output/`, `code/business_entity_resolution/`, and `Documentation_template.md`.
