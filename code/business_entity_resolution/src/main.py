"""
main.py
=======
Entry point for the Amazon ML Challenge 2026 — Business Entity Resolution pipeline.

Run from the Amazon_ML_Challenge_submission/ root directory:

    # Dry-run on built-in sample data:
    python code/business_entity_resolution/src/main.py --mode sample

    # Full run on the official competition dataset:
    python code/business_entity_resolution/src/main.py \\
        --mode full \\
        --train-dir dataset/train \\
        --test-dir  dataset/test  \\
        --threshold 0.85

Output files are always written to the root-level output/ directory:
    output/candidate_pairs.tsv
    output/matching_results.tsv

Validate after running:
    python utils/validate_submission.py \\
        --matching  output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv  \\
        --test-dir  dataset/test
"""

import os
import sys
import time
import argparse

# ---------------------------------------------------------------------------
# Resolve the repository root (Amazon_ML_Challenge_submission/) regardless of
# where Python is invoked from, so all relative paths are anchored correctly.
# ---------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)                         # .../src/main.py
_SRC_DIR   = os.path.dirname(_THIS_FILE)                      # .../src/
_PKG_DIR   = os.path.dirname(_SRC_DIR)                        # .../business_entity_resolution/
_CODE_DIR  = os.path.dirname(_PKG_DIR)                        # .../code/
_ROOT_DIR  = os.path.dirname(_CODE_DIR)                       # Amazon_ML_Challenge_submission/

# Ensure the package is importable without installing it
sys.path.insert(0, _CODE_DIR)

# ---------------------------------------------------------------------------
# Internal module imports (all relative to code/business_entity_resolution/src/)
# ---------------------------------------------------------------------------
from business_entity_resolution.src.ingestion import partition_dataset_pipeline
from business_entity_resolution.src.blocking  import run_blocking_pipeline
from business_entity_resolution.src.model     import (
    prepare_training_features,
    train_lightgbm_model,
    predict_matches,
)

# ---------------------------------------------------------------------------
# Validator lives at utils/validate_submission.py (root-relative)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(_ROOT_DIR, "utils"))
from validate_submission import validate


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    train_dir: str,
    test_dir:  str,
    output_dir: str,
    work_dir:   str,
    threshold:  float,
    team_name:  str,
) -> bool:
    """
    Executes the full entity resolution pipeline end-to-end.

    Args:
        train_dir:   Path to the folder containing train_source1/2/3.tsv
                     and train_ground_truth.tsv.
        test_dir:    Path to the folder containing test_source1/2/3.tsv.
        output_dir:  Destination for candidate_pairs.tsv and
                     matching_results.tsv.  Always rooted at the repo root,
                     NOT inside code/.
        work_dir:    Scratch directory for intermediate Parquet partitions.
        threshold:   LightGBM probability threshold for the match decision.
        team_name:   Used to name the final submission .zip archive.

    Returns:
        True on success, False if the validator finds issues.
    """
    t0 = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION")
    print(f"  Output directory : {output_dir}")
    print(f"  Work directory   : {work_dir}")
    print("=" * 70)

    # Derived paths
    train_proc_dir  = os.path.join(work_dir,    "train")
    test_proc_dir   = os.path.join(work_dir,    "test")
    candidate_tsv   = os.path.join(output_dir,  "candidate_pairs.tsv")
    matching_tsv    = os.path.join(output_dir,  "matching_results.tsv")
    model_path      = os.path.join(output_dir,  "lgb_entity_matcher.txt")
    gt_file         = os.path.join(train_dir,   "train_ground_truth.tsv")

    # Guarantee output and work directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir,   exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1: DuckDB ingestion — stream TSVs → country-partitioned Parquet
    # ------------------------------------------------------------------
    t1 = time.time()
    print("\n>>> STEP 1: Memory-Safe Ingestion & Partitioning (DuckDB)")
    partition_dataset_pipeline(train_dir, train_proc_dir, is_train=True)
    partition_dataset_pipeline(test_dir,  test_proc_dir,  is_train=False)
    print(f">>> STEP 1 completed in {time.time() - t1:.2f}s")

    # ------------------------------------------------------------------
    # STEP 2: Feature extraction on training candidates + LightGBM training
    # ------------------------------------------------------------------
    t2 = time.time()
    print("\n>>> STEP 2: Training Feature Extraction & LightGBM Training")
    X_train, y_train = prepare_training_features(
        train_processed_dir=train_proc_dir,
        ground_truth_tsv=gt_file,
        top_k=15,
        min_similarity=0.10,   # Must match Step 3 min_similarity
    )
    model = train_lightgbm_model(X_train, y_train, model_save_path=model_path)
    del X_train, y_train
    print(f">>> STEP 2 completed in {time.time() - t2:.2f}s")

    # ------------------------------------------------------------------
    # STEP 3: Candidate generation on test data → output/candidate_pairs.tsv
    # ------------------------------------------------------------------
    t3 = time.time()
    print("\n>>> STEP 3: Test Candidate Generation (Inverted-Index Blocking)")
    print(f"    Writing: {candidate_tsv}")
    candidate_map = run_blocking_pipeline(
        processed_dir=test_proc_dir,
        output_candidate_tsv=candidate_tsv,   # explicitly rooted in output/
        top_k=15,
        min_similarity=0.10,  # Aligned with training min_similarity
    )
    print(f">>> STEP 3 completed in {time.time() - t3:.2f}s")

    # ------------------------------------------------------------------
    # STEP 4: Batch inference → output/matching_results.tsv
    # ------------------------------------------------------------------
    t4 = time.time()
    print(f"\n>>> STEP 4: Inference (threshold = {threshold})")
    print(f"    Writing: {matching_tsv}")
    predict_matches(
        model=model,
        test_processed_dir=test_proc_dir,
        candidate_map=candidate_map,
        output_tsv=matching_tsv,              # explicitly rooted in output/
        threshold=threshold,
    )
    print(f">>> STEP 4 completed in {time.time() - t4:.2f}s")

    # ------------------------------------------------------------------
    # STEP 5: Official format validation
    # ------------------------------------------------------------------
    print("\n>>> STEP 5: Official Submission Validator")
    is_valid = validate(
        matching_file=matching_tsv,
        candidate_file=candidate_tsv,
        test_dir=test_dir,
    )

    if not is_valid:
        print("\n[ERROR] Validation failed. Check the issues listed above.")
        return False

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE  ({time.time() - t0:.1f}s total)")
    print(f"  Candidate pairs  → {candidate_tsv}")
    print(f"  Matching results → {matching_tsv}")
    print(f"{'=' * 70}")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 — Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["sample", "full"], default="sample",
        help=(
            "'sample' → dry-run on built-in synthetic dataset (no extra files needed). "
            "'full'   → run on the official competition dataset (requires --train-dir / --test-dir)."
        ),
    )
    parser.add_argument(
        "--train-dir", default=None,
        help="[full mode] Folder containing train_source1/2/3.tsv + train_ground_truth.tsv.",
    )
    parser.add_argument(
        "--test-dir", default=None,
        help="[full mode] Folder containing test_source1/2/3.tsv.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Directory for output files.  Defaults to <repo-root>/output/. "
            "Always resolved relative to the repo root, never inside code/."
        ),
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Scratch directory for intermediate Parquet files. Defaults to <repo-root>/data/processed/.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="LightGBM match probability threshold (default 0.85). Run run_train_val.py to find the optimal value.",
    )
    parser.add_argument(
        "--team-name", default="EntityForge",
        help="Team name used for the submission zip archive name.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve train_dir / test_dir
    # ------------------------------------------------------------------
    if args.mode == "sample":
        # Built-in synthetic dataset lives at dataset/sample/ inside the repo
        # Generate it first if not already present
        sample_train = os.path.join(_ROOT_DIR, "dataset", "sample", "train")
        sample_test  = os.path.join(_ROOT_DIR, "dataset", "sample", "test")

        if not os.path.exists(os.path.join(sample_train, "train_source1.tsv")):
            print("[INFO] Sample dataset not found — generating it now …")
            from business_entity_resolution.src.generate_sample_data import create_sample_datasets
            create_sample_datasets(base_dir=os.path.join(_ROOT_DIR, "dataset", "sample"))

        train_dir = sample_train
        test_dir  = sample_test

    else:  # full
        if not args.train_dir or not args.test_dir:
            parser.error(
                "--mode full requires both --train-dir and --test-dir.\n"
                "Example:\n"
                "  python code/business_entity_resolution/src/main.py \\\n"
                "      --mode full \\\n"
                "      --train-dir dataset/train \\\n"
                "      --test-dir  dataset/test"
            )
        train_dir = os.path.abspath(args.train_dir)
        test_dir  = os.path.abspath(args.test_dir)

    # Validate directories exist
    for label, path in [("train-dir", train_dir), ("test-dir", test_dir)]:
        if not os.path.isdir(path):
            sys.exit(f"[ERROR] {label} not found: {path}")

    # ------------------------------------------------------------------
    # Resolve output_dir and work_dir — always rooted at _ROOT_DIR
    # ------------------------------------------------------------------
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(_ROOT_DIR, "output")
    )
    work_dir = (
        os.path.abspath(args.work_dir)
        if args.work_dir
        else os.path.join(_ROOT_DIR, "data", "processed")
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    success = run_pipeline(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        work_dir=work_dir,
        threshold=args.threshold,
        team_name=args.team_name,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
