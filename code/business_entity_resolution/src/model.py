"""
Module 4: LightGBM Classification & F_0.5 Optimization
Memory-Constrained Gradient Boosting for 8GB-12GB RAM Windows Environments.

Trains a LightGBM binary classifier on candidate pairs, applies a calibrated probability
threshold to maximize macro F_0.5, and formats output for official submission.
"""

import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
from .features import extract_pairwise_features, FEATURE_COLUMNS


def prepare_training_features(
    train_processed_dir: str,
    ground_truth_tsv: str,
    top_k: int = 6,
    min_similarity: float = 0.20,
    max_entities: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts pairwise features and binary labels (1=match, 0=non-match) from training partitions.
    High performance: pre-filters candidates into python dicts for sub-millisecond lookups.
    """
    from .blocking import generate_candidates_for_country

    print(f"\n[Module 4 - Training Data] Loading ground truth from: {ground_truth_tsv}")
    import duckdb
    con = duckdb.connect()
    norm_gt = ground_truth_tsv.replace("\\", "/")
    gt_rows = con.execute(f"SELECT source1_entity_id, coalesce(matched_entity_ids, '') FROM read_csv('{norm_gt}', delim='\t', header=True)").fetchall()
    con.close()

    gt_map: Dict[str, set] = {}
    for s1_id, matched_str in gt_rows:
        if s1_id:
            matched = set(x.strip() for x in matched_str.split(",") if x.strip())
            gt_map[s1_id] = matched

    X_list: List[List[float]] = []
    y_list: List[int] = []

    countries = sorted([d for d in os.listdir(train_processed_dir) if os.path.isdir(os.path.join(train_processed_dir, d))])
    per_country_max = (max_entities // len(countries)) + 1 if max_entities else None

    for country in countries:
        country_dir = os.path.join(train_processed_dir, country)
        s1_path = os.path.join(country_dir, "source1.parquet")
        s2_path = os.path.join(country_dir, "source2.parquet")
        s3_path = os.path.join(country_dir, "source3.parquet")

        if not os.path.exists(s1_path):
            continue

        print(f"  Generating candidates for training partition: {country} (budget={per_country_max})...")
        country_candidates = generate_candidates_for_country(
            country_dir,
            top_k=top_k,
            min_similarity=min_similarity,
            max_s1_entities=per_country_max
        )

        df_s1 = pd.read_parquet(s1_path).set_index("entity_id")
        if per_country_max and len(df_s1) > per_country_max:
            df_s1 = df_s1.iloc[:per_country_max]
        s1_dict = df_s1.to_dict(orient="index")
        del df_s1

        needed_cand_ids = {cand_id for _, cands in country_candidates for cand_id, _ in cands}
        print(f"  Filtering {len(needed_cand_ids):,} target candidate records for {country}...")

        df_s2 = pd.read_parquet(s2_path).set_index("entity_id") if os.path.exists(s2_path) else pd.DataFrame()
        df_s3 = pd.read_parquet(s3_path).set_index("entity_id") if os.path.exists(s3_path) else pd.DataFrame()
        df_s23 = pd.concat([df_s2, df_s3])
        del df_s2, df_s3

        # Single hash-set filter + dict conversion
        df_s23_needed = df_s23[df_s23.index.isin(needed_cand_ids)]
        cand_dict = df_s23_needed.to_dict(orient="index")
        del df_s23, df_s23_needed
        gc.collect()

        print(f"  Extracting pairwise features for {len(country_candidates):,} entities in {country}...")
        for s1_id, cands in country_candidates:
            if s1_id not in s1_dict:
                continue
            entity_a = s1_dict[s1_id]
            true_matches = gt_map.get(s1_id, set())

            for cand_id, score in cands:
                if cand_id not in cand_dict:
                    continue
                entity_b = cand_dict[cand_id]
                feats = extract_pairwise_features(entity_a, entity_b, sim_score=score)

                label = 1 if cand_id in true_matches else 0
                X_list.append([feats[col] for col in FEATURE_COLUMNS])
                y_list.append(label)

        del s1_dict, cand_dict
        gc.collect()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"[Module 4 - Training Data] Built feature matrix: {X.shape}, Positives: {int(np.sum(y)):,}, Negatives: {len(y) - int(np.sum(y)):,}")
    return X, y


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_save_path: Optional[str] = None
) -> lgb.Booster:
    """
    Trains a LightGBM binary classifier using parameters optimized for 8GB-12GB RAM Windows.
    Enforces free_raw_data=True to minimize peak memory consumption.
    """
    print("\n[Module 4 - Model] Initializing LightGBM Dataset with free_raw_data=True...")
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=FEATURE_COLUMNS,
        free_raw_data=True
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "max_bin": 127,
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "n_jobs": 4
    }

    print("[Module 4 - Model] Training LightGBM Booster (200 rounds)...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200
    )

    if model_save_path:
        dirname = os.path.dirname(model_save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        model.save_model(model_save_path)
        print(f"[Module 4 - Model] Saved model to: {model_save_path}")

    return model


def predict_matches(
    model: lgb.Booster,
    test_processed_dir: str,
    candidate_map: Dict[str, List[Tuple[str, float]]],
    output_tsv: str,
    threshold: float = 0.80
) -> Dict[str, List[str]]:
    """
    Runs batch inference on candidate pairs, applies strict threshold,
    and writes matching_results.tsv: source1_entity_id TAB matched_entity_ids.
    """
    out_dir = os.path.dirname(output_tsv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    predictions: Dict[str, List[str]] = {}

    print(f"\n[Module 4 - Inference] Running inference with threshold = {threshold:.2f}...")
    countries = sorted([d for d in os.listdir(test_processed_dir) if os.path.isdir(os.path.join(test_processed_dir, d))])

    total_s1 = 0
    total_matched = 0
    total_singletons = 0

    with open(output_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")

        for country in countries:
            country_dir = os.path.join(test_processed_dir, country)
            s1_path = os.path.join(country_dir, "source1.parquet")
            s2_path = os.path.join(country_dir, "source2.parquet")
            s3_path = os.path.join(country_dir, "source3.parquet")

            if not os.path.exists(s1_path):
                continue

            df_s1 = pd.read_parquet(s1_path).set_index("entity_id")
            s1_dict = df_s1.to_dict(orient="index")
            s1_ids_order = list(df_s1.index)
            del df_s1

            needed_cand_ids = {cand_id for sid in s1_ids_order for cand_id, _ in candidate_map.get(sid, [])}

            df_s2 = pd.read_parquet(s2_path).set_index("entity_id") if os.path.exists(s2_path) else pd.DataFrame()
            df_s3 = pd.read_parquet(s3_path).set_index("entity_id") if os.path.exists(s3_path) else pd.DataFrame()
            df_s23 = pd.concat([df_s2, df_s3])
            del df_s2, df_s3

            df_s23_needed = df_s23[df_s23.index.isin(needed_cand_ids)]
            cand_dict = df_s23_needed.to_dict(orient="index")
            del df_s23, df_s23_needed
            gc.collect()

            for s1_id in s1_ids_order:
                total_s1 += 1
                cands = candidate_map.get(s1_id, [])

                if not cands or s1_id not in s1_dict:
                    predictions[s1_id] = []
                    total_singletons += 1
                    out_f.write(f"{s1_id}\t\n")
                    continue

                entity_a = s1_dict[s1_id]
                pair_features: List[List[float]] = []
                cand_ids: List[str] = []

                for cand_id, score in cands:
                    if cand_id not in cand_dict:
                        continue
                    entity_b = cand_dict[cand_id]
                    feats = extract_pairwise_features(entity_a, entity_b, sim_score=score)
                    pair_features.append([feats[col] for col in FEATURE_COLUMNS])
                    cand_ids.append(cand_id)

                if not pair_features:
                    predictions[s1_id] = []
                    total_singletons += 1
                    out_f.write(f"{s1_id}\t\n")
                    continue

                X_pairs = np.array(pair_features, dtype=np.float32)
                probs = model.predict(X_pairs)

                matched_candidates = [
                    (cand_id, float(prob))
                    for cand_id, prob in zip(cand_ids, probs)
                    if prob >= threshold
                ]
                matched_candidates.sort(key=lambda x: x[1], reverse=True)
                final_ids = [m[0] for m in matched_candidates]
                predictions[s1_id] = final_ids

                if final_ids:
                    total_matched += 1
                    out_f.write(f"{s1_id}\t{','.join(final_ids)}\n")
                else:
                    total_singletons += 1
                    out_f.write(f"{s1_id}\t\n")

            del s1_dict, cand_dict
            gc.collect()

    print(f"[Module 4 - Inference] Results saved to: {output_tsv}")
    print(f"  Total S1 Entities Evaluated: {total_s1:,}")
    print(f"  Entities with Matches:      {total_matched:,}")
    print(f"  Singletons (Empty Matches): {total_singletons:,}\n")

    return predictions
