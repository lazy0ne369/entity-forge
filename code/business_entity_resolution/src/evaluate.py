"""
Evaluation script for Amazon ML Challenge 2026.
Calculates macro-averaged F_0.5 score with strict singleton handling according to the official formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
"""

from typing import Dict, List, Set, Union

def compute_entity_f05(pred_ids: Set[str], true_ids: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    Includes singleton edge case handling as specified in the competition rules.
    """
    # Singleton check
    if not true_ids:
        # True entity is a singleton (has no matches)
        # Correctly predicting empty yields 1.0, predicting any match yields 0.0
        return 1.0 if not pred_ids else 0.0
    
    if not pred_ids:
        # Ground truth has matches, but model predicted none
        return 0.0

    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0

    precision = float(tp) / float(len(pred_ids))
    recall = float(tp) / float(len(true_ids))

    denom = 0.25 * precision + recall
    if denom <= 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05

def evaluate_predictions(
    predictions: Dict[str, Union[List[str], Set[str], str]],
    ground_truth: Dict[str, Union[List[str], Set[str], str]]
) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth across all Source 1 entities.

    Args:
        predictions: Mapping from source1_entity_id to predicted match list / set / comma-string
        ground_truth: Mapping from source1_entity_id to true match list / set / comma-string

    Returns:
        Dictionary containing macro_f05, singleton_accuracy, non_singleton_f05, precision, recall
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []

    total_tp = 0
    total_pred = 0
    total_true = 0

    def parse_ids(val) -> Set[str]:
        if isinstance(val, (list, tuple, set)):
            return set(x.strip() for x in val if str(x).strip())
        elif isinstance(val, str):
            return set(x.strip() for x in val.split(",") if x.strip())
        return set()

    for s1_id, true_val in ground_truth.items():
        true_set = parse_ids(true_val)
        pred_set = parse_ids(predictions.get(s1_id, set()))

        score = compute_entity_f05(pred_set, true_set)
        scores.append(score)

        if not true_set:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)
            tp = len(pred_set & true_set)
            total_tp += tp
            total_pred += len(pred_set)
            total_true += len(true_set)

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    singleton_acc = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 1.0
    non_singleton_f05 = sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0
    micro_precision = total_tp / total_pred if total_pred > 0 else 1.0
    micro_recall = total_tp / total_true if total_true > 0 else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_f05,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "total_evaluated": len(scores),
        "total_singletons": len(singleton_scores),
        "total_non_singletons": len(non_singleton_scores)
    }

if __name__ == "__main__":
    # Test example from official document:
    # Model predicts [S2-00047, S2-00193, S3-00812], GT says [S2-00047, S3-00812]
    # Expected F_0.5 = 0.714
    score = compute_entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    print(f"Official example score: {score:.3f} (Expected: 0.714)")
    assert abs(score - 0.714) < 0.002, "Score calculation mismatch!"
    print("F_0.5 Evaluator verified successfully!")
