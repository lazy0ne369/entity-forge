"""
Module 3: Pairwise Feature Extraction
High-Speed Tabular Feature Extraction for Entity Pairs using RapidFuzz & Regex.

Calculates string similarities, n-gram Jaccard metrics, numeric address comparisons,
and integrates candidate generation similarity scores.
"""

import re
from typing import Dict, Any, Set
from rapidfuzz import fuzz, distance

NUMERIC_REGEX = re.compile(r"\b\d+\b")


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extracts character n-grams from text."""
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def get_numeric_tokens(text: str) -> Set[str]:
    """Extracts numeric tokens from text."""
    if not text:
        return set()
    return set(NUMERIC_REGEX.findall(text))


def extract_pairwise_features(
    entity_a: Dict[str, Any],
    entity_b: Dict[str, Any],
    sim_score: float = 0.0
) -> Dict[str, float]:
    """
    Calculates pairwise string similarity features for a candidate pair.

    Args:
        entity_a: Dictionary with keys 'business_name', 'business_address'
        entity_b: Dictionary with keys 'business_name', 'business_address'
        sim_score: Cosine similarity score from blocking stage (default 0.0)

    Returns:
        Flat dictionary of numerical features (19 dimensions).
    """
    name_a = str(entity_a.get("business_name", "") or "").lower().strip()
    name_b = str(entity_b.get("business_name", "") or "").lower().strip()

    addr_a = str(entity_a.get("business_address", "") or "").lower().strip()
    addr_b = str(entity_b.get("business_address", "") or "").lower().strip()

    # --- Name Features ---
    lev_dist = float(distance.Levenshtein.distance(name_a, name_b))
    lev_sim = float(distance.Levenshtein.normalized_similarity(name_a, name_b))
    jw_sim = float(distance.JaroWinkler.similarity(name_a, name_b))
    token_set_ratio = float(fuzz.token_set_ratio(name_a, name_b)) / 100.0
    token_sort_ratio = float(fuzz.token_sort_ratio(name_a, name_b)) / 100.0
    partial_ratio = float(fuzz.partial_ratio(name_a, name_b)) / 100.0

    # --- Address Features ---
    ngrams_a = get_char_ngrams(addr_a, n=3)
    ngrams_b = get_char_ngrams(addr_b, n=3)
    if ngrams_a or ngrams_b:
        ngram_union = len(ngrams_a | ngrams_b)
        addr_3gram_jaccard = float(len(ngrams_a & ngrams_b)) / float(ngram_union) if ngram_union > 0 else 0.0
    else:
        addr_3gram_jaccard = 0.0

    nums_a = get_numeric_tokens(addr_a)
    nums_b = get_numeric_tokens(addr_b)

    common_nums = nums_a & nums_b
    num_common = len(common_nums)
    num_union = len(nums_a | nums_b)

    numeric_jaccard = float(num_common) / float(num_union) if num_union > 0 else (1.0 if not nums_a and not nums_b else 0.0)
    numeric_exact_match = 1.0 if (nums_a and nums_a == nums_b) else 0.0
    numeric_overlap_count = float(num_common)
    numeric_count_diff = float(abs(len(nums_a) - len(nums_b)))

    addr_lev_sim = float(distance.Levenshtein.normalized_similarity(addr_a, addr_b))
    addr_jw_sim = float(distance.JaroWinkler.similarity(addr_a, addr_b))
    addr_token_set_ratio = float(fuzz.token_set_ratio(addr_a, addr_b)) / 100.0

    # --- Structural / Length Features ---
    len_a, len_b = len(name_a), len(name_b)
    name_len_diff = float(abs(len_a - len_b)) / max(1.0, float(max(len_a, len_b)))

    addr_len_a, addr_len_b = len(addr_a), len(addr_b)
    addr_len_diff = float(abs(addr_len_a - addr_len_b)) / max(1.0, float(max(addr_len_a, addr_len_b)))

    first_char_match = 1.0 if (name_a and name_b and name_a[0] == name_b[0]) else 0.0
    exact_name_match = 1.0 if (name_a and name_a == name_b) else 0.0

    return {
        "levenshtein_distance": lev_dist,
        "levenshtein_similarity": lev_sim,
        "jaro_winkler_similarity": jw_sim,
        "token_set_ratio": token_set_ratio,
        "token_sort_ratio": token_sort_ratio,
        "partial_ratio": partial_ratio,
        "address_3gram_jaccard": addr_3gram_jaccard,
        "numeric_jaccard": numeric_jaccard,
        "numeric_exact_match": numeric_exact_match,
        "numeric_overlap_count": numeric_overlap_count,
        "numeric_count_diff": numeric_count_diff,
        "address_lev_similarity": addr_lev_sim,
        "address_jw_similarity": addr_jw_sim,
        "address_token_set_ratio": addr_token_set_ratio,
        "name_len_diff": name_len_diff,
        "addr_len_diff": addr_len_diff,
        "first_char_match": first_char_match,
        "exact_name_match": exact_name_match,
        "tfidf_cosine_sim": float(sim_score)
    }


FEATURE_COLUMNS = [
    "levenshtein_distance",
    "levenshtein_similarity",
    "jaro_winkler_similarity",
    "token_set_ratio",
    "token_sort_ratio",
    "partial_ratio",
    "address_3gram_jaccard",
    "numeric_jaccard",
    "numeric_exact_match",
    "numeric_overlap_count",
    "numeric_count_diff",
    "address_lev_similarity",
    "address_jw_similarity",
    "address_token_set_ratio",
    "name_len_diff",
    "addr_len_diff",
    "first_char_match",
    "exact_name_match",
    "tfidf_cosine_sim"
]
