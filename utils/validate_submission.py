"""
validate_submission.py
Official submission format validator for Amazon ML Challenge 2026.
Uses Python standard library only (no external dependencies).

Checks matching_results.tsv and candidate_pairs.tsv against all competition rules:
1. File format (tab-separated, UTF-8, exact headers).
2. Exactly one row per test Source 1 entity (no missing, no duplicate S1 IDs).
3. Candidate/Match ID syntax (must begin with S2- or S3- only; no self-matches to S1-).
4. No duplicate candidate or match IDs within any row.
5. All referenced candidate and match IDs exist in test source2 or source3.
6. Matches subset of candidates rule (every matched ID must be in candidates).
7. Singletons (empty list) properly formatted.

Exit code 0: PASS
Exit code 1: FAIL with numbered issues list.
"""

import sys
import os
import argparse


def read_test_ids(test_dir: str):
    """Loads all valid entity IDs from test_source1.tsv, test_source2.tsv, and test_source3.tsv."""
    s1_ids = []
    s2_ids = set()
    s3_ids = set()

    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")

    for path in (s1_file, s2_file, s3_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required test file: {path}")

    def _load_ids(filepath: str) -> list:
        ids = []
        with open(filepath, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts and parts[id_idx]:
                    ids.append(parts[id_idx].strip())
        return ids

    s1_ids = _load_ids(s1_file)
    s2_ids = set(_load_ids(s2_file))
    s3_ids = set(_load_ids(s3_file))
    return s1_ids, s2_ids, s3_ids


def _report_and_return(errors: list) -> bool:
    """Prints the error list and returns False."""
    print("FAIL (exit 1)")
    for idx, err in enumerate(errors, 1):
        print(f"  {idx}. {err}")
    return False


def validate(matching_file: str, candidate_file: str, test_dir: str) -> bool:
    errors = []
    # FIX #3: Removed dead `warnings = []` — nothing was ever appended or printed

    if not os.path.exists(matching_file):
        return _report_and_return([f"Matching results file not found: {matching_file}"])

    if not os.path.exists(candidate_file):
        return _report_and_return([f"Candidate pairs file not found: {candidate_file}"])

    try:
        s1_ids_list, s2_ids_set, s3_ids_set = read_test_ids(test_dir)
    except Exception as e:
        return _report_and_return([f"Failed to load test source files from {test_dir}: {e}"])

    expected_s1_set = set(s1_ids_list)
    valid_cand_pool = s2_ids_set | s3_ids_set

    # --- 1. Parse candidate_pairs.tsv ---
    cand_dict: dict[str, set] = {}
    seen_cand_s1: set[str] = set()
    with open(candidate_file, "r", encoding="utf-8") as f:
        cand_header = f.readline().rstrip("\r\n").split("\t")
        if cand_header != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"candidate_pairs.tsv header must be 'source1_entity_id\\tcandidate_entity_ids', got: {cand_header}")

        for line_num, line in enumerate(f, 2):
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0].strip()
            cand_str = parts[1].strip() if len(parts) > 1 else ""

            if s1_id in seen_cand_s1:
                errors.append(f"Line {line_num} in candidate_pairs.tsv: duplicate source1_entity_id: {s1_id}")
            seen_cand_s1.add(s1_id)

            cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]

            if len(cand_ids) != len(set(cand_ids)):
                errors.append(f"Line {line_num} in candidate_pairs.tsv: duplicate candidate IDs for {s1_id}")

            for c_id in cand_ids:
                if not (c_id.startswith("S2-") or c_id.startswith("S3-")):
                    errors.append(f"Line {line_num} in candidate_pairs.tsv: invalid prefix '{c_id}'. Only S2- and S3- allowed.")
                if c_id not in valid_cand_pool:
                    errors.append(f"Line {line_num} in candidate_pairs.tsv: '{c_id}' not found in test source2/source3.")

            cand_dict[s1_id] = set(cand_ids)

    missing_cand_s1 = expected_s1_set - seen_cand_s1
    if missing_cand_s1:
        errors.append(f"candidate_pairs.tsv missing {len(missing_cand_s1)} S1 entities (e.g. {list(missing_cand_s1)[:3]})")

    # --- 2. Parse matching_results.tsv ---
    match_dict: dict[str, set] = {}
    seen_match_s1: set[str] = set()
    with open(matching_file, "r", encoding="utf-8") as f:
        match_header = f.readline().rstrip("\r\n").split("\t")
        if match_header != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"matching_results.tsv header must be 'source1_entity_id\\tmatched_entity_ids', got: {match_header}")

        for line_num, line in enumerate(f, 2):
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0].strip()
            match_str = parts[1].strip() if len(parts) > 1 else ""

            if s1_id in seen_match_s1:
                errors.append(f"Line {line_num} in matching_results.tsv: duplicate source1_entity_id: {s1_id}")
            seen_match_s1.add(s1_id)

            match_ids = [m.strip() for m in match_str.split(",") if m.strip()]

            if len(match_ids) != len(set(match_ids)):
                errors.append(f"Line {line_num} in matching_results.tsv: duplicate matched IDs for {s1_id}")

            for m_id in match_ids:
                if not (m_id.startswith("S2-") or m_id.startswith("S3-")):
                    errors.append(f"Line {line_num} in matching_results.tsv: invalid prefix '{m_id}'. Only S2- and S3- allowed.")
                if m_id not in valid_cand_pool:
                    errors.append(f"Line {line_num} in matching_results.tsv: '{m_id}' not found in test source2/source3.")

            match_dict[s1_id] = set(match_ids)

    missing_match_s1 = expected_s1_set - seen_match_s1
    if missing_match_s1:
        errors.append(f"matching_results.tsv missing {len(missing_match_s1)} S1 entities (e.g. {list(missing_match_s1)[:3]})")

    # --- 3. Matches ⊆ Candidates ---
    for s1_id, match_set in match_dict.items():
        diff = match_set - cand_dict.get(s1_id, set())
        if diff:
            errors.append(f"Entity {s1_id}: matched IDs not in candidate_pairs.tsv: {diff}")

    # --- Report ---
    if errors:
        return _report_and_return(errors)

    print("PASS (exit 0)")
    print(f"Successfully validated:")
    print(f"  Total Source 1 entities verified: {len(seen_match_s1)}")
    print(f"  Matching results: {matching_file}")
    print(f"  Candidate pairs:  {candidate_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate Amazon ML Challenge 2026 Submissions")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Directory containing test_source1/2/3.tsv")
    args = parser.parse_args()

    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
