"""
main.py  —  Amazon ML Challenge 2026: Business Entity Resolution
================================================================
Single entry point for the full pipeline.

Run from Amazon_ML_Challenge_submission/ (the project root):

    # Dry-run on built-in synthetic sample data (no dataset required):
    python code/business_entity_resolution/src/main.py --mode sample

    # Full run on the competition dataset placed under dataset/:
    python code/business_entity_resolution/src/main.py \\
        --mode full \\
        --train-dir dataset/train \\
        --test-dir  dataset/test  \\
        --threshold 0.85

Output files are ALWAYS written to the root-level output/ directory:
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
import gc
import re
import time
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Anchor every path to the repository root  (Amazon_ML_Challenge_submission/)
# This is computed at import time from __file__, so it works correctly
# regardless of the CWD the user runs Python from.
# ---------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)          # .../src/main.py
_SRC_DIR   = os.path.dirname(_THIS_FILE)        # .../src/
_PKG_DIR   = os.path.dirname(_SRC_DIR)          # .../business_entity_resolution/
_CODE_DIR  = os.path.dirname(_PKG_DIR)          # .../code/
_ROOT_DIR  = os.path.dirname(_CODE_DIR)         # Amazon_ML_Challenge_submission/  ← root

# Ensure the code/ package directory is importable
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)


# ---------------------------------------------------------------------------
# Lazy import wrappers so we give a clear message if deps are missing
# ---------------------------------------------------------------------------
def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError:
        sys.exit(
            f"\n[ERROR] Required package '{pkg}' is not installed.\n"
            f"  Run:  pip install -r requirements.txt\n"
        )


# ===========================================================================
# INLINE IMPLEMENTATIONS
# All pipeline logic is reproduced here so that this single file is fully
# self-contained and works even when relative imports across the package
# boundary are tricky (e.g. when __main__ is inside src/).
# ===========================================================================

# ── Regex constants ──────────────────────────────────────────────────────────
_LEGAL_RE    = re.compile(
    r"\b(ltd|limited|pvt|private|corp|corporation|inc|incorporated|"
    r"llc|llp|gmbh|co|company|enterprises|enterprise|industries|"
    r"group|holdings|services|solutions|technologies|tech)\b",
    flags=re.IGNORECASE,
)
_ADDR_ABBREV = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bln\b": "lane", r"\bdr\b": "drive",
    r"\bhwy\b": "highway", r"\bapt\b": "apartment", r"\bste\b": "suite",
    r"\bfl\b": "floor", r"\bdept\b": "department", r"\bpk\b": "park",
    r"\bct\b": "court", r"\bpl\b": "place",
}
_ADDR_REGEXES   = [(re.compile(p, re.IGNORECASE), r) for p, r in _ADDR_ABBREV.items()]
_PUNCT_RE       = re.compile(r"[^\w\s]")
_NUMERIC_RE     = re.compile(r"\b\d+\b")
_SAFE_COUNTRY_RE = re.compile(r"[^A-Za-z0-9_\- ]")
_ALLOWED_SOURCES = frozenset({"source1", "source2", "source3", "ground_truth"})

_BLOCK_STOPS = frozenset({
    "pvt", "ltd", "limited", "private", "corp", "corporation", "inc",
    "incorporated", "llc", "llp", "gmbh", "co", "company", "enterprises",
    "enterprise", "industries", "group", "holdings", "services", "solutions",
    "technologies", "tech", "road", "rd", "street", "st", "lane", "ln",
    "avenue", "ave", "boulevard", "blvd", "near", "opp", "floor", "fl",
    "suite", "ste", "india", "us", "state", "city", "and", "the", "of",
    "in", "to", "for", "at", "by", "from", "with", "no", "plot", "block",
    "sector", "phase", "sa", "sarl", "sas", "sasu", "eurl", "rue",
    "chemin", "route", "allee", "france", "cedex",
})


# ── Text cleaning ────────────────────────────────────────────────────────────

def _safe_country(c: str) -> str:
    return _SAFE_COUNTRY_RE.sub("", c).strip() or "UNKNOWN"


def _clean_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip() or name.lower() == "none":
        return ""
    s = name.lower().replace("&", " and ")
    s = _LEGAL_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _clean_addr(addr: str) -> str:
    if not isinstance(addr, str) or not addr.strip() or addr.lower() == "none":
        return ""
    s = addr.lower().replace("&", " and ")
    for pat, rep in _ADDR_REGEXES:
        s = pat.sub(rep, s)
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _composite(name: str, addr: str) -> str:
    cn, ca = _clean_name(name), _clean_addr(addr)
    if cn and ca:
        return f"{cn} {ca}"
    return cn or ca or ""


def _block_tokens(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    return [w for w in words if w not in _BLOCK_STOPS]


# ── Feature extraction ───────────────────────────────────────────────────────

def _ngrams(text: str, n: int = 3) -> Set[str]:
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _numeric_tokens(text: str) -> Set[str]:
    return set(_NUMERIC_RE.findall(text)) if text else set()


FEATURE_COLUMNS = [
    "levenshtein_distance", "levenshtein_similarity", "jaro_winkler_similarity",
    "token_set_ratio", "token_sort_ratio", "partial_ratio",
    "address_3gram_jaccard", "numeric_jaccard", "numeric_exact_match",
    "numeric_overlap_count", "numeric_count_diff",
    "address_lev_similarity", "address_jw_similarity", "address_token_set_ratio",
    "name_len_diff", "addr_len_diff", "first_char_match", "exact_name_match",
    "tfidf_cosine_sim",
]


def _extract_features(ea: Dict[str, Any], eb: Dict[str, Any], sim: float = 0.0) -> List[float]:
    fuzz      = _require("rapidfuzz").fuzz
    distance  = _require("rapidfuzz").distance

    na = str(ea.get("business_name", "") or "").lower().strip()
    nb = str(eb.get("business_name", "") or "").lower().strip()
    aa = str(ea.get("business_address", "") or "").lower().strip()
    ab = str(eb.get("business_address", "") or "").lower().strip()

    lev_d  = float(distance.Levenshtein.distance(na, nb))
    lev_s  = float(distance.Levenshtein.normalized_similarity(na, nb))
    jw_s   = float(distance.JaroWinkler.similarity(na, nb))
    tsr    = float(fuzz.token_set_ratio(na, nb)) / 100.0
    tso    = float(fuzz.token_sort_ratio(na, nb)) / 100.0
    pr     = float(fuzz.partial_ratio(na, nb)) / 100.0

    nga, ngb = _ngrams(aa), _ngrams(ab)
    nu  = len(nga | ngb)
    a3j = float(len(nga & ngb)) / nu if nu > 0 else 0.0

    na_nums, nb_nums = _numeric_tokens(aa), _numeric_tokens(ab)
    nc   = len(na_nums & nb_nums)
    nnu  = len(na_nums | nb_nums)
    njac = float(nc) / nnu if nnu > 0 else (1.0 if not na_nums and not nb_nums else 0.0)
    nem  = 1.0 if na_nums and na_nums == nb_nums else 0.0
    nov  = float(nc)
    ndf  = float(abs(len(na_nums) - len(nb_nums)))

    als  = float(distance.Levenshtein.normalized_similarity(aa, ab))
    ajs  = float(distance.JaroWinkler.similarity(aa, ab))
    atsr = float(fuzz.token_set_ratio(aa, ab)) / 100.0

    la, lb = len(na), len(nb)
    nld  = float(abs(la - lb)) / max(1.0, float(max(la, lb)))
    ala, alb = len(aa), len(ab)
    ald  = float(abs(ala - alb)) / max(1.0, float(max(ala, alb)))

    fcm  = 1.0 if na and nb and na[0] == nb[0] else 0.0
    exm  = 1.0 if na and na == nb else 0.0

    return [
        lev_d, lev_s, jw_s, tsr, tso, pr,
        a3j, njac, nem, nov, ndf,
        als, ajs, atsr,
        nld, ald, fcm, exm, float(sim),
    ]


# ── DuckDB ingestion ─────────────────────────────────────────────────────────

def _get_duckdb(temp_dir: str = "scratch/duckdb_temp", memory_limit: str = "3GB"):
    duckdb = _require("duckdb")
    os.makedirs(temp_dir, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA memory_limit='{memory_limit}';")
    con.execute(f"PRAGMA temp_directory='{temp_dir}';")
    con.execute("PRAGMA preserve_insertion_order=false;")
    return con


def _partition_tsv(tsv_file: str, out_base: str, source_name: str, con) -> List[str]:
    if source_name not in _ALLOWED_SOURCES:
        raise ValueError(f"Invalid source_name '{source_name}'")
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"Source file not found: {tsv_file}")

    norm = tsv_file.replace("\\", "/")
    vn   = f"view_{source_name}"
    con.execute(f"""
        CREATE OR REPLACE VIEW {vn} AS
        SELECT
            trim(entity_id)                             as entity_id,
            coalesce(trim(business_name), '')           as business_name,
            coalesce(trim(business_address), '')        as business_address,
            coalesce(trim(country), 'UNKNOWN')          as country
        FROM read_csv('{norm}', delim='\\t', header=True,
                      quote='"', escape='\\', all_varchar=True, ignore_errors=True);
    """)
    raw_countries = [r[0] for r in con.execute(
        f"SELECT DISTINCT country FROM {vn} WHERE country IS NOT NULL"
    ).fetchall()]

    for rc in raw_countries:
        country = _safe_country(rc)
        cdir    = os.path.join(out_base, country)
        os.makedirs(cdir, exist_ok=True)
        dest    = os.path.join(cdir, f"{source_name}.parquet").replace("\\", "/")
        con.execute(f"""
            COPY (SELECT entity_id, business_name, business_address, country
                  FROM {vn} WHERE country = '{country}')
            TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
    return [_safe_country(c) for c in raw_countries]


def _ingest(input_dir: str, proc_dir: str, is_train: bool, tmp_dir: str) -> None:
    import shutil
    shutil.rmtree(proc_dir, ignore_errors=True)
    os.makedirs(proc_dir, exist_ok=True)
    prefix = "train" if is_train else "test"
    print(f"\n[Ingestion] Partitioning {prefix.upper()} data → {proc_dir}")
    con = _get_duckdb(temp_dir=tmp_dir)
    try:
        for src in ("source1", "source2", "source3"):
            tsv = os.path.join(input_dir, f"{prefix}_{src}.tsv")
            countries = _partition_tsv(tsv, proc_dir, src, con)
            print(f"  {src}: countries={countries}")
        if is_train:
            gt  = os.path.join(input_dir, "train_ground_truth.tsv")
            if os.path.exists(gt):
                norm_gt  = gt.replace("\\", "/")
                dest_gt  = os.path.join(proc_dir, "ground_truth.parquet").replace("\\", "/")
                con.execute(f"""
                    COPY (SELECT trim(source1_entity_id) as source1_entity_id,
                                 coalesce(trim(matched_entity_ids), '') as matched_entity_ids
                          FROM read_csv('{norm_gt}', delim='\\t', header=True,
                                        all_varchar=True, ignore_errors=True))
                    TO '{dest_gt}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
    finally:
        con.close()
    print("[Ingestion] Done.\n")


# ── Blocking ─────────────────────────────────────────────────────────────────

def _candidates_for_country(
    country_dir: str,
    top_k: int = 15,
    min_sim: float = 0.10,
    max_s1: Optional[int] = None,
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    pd = _require("pandas")
    fuzz = _require("rapidfuzz").fuzz

    s1p = os.path.join(country_dir, "source1.parquet")
    s2p = os.path.join(country_dir, "source2.parquet")
    s3p = os.path.join(country_dir, "source3.parquet")

    if not os.path.exists(s1p):
        return []

    df1 = pd.read_parquet(s1p)
    if max_s1 and len(df1) > max_s1:
        df1 = df1.iloc[:max_s1]
    if len(df1) == 0:
        return []

    df2 = pd.read_parquet(s2p) if os.path.exists(s2p) else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address"])
    df3 = pd.read_parquet(s3p) if os.path.exists(s3p) else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address"])

    df23 = pd.concat([df2, df3], ignore_index=True)
    del df2, df3
    gc.collect()

    if len(df23) == 0:
        return [(sid, []) for sid in df1["entity_id"]]

    s1_ids   = df1["entity_id"].tolist()
    s1_texts = [_composite(n, a) for n, a in zip(df1["business_name"], df1["business_address"])]
    del df1

    s23_ids   = df23["entity_id"].tolist()
    s23_texts = [_composite(n, a) for n, a in zip(df23["business_name"], df23["business_address"])]
    del df23
    gc.collect()

    # Build inverted index on target corpus
    inv: Dict[str, List[int]] = defaultdict(list)
    for idx, txt in enumerate(s23_texts):
        for tok in set(_block_tokens(txt)):
            inv[tok].append(idx)

    # Prune ultra-frequent tokens (> 15 000 postings)
    pruned = {t: v for t, v in inv.items() if len(v) <= 15000}
    del inv
    gc.collect()

    results = []
    for s1id, q_txt in zip(s1_ids, s1_texts):
        cand_idxs: Set[int] = set()
        for tok in _block_tokens(q_txt):
            if tok in pruned:
                cand_idxs.update(pruned[tok])
                if len(cand_idxs) > 1000:
                    break

        if not cand_idxs:
            results.append((s1id, []))
            continue

        scored = []
        for d_idx in cand_idxs:
            sc = fuzz.token_sort_ratio(q_txt, s23_texts[d_idx]) / 100.0
            if sc >= min_sim:
                scored.append((s23_ids[d_idx], sc))

        scored.sort(key=lambda x: x[1], reverse=True)
        results.append((s1id, scored[:top_k]))

    del pruned, s23_texts, s23_ids, s1_texts, s1_ids
    gc.collect()
    return results


def _run_blocking(
    proc_dir: str,
    output_tsv: str,
    top_k: int = 15,
    min_sim: float = 0.10,
) -> Dict[str, List[Tuple[str, float]]]:
    """Generate candidate pairs and write output/candidate_pairs.tsv."""
    os.makedirs(os.path.dirname(output_tsv) or ".", exist_ok=True)
    cand_map: Dict[str, List[Tuple[str, float]]] = {}

    countries = sorted(
        d for d in os.listdir(proc_dir) if os.path.isdir(os.path.join(proc_dir, d))
    )
    print(f"\n[Blocking] Countries: {countries}")
    print(f"[Blocking] Writing candidates → {output_tsv}")

    total_s1, total_cands = 0, 0
    with open(output_tsv, "w", encoding="utf-8", newline="") as fout:
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        fout.flush()

        for country in countries:
            cdir  = os.path.join(proc_dir, country)
            pairs = _candidates_for_country(cdir, top_k=top_k, min_sim=min_sim)

            c_count = 0
            for s1id, cands in pairs:
                ids = [c[0] for c in cands]
                cand_map[s1id] = cands
                c_count += len(ids)
                fout.write(f"{s1id}\t{','.join(ids)}\n")

            fout.flush()   # ← explicit flush after each country
            total_s1    += len(pairs)
            total_cands += c_count
            print(f"  {country}: {len(pairs):,} S1 entities, {c_count:,} candidates")

    print(f"[Blocking] Total S1={total_s1:,}, Total candidates={total_cands:,}")
    print(f"[Blocking] candidate_pairs.tsv written ({os.path.getsize(output_tsv):,} bytes)\n")
    return cand_map


# ── Model training ───────────────────────────────────────────────────────────

def _build_train_features(
    train_proc_dir: str,
    gt_tsv: str,
    top_k: int = 15,
    min_sim: float = 0.10,
    max_entities: Optional[int] = None,
):
    np     = _require("numpy")
    pd     = _require("pandas")
    duckdb = _require("duckdb")

    print(f"\n[Training] Loading ground truth from: {gt_tsv}")
    con = duckdb.connect()
    norm = gt_tsv.replace("\\", "/")
    rows = con.execute(
        f"SELECT source1_entity_id, coalesce(matched_entity_ids, '') "
        f"FROM read_csv('{norm}', delim='\\t', header=True)"
    ).fetchall()
    con.close()

    gt: Dict[str, set] = {}
    for sid, ms in rows:
        if sid:
            gt[sid] = {x.strip() for x in ms.split(",") if x.strip()}

    X_list, y_list = [], []
    countries = sorted(
        d for d in os.listdir(train_proc_dir) if os.path.isdir(os.path.join(train_proc_dir, d))
    )
    per_c = (max_entities // max(len(countries), 1)) + 1 if max_entities else None

    for country in countries:
        cdir = os.path.join(train_proc_dir, country)
        s1p  = os.path.join(cdir, "source1.parquet")
        s2p  = os.path.join(cdir, "source2.parquet")
        s3p  = os.path.join(cdir, "source3.parquet")
        if not os.path.exists(s1p):
            continue

        print(f"  [Training] Generating candidates for {country} (budget={per_c})…")
        pairs = _candidates_for_country(cdir, top_k=top_k, min_sim=min_sim, max_s1=per_c)

        df1 = pd.read_parquet(s1p).set_index("entity_id")
        if per_c and len(df1) > per_c:
            df1 = df1.iloc[:per_c]
        s1d = df1.to_dict(orient="index")
        del df1

        need = {cid for _, cands in pairs for cid, _ in cands}
        df2  = pd.read_parquet(s2p).set_index("entity_id") if os.path.exists(s2p) else pd.DataFrame()
        df3  = pd.read_parquet(s3p).set_index("entity_id") if os.path.exists(s3p) else pd.DataFrame()
        df23 = pd.concat([df2, df3])
        del df2, df3
        cd   = df23[df23.index.isin(need)].to_dict(orient="index")
        del df23
        gc.collect()

        for s1id, cands in pairs:
            if s1id not in s1d:
                continue
            ea   = s1d[s1id]
            true = gt.get(s1id, set())
            for cid, sc in cands:
                if cid not in cd:
                    continue
                feats = _extract_features(ea, cd[cid], sc)
                X_list.append(feats)
                y_list.append(1 if cid in true else 0)

        del s1d, cd
        gc.collect()

    X = _require("numpy").array(X_list, dtype="float32")
    y = _require("numpy").array(y_list, dtype="int32")
    print(f"[Training] Feature matrix: {X.shape}, positives={int(y.sum()):,}, negatives={len(y)-int(y.sum()):,}")
    return X, y


def _train_lgb(X, y, model_path: Optional[str] = None):
    lgb = _require("lightgbm")
    print("\n[Training] Fitting LightGBM (200 rounds)…")
    ds = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS, free_raw_data=True)
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "max_bin": 127, "num_leaves": 31,
        "max_depth": 6, "learning_rate": 0.05,
        "feature_fraction": 0.85, "bagging_fraction": 0.85,
        "bagging_freq": 1, "verbose": -1, "n_jobs": 4,
    }
    model = lgb.train(params, ds, num_boost_round=200)
    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        model.save_model(model_path)
        print(f"[Training] Model saved → {model_path}")
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def _predict(
    model,
    test_proc_dir: str,
    cand_map: Dict[str, List[Tuple[str, float]]],
    output_tsv: str,
    threshold: float = 0.85,
) -> Dict[str, List[str]]:
    np = _require("numpy")
    pd = _require("pandas")

    os.makedirs(os.path.dirname(output_tsv) or ".", exist_ok=True)
    preds: Dict[str, List[str]] = {}

    print(f"\n[Inference] threshold={threshold}  → {output_tsv}")
    countries = sorted(
        d for d in os.listdir(test_proc_dir) if os.path.isdir(os.path.join(test_proc_dir, d))
    )

    total_s1 = total_matched = total_singletons = 0

    with open(output_tsv, "w", encoding="utf-8", newline="") as fout:
        # Header — tab-separated, no trailing spaces
        fout.write("source1_entity_id\tmatched_entity_ids\n")
        fout.flush()

        for country in countries:
            cdir = os.path.join(test_proc_dir, country)
            s1p  = os.path.join(cdir, "source1.parquet")
            s2p  = os.path.join(cdir, "source2.parquet")
            s3p  = os.path.join(cdir, "source3.parquet")
            if not os.path.exists(s1p):
                continue

            df1  = pd.read_parquet(s1p).set_index("entity_id")
            s1d  = df1.to_dict(orient="index")
            s1ids = list(df1.index)
            del df1

            need = {cid for sid in s1ids for cid, _ in cand_map.get(sid, [])}
            df2  = pd.read_parquet(s2p).set_index("entity_id") if os.path.exists(s2p) else pd.DataFrame()
            df3  = pd.read_parquet(s3p).set_index("entity_id") if os.path.exists(s3p) else pd.DataFrame()
            df23 = pd.concat([df2, df3])
            del df2, df3
            cd   = df23[df23.index.isin(need)].to_dict(orient="index")
            del df23
            gc.collect()

            for s1id in s1ids:
                total_s1 += 1
                cands = cand_map.get(s1id, [])

                if not cands or s1id not in s1d:
                    # Singleton — write empty match list  (tab then nothing)
                    preds[s1id] = []
                    total_singletons += 1
                    fout.write(f"{s1id}\t\n")
                    continue

                ea = s1d[s1id]
                pair_feats, cids = [], []
                for cid, sc in cands:
                    if cid not in cd:
                        continue
                    pair_feats.append(_extract_features(ea, cd[cid], sc))
                    cids.append(cid)

                if not pair_feats:
                    preds[s1id] = []
                    total_singletons += 1
                    fout.write(f"{s1id}\t\n")
                    continue

                probs  = model.predict(np.array(pair_feats, dtype="float32"))
                matched = sorted(
                    [(cid, float(p)) for cid, p in zip(cids, probs) if p >= threshold],
                    key=lambda x: x[1], reverse=True,
                )
                final = [m[0] for m in matched]
                preds[s1id] = final

                if final:
                    total_matched += 1
                    fout.write(f"{s1id}\t{','.join(final)}\n")
                else:
                    total_singletons += 1
                    fout.write(f"{s1id}\t\n")

            fout.flush()   # ← explicit flush after every country
            del s1d, cd
            gc.collect()

    print(f"[Inference] S1 total={total_s1:,}, matched={total_matched:,}, singletons={total_singletons:,}")
    print(f"[Inference] matching_results.tsv written ({os.path.getsize(output_tsv):,} bytes)\n")
    return preds


# ── Sample dataset generation (--mode sample) ────────────────────────────────

def _generate_sample(base_dir: str) -> None:
    """Generates a tiny synthetic dataset for smoke-testing the pipeline."""
    train_dir = os.path.join(base_dir, "train")
    test_dir  = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    random.seed(42)

    us_names  = ["Apex Global Logistics", "Summit Health Systems", "Blue Horizon Technologies",
                 "Pinnacle Financial Services", "Vanguard Retail Partners", "Starlight Media Group",
                 "Beacon Industrial Supply", "Atlas Cloud Solutions", "Ironclad Construction",
                 "Pacific Coast Marketing", "NextGen Robotics", "Omni Health Diagnostics"]
    us_addrs  = ["1042 Market St Suite 400 San Francisco CA 94103",
                 "780 Broadway Ave Floor 12 New York NY 10003",
                 "2400 Peachtree Rd NW Atlanta GA 30309",
                 "500 Michigan Ave Chicago IL 60611",
                 "1200 Westheimer Rd Houston TX 77006",
                 "8500 Wilshire Blvd Beverly Hills CA 90211"]
    in_names  = ["Tata Consultancy Services", "Reliance Retail Enterprises", "Infosys Digital Solutions",
                 "Mahindra Aerospace Ltd", "Adani Logistics Solutions", "Bharti Global Communications",
                 "HDFC Financial Corporation", "Wipro Infotech Systems", "Larsen and Toubro Engineering",
                 "Bajaj Auto Components", "Sun Pharma Laboratories", "Godrej Consumer Products"]
    in_addrs  = ["Plot 42 Electronic City Phase 1 Hosur Road Bengaluru Karnataka 560100",
                 "Bandra Kurla Complex G Block Bandra East Mumbai Maharashtra 400051",
                 "DLF Cyber City Tower B Sector 24 Gurugram Haryana 122002",
                 "Hi-Tech City Madhapur Near Cyber Towers Hyderabad Telangana 500081",
                 "Salt Lake Sector V Block EP GP Bidhannagar Kolkata West Bengal 700091",
                 "Anna Salai Mount Road Near Spencer Plaza Chennai Tamil Nadu 600002"]

    def noisy_name(n): return random.choice([
        n.replace("Logistics", "Logistics Pvt Ltd"), n.replace("Technologies", "Tech Inc"),
        n.replace("Solutions", "Solns Corp"), n.replace("Services", "Svcs LLC"),
        n.lower(), n.upper(), n + " & Co", n.replace("and", "&"),
    ])
    def noisy_addr(a): return random.choice([
        a.replace("Road", "Rd").replace("Avenue", "Ave"),
        a.replace("Floor", "Flr"), a.replace("Suite", "Ste"),
        a.split(",")[0] if "," in a else a, a,
    ])

    def gen_split(pools, s1c, s2c, s3c):
        s1, s2, s3, gt = [], [], [], []
        for country, names, addrs in pools:
            for i, bn in enumerate(names):
                ba  = addrs[i % len(addrs)]
                sid = f"S1-{s1c:05d}"; s1c += 1
                s1.append((sid, bn, ba, country))
                mt  = random.choice([0, 1, 2, 3])
                mids = []
                if mt in (1, 3):
                    did = f"S2-{s2c:05d}"; s2c += 1
                    s2.append((did, noisy_name(bn), noisy_addr(ba), country))
                    mids.append(did)
                if mt in (2, 3):
                    did = f"S3-{s3c:05d}"; s3c += 1
                    s3.append((did, noisy_name(bn), noisy_addr(ba), country))
                    mids.append(did)
                gt.append((sid, ",".join(mids)))
        return s1, s2, s3, gt, s1c, s2c, s3c

    tr_s1, tr_s2, tr_s3, tr_gt, _, _, _ = gen_split(
        [("US", us_names, us_addrs), ("India", in_names, in_addrs)], 1, 1, 1)
    te_s1, te_s2, te_s3, te_gt, _, _, _ = gen_split(
        [("US", us_names, us_addrs), ("India", in_names, in_addrs)],
        1, 1, 1)

    def write_tsv(path, rows, header):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(header + "\n")
            for row in rows:
                f.write("\t".join(str(c) for c in row) + "\n")

    hdr4 = "entity_id\tbusiness_name\tbusiness_address\tcountry"
    write_tsv(os.path.join(train_dir, "train_source1.tsv"),      tr_s1, hdr4)
    write_tsv(os.path.join(train_dir, "train_source2.tsv"),      tr_s2, hdr4)
    write_tsv(os.path.join(train_dir, "train_source3.tsv"),      tr_s3, hdr4)
    write_tsv(os.path.join(train_dir, "train_ground_truth.tsv"), tr_gt, "source1_entity_id\tmatched_entity_ids")
    write_tsv(os.path.join(test_dir,  "test_source1.tsv"),       te_s1, hdr4)
    write_tsv(os.path.join(test_dir,  "test_source2.tsv"),       te_s2, hdr4)
    write_tsv(os.path.join(test_dir,  "test_source3.tsv"),       te_s3, hdr4)
    write_tsv(os.path.join(test_dir,  "hidden_test_ground_truth.tsv"), te_gt, "source1_entity_id\tmatched_entity_ids")

    print(f"[Sample] Generated: train S1={len(tr_s1)}, S2={len(tr_s2)}, S3={len(tr_s3)}")
    print(f"[Sample] Generated: test  S1={len(te_s1)}, S2={len(te_s2)}, S3={len(te_s3)}")


# ── Validator wrapper ────────────────────────────────────────────────────────

def _validate(matching_tsv: str, candidate_tsv: str, test_dir: str) -> bool:
    """Runs the official format validator from utils/validate_submission.py."""
    val_script = os.path.join(_ROOT_DIR, "utils", "validate_submission.py")
    if not os.path.exists(val_script):
        print(f"[Validator] WARNING: {val_script} not found — skipping validation.")
        return True

    # Import dynamically using importlib so no sys.path manipulation needed
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location("validate_submission", val_script)
    mod  = types.ModuleType(spec.name)
    spec.loader.exec_module(mod)
    return mod.validate(
        matching_file=matching_tsv,
        candidate_file=candidate_tsv,
        test_dir=test_dir,
    )


# ===========================================================================
# MAIN PIPELINE ORCHESTRATOR
# ===========================================================================

def run_pipeline(
    train_dir: str,
    test_dir:  str,
    output_dir: str,
    work_dir:   str,
    threshold:  float,
) -> bool:
    t0 = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION")
    print(f"  output_dir : {output_dir}")
    print(f"  work_dir   : {work_dir}")
    print("=" * 70)

    # Paths — everything rooted in output_dir and work_dir (both under ROOT)
    train_proc  = os.path.join(work_dir, "train")
    test_proc   = os.path.join(work_dir, "test")
    cand_tsv    = os.path.join(output_dir, "candidate_pairs.tsv")
    match_tsv   = os.path.join(output_dir, "matching_results.tsv")
    model_path  = os.path.join(output_dir, "lgb_entity_matcher.txt")
    gt_file     = os.path.join(train_dir,  "train_ground_truth.tsv")

    # ── Step 0: guarantee output/ and work/ directories exist ────────────────
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir,   exist_ok=True)
    print(f"\n[Setup] output_dir created/verified: {output_dir}")
    print(f"[Setup] work_dir   created/verified: {work_dir}")

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    t1 = time.time()
    print("\n>>> STEP 1: DuckDB Ingestion & Partitioning")
    tmp = os.path.join(work_dir, "duckdb_temp")
    _ingest(train_dir, train_proc, is_train=True,  tmp_dir=tmp)
    _ingest(test_dir,  test_proc,  is_train=False, tmp_dir=tmp)
    print(f">>> STEP 1 done in {time.time()-t1:.1f}s")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    t2 = time.time()
    print("\n>>> STEP 2: Feature Extraction & LightGBM Training")
    X, y   = _build_train_features(train_proc, gt_file, top_k=15, min_sim=0.10)
    model  = _train_lgb(X, y, model_path=model_path)
    del X, y
    gc.collect()
    print(f">>> STEP 2 done in {time.time()-t2:.1f}s")

    # ── Step 3: Block (test) → output/candidate_pairs.tsv ────────────────────
    t3 = time.time()
    print("\n>>> STEP 3: Candidate Generation (Blocking)")
    cand_map = _run_blocking(test_proc, cand_tsv, top_k=15, min_sim=0.10)
    print(f">>> STEP 3 done in {time.time()-t3:.1f}s")

    # ── Step 4: Predict → output/matching_results.tsv ────────────────────────
    t4 = time.time()
    print(f"\n>>> STEP 4: Inference (threshold={threshold})")
    _predict(model, test_proc, cand_map, match_tsv, threshold=threshold)
    print(f">>> STEP 4 done in {time.time()-t4:.1f}s")

    # ── Step 5: Verify both files exist and are non-empty ─────────────────────
    print("\n>>> STEP 5: File existence & size check")
    for label, path in [("candidate_pairs.tsv", cand_tsv), ("matching_results.tsv", match_tsv)]:
        if not os.path.exists(path):
            print(f"  [FAIL] {label} NOT FOUND at {path}")
            return False
        sz = os.path.getsize(path)
        if sz == 0:
            print(f"  [FAIL] {label} is EMPTY at {path}")
            return False
        print(f"  [OK]   {label}  ({sz:,} bytes)")

    # ── Step 6: Official validator ────────────────────────────────────────────
    print("\n>>> STEP 6: Official Submission Validator")
    ok = _validate(match_tsv, cand_tsv, test_dir)
    if not ok:
        print("[ERROR] Validation FAILED — see issues above.")
        return False

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  ({time.time()-t0:.1f}s total)")
    print(f"  candidate_pairs.tsv  → {cand_tsv}")
    print(f"  matching_results.tsv → {match_tsv}")
    print(f"{'='*70}")
    return True


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 — Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--mode", choices=["sample", "full"], default="sample",
                    help="'sample' = dry-run on synthetic data; 'full' = competition dataset")
    ap.add_argument("--train-dir", default=None)
    ap.add_argument("--test-dir",  default=None)
    ap.add_argument("--output-dir", default=None,
                    help="Defaults to <repo-root>/output/")
    ap.add_argument("--work-dir", default=None,
                    help="Defaults to <repo-root>/data/processed/")
    ap.add_argument("--threshold", type=float, default=0.85)
    args = ap.parse_args()

    # ── Resolve train/test directories ────────────────────────────────────────
    if args.mode == "sample":
        sample_base  = os.path.join(_ROOT_DIR, "dataset", "sample")
        sample_train = os.path.join(sample_base, "train")
        sample_test  = os.path.join(sample_base, "test")
        if not os.path.exists(os.path.join(sample_train, "train_source1.tsv")):
            print("[INFO] Sample data not found — generating…")
            _generate_sample(sample_base)
        train_dir = sample_train
        test_dir  = sample_test
    else:
        if not args.train_dir or not args.test_dir:
            ap.error("--mode full requires --train-dir and --test-dir")
        train_dir = os.path.abspath(args.train_dir)
        test_dir  = os.path.abspath(args.test_dir)

    for label, p in [("train-dir", train_dir), ("test-dir", test_dir)]:
        if not os.path.isdir(p):
            sys.exit(f"[ERROR] {label} not found: {p}")

    # ── Resolve output/work directories — ALWAYS rooted under _ROOT_DIR ───────
    output_dir = (
        os.path.abspath(args.output_dir) if args.output_dir
        else os.path.join(_ROOT_DIR, "output")
    )
    work_dir = (
        os.path.abspath(args.work_dir) if args.work_dir
        else os.path.join(_ROOT_DIR, "data", "processed")
    )

    print(f"[Config] ROOT        = {_ROOT_DIR}")
    print(f"[Config] train_dir   = {train_dir}")
    print(f"[Config] test_dir    = {test_dir}")
    print(f"[Config] output_dir  = {output_dir}")
    print(f"[Config] work_dir    = {work_dir}")
    print(f"[Config] threshold   = {args.threshold}")

    ok = run_pipeline(train_dir, test_dir, output_dir, work_dir, args.threshold)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
