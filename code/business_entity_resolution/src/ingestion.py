"""
Module 1: Data Ingestion & Memory-Safe Partitioning
Optimized for 8GB RAM Windows Environments using DuckDB out-of-core execution.

Partition large multi-gigabyte TSV files into country-based Parquet folders
without loading entire datasets into RAM.
"""

import os
import re
import duckdb
from typing import List, Optional

# FIX #2: Strict allowlist for source_name to prevent SQL injection / path traversal
_ALLOWED_SOURCE_NAMES = frozenset({"source1", "source2", "source3", "ground_truth"})

# FIX #1: Allowlist pattern for country strings used in SQL/filesystem paths
_SAFE_COUNTRY_RE = re.compile(r"[^A-Za-z0-9_\- ]")


def _safe_country(country: str) -> str:
    """Strips any character that is not alphanumeric, underscore, hyphen, or space."""
    return _SAFE_COUNTRY_RE.sub("", country).strip() or "UNKNOWN"


def get_duckdb_connection(
    temp_dir: str = "scratch/duckdb_temp",
    memory_limit: str = "3GB"
) -> duckdb.DuckDBPyConnection:
    """Initializes DuckDB connection with strict memory limits and disk-backed spillover."""
    os.makedirs(temp_dir, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA memory_limit='{memory_limit}';")
    con.execute(f"PRAGMA temp_directory='{temp_dir}';")
    con.execute("PRAGMA preserve_insertion_order=false;")
    return con


def partition_tsv_by_country(
    tsv_file: str,
    output_base_dir: str,
    source_name: str,
    con: Optional[duckdb.DuckDBPyConnection] = None
) -> List[str]:
    """
    Reads a large TSV file via DuckDB streaming and writes out partitioned Parquet files
    organized as: {output_base_dir}/{country}/{source_name}.parquet

    Args:
        tsv_file: Path to input TSV file
        output_base_dir: Directory where country subfolders will be placed
        source_name: File identifier — must be one of 'source1', 'source2', 'source3'
        con: Optional shared DuckDB connection

    Returns:
        List of distinct countries discovered
    """
    # FIX #2: Validate source_name against allowlist before any SQL or FS use
    if source_name not in _ALLOWED_SOURCE_NAMES:
        raise ValueError(
            f"Invalid source_name '{source_name}'. Must be one of: {sorted(_ALLOWED_SOURCE_NAMES)}"
        )

    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"Source file not found: {tsv_file}")

    close_con = con is None
    if close_con:
        con = get_duckdb_connection()

    # FIX #9: Guarantee connection is released even if an exception is raised
    try:
        # Normalize path for DuckDB SQL (forward slashes required on Windows)
        norm_tsv_path = tsv_file.replace("\\", "/")

        view_name = f"view_{source_name}"
        con.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                trim(entity_id)                                as entity_id,
                coalesce(trim(business_name), '')             as business_name,
                coalesce(trim(business_address), '')          as business_address,
                coalesce(trim(country), 'UNKNOWN')            as country
            FROM read_csv(
                '{norm_tsv_path}',
                delim='\t',
                header=True,
                quote='',
                escape='',
                all_varchar=True,
                ignore_errors=True
            );
        """)

        raw_countries = [
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT country FROM {view_name} WHERE country IS NOT NULL"
            ).fetchall()
        ]

        for raw_country in raw_countries:
            # FIX #1: Sanitize the country value before embedding in SQL and filesystem paths
            country = _safe_country(raw_country)

            country_dir = os.path.join(output_base_dir, country)
            os.makedirs(country_dir, exist_ok=True)
            dest_parquet = os.path.join(country_dir, f"{source_name}.parquet").replace("\\", "/")

            # Use Python string equality — country is now sanitized, safe to embed
            con.execute(f"""
                COPY (
                    SELECT entity_id, business_name, business_address, country
                    FROM {view_name}
                    WHERE country = '{country}'
                ) TO '{dest_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """)

        return [_safe_country(c) for c in raw_countries]

    finally:
        if close_con:
            con.close()


def partition_dataset_pipeline(
    input_dir: str,
    output_processed_dir: str,
    is_train: bool = False
):
    """
    End-to-end ingestion and partitioning for all source files in a dataset folder.
    Handles test_source1/2/3.tsv or train_source1/2/3.tsv + train_ground_truth.tsv.
    """
    con = get_duckdb_connection()
    prefix = "train" if is_train else "test"

    print(f"\n[Module 1 - Ingestion] Partitioning {prefix.upper()} dataset from: {input_dir}")
    print(f"[Module 1 - Ingestion] Destination: {output_processed_dir}")

    try:
        s1_path = os.path.join(input_dir, f"{prefix}_source1.tsv")
        s2_path = os.path.join(input_dir, f"{prefix}_source2.tsv")
        s3_path = os.path.join(input_dir, f"{prefix}_source3.tsv")

        countries_s1 = partition_tsv_by_country(s1_path, output_processed_dir, "source1", con=con)
        print(f"  Processed Source 1. Countries found: {countries_s1}")

        countries_s2 = partition_tsv_by_country(s2_path, output_processed_dir, "source2", con=con)
        print(f"  Processed Source 2. Countries found: {countries_s2}")

        countries_s3 = partition_tsv_by_country(s3_path, output_processed_dir, "source3", con=con)
        print(f"  Processed Source 3. Countries found: {countries_s3}")

        if is_train:
            gt_path = os.path.join(input_dir, "train_ground_truth.tsv")
            if os.path.exists(gt_path):
                norm_gt = gt_path.replace("\\", "/")
                dest_gt = os.path.join(output_processed_dir, "ground_truth.parquet").replace("\\", "/")
                con.execute(f"""
                    COPY (
                        SELECT
                            trim(source1_entity_id) as source1_entity_id,
                            coalesce(trim(matched_entity_ids), '') as matched_entity_ids
                        FROM read_csv('{norm_gt}', delim='\t', header=True, all_varchar=True, ignore_errors=True)
                    ) TO '{dest_gt}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
                print(f"  Processed Ground Truth -> {dest_gt}")
    finally:
        con.close()

    print("[Module 1 - Ingestion] Partitioning complete!\n")


if __name__ == "__main__":
    # FIX #13: Use paths relative to this file instead of hardcoded absolute paths
    _base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sample_dir = os.path.join(_base, "data", "sample_dataset", "test")
    processed_dir = os.path.join(_base, "data", "processed", "sample_test")
    partition_dataset_pipeline(sample_dir, processed_dir, is_train=False)
