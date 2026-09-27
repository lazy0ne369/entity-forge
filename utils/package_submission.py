"""
package_submission.py
Builds and verifies the official Amazon ML Challenge 2026 final submission package.

Required Submission Archive Structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def package_submission(team_name: str, base_dir: str = ".") -> str:
    base_dir = os.path.abspath(base_dir)
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(base_dir, zip_filename)

    matching_tsv = os.path.join(base_dir, "output", "matching_results.tsv")
    candidate_tsv = os.path.join(base_dir, "output", "candidate_pairs.tsv")
    code_pkg_dir = os.path.join(base_dir, "code", "business_entity_resolution")
    src_dir = os.path.join(code_pkg_dir, "src")
    readme_file = os.path.join(code_pkg_dir, "README.md")
    req_file = os.path.join(code_pkg_dir, "requirements.txt")
    doc_file = os.path.join(base_dir, "Documentation_template.md")

    # 1. Verification of required components
    missing = []
    for p, desc in [
        (matching_tsv, "output/matching_results.tsv"),
        (candidate_tsv, "output/candidate_pairs.tsv"),
        (src_dir, "code/business_entity_resolution/src/"),
        (readme_file, "code/business_entity_resolution/README.md"),
        (req_file, "code/business_entity_resolution/requirements.txt"),
        (doc_file, "Documentation_template.md"),
    ]:
        if not os.path.exists(p):
            missing.append(f"Missing {desc} at: {p}")

    if missing:
        print("[ERROR] Cannot package submission. Required components missing:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"\n[Packaging] Assembling final submission package: {zip_filename}")
    print("=" * 70)

    # 2. Build the ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # output/
        zf.write(matching_tsv, arcname=os.path.join("output", "matching_results.tsv"))
        zf.write(candidate_tsv, arcname=os.path.join("output", "candidate_pairs.tsv"))

        # code/business_entity_resolution/README.md & requirements.txt
        zf.write(readme_file, arcname=os.path.join("code", "business_entity_resolution", "README.md"))
        zf.write(req_file, arcname=os.path.join("code", "business_entity_resolution", "requirements.txt"))

        # code/business_entity_resolution/src/
        for root, _, files in os.walk(src_dir):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith((".py", ".txt", ".md", ".sh")) and not f.endswith(".pyc"):
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(fp, code_pkg_dir)
                    arc = os.path.join("code", "business_entity_resolution", rel)
                    zf.write(fp, arcname=arc)

        # Documentation_template.md
        zf.write(doc_file, arcname="Documentation_template.md")

    # 3. Audit archive contents
    print("[Packaging] Verifying archive structure:")
    total_uncompressed = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = sorted(zf.namelist())
        for name in namelist:
            info = zf.getinfo(name)
            total_uncompressed += info.file_size
            print(f"  [OK] {name} ({info.file_size:,} bytes)")

    zip_size = os.path.getsize(zip_path)
    print("=" * 70)
    print(f"[SUCCESS] Submission archive created successfully!")
    print(f"  Archive path:       {zip_path}")
    print(f"  Total files:        {len(namelist)}")
    print(f"  Compressed size:    {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"  Uncompressed size:  {total_uncompressed:,} bytes ({total_uncompressed / (1024*1024):.2f} MB)")
    print("=" * 70)
    return zip_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge Final Submission")
    parser.add_argument("--team-name", type=str, default="entity_forge",
                        help="Your official competition team name")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base workspace directory")
    args = parser.parse_args()
    package_submission(args.team_name, args.base_dir)
