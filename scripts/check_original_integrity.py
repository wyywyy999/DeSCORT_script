#!/usr/bin/env python3
"""Read-only SHA-256 check of the original project against the pre-organization snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=ROOT.parents[1])
    args = parser.parse_args()
    rows = json.loads((ROOT / "docs/reorganization/original_files.json").read_text())
    changed = []
    for row in rows:
        path = args.original_root / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            changed.append(row["path"])
    print(json.dumps({"checked": len(rows), "changed_or_missing": changed}, ensure_ascii=False, indent=2))
    return bool(changed)


if __name__ == "__main__":
    sys.exit(main())
