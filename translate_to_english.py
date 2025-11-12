#!/usr/bin/env python3
"""
Report generator that lists every line containing Han characters.
Useful when performing manual translations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from translate_aggressive import DEFAULT_EXTENSIONS, scan_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a report of non-English lines.")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent,
                        help="Repository root.")
    parser.add_argument("--ext", nargs="*", default=sorted(DEFAULT_EXTENSIONS),
                        help="File extensions to include.")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON file for the report.")
    args = parser.parse_args()

    base_dir = args.base
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.ext}

    offenders = scan_repo(base_dir, extensions)
    if not offenders:
        print("✅ Repository already contains English-only content.")
        if args.output:
            args.output.write_text(json.dumps([], indent=2), encoding="utf-8")
        return

    print("⚠️  Detected files that still contain Han characters:")
    report = []
    for filepath, matches in offenders:
        entry = {
            "file": str(filepath),
            "occurrences": [{"line": line_no, "text": text} for line_no, text in matches]
        }
        report.append(entry)
        print(f"- {filepath}: {len(matches)} line(s)")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nDetailed report saved to {args.output}")
    else:
        print("\nRe-run with --output report.json to save the details.")


if __name__ == "__main__":
    main()
