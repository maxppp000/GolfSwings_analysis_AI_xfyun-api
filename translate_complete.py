#!/usr/bin/env python3
"""
High-level localization helper.
Ensures markup uses English locales and reports any remaining Han characters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

from translate_aggressive import (
    DEFAULT_EXTENSIONS,
    rewrite_locale_artifacts,
    scan_repo,
)

FONT_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("'PingFang SC'", "'Segoe UI'"),
    ('"PingFang SC"', '"Segoe UI"'),
    ("'Microsoft YaHei'", "'Segoe UI'"),
)


def harmonize_fonts(base_dir: Path, extensions: Iterable[str]) -> int:
    """Replace region-specific fonts with cross-platform fallbacks."""
    updated = 0
    for path in base_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        new_text = text
        for source, target in FONT_REPLACEMENTS:
            new_text = new_text.replace(source, target)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            updated += 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize locale artifacts and report Han characters.")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent,
                        help="Repository root (defaults to script directory).")
    parser.add_argument("--ext", nargs="*", default=sorted(DEFAULT_EXTENSIONS),
                        help="File extensions to include in the scan.")
    args = parser.parse_args()

    base_dir = args.base
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.ext}

    lang_updates = rewrite_locale_artifacts(base_dir, extensions)
    font_updates = harmonize_fonts(base_dir, {".html", ".css"})
    print(f"Updated lang attributes in {lang_updates} file(s).")
    print(f"Replaced locale-specific fonts in {font_updates} file(s).")

    offenders = scan_repo(base_dir, extensions)
    if offenders:
        print("\nRemaining files still contain Han characters. Please translate them manually:")
        for filepath, matches in offenders:
            print(f"- {filepath} ({len(matches)} line(s))")
    else:
        print("\nAll scanned files are English-only.")


if __name__ == "__main__":
    main()
