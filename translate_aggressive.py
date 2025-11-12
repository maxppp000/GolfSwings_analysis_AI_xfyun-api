#!/usr/bin/env python3
"""
Aggressively scan the repository for non-English (Han) characters.
Optionally apply common locale fixes such as updating lang attributes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
DEFAULT_EXTENSIONS = {".py", ".html", ".css", ".js", ".json", ".md", ".txt"}


def find_chinese_segments(text: str) -> List[Tuple[int, str]]:
    """Return line numbers and text that contain Han characters."""
    matches: List[Tuple[int, str]] = []
    for idx, line in enumerate(text.splitlines(), 1):
        if CHINESE_PATTERN.search(line):
            matches.append((idx, line.strip()))
    return matches


def fix_common_locale_artifacts(text: str) -> str:
    """
    Apply safe replacements that convert zh-CN specific markup to English defaults.
    """
    replacements = {
        'lang="zh-CN"': 'lang="en"',
        "lang='zh-CN'": "lang='en'",
        'lang="zh"': 'lang="en"',
        "lang='zh'": "lang='en'",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def iter_files(base_dir: Path, extensions: Iterable[str]) -> Iterable[Path]:
    """Yield files with the desired extensions."""
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in extensions:
            yield path


def scan_repo(base_dir: Path, extensions: Iterable[str]) -> List[Tuple[Path, List[Tuple[int, str]]]]:
    """Scan every matching file and collect lines that contain Han characters."""
    offenders: List[Tuple[Path, List[Tuple[int, str]]]] = []
    for filepath in iter_files(base_dir, extensions):
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        matches = find_chinese_segments(content)
        if matches:
            offenders.append((filepath, matches))
    return offenders


def rewrite_locale_artifacts(base_dir: Path, extensions: Iterable[str]) -> int:
    """Rewrite lang attributes to English for each matching file."""
    touched = 0
    for filepath in iter_files(base_dir, extensions):
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        fixed = fix_common_locale_artifacts(content)
        if fixed != content:
            filepath.write_text(fixed, encoding="utf-8")
            touched += 1
    return touched


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan repository for non-English characters.")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent,
                        help="Repository root (defaults to script directory).")
    parser.add_argument("--ext", nargs="*", default=sorted(DEFAULT_EXTENSIONS),
                        help="File extensions to include.")
    parser.add_argument("--fix-lang", action="store_true",
                        help="Rewrite lang attributes from zh-CN to en before scanning.")
    args = parser.parse_args()

    base_dir = args.base
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.ext}

    if args.fix_lang:
        updated = rewrite_locale_artifacts(base_dir, extensions)
        print(f"Updated lang attributes in {updated} file(s).")

    offenders = scan_repo(base_dir, extensions)
    if not offenders:
        print("✅ No Han characters detected.")
        return

    print("⚠️  Found potential non-English text:")
    for filepath, matches in offenders:
        print(f"\n{filepath}:")
        for line_no, text in matches[:5]:
            preview = (text[:80] + "…") if len(text) > 80 else text
            print(f"  Line {line_no}: {preview}")
        if len(matches) > 5:
            print(f"  ... {len(matches) - 5} more line(s)")

    raise SystemExit(1)


if __name__ == "__main__":
    main()
