#!/usr/bin/env python
"""Code-health gate: flag files that are too big or too deeply indented.

Why: small/quantized LLMs (and humans) degrade on very long files and deeply
nested code. This keeps the repo in a shape that stays easy to edit correctly.

Checks
------
* line count      -- Python files over MAX_LINES
* byte size       -- any file over MAX_BYTES (catches stray footage/npz/binaries)
* indentation     -- Python lines indented deeper than MAX_INDENT levels
                     (a proxy for tangled control flow; see CLAUDE.md "never nest")

Usage
-----
    python scripts/check_code_health.py            # check git-staged files
    python scripts/check_code_health.py --all      # check all tracked files
    python scripts/check_code_health.py a.py b.py   # check specific files

Exits non-zero (and prints offenders) if anything fails, so it works as a git
pre-commit hook. Thresholds can be overridden with env vars CH_MAX_LINES,
CH_MAX_BYTES, CH_MAX_INDENT, CH_INDENT_WIDTH.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import tokenize
from pathlib import Path
from typing import List

MAX_LINES = int(os.environ.get("CH_MAX_LINES", "500"))
MAX_BYTES = int(os.environ.get("CH_MAX_BYTES", str(1_000_000)))  # 1 MB
MAX_INDENT = int(os.environ.get("CH_MAX_INDENT", "6"))           # block-nesting levels

# Directories/globs never worth checking (generated or vendored).
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "work", "dropzone", "htmlcov"}


def _git(*args: str) -> List[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _staged_files() -> List[Path]:
    # Added/Copied/Modified staged paths only -- what this commit will introduce.
    return [Path(p) for p in _git("diff", "--cached", "--name-only", "--diff-filter=ACM")]


def _all_tracked() -> List[Path]:
    return [Path(p) for p in _git("ls-files")]


def _skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def check_file(path: Path) -> List[str]:
    """Return a list of human-readable problems for one file (empty = healthy)."""
    problems: List[str] = []
    if not path.is_file() or _skip(path):
        return problems

    size = path.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"{path}: {size} bytes > {MAX_BYTES} (too large to commit)")

    if path.suffix != ".py":
        return problems  # line/indent checks only make sense for source

    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return problems
    lines = text.splitlines()

    if len(lines) > MAX_LINES:
        problems.append(f"{path}: {len(lines)} lines > {MAX_LINES} (split this file)")

    problems.extend(_indent_violations(text, path))
    return problems


def _indent_violations(text: str, path: Path) -> List[str]:
    """Flag lines whose *block* nesting exceeds MAX_INDENT.

    Uses tokenize's INDENT/DEDENT tokens, so it measures real control-flow depth
    -- not the visual column. That means docstring tables and arguments aligned
    under an open paren (continuation lines) are correctly ignored; only genuine
    nested blocks (for/if/with/try inside each other) count.
    """
    problems: List[str] = []
    level = 0
    reported_at_level = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.INDENT:
                level += 1
                if level > MAX_INDENT and level not in reported_at_level:
                    reported_at_level.add(level)
                    problems.append(
                        f"{path}:{tok.start[0]}: nested {level} blocks deep "
                        f"> {MAX_INDENT} (extract a helper / invert a guard)"
                    )
            elif tok.type == tokenize.DEDENT:
                level = max(0, level - 1)
                reported_at_level.discard(level + 1)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # partial/unparseable file -- skip the indent check, keep other checks
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="specific files to check")
    ap.add_argument("--all", action="store_true", help="check all git-tracked files")
    args = ap.parse_args(argv)

    if args.files:
        targets = [Path(f) for f in args.files]
    elif args.all:
        targets = _all_tracked()
    else:
        targets = _staged_files()

    problems: List[str] = []
    for path in targets:
        problems.extend(check_file(path))

    if problems:
        print("Code-health check FAILED:")
        for p in problems:
            print(f"  {p}")
        print(
            f"\nLimits: {MAX_LINES} lines, {MAX_BYTES} bytes, {MAX_INDENT} indent levels. "
            "Override with CH_MAX_LINES / CH_MAX_BYTES / CH_MAX_INDENT if truly justified."
        )
        return 1

    print(f"Code-health check passed ({len(targets)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
