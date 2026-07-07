"""Tests for the code-health checker (scripts/check_code_health.py).

Runs under pytest OR directly: `python tests/test_code_health.py`.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# scripts/ is not a package; load the module by path.
_spec = importlib.util.spec_from_file_location("check_code_health", ROOT / "scripts" / "check_code_health.py")
ch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch)


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text)
    return p


def test_clean_file_has_no_problems():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "ok.py", "def f():\n    return 1\n")
        assert ch.check_file(p) == []


def test_flags_too_many_lines():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "big.py", "x = 1\n" * (ch.MAX_LINES + 5))
        problems = ch.check_file(p)
        assert any("lines >" in msg for msg in problems)


def test_flags_deep_nesting_but_not_docstrings_or_continuations():
    deep = (
        "def f():\n"
        + "".join("    " * (i + 1) + f"for x{i} in y:\n" for i in range(ch.MAX_INDENT + 1))
        + "    " * (ch.MAX_INDENT + 2) + "pass\n"
    )
    docstring_table = (
        'def g():\n'
        '    """Doc.\n\n'
        '        col1            deeply aligned table text that is not code\n'
        '                        continues even further to the right\n'
        '    """\n'
        '    return (1 +\n'
        '            2 +\n'
        '            3)\n'  # continuation lines aligned under the paren
    )
    with tempfile.TemporaryDirectory() as tmp:
        pd = _write(Path(tmp), "deep.py", deep)
        assert any("nested" in m for m in ch.check_file(pd))
        pg = _write(Path(tmp), "doc.py", docstring_table)
        assert ch.check_file(pg) == []  # docstring/continuation lines don't count


def test_flags_oversized_nonpython_file():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(Path(tmp), "blob.bin", "")
        p.write_bytes(b"0" * (ch.MAX_BYTES + 1))
        assert any("bytes >" in m for m in ch.check_file(p))


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} code-health tests passed")


if __name__ == "__main__":
    _run_all()
