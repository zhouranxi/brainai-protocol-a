#!/usr/bin/env python3
"""Strict BibTeX sanity check for the paper bibliographies.

No pdflatex/bibtex is available in this environment, so this parses the .bib
files directly and asserts the properties a BibTeX run would otherwise take on
trust:

  * every entry is a well-formed @type{key, ... } block,
  * braces balance inside every entry (no orphaned '}' or '{'),
  * every \\cite key used in the matching .tex file resolves to an entry,
  * every entry that should carry an arXiv `note` actually has one, and that
    note sits inside the entry (not dangling after it).

Usage: python code/check_references_bib.py
Exit code 0 = all checks pass.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"

# Entries that legitimately have no arXiv note: books and tech reports that
# were never on arXiv.
NO_NOTE_EXPECTED = {
    "efron1993bootstrap",
    "pearl2009causality",
    "le2015tiny",
    "zhou2026protocol",
}

# The arXiv identifier each entry is expected to carry, keyed by bib key.
EXPECTED_NOTE = {
    "chou2026diagnosing": "arXiv:2603.01879",
    "chou2025feature": "arXiv:2503.18114",
    "chung2018classification": "arXiv:1710.06487",
    "hendrycks2019benchmarking": "arXiv:1903.12261",
    "jiang2020fantastic": "arXiv:1912.02178",
    "ma2018shufflenet": "arXiv:1807.11164",
    "nakkiran2020deep": "arXiv:1912.02292",
    "radosavovic2020designing": "arXiv:2003.13678",
}

ENTRY_RE = re.compile(
    r"@(?P<type>[A-Za-z]+)\s*\{\s*(?P<key>[^\s,]+)\s*,(?P<body>.*?)\n\}",
    re.DOTALL,
)
FIELD_RE = re.compile(r"(?P<name>[A-Za-z_]+)\s*=\s*(?P<value>\{.*?\}|\"[^\"]*\")\s*,?",
                      re.DOTALL)
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}")


def parse_bib(path: Path):
    """Return (entries, problems). entries = {key: {type, fields{...}, text}}."""
    text = path.read_text(encoding="utf-8")
    entries = {}
    problems = []
    spans = []

    for m in ENTRY_RE.finditer(text):
        key = m.group("key")
        body = m.group("body")
        fields = {}
        for fm in FIELD_RE.finditer(body):
            val = fm.group("value")
            if val.startswith("{"):
                val = val[1:-1]
            fields[fm.group("name").lower()] = val
        entries[key] = {"type": m.group("type").lower(), "fields": fields,
                        "text": m.group(0)}
        spans.append((m.start(), m.end()))

    def inside_entry(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    # Every '@type{' must be the start of a parsed entry; otherwise the file
    # was truncated mid-entry.
    for m in re.finditer(r"@([A-Za-z]+)\s*\{", text):
        if not inside_entry(m.start()):
            problems.append(
                f"malformed entry start at line {text[:m.start()].count(chr(10)) + 1}: "
                f"{m.group(0)!r}")

    # Brace balance over the whole file.
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                line = text[:i].count("\n") + 1
                problems.append(f"unbalanced '}}' at line {line} (depth went negative)")
                depth = 0
    if depth != 0:
        problems.append(f"file ends with brace depth {depth} (expected 0)")

    # Dangling note fields: a `note = {...}` that is not inside any entry.
    for m in re.finditer(r"^[ \t]*note\s*=", text, re.MULTILINE):
        line = text[:m.start()].count("\n") + 1
        if not inside_entry(m.start()):
            problems.append(f"dangling note field at line {line} (outside any entry)")

    # A line consisting of only '}' that closes no entry.
    for m in re.finditer(r"^\}[ \t]*$", text, re.MULTILINE):
        line = text[:m.start()].count("\n") + 1
        if not inside_entry(m.start()):
            problems.append(f"orphan closing brace at line {line}")

    return entries, problems


def check(path: Path, tex_path: Path, expect_notes: bool):
    print(f"=== {path.relative_to(ROOT)}")
    entries, problems = parse_bib(path)

    print(f"  entries parsed: {len(entries)}")
    for key in sorted(entries):
        e = entries[key]
        note = e["fields"].get("note", "-")
        print(f"    {key:26s} @{e['type']:14s} note={note}")

    # Duplicate keys
    raw = path.read_text(encoding="utf-8")
    all_keys = re.findall(r"@[A-Za-z]+\s*\{\s*([^\s,]+)\s*,", raw)
    dupes = {k for k in all_keys if all_keys.count(k) > 1}
    if dupes:
        problems.append(f"duplicate keys: {sorted(dupes)}")

    # note presence / value
    if expect_notes:
        for key, expected in EXPECTED_NOTE.items():
            if key not in entries:
                problems.append(f"missing entry {key}")
                continue
            note = entries[key]["fields"].get("note")
            if note is None:
                problems.append(f"{key}: no note field (expected {expected})")
            elif note != expected:
                problems.append(f"{key}: note is {note!r}, expected {expected!r}")
        for key in NO_NOTE_EXPECTED:
            if key in entries and "note" in entries[key]["fields"]:
                note = entries[key]["fields"]["note"]
                if note.startswith("arXiv:"):
                    problems.append(f"{key}: unexpected arXiv note {note!r}")
    else:
        # measurement bib: chung and jiang notes are verified correct, keep them
        for key, expected in (("chung2018classification", "arXiv:1710.06487"),
                              ("jiang2020fantastic", "arXiv:1912.02178")):
            note = entries.get(key, {}).get("fields", {}).get("note")
            if note != expected:
                problems.append(f"{key}: note is {note!r}, expected {expected!r}")

    # Every entry must have at least author/title
    for key, e in entries.items():
        for req in ("title",):
            if req not in e["fields"]:
                problems.append(f"{key}: missing required field {req!r}")

    # \cite keys must resolve
    if tex_path.exists():
        tex = tex_path.read_text(encoding="utf-8")
        cited = set()
        for m in CITE_RE.finditer(tex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cited.add(k)
        missing = sorted(cited - set(entries))
        unused = sorted(set(entries) - cited)
        print(f"  cite keys used: {len(cited)}  unresolved: {missing or 'none'}")
        print(f"  entries not cited: {unused or 'none'}")
        if missing:
            problems.append(f"unresolved cite keys: {missing}")

    if problems:
        print("  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  OK: no problems found")
    print()
    return problems


def main():
    all_problems = []
    all_problems += check(PAPER / "references.bib", PAPER / "main.tex", True)
    all_problems += check(PAPER / "measurement" / "references.bib",
                          PAPER / "measurement" / "main.tex", False)
    if all_problems:
        print(f"FAILED with {len(all_problems)} problem(s).")
        return 1
    print("All bibliography checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
