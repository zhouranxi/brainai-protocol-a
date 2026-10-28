#!/usr/bin/env python3
"""Fill the Zenodo identifiers into the manuscript and the data availability file.

Usage
-----
    python fill_doi.py --version-doi 10.5281/zenodo.12345678 \
                       --concept-doi 10.5281/zenodo.1234567 \
                       --record-id 12345678

The script substitutes the XXXXXXX placeholders that were left in place when
the deposition was created, recompiles the paper, and reports what changed.
Run it from the repository root.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_TEX = ROOT / "paper" / "main.tex"
PAPER_PDF = ROOT / "paper" / "main.pdf"
DATA_MD = ROOT / "DATA_AVAILABILITY.md"

TEX_OLD = ("\\href{https://doi.org/10.5281/zenodo.XXXXXXX}{doi:10.5281/zenodo.XXXXXXX} "
           "(concept DOI \\href{https://doi.org/10.5281/zenodo.XXXXXXX}{10.5281/zenodo.XXXXXXX}); "
           "the version DOI of this record is 10.5281/zenodo.XXXXXXX")
TEX_NEW = ("\\href{https://doi.org/@CONCEPT@}{doi:@CONCEPT@} "
           "(concept DOI \\href{https://doi.org/@CONCEPT@}{@CONCEPT@}); "
           "the version DOI of this record is @VERSION@")

MD_OLD = """> **Zenodo record:** `10.5281/zenodo.XXXXXXX` (version DOI)
> **Concept DOI:** `10.5281/zenodo.XXXXXXX`
> **Record page:** `https://zenodo.org/records/XXXXXXX`
> **Direct download:** `https://zenodo.org/records/XXXXXXX/files/`

> **Note.** The `XXXXXXX` placeholders are replaced with the real identifiers once
> the record is published. The concept DOI is stable across versions, so it is the
> one to cite; the version DOI points at this exact release."""
MD_NEW = """> **Zenodo record:** `@VERSION@` (version DOI)
> **Concept DOI:** `@CONCEPT@`
> **Record page:** `https://zenodo.org/records/@RID@`
> **Direct download:** `https://zenodo.org/records/@RID@/files/`

> The concept DOI is stable across versions, so it is the one to cite; the version
> DOI points at this exact release."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version-doi", required=True, help="e.g. 10.5281/zenodo.12345678")
    ap.add_argument("--concept-doi", required=True, help="e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--record-id", required=True, help="e.g. 12345678")
    ap.add_argument("--no-compile", action="store_true", help="skip the LaTeX rebuild")
    args = ap.parse_args()

    version = args.version_doi.strip()
    concept = args.concept_doi.strip()
    rid = args.record_id.strip()

    for name, val in (("version DOI", version), ("concept DOI", concept)):
        if not re.fullmatch(r"10\.5281/zenodo\.\d+", val):
            sys.exit(f"{name} does not look like a Zenodo DOI: {val!r}")
    if not rid.isdigit():
        sys.exit(f"record id must be numeric: {rid!r}")

    # --- the manuscript ---
    tex = PAPER_TEX.read_text(encoding="utf-8")
    if "XXXXXXX" not in tex:
        print("main.tex: no placeholders left; nothing to do")
    else:
        if TEX_OLD not in tex:
            print("main.tex: the expected placeholder block was not found verbatim.",
                  file=sys.stderr)
            print("        Inspect paper/main.tex around the reproducibility statement.",
                  file=sys.stderr)
            return 1
        tex = tex.replace(TEX_OLD, TEX_NEW.replace("@CONCEPT@", concept).replace("@VERSION@", version))
        PAPER_TEX.write_text(tex, encoding="utf-8")
        print(f"main.tex: concept DOI {concept}, version DOI {version}")

    # --- the data availability file ---
    md = DATA_MD.read_text(encoding="utf-8")
    if MD_OLD in md:
        DATA_MD.write_text(md.replace(MD_OLD, MD_NEW.replace("@VERSION@", version)
                                 .replace("@CONCEPT@", concept)
                                 .replace("@RID@", rid)),
                           encoding="utf-8")
        print(f"DATA_AVAILABILITY.md: record {rid}")
    else:
        left = md.count("XXXXXXX")
        if left:
            print(f"DATA_AVAILABILITY.md: expected block not found, {left} placeholders "
                  f"remain; edit by hand", file=sys.stderr)
        else:
            print("DATA_AVAILABILITY.md: no placeholders left; nothing to do")

    if args.no_compile:
        return 0

    # --- rebuild ---
    if not shutil.which("pdftex"):
        print("pdftex not on PATH; skipping the rebuild (TinyTeX: "
              "export PATH=\"/c/c/tinytex/bin/windows:$PATH\")")
        return 0
    paper_dir = ROOT / "paper"
    for cmd in (["pdftex", "-fmt=pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", "main"],
                ["pdftex", "-fmt=pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdftex", "-fmt=pdflatex", "-interaction=nonstopmode", "main.tex"]):
        r = subprocess.run(cmd, cwd=paper_dir, capture_output=True, text=True)
        if cmd[0] == "bibtex" and r.returncode != 0:
            print("bibtex failed:", r.stdout[-500:], file=sys.stderr)
        if cmd[0] == "pdftex":
            log = r.stdout
            errs = log.count("\n!")
            if errs:
                print(f"  {log.count('Output written on main.pdf')} pages, {errs} errors",
                      file=sys.stderr)
                for line in log.splitlines():
                    if line.startswith("!"):
                        print("   ", line, file=sys.stderr)
            else:
                m = re.search(r"Output written on main\.pdf \((\d+) pages", log)
                print(f"  rebuilt: {m.group(1) if m else '?'} pages, 0 errors")
    if PAPER_PDF.exists():
        print(f"  {PAPER_PDF.relative_to(ROOT)}  {PAPER_PDF.stat().st_size} bytes")

    # --- verify the identifiers really landed in the PDF ---
    try:
        from pypdf import PdfReader
        txt = "\n".join((p.extract_text() or "") for p in PdfReader(PAPER_PDF).pages)
        for name, val in (("concept DOI", concept), ("version DOI", version)):
            print(f"  {name} in the rendered PDF: "
                  f"{'yes' if val in txt else 'NOT FOUND'}")
    except Exception as e:
        print(f"  (could not check the rendered PDF: {e})")

    print("\nRemaining XXXXXXX placeholders:")
    for f in (PAPER_TEX, DATA_MD):
        n = f.read_text(encoding="utf-8").count("XXXXXXX")
        print(f"  {f.relative_to(ROOT)}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
