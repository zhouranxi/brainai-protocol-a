#!/usr/bin/env python3
"""Build anonymized TMLR submission packages.

TMLR requires the submission PDF and the supplementary material to be
anonymized, and forbids linking to a de-anonymized version (editorial
policies: "submissions must be anonymized", supplementary material "must be
anonymized", and double blind "must be maintained by not linking to another
version that includes the authors' names").

The public repository and the Zenodo record carry the authors' identity, so
the reproducibility statements in the submission must not name them. This
script produces, from the current sources:

  submission/<paper>/main.tex   the anonymized LaTeX source
  submission/<paper>/main.pdf   compiled from that source
  submission/supplementary.zip  code + data + results + figure sources,
                                anonymized, self-contained, < 100 MB

and then scans every produced artifact for identity leaks. The scan fails
the build on any hit, so a leak cannot pass silently.

What the anonymization does
---------------------------
1. Rewrites identity-bearing fragments (GitHub URL, Zenodo DOIs, the
   named self-citation of the companion paper, the named copyright line)
   into neutral forms in every text file it ships -- the submission .tex,
   and every text file inside the supplementary zip.  The public sources
   are never edited; only the copies under submission/ are.
2. Excludes files that exist only to publish or document the
   de-anonymized artifacts (zenodo-upload/, the publishing scripts, the
   verification-email record, the pre-submission audit reports).
3. Scans the produced PDF's embedded strings, the anonymized .tex, and
   every text member of the zip for identity markers, and fails on any
   hit.

The camera-ready flip is a one-line change (run with --de-anonymize not
implemented on purpose: the public sources are the camera-ready).

Usage:
    python code/make_submission.py            # build both papers
    python code/make_submission.py --paper r1
    python code/make_submission.py --scan-only
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission"
PREPRINT_OUT = ROOT / "preprint"

PDFTEX = Path("C:/Users/Lance/TinyTeX/bin/windows/pdftex.exe")
BIBTEX = Path("C:/Users/Lance/TinyTeX/bin/windows/bibtex.exe")

PAPERS = {
    "r1": ROOT / "paper",
    "measurement": ROOT / "paper" / "measurement",
}

# ---------------------------------------------------------------------------
# Identity markers the scanner looks for.  Deliberately broad: a false
# positive costs a manual look; a false negative costs the review.
# ---------------------------------------------------------------------------
IDENTITY_MARKERS = [
    r"zhouranxi",
    r"Zhou,?\s+Ranxi",
    r"Ranxi\s+Zhou",
    r"foxmail\.com",
    r"10\.5281/zenodo\.2372888[34]",
    r"10\.5281/zenodo\.23901486",
    r"10\.5281/zenodo\.23918659",
    r"zenodo\.org/records/23[79]",
    r"Hunan University of Chinese Medicine",
    r"湖南中医药大学",
    r"周燃犀",
]

# ---------------------------------------------------------------------------
# Regex anonymization rules, applied to every text artifact we ship.
# Each rule: (pattern, replacement, description).
# ---------------------------------------------------------------------------
ANON_RULES = [
    # Reproducibility statements: replace the archive sentence wholesale.
    # Matches across the line-wrapping differences between the two papers.
    (re.compile(
        r"and archived at \\href\{https://doi\.org/10\.5281/zenodo\.[0-9]+\}"
        r"\{doi:10\.5281/zenodo\.[0-9]+\}\s*"
        r"\(concept DOI \\href\{https://doi\.org/10\.5281/zenodo\.[0-9]+\}"
        r"\{10\.5281/zenodo\.[0-9]+\}\);?\s*"
        r"the version DOI of this record is 10\.5281/zenodo\.[0-9]+\."),
     "An anonymized archive containing all of these artifacts is included "
     "as supplementary material; the public repository and its archival "
     "DOI will be added in the camera-ready version.",
     "reproducibility archive sentence"),
    # GitHub links, in either line-wrapping.
    (re.compile(
        r"(\\href\{https://github\.com/zhouranxi/brainai-protocol-a\}"
        r"\{github\.com/zhouranxi/brainai-protocol-a\})"),
     "the anonymized supplementary material",
     "github link"),
    # The named self-citation of the companion paper (measurement paper).
    # In an anonymous submission the companion is a concurrent submission;
    # cite it third-person without the name.
    (re.compile(r"\\citep\{zhou2026protocol\}"),
     "(a concurrent submission, anonymized here)",
     "companion self-citation in text"),
    # The second companion paper (the measurement study) is likewise cited
    # by name in the causality paper; anonymize it the same way.
    (re.compile(r"\\citep\{zhou2026measuring\}"),
     "(a concurrent submission, anonymized here)",
     "companion-2 self-citation in text"),
    (re.compile(r"@article\{zhou2026protocol,.*?\n\}\n", re.S),
     "",  # the entry is no longer cited; drop it from the bib
     "companion bib entry"),
    (re.compile(r"@article\{zhou2026measuring,.*?\n\}\n", re.S),
     "",
     "companion-2 bib entry"),
    # LICENSE copyright line.
    (re.compile(r"Copyright \(c\) 2026 Ranxi Zhou, Hunan University of "
                r"Chinese Medicine"),
     "Copyright (c) 2026, the authors of this submission (license text "
     "identical to the MIT license; the copyright holder's identity is "
     "withheld for double-blind review)",
     "license copyright line"),
    # DATA_AVAILABILITY.md: the whole DOI block points at named records.
    (re.compile(r"> \*\*Zenodo record:\*\*.*?Direct download:.*?\n", re.S),
     "> **Archival record:** an anonymized archive is included as the "
     "submission's supplementary material; the archival DOI will be added "
     "in the camera-ready version.\n",
     "DATA_AVAILABILITY DOI block"),
    # paper_draft_v1.md / outlines may name the author; blanket-neutralize
    # the remaining direct markers in every file.
    (re.compile(r"github\.com/zhouranxi/brainai-protocol-a"),
     "github.com/[anonymized-for-review]",
     "bare github url"),
]

# Files/dirs excluded from the supplementary zip entirely.
ZIP_EXCLUDES = [
    "zenodo-upload", "submission", "checkpoints", ".github", "__pycache__",
    # publishing infrastructure for the public artifacts
    "fill_doi.py", "publish_zenodo.py", "push_via_api.py",
    "repair_remote.py", "finalize_zenodo.sh", "upload_zenodo_chunked.sh",
    "upload_zenodo_curl.sh", "watch_and_finalize.sh", "make_submission.py",
    "check_references_bib.py",
    "FILL_DOI_CHECKLIST.md", "ZENODO_PUBLISHING.md", "zenodo_metadata.json",
    # internal records that name the author
    "paper/protocol_verification_email.md",
    "paper/paper_draft_v1.md", "paper/paper_outline.md",
    "paper/SUBMISSION_TIMELINE.md", "paper/adr_0016.md",
    "results/pre_submission_audit_2026-11-02.md",
    "results/references_check_report.md",
    # the citation metadata file is nothing but author identity
    "CITATION.cff",
    # drafts / working notes in the root
    "repair_remote.py",
]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".tex", ".bib", ".json", ".yml",
                 ".cff", ".txt", ".sty", ".bst", ".toml", ".gitattributes",
                 ".gitignore", ".md5", ""}


def anon_text(s: str) -> tuple[str, list]:
    applied = []
    for pat, rep, desc in ANON_RULES:
        s2, n = pat.subn(rep, s)
        if n:
            applied.append(f"{desc} x{n}")
            s = s2
    return s, applied


def scan_text(name: str, text: str, failures: list) -> None:
    for pat in IDENTITY_MARKERS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            snippet = text[max(0, m.start() - 40):m.end() + 40].replace("\n", "\\n")
            failures.append(f"{name}: marker {pat!r} at ...{snippet}...")


def pdf_embedded_strings(pdf: Path) -> str:
    """Best-effort string extraction from a PDF without external tools."""
    data = pdf.read_bytes()
    out = []
    for m in re.finditer(rb"\(([^()\\]{4,})\)", data):
        try:
            out.append(m.group(1).decode("latin-1"))
        except Exception:
            pass
    return "\n".join(out)


def compile_paper(d: Path, log_name: str) -> list:
    failures = []
    env = dict(os.environ)
    env["PATH"] = str(PDFTEX.parent) + os.pathsep + env.get("PATH", "")
    def run(cmd):
        return subprocess.run(cmd, cwd=d, capture_output=True, text=True,
                              env=env, timeout=600)
    run([str(PDFTEX), "-fmt=pdflatex", "-interaction=nonstopmode", "main.tex"])
    run([str(BIBTEX), "main"])
    for _ in range(2):
        run([str(PDFTEX), "-fmt=pdflatex", "-interaction=nonstopmode",
             "main.tex"])
    if not (d / "main.pdf").exists():
        failures.append(f"{log_name}: compile produced no PDF")
        return failures
    log = d / "main.log"
    if log.exists():
        t = log.read_text(encoding="latin-1", errors="ignore")
        errs = re.findall(r"^! (.*)$", t, re.M)
        for e in errs:
            failures.append(f"{log_name}: LaTeX error: {e}")
        for m in re.finditer(r"(Reference|Citation) `([^']+)'.*undefined", t):
            failures.append(f"{log_name}: undefined {m.group(1)} "
                            f"{m.group(2)}")
    return failures


def is_excluded(rel: str) -> bool:
    for e in ZIP_EXCLUDES:
        # e matches the full relative path, a top-level prefix, any path
        # component (e.g. __pycache__ at any depth), or the basename.
        if rel == e or rel.startswith(e + "/") or rel.rsplit("/", 1)[-1] == e \
                or f"/{e}/" in f"/{rel}":
            return True
    return False


def build_zip(dst: Path, report: list) -> list:
    failures = []
    if dst.exists():
        dst.unlink()
    # First pass: collect the files to ship and anonymize text members.
    members: dict[str, bytes] = {}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_dir() or is_excluded(rel):
            continue
        if rel.startswith(".git") or rel.startswith(".workbuddy"):
            continue
        data = p.read_bytes()
        if Path(rel).suffix.lower() in TEXT_SUFFIXES and p.stat().st_size < 1_000_000:
            try:
                t = data.decode("utf-8")
            except UnicodeDecodeError:
                members[rel] = data
                continue
            t2, applied = anon_text(t)
            if applied:
                report.append(f"zip: {rel}: " + ", ".join(applied))
            data = t2.encode("utf-8")
        members[rel] = data
    # Drop the compiled public PDFs (the submission ships the anonymized ones).
    for rel in [r for r in members if r.endswith("main.pdf")]:
        del members[rel]
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in members.items():
            z.writestr(rel, data)
    # Second pass: scan every text member post-anonymization.
    with zipfile.ZipFile(dst) as z:
        for info in z.infolist():
            if Path(info.filename).suffix.lower() not in TEXT_SUFFIXES:
                continue
            scan_text(f"supplementary.zip::{info.filename}",
                      z.read(info).decode("utf-8", errors="ignore"), failures)
    return failures


# ---------------------------------------------------------------------------
# Preprint variants (identity version).  The sources keep the anonymous
# \usepackage{tmlr} as the default; the preprint build flips it in the
# staging copy at build time, exactly like the anonymization rules:
# never hand-edit the sources.
# ---------------------------------------------------------------------------
PREPRINT_TEX_RULES = [
    # (description, line-pattern, replacement).  Replacements are applied
    # via a lambda so backslashes in them are never parsed as escapes.
    ("tmlr preprint mode",
     re.compile(r"^\\usepackage\{tmlr\}", re.M),
     "\\usepackage[preprint]{tmlr}"),
    ("real author block",
     re.compile(r"^\\author\{\\name Anonymous Authors \\AND \\addr Anonymous Affiliation\}", re.M),
     "\\author{\\name Ranxi Zhou \\AND \\addr Department of Computer Science "
     "and Technology, Hunan University of Chinese Medicine, Changsha, China "
     "\\AND \\email see the ORCID profile}"),
]


def build_preprint(paper_name: str, report: list, failures: list) -> None:
    """Build the identity-version preprint for one paper into preprint/."""
    src = PAPERS[paper_name]
    dst = PREPRINT_OUT / paper_name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.suffix in {".tex", ".sty", ".bst", ".bib"} \
                or item.name in ("figures", "figs"):
            if item.is_dir():
                shutil.copytree(item, dst / item.name)
            else:
                shutil.copy2(item, dst)
    tex = (src / "main.tex").read_text(encoding="utf-8")
    applied = []
    for desc, pat, rep in PREPRINT_TEX_RULES:
        tex, n = pat.subn(lambda m, r=rep: r, tex)
        if n:
            applied.append(f"{desc} x{n}")
    if len(applied) != len(PREPRINT_TEX_RULES):
        failures.append(f"preprint/{paper_name}: expected "
                        f"{len(PREPRINT_TEX_RULES)} rule applications, got "
                        f"{len(applied)} -- check PREPRINT_TEX_RULES against "
                        f"the source")
    (dst / "main.tex").write_text(tex, encoding="utf-8", newline="\n")
    report.append(f"preprint/{paper_name}/main.tex: "
                  + ", ".join(applied) if applied else "no rules applied")
    failures += compile_paper(dst, f"preprint/{paper_name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", choices=["r1", "measurement"])
    ap.add_argument("--preprint", action="store_true",
                    help="build the identity-version preprint(s) instead")
    ap.add_argument("--scan-only", action="store_true")
    args = ap.parse_args()

    targets = [args.paper] if args.paper else list(PAPERS)
    failures: list = []
    report: list = []

    if args.preprint:
        if PREPRINT_OUT.exists():
            shutil.rmtree(PREPRINT_OUT)
        PREPRINT_OUT.mkdir(parents=True)
        for name in targets:
            build_preprint(name, report, failures)
        print("\n".join(report))
        if failures:
            print()
            print("\n".join("FAIL " + f for f in failures))
            return 1
        print("\npreprints built:")
        for name in targets:
            print(f"  preprint/{name}/main.pdf")
        return 0

    if not args.scan_only:
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)

        for name in targets:
            src = PAPERS[name]
            dst = OUT / name
            dst.mkdir()
            for item in src.iterdir():
                if item.suffix in {".tex", ".sty", ".bst", ".bib"} \
                        or item.name == "figures":
                    if item.is_dir():
                        shutil.copytree(item, dst / item.name)
                    else:
                        shutil.copy2(item, dst)
            tex = (src / "main.tex").read_text(encoding="utf-8")
            tex2, applied = anon_text(tex)
            # every paper must have had its links rewritten; otherwise the
            # regexes have drifted from the sources
            if not applied:
                failures.append(f"{name}: no anonymization rule applied -- "
                                f"check ANON_RULES against the sources")
            (dst / "main.tex").write_text(tex2, encoding="utf-8",
                                          newline="\n")
            report.append(f"{name}/main.tex: " + ", ".join(applied))
            failures += compile_paper(dst, name)

        failures += build_zip(OUT / "supplementary.zip", report)

    for name in targets:
        d = OUT / name
        if not d.exists():
            continue
        scan_text(f"{name}/main.tex",
                  (d / "main.tex").read_text(encoding="utf-8"), failures)
        if (d / "main.pdf").exists():
            scan_text(f"{name}/main.pdf (embedded strings)",
                      pdf_embedded_strings(d / "main.pdf"), failures)

    print("\n".join(report))
    if failures:
        print()
        print("\n".join("FAIL " + f for f in failures))
        print(f"\n{len(failures)} failure(s); submission NOT ready")
        return 1
    print("\nsubmission built and clean:")
    for name in targets:
        print(f"  submission/{name}/main.pdf")
    print("  submission/supplementary.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
