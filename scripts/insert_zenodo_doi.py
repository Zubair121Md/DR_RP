#!/usr/bin/env python3
"""Insert a Zenodo version DOI into manuscript, README and CITATION.cff, then rebuild.

Usage:
  export ZENODO_DOI=10.5281/zenodo.12345678
  python3 scripts/insert_zenodo_doi.py

Optional:
  export ZENODO_CONCEPT_DOI=10.5281/zenodo.12345677
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOI = os.environ.get("ZENODO_DOI", "").strip().removeprefix("https://doi.org/")
CONCEPT = os.environ.get("ZENODO_CONCEPT_DOI", "").strip().removeprefix("https://doi.org/")

if not re.fullmatch(r"10\.5281/zenodo\.\d+", DOI or ""):
    raise SystemExit("Set ZENODO_DOI=10.5281/zenodo.NNNNNNN (version DOI from Zenodo)")


def sub_file(path: Path, patterns: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in patterns:
        if old not in text and not re.search(old, text):
            raise SystemExit(f"Pattern not found in {path}: {old!r}")
        text = re.sub(old, new, text) if old.startswith("^") or ".*" in old else text.replace(old, new)
    path.write_text(text)
    print(f"UPDATED {path.relative_to(ROOT)}")


def main() -> None:
    url = f"https://doi.org/{DOI}"
    # Manuscript template Code availability
    tpl = ROOT / "paper" / "MANUSCRIPT_TEMPLATE.md"
    text = tpl.read_text()
    text2, n = re.subn(
        r"\*\*Code availability\.\*\*.+?(?=\n\*\*|\n## |\Z)",
        "**Code availability.** The training and evaluation code, the configuration file of every run, "
        "the seed list, the metric files, the environment record and the scripts that generate every "
        f"table and figure are archived at {url} (GitHub: https://github.com/Zubair121Md/DR_RP, "
        "release v1.0.0-exploratory). Per-image predictions and logits are available from the author "
        "on reasonable request (zubairishaq12@gmail.com).\n",
        text,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise SystemExit("Could not replace Code availability block in MANUSCRIPT_TEMPLATE.md")
    tpl.write_text(text2)
    print("UPDATED paper/MANUSCRIPT_TEMPLATE.md")

    readme = ROOT / "README.md"
    r = readme.read_text()
    r = re.sub(r"Zenodo DOI: \*\*TODO\*\*[^\n]*", f"Zenodo DOI: **{DOI}** ({url}).", r)
    readme.write_text(r)
    print("UPDATED README.md")

    cff = ROOT / "CITATION.cff"
    c = cff.read_text()
    if "doi:" in c:
        c = re.sub(r"(?m)^doi:.*$", f'doi: "{DOI}"', c)
    else:
        c = c.replace('repository-code:', f'doi: "{DOI}"\nrepository-code:')
    if "update DOI after Zenodo release" in c:
        c = c.replace("update DOI after Zenodo release", f"DOI {DOI}")
    cff.write_text(c)
    print("UPDATED CITATION.cff")

    zc = ROOT / "ZENODO_RELEASE_CHECKLIST.md"
    z = zc.read_text()
    for step in ("4. Connect GitHub to Zenodo", "5. Zenodo release from tag",
                 "6. Obtain Zenodo DOI", "7. DOI badge in README",
                 "8. DOI in manuscript Code availability", "9. DOI in cover letter"):
        z = z.replace(f"| {step} | TODO", f"| {step} | Done ({DOI})")
        z = z.replace(f"| {step} | TODO — do not invent", f"| {step} | Done ({DOI})")
    zc.write_text(z)
    print("UPDATED ZENODO_RELEASE_CHECKLIST.md")

    if CONCEPT:
        print(f"Concept DOI recorded for reference only: {CONCEPT}")

    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_manuscript_md.py")])
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_manuscript_docx.py")])
    subprocess.check_call([sys.executable, str(ROOT / "paper" / "check_manuscript.py")])
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_submission_package.py")])
    print(f"ZENODO_DOI_INSERTED {DOI}")


if __name__ == "__main__":
    main()
