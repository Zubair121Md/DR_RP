#!/usr/bin/env python3
"""
Check paper/MANUSCRIPT.docx against paper/tables/*.csv. Reads the built document; never
writes or regenerates it. Fails loudly; does not auto-fix.

Checks
  1. Every decimal number (and every a ± b pair) in body text, captions and table cells
     matches a value in paper/tables/*.csv.
  2. Every numbered figure and table (including supplementary figures) has exactly one
     caption and is cited in the body text outside its own caption.
  3. Figures, supplementary figures and tables are first cited in numeric order.
  4. Every reference in the list is cited in the text before the reference list, and
     every citation has a reference entry (ranges such as [4-6] are expanded).
  5. No duplicated captions, placeholder text, file paths or code formatting in the body.
  6. Every embedded image is present (the document must contain one picture per figure).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "paper" / "MANUSCRIPT.docx"
TABLE_DIR = ROOT / "paper" / "tables"

NUM_RE = re.compile(r"-?\d[\d,]*\.\d+(?:±\d+\.\d+)?")
AUTHOR_PLACEHOLDER = re.compile(r"\[(Author Name|Institution|email|To be completed|Declare|CRediT)", re.I)


def norm(text: str) -> str:
    text = re.sub(r"\bSections? \d+\.\d+(?:\s*(?:and|–|-)\s*\d+\.\d+)?", "Section", text)
    return text.replace("\u2212", "-").replace(" ± ", "±").replace("\u00a0", " ")


def load_source_numbers() -> set[float]:
    vals: set[float] = set()
    for p in sorted(TABLE_DIR.glob("*.csv")):
        n = 0
        with p.open() as f:
            for row in csv.reader(f):
                for cell in row:
                    for tok in re.findall(r"[-+]?\d+(?:\.\d+)?(?:e-?\d+)?", norm(cell).replace(",", "")):
                        try:
                            vals.add(float(tok))
                            n += 1
                        except ValueError:
                            pass
        print("READ", p.relative_to(ROOT), "numeric_tokens", n)
    return vals


def matches_source(tok: str, src: set[float]) -> bool:
    parts = tok.replace(",", "").split("±")
    for part in parts:
        x = float(part)
        decimals = len(part.split(".")[1]) if "." in part else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(abs(x) - abs(v)) <= tol for v in src):
            return False
    return True


def expand_citation(body: str) -> set[int]:
    out: set[int] = set()
    for grp in re.findall(r"\[(\d+(?:\s*[–-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[–-]\s*\d+)?)*)\]", body):
        for part in grp.split(","):
            part = part.strip()
            m = re.match(r"(\d+)\s*[–-]\s*(\d+)", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                out.update(range(a, b + 1))
            else:
                out.add(int(part))
    return out


def main() -> None:
    errors: list[str] = []
    docx_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DOCX
    if not docx_path.is_file():
        print("MISSING", docx_path)
        raise SystemExit(1)
    print("READ", docx_path)
    doc = Document(str(docx_path))
    src = load_source_numbers()

    paras = [p.text.strip() for p in doc.paragraphs]
    try:
        ref_start = next(i for i, t in enumerate(paras) if t == "References")
    except StopIteration:
        raise SystemExit("CHECK_MANUSCRIPT_FAIL no References heading")
    supp_start = next((i for i, t in enumerate(paras) if t.lower().startswith("supplementary material")), len(paras))
    body_paras = paras[:ref_start]
    ref_paras = paras[ref_start + 1:supp_start]
    supp_paras = paras[supp_start:]

    cap_re = re.compile(r"^(Figure S?\d+|Table S?\d+)\.\s")
    captions = [t for t in body_paras + supp_paras if cap_re.match(t)]
    running = [t for t in body_paras if t and not cap_re.match(t)]

    # 1. numbers
    unmatched = []
    heading_idx = {i for i, p in enumerate(doc.paragraphs) if p.style.name.startswith(("Heading", "Title"))}
    for i, t in enumerate(body_paras + supp_paras):
        if not t or AUTHOR_PLACEHOLDER.search(t):
            continue
        if (i if i < ref_start else i + (supp_start - ref_start)) in heading_idx:
            continue
        for m in NUM_RE.finditer(norm(t)):
            if not matches_source(m.group(0), src):
                unmatched.append((f"para {i}", m.group(0), t[:70]))
    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                for m in NUM_RE.finditer(norm(cell.text)):
                    if not matches_source(m.group(0), src):
                        unmatched.append((f"table {ti + 1} r{ri}c{ci}", m.group(0), cell.text[:40]))
    for loc, tok, ctx in unmatched:
        errors.append(f"NO_SOURCE {loc}: {tok}  «{ctx}»")

    # 2 & 3. captions, citations and order
    labels = [cap_re.match(c).group(1) for c in captions]
    for lab in sorted(set(labels)):
        if labels.count(lab) > 1:
            errors.append(f"DUPLICATE caption for {lab}")
    cap_texts = [c.split(".", 1)[1].strip() for c in captions]
    for c in set(cap_texts):
        if cap_texts.count(c) > 1:
            errors.append(f"DUPLICATE caption text: {c[:60]}")

    body_text = "\n".join(running)
    cited_fig = [int(x) for x in re.findall(r"\bFigures?\s+(\d+)(?![\d])", body_text)]
    cited_fig += [int(b) for a, b in re.findall(r"\bFigures?\s+\d+\s*(?:and|–|-)\s*(\d+)", body_text)]
    cited_sup: list[int] = []
    for m in re.finditer(r"Supplementary Figures?\s+S(\d+)(?:\s*[–-]\s*S(\d+))?", body_text):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        cited_sup.extend(range(a, b + 1))
    cited_tab = [int(x) for x in re.findall(r"\bTables?\s+(\d+)", body_text)]
    cited_supt: list[int] = []
    for m in re.finditer(r"Supplementary Tables?\s+S(\d+)(?:\s*(?:[–-]|and)\s*S(\d+))?", body_text):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        cited_supt.extend(range(a, b + 1))

    for lab in labels:
        kind, n = lab.rsplit(" ", 1)
        if n.startswith("S"):
            pool = cited_supt if kind == "Table" else cited_sup
            if int(n[1:]) not in pool:
                errors.append(f"NOT_CITED in text: {lab}")
        elif kind == "Figure" and int(n) not in cited_fig:
            errors.append(f"NOT_CITED in text: {lab}")
        elif kind == "Table" and int(n) not in cited_tab:
            errors.append(f"NOT_CITED in text: {lab}")
    for n in set(cited_fig):
        if f"Figure {n}" not in labels:
            errors.append(f"CITED but no caption: Figure {n}")
    for n in set(cited_sup):
        if f"Figure S{n}" not in labels:
            errors.append(f"CITED but no caption: Figure S{n}")
    for n in set(cited_tab):
        if f"Table {n}" not in labels:
            errors.append(f"CITED but no caption: Table {n}")
    for n in set(cited_supt):
        if f"Table S{n}" not in labels:
            errors.append(f"CITED but no caption: Table S{n}")

    def first_order(seq: list[int]) -> list[int]:
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def order_in_text(pattern: str) -> list[int]:
        out = []
        for m in re.finditer(pattern, body_text):
            a = int(m.group(1))
            b = int(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else a
            out.extend(range(a, b + 1))
        return first_order(out)

    for name, pat in (("Figure", r"(?<!Supplementary )\bFigures?\s+(\d+)(?:\s*(?:and|–|-)\s*(\d+))?"),
                      ("Supplementary figure", r"Supplementary Figures?\s+S(\d+)(?:\s*[–-]\s*S(\d+))?"),
                      ("Supplementary table", r"Supplementary Tables?\s+S(\d+)(?:\s*(?:[–-]|and)\s*S(\d+))?"),
                      ("Table", r"\bTables?\s+(\d+)(?:\s*(?:and|–|-)\s*(\d+))?")):
        order = order_in_text(pat)
        if order != sorted(order):
            errors.append(f"ORDER: {name} first citations not in numeric order: {order}")
        if order and order != list(range(1, max(order) + 1)):
            errors.append(f"ORDER: {name} numbering has gaps: {order}")

    # 4. references
    listed = set()
    for t in ref_paras:
        m = re.match(r"^\[(\d+)\]\s", t)
        if m:
            listed.add(int(m.group(1)))
    table_text = "\n".join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    cited_refs = expand_citation("\n".join(body_paras) + "\n" + table_text)
    for n in sorted(listed - cited_refs):
        errors.append(f"REFERENCE never cited: [{n}]")
    for n in sorted(cited_refs - listed):
        errors.append(f"CITATION with no reference entry: [{n}]")
    if listed and listed != set(range(1, max(listed) + 1)):
        errors.append(f"REFERENCE numbering has gaps: {sorted(listed)}")

    # 5. placeholders, paths, raw formatting
    for i, t in enumerate(body_paras + supp_paras):
        if AUTHOR_PLACEHOLDER.search(t):
            continue
        if re.search(r"\bTODO\b|\bTBD\b|\bXXX\b|placeholder|Insert figure|\{\{|\}\}", t, re.I):
            errors.append(f"PLACEHOLDER para {i}: {t[:80]}")
        if re.search(r"`|\.csv\b|\.json\b|\.yaml\b|\.py\b|paper/|results/|configs/", t):
            errors.append(f"RAW path or code formatting in text, para {i}: {t[:80]}")
        if re.search(r"\*\*|\\\(|\\\[|\$", t):
            errors.append(f"UNRENDERED markup, para {i}: {t[:80]}")

    # 6. embedded images
    n_pics = len(doc.inline_shapes)
    n_fig_caps = sum(1 for lab in labels if lab.startswith("Figure"))
    n_eq = len(re.findall(r"\(\d+\)$", "\n".join(p.text for p in doc.paragraphs), flags=re.M))
    print(f"INFO figures_captioned={n_fig_caps} tables_captioned={sum(1 for l in labels if l.startswith('Table'))} "
          f"inline_images={n_pics} equations={n_eq} references={len(listed)}")
    if n_pics < n_fig_caps + n_eq:
        errors.append(f"EMBEDDED images ({n_pics}) fewer than figure captions + equations ({n_fig_caps + n_eq})")

    if errors:
        print("CHECK_MANUSCRIPT_FAIL")
        for e in errors:
            print(e)
        raise SystemExit(1)
    print("CHECK_MANUSCRIPT_OK")


if __name__ == "__main__":
    sys.exit(main())
