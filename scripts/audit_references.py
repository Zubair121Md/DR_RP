#!/usr/bin/env python3
"""
Audit the reference list of paper/MANUSCRIPT.md against Crossref / arXiv metadata.

For every numbered reference: extract DOI or URL, query Crossref (DOI) or the arXiv API
(arXiv id), compare the returned title with the cited title, and record whether the
reference number is cited in the body text before the reference list.

Writes REFERENCE_AUDIT.csv at the repository root.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "paper" / "MANUSCRIPT.md"
OUT = ROOT / "REFERENCE_AUDIT.csv"
UA = {"User-Agent": "dr-research-reference-audit/1.0 (mailto:zubairishaq12@gmail.com)"}


# Claim each reference is cited for in the manuscript. "Context" means the citing sentence was
# checked against the retrieved title/abstract only; full-text support was not re-read.
CLAIMS = {
    1: "Intro/Related work: DL detects referable DR at clinician-level operating points (context)",
    2: "Related work: retrospective algorithm-vs-clinician screening validation (context)",
    3: "Related work: DL system across the full DR spectrum (context)",
    4: "Intro/Related work: DR DL survey; CLAHE/green-channel common preprocessing (context)",
    5: "Intro/Related work: survey of challenges and gaps in DR DL (context)",
    6: "Intro/Related work: survey of DR screening DL evolution; arXiv preprint (context)",
    7: "Intro: earlier DL methods perform poorly at early (mild) stages (context)",
    8: "Intro/Related work: lightweight CNNs for DR; CLAHE preprocessing (context)",
    9: "Intro: EfficientNet proposed for DR in low-resource settings (context)",
    10: "Intro: EfficientNet-B0 applied to DR severity grading; niche venue (context)",
    11: "Intro/Related work: MobileViT-based hybrid for DR grading (context)",
    12: "Intro/Related work: hybrid conv-transformer with ordinal-sensitive loss for DR (context)",
    13: "Intro/Related work: CNN-ViT hybrid for ocular disease diagnosis (context)",
    14: "Intro/Related work: green-channel CLAHE in DR pipelines (context)",
    15: "Ordinal methods: CORAL definition; rank consistency needs non-increasing biases (checked)",
    16: "Ordinal methods: CORN conditional ordinal regression (context)",
    17: "Ordinal methods: neuron stick-breaking ordinal regression (context)",
    18: "Related work: class imbalance makes mild DR fragile for lightweight CNNs (context)",
    19: "Related work: CLAHE + EfficientNet + Grad-CAM pipelines (context)",
    20: "Related work: uncertainty-aware ordinal head with multi-dataset training; arXiv preprint (context)",
    21: "Related work: cross-dataset generalisability analysis (context)",
    22: "Related work: cross-dataset generalisation in DR grading (context)",
    23: "Related work: explanation maps expected in clinical pipelines (context)",
    24: "Methods: frozen BatchNorm / batch-statistics behaviour; arXiv preprint (context)",
    25: "Methods: APTOS third-party partition actually used (verified against local files)",
    26: "Methods: IDRiD dataset (checked)",
    27: "Methods: Messidor database (images) (checked)",
    28: "Methods: Messidor-2 clinical context (context)",
    29: "Methods: adjudicated reference standard for Messidor-2 labels (checked)",
    30: "Methods: DeepDRiD dataset (checked)",
    31: "Methods: AdamW optimiser (checked)",
    32: "Methods: timm library (checked)",
    33: "Methods: CLAHE algorithm (checked)",
    34: "Methods: MobileViT architecture; ICLR 2022 (arXiv URL given) (checked)",
    35: "Methods: EfficientNet architecture (checked)",
    36: "Methods: Swin Transformer architecture (checked)",
    37: "Methods: Grad-CAM (checked)",
    38: "Methods: temperature scaling / ECE (checked)",
}


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001
        print("FETCH_FAIL", url, type(e).__name__, e)
        return None


def norm(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("—", " ").replace("–", " "))


def expand(body: str) -> set[int]:
    out: set[int] = set()
    for grp in re.findall(r"\[(\d+(?:\s*[–-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[–-]\s*\d+)?)*)\]", body):
        for part in grp.split(","):
            m = re.match(r"\s*(\d+)\s*[–-]\s*(\d+)", part)
            out.update(range(int(m.group(1)), int(m.group(2)) + 1) if m else [int(part)])
    return out


def main() -> None:
    text = MD.read_text()
    body, rest = text.split("\n## References", 1)
    refs_block = rest.split("\n## Supplementary", 1)[0]
    cited = expand(body)
    rows = []
    for m in re.finditer(r"^\[(\d+)\] (.+)$", refs_block, flags=re.M):
        n, ref = int(m.group(1)), m.group(2)
        first_author = ref.split(",")[0].split(" ")[0] if not ref.startswith("Asia") else "APTOS"
        year = (re.findall(r"\b(19|20)\d{2}\b", ref) and re.search(r"\b((?:19|20)\d{2})\b", ref).group(1)) or ""
        parts = ref.split(". ")
        title = parts[1] if len(parts) > 1 else ""
        venue = re.search(r"\*([^*]+)\*", ref)
        venue = venue.group(1) if venue else ""
        doi = re.search(r"https://doi\.org/(\S+)", ref)
        arx = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", ref)
        url = re.search(r"https?://\S+", ref)
        link = (f"https://doi.org/{doi.group(1)}" if doi else url.group(0) if url else "")
        status, found_title, kind, note = "no_identifier", "", "", ""
        if doi and "zenodo" not in doi.group(1):
            raw = fetch(f"https://api.crossref.org/works/{doi.group(1)}")
            if raw:
                msg = json.loads(raw)["message"]
                found_title = (msg.get("title") or [""])[0]
                kind = msg.get("type", "")
                cr_year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
                note = f"crossref_type={kind}; crossref_year={cr_year}; container={(msg.get('container-title') or [''])[0]}"
            else:
                status = "doi_not_found_in_crossref"
            time.sleep(0.3)
        elif doi:
            raw = fetch(f"https://zenodo.org/api/records?q=doi:%22{doi.group(1)}%22")
            if raw:
                hits = json.loads(raw).get("hits", {}).get("hits", [])
                found_title = hits[0]["metadata"]["title"] if hits else ""
                kind = "software"
        elif arx:
            raw = fetch(f"http://export.arxiv.org/api/query?id_list={arx.group(1)}")
            if raw:
                t = re.findall(r"<title>(.*?)</title>", raw.decode(), flags=re.S)
                found_title = t[1].strip() if len(t) > 1 else ""
                kind = "arXiv preprint"
            time.sleep(3)
        if found_title:
            sim = SequenceMatcher(None, norm(title), norm(found_title)).ratio()
            status = "resolves_title_match" if sim >= 0.85 else f"resolves_TITLE_MISMATCH(sim={sim:.2f})"
        elif status == "no_identifier":
            status = "no_doi_listed" if not url else "url_only"
        if "ICLR" in ref or "International Conference on Learning Representations" in ref:
            pub = "conference paper"
        elif "arxiv" in ref.lower():
            pub = "arXiv preprint" + ("" if "arXiv preprint" in ref else " (NOT LABELLED)")
        elif "Kaggle" in ref:
            pub = "dataset webpage"
        elif "GitHub" in ref or kind == "software":
            pub = "software repository"
        elif kind == "book-chapter" or "Graphics Gems" in ref:
            pub = "book chapter"
        elif kind == "proceedings-article" or "Proceedings" in ref or "In: " in ref or "ICLR" in ref:
            pub = "conference paper"
        else:
            pub = "published journal article"
        rows.append({"ref_number": n, "first_author": first_author, "year": year, "title": title,
                     "journal_or_conference": venue, "doi_or_url": link, "doi_resolves": status,
                     "published_or_preprint": pub, "cited_in_text": "yes" if n in cited else "NO",
                     "claim_checked": CLAIMS.get(n, ""), "notes": f"{note}; retrieved_title={found_title}".strip("; ")})
        print(n, status, pub)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("WROTE", OUT.relative_to(ROOT), len(rows))


if __name__ == "__main__":
    sys.exit(main())
