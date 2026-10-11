#!/usr/bin/env python3
"""
Superseded. The manuscript is now generated from paper/MANUSCRIPT_TEMPLATE.md by
scripts/build_manuscript_md.py (numbers) and scripts/build_manuscript_docx.py (Word file).
Editing paper/MANUSCRIPT.md in place with regexes would be overwritten, so this entry point
simply runs the new builders.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for script in ("scripts/build_manuscript_md.py", "scripts/build_manuscript_docx.py"):
    print(">>>", script, flush=True)
    if subprocess.run([sys.executable, "-u", script], cwd=str(ROOT)).returncode != 0:
        raise SystemExit(f"FAILED: {script}")
