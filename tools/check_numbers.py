"""Machine-check numerical claims in README, CHANGELOG, and ADRs against docs/metrics.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def verify_numbers(root: Path = ROOT_DIR) -> list[str]:
    mf = root / "docs" / "metrics.json"
    if not mf.exists():
        return ["Missing docs/metrics.json"]
    data = json.loads(mf.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    anchors = data.get("anchors", {})
    doc_paths = data.get("checked_documents", ["README.md"])
    combined = "\n".join((root / d).read_text(encoding="utf-8") for d in doc_paths if (root / d).exists())
    norm = re.sub(r"\s+", " ", combined).lower()
    errs: list[str] = []
    for k, v in metrics.items():
        if isinstance(v, bool):
            continue
        pats = anchors.get(k)
        if not pats:
            continue
        matched = False
        cands_all: list[str] = []
        for tmpl in pats:
            cands: list[str] = []
            if "{:,.2f}" in tmpl and isinstance(v, (int, float)):
                cands.append(tmpl.format(float(v)).lower())
            elif "{:,}" in tmpl and isinstance(v, (int, float)):
                cands.append(tmpl.format(v).lower())
            else:
                cands.append(tmpl.format(v).lower())
                if isinstance(v, float) and v.is_integer():
                    cands.append(tmpl.format(int(v)).lower())
            cands_all.extend(cands)
            if any(c in norm for c in cands):
                matched = True
                break
        if not matched:
            errs.append(f"Metric {k}={v!r} failed anchor match; expected {cands_all}")
    return errs


def main() -> int:
    errs = verify_numbers()
    if errs:
        for e in errs:
            sys.stderr.write(f"Number check failed: {e}\n")
        return 1
    print("Number receipt verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
