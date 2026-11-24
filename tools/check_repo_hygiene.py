"""Repository hygiene checker wired into make verify and CI."""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_NAME = "inference-quant-tradeoff-lab"
EM_DASH = chr(8212)
EN_DASH = chr(8211)
BANNED = ("defense", "study_guide", "interview_prep", "cheat_sheet")
SKIP = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build", "reports"}
AI_WORDS = ("co-authored-by:", "generated with", "claude", "anthropic", "openai", "chatgpt", "gemini")


def check_lint(root: Path = ROOT_DIR) -> list[str]:
    errs: list[str] = []
    for folder in ("src", "tests", "experiments", "tools"):
        base = root / folder
        if not base.exists():
            continue
        for pyf in sorted(base.rglob("*.py")):
            raw = pyf.read_text(encoding="utf-8")
            try:
                compile(raw, str(pyf), "exec")
            except SyntaxError as exc:
                errs.append(f"Syntax error in {pyf.relative_to(root)}: {exc}")
                continue
            for idx, ln in enumerate(raw.splitlines(), start=1):
                if len(ln) > 120:
                    errs.append(f"Line too long ({len(ln)} > 120) in {pyf.relative_to(root)}:{idx}")
    return errs


def check_hygiene(root: Path = ROOT_DIR) -> list[str]:
    errs: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parts = p.relative_to(root).parts
        if any(pt in SKIP or pt.endswith(".egg-info") for pt in parts):
            continue
        low = str(p.relative_to(root)).lower()
        for b in BANNED:
            if b in low:
                errs.append(f"Forbidden study/defense file: {low}")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".zip"}:
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        if EM_DASH in txt or EN_DASH in txt:
            errs.append(f"Em/en dash found in {p.relative_to(root)}")
        if p.name == "README.md":
            for ph in ("YOUR_USERNAME", "<USERNAME>"):
                if ph in txt:
                    errs.append(f"Placeholder {ph!r} in README.md")
            wf_pat = r"https://github\.com/[^\s)\]\"]+/actions/workflows/[^\s)\]\"]+"
            for m in re.finditer(wf_pat, txt):
                if REPO_NAME not in m.group(0):
                    errs.append(f"Workflow badge mismatch: {m.group(0)}")
    head = root / ".git" / "refs" / "heads" / "main"
    if head.exists():
        oid = head.read_text(encoding="utf-8").strip()
        seen: set[str] = set()
        while oid and oid not in seen:
            seen.add(oid)
            obj = root / ".git" / "objects" / oid[:2] / oid[2:]
            if not obj.exists():
                break
            raw = zlib.decompress(obj.read_bytes()).split(b"\x00", 1)[1]
            hdr, msg = raw.decode("utf-8", "replace").split("\n\n", 1)
            if EM_DASH in msg or EN_DASH in msg:
                errs.append("Dash found in git commit message")
            for fw in AI_WORDS:
                if fw in msg.lower():
                    errs.append(f"AI attribution {fw!r} in git log")
            parent = None
            for ln in hdr.splitlines():
                if ln.startswith("parent "):
                    parent = ln[7:].strip()
            oid = parent
    for wf in sorted((root / ".github" / "workflows").glob("*.yml")):
        wt = wf.read_text(encoding="utf-8")
        for key in ("name:", "on:", "jobs:"):
            if key not in wt:
                errs.append(f"Workflow missing {key} in {wf.name}")
        for idx, ln in enumerate(wt.splitlines(), start=1):
            if "\t" in ln:
                errs.append(f"Tab indentation in {wf.name}:{idx}")
    return errs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Repo hygiene and lint checker.")
    p.add_argument("--lint-only", action="store_true")
    args = p.parse_args(argv)
    errs = check_lint() + ([] if args.lint_only else check_hygiene())
    if errs:
        for e in errs:
            sys.stderr.write(f"Hygiene error: {e}\n")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
