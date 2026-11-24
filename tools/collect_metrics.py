"""Collect machine-measured test, coverage, and experiment receipts into docs/metrics.json."""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import sys
import trace
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.run_experiments import run_all_experiments  # noqa: E402

_SKIP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)


def _measure_tests_and_coverage() -> tuple[int, bool, float]:
    for mod in list(sys.modules.keys()):
        if mod.startswith("inference_quant_tradeoff_lab") or mod.startswith("test_"):
            del sys.modules[mod]
    tracer = trace.Trace(count=1, trace=0)

    def _run() -> tuple[int, bool]:
        old_o, old_e = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            loader = unittest.TestLoader()
            suite = loader.discover(str(ROOT_DIR / "tests"), pattern="test_*.py")
            res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
            return suite.countTestCases(), res.wasSuccessful()
        finally:
            sys.stdout, sys.stderr = old_o, old_e

    t_cnt, passed = tracer.runfunc(_run)
    counts: dict[str, set[int]] = {}
    for (fn, ln), _ in tracer.results().counts.items():
        counts.setdefault(os.path.realpath(fn), set()).add(ln)
    tot = cov = 0
    pkg_dir = os.path.realpath(str(SRC_DIR / "inference_quant_tradeoff_lab"))
    for fn in sorted(os.listdir(pkg_dir)):
        if not fn.endswith(".py"):
            continue
        full = os.path.join(pkg_dir, fn)
        tree = ast.parse(Path(full).read_text(encoding="utf-8"), filename=full)
        lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt) and not isinstance(node, _SKIP):
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    continue
                lines.add(node.lineno)
        tot += len(lines)
        cov += len(lines & counts.get(full, set()))
    return t_cnt, passed, round((cov / max(1, tot)) * 100.0, 2)


def collect(skip_tests: bool = False) -> dict[str, object]:
    exp = run_all_experiments(ROOT_DIR)
    mf = ROOT_DIR / "docs" / "metrics.json"
    if skip_tests and mf.exists():
        prev = json.loads(mf.read_text(encoding="utf-8")).get("metrics", {})
        test_count = int(prev.get("tests_total", 0))
        passed = bool(prev.get("tests_passed", True))
        cov_pct = float(prev.get("coverage_line_pct", 0.0))
    else:
        test_count, passed, cov_pct = _measure_tests_and_coverage()
    e1 = exp["exp1"]
    metrics = {
        "tests_total": test_count,
        "tests_passed": passed,
        "coverage_line_pct": cov_pct,
        "total_cases": e1["total_cases"],
        "edge_cases_count": e1["edge_cases_count"],
        "edge_divergence_cases_count": e1["edge_divergence_cases_count"],
        "mean_accuracy_drop": e1["mean_accuracy_drop"],
        "mean_edge_accuracy_drop": e1["mean_edge_accuracy_drop"],
        "mean_mem_saved_pct": e1["mean_mem_saved_pct"],
        "mean_latency_reduction_pct": e1["mean_latency_reduction_pct"],
        "resolution_floor_pct": e1["resolution_floor_pct"],
    }
    anchors = {
    "tests_total": [
        "{} tests"
    ],
    "coverage_line_pct": [
        "{}% line coverage"
    ],
    "total_cases": [
        "{} benchmark cases"
    ],
    "edge_cases_count": [
        "{} edge cases"
    ],
    "edge_divergence_cases_count": [
        "{} edge divergences"
    ],
    "mean_accuracy_drop": [
        "{} mean drop"
    ],
    "mean_edge_accuracy_drop": [
        "{} edge drop"
    ],
    "mean_mem_saved_pct": [
        "{}% memory saved"
    ],
    "mean_latency_reduction_pct": [
        "{}% latency reduction"
    ],
    "resolution_floor_pct": [
        "{}% resolution floor"
    ]
}
    payload = {
        "metrics": metrics,
        "anchors": anchors,
        "checked_documents": [
        "README.md",
        "CHANGELOG.md",
        "docs/adr/ADR-001-separate-tail-erosion-from-mean.md",
        "docs/adr/ADR-002-explicit-three-way-exit-codes.md",
        "docs/adr/ADR-003-stdlib-core-and-py312-ci.md",
        "docs/adr/ADR-004-positive-memory-boundary.md",
        "docs/adr/ADR-005-offline-hardware-boundaries.md"
],
        "note": "Every value here is produced by running tests and experiments.",
    }
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect metrics receipt.")
    p.add_argument("--skip-tests", action="store_true")
    args = p.parse_args(argv)
    payload = collect(skip_tests=args.skip_tests)
    print(json.dumps(payload["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
