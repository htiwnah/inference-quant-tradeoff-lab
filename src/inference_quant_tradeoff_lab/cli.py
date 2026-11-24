"""CLI for inference-quant-tradeoff-lab with 0/1/2 exit codes and trade-off table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inference_quant_tradeoff_lab.analyzer import QuantSummary, analyze_cases, load_cases
from inference_quant_tradeoff_lab.config import load_policy

EXIT_PASS = 0
EXIT_ANSWER_NO = 1
EXIT_FAILURE = 2


def render_table(s: QuantSummary, max_rows: int = 12) -> str:
    hdr = f"{'CaseID':<7} {'Slice':<18} {'Edge':<5} {'FP16':>6} {'Quant':>6} {'Drop':>7} {'Mem%':>6} {'Flag'}"
    lines = [
        "Quantization Trade-Off Audit: Memory Savings vs Edge Accuracy Erosion",
        (
            f"Cases: {s.total_cases} | Edge Divergences: {s.edge_divergence_cases_count}/"
            f"{s.edge_cases_count} | Mean Drop: {s.mean_accuracy_drop:.4f} "
            f"(Edge: {s.mean_edge_accuracy_drop:.4f} vs Bulk: {s.mean_bulk_accuracy_drop:.4f})"
        ),
        (
            f"Mem Saved: {s.mean_mem_saved_pct:.2f}% | Latency Cut: "
            f"{s.mean_latency_reduction_pct:.2f}% | Naive Mean Gate: "
            f"{'PASS' if s.naive_mean_gate_would_pass else 'BLOCK'} | "
            f"Tail Gate: {'PASS' if s.gate_passed else 'BLOCK'}"
        ),
        hdr,
        "-" * len(hdr),
    ]
    ordered = sorted(s.cases, key=lambda c: (0 if c.edge_divergence else 1, -c.accuracy_drop, c.case_id))
    for c in ordered[:max_rows]:
        flag = "EDGE_LOSS" if c.edge_divergence else "OK"
        lines.append(
            f"{c.case_id:<7} {c.slice_name:<18} {str(c.is_edge_case):<5} "
            f"{c.fp16_score:>6.3f} {c.quant_score:>6.3f} {c.accuracy_drop:>+7.3f} "
            f"{c.mem_saved_pct:>5.1f}% {flag}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quant-lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    ap = sub.add_parser("audit")
    ap.add_argument("--fixture", default="fixtures/quant_eval_cases.jsonl")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--out", default="reports/quant_audit.json")
    ap.add_argument("--max-rows", type=int, default=12)
    ap.add_argument("--json-only", action="store_true")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return EXIT_PASS if exc.code == 0 else EXIT_FAILURE
    try:
        cfg = load_policy(args.policy)
        summary = analyze_cases(load_cases(args.fixture), cfg)
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
        sys.stderr.write(f"Input error: {exc}\n")
        return EXIT_FAILURE
    if args.json_only:
        sys.stdout.write(json.dumps(summary.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(render_table(summary, max_rows=args.max_rows) + "\n")
    return EXIT_PASS if summary.gate_passed else EXIT_ANSWER_NO


if __name__ == "__main__":
    raise SystemExit(main())
