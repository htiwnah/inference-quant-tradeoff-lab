"""Deterministic offline quantization trade-off experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inference_quant_tradeoff_lab.analyzer import analyze_cases, load_cases  # noqa: E402
from inference_quant_tradeoff_lab.config import QuantPolicy, load_policy  # noqa: E402


def run_all_experiments(repo_root: Path | None = None) -> dict[str, object]:
    root = repo_root or ROOT
    out_dir = root / "docs" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(root / "fixtures" / "quant_eval_cases.jsonl")
    s = analyze_cases(cases, load_policy(root / "policy" / "quant_policy.json"))
    e1 = {
        "total_cases": s.total_cases,
        "edge_cases_count": s.edge_cases_count,
        "edge_divergence_cases_count": s.edge_divergence_cases_count,
        "mean_accuracy_drop": s.mean_accuracy_drop,
        "mean_edge_accuracy_drop": s.mean_edge_accuracy_drop,
        "mean_bulk_accuracy_drop": s.mean_bulk_accuracy_drop,
        "mean_mem_saved_pct": s.mean_mem_saved_pct,
        "mean_latency_reduction_pct": s.mean_latency_reduction_pct,
        "resolution_floor_pct": s.resolution_floor_pct,
    }
    sweep = [
        {
            "edge_drop_threshold": t,
            "divergences": analyze_cases(
                cases, QuantPolicy(edge_drop_threshold=t)
            ).edge_divergence_cases_count,
        }
        for t in (0.02, 0.05, 0.10, 0.20)
    ]
    (out_dir / "exp1_quant_tradeoff.json").write_text(json.dumps(e1, indent=2) + "\n", encoding="utf-8")
    (out_dir / "exp2_drop_sweep.json").write_text(json.dumps({"rows": sweep}, indent=2) + "\n", encoding="utf-8")
    return {"exp1": e1, "exp2": {"rows": sweep}}


if __name__ == "__main__":
    res = run_all_experiments()["exp1"]
    print(
        f"Exp1 | cases={res['total_cases']} | edge_div={res['edge_divergence_cases_count']}/"
        f"{res['edge_cases_count']} | mem={res['mean_mem_saved_pct']}% | "
        f"lat={res['mean_latency_reduction_pct']}%"
    )
