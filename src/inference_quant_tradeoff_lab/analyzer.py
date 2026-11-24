"""Deterministic FP16 vs quantized trade-off analyzer isolating edge-case erosion."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from inference_quant_tradeoff_lab.config import QuantPolicy


@dataclass(frozen=True)
class QuantCase:
    case_id: str
    slice_name: str
    is_edge_case: bool
    fp16_score: float
    quant_score: float
    fp16_mem_mb: float
    quant_mem_mb: float
    fp16_latency_ms: float
    quant_latency_ms: float


@dataclass(frozen=True)
class CaseVerdict:
    case_id: str
    slice_name: str
    is_edge_case: bool
    fp16_score: float
    quant_score: float
    accuracy_drop: float
    mem_saved_pct: float
    latency_cut_pct: float
    edge_divergence: bool


@dataclass(frozen=True)
class QuantSummary:
    total_cases: int
    edge_cases_count: int
    edge_divergence_cases_count: int
    mean_fp16_score: float
    mean_quant_score: float
    mean_accuracy_drop: float
    mean_edge_accuracy_drop: float
    mean_bulk_accuracy_drop: float
    mean_mem_saved_pct: float
    mean_latency_reduction_pct: float
    resolution_floor_pct: float
    naive_mean_gate_would_pass: bool
    gate_passed: bool
    cases: list[CaseVerdict] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_cases": self.total_cases,
            "edge_cases_count": self.edge_cases_count,
            "edge_divergence_cases_count": self.edge_divergence_cases_count,
            "mean_fp16_score": self.mean_fp16_score,
            "mean_quant_score": self.mean_quant_score,
            "mean_accuracy_drop": self.mean_accuracy_drop,
            "mean_edge_accuracy_drop": self.mean_edge_accuracy_drop,
            "mean_bulk_accuracy_drop": self.mean_bulk_accuracy_drop,
            "mean_mem_saved_pct": self.mean_mem_saved_pct,
            "mean_latency_reduction_pct": self.mean_latency_reduction_pct,
            "resolution_floor_pct": self.resolution_floor_pct,
            "naive_mean_gate_would_pass": self.naive_mean_gate_would_pass,
            "gate_passed": self.gate_passed,
            "cases": [
                {
                    "case_id": c.case_id,
                    "slice_name": c.slice_name,
                    "is_edge_case": c.is_edge_case,
                    "fp16_score": c.fp16_score,
                    "quant_score": c.quant_score,
                    "accuracy_drop": c.accuracy_drop,
                    "mem_saved_pct": c.mem_saved_pct,
                    "latency_cut_pct": c.latency_cut_pct,
                    "edge_divergence": c.edge_divergence,
                }
                for c in self.cases
            ],
        }


def load_cases(path: str | Path) -> list[QuantCase]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fixture not found: {p}")
    out: list[QuantCase] = []
    for ln_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        r = json.loads(s)
        if not isinstance(r, dict):
            raise ValueError(f"Line {ln_no} must be a JSON object")
        for req in ("case_id", "slice_name", "fp16_score", "quant_score", "fp16_mem_mb", "quant_mem_mb"):
            if req not in r:
                raise ValueError(f"Missing {req!r} on line {ln_no}")
        fp_m = float(r["fp16_mem_mb"])
        q_m = float(r["quant_mem_mb"])
        if fp_m <= 0.0 or q_m <= 0.0:
            raise ValueError(f"Memory values must be positive on line {ln_no}")
        out.append(
            QuantCase(
                case_id=str(r["case_id"]),
                slice_name=str(r["slice_name"]),
                is_edge_case=bool(r.get("is_edge_case", False)),
                fp16_score=0.0 if r["fp16_score"] is None else float(r["fp16_score"]),
                quant_score=0.0 if r["quant_score"] is None else float(r["quant_score"]),
                fp16_mem_mb=fp_m,
                quant_mem_mb=q_m,
                fp16_latency_ms=float(r.get("fp16_latency_ms", 140.0)),
                quant_latency_ms=float(r.get("quant_latency_ms", 95.0)),
            )
        )
    if not out:
        raise ValueError(f"No cases in {p}")
    return out


def analyze_cases(cases: list[QuantCase], policy: QuantPolicy) -> QuantSummary:
    if not cases:
        raise ValueError("Cannot analyze empty cases")
    policy.validate()
    verdicts: list[CaseVerdict] = []
    edge_drops: list[float] = []
    bulk_drops: list[float] = []
    divs = 0
    for c in cases:
        drop = round(c.fp16_score - c.quant_score, 4)
        mem_s = round(((c.fp16_mem_mb - c.quant_mem_mb) / c.fp16_mem_mb) * 100.0, 2)
        lat_s = (
            round(((c.fp16_latency_ms - c.quant_latency_ms) / c.fp16_latency_ms) * 100.0, 2)
            if c.fp16_latency_ms > 0
            else 0.0
        )
        is_div = bool(c.is_edge_case and drop > policy.edge_drop_threshold)
        divs += int(is_div)
        if c.is_edge_case:
            edge_drops.append(drop)
        else:
            bulk_drops.append(drop)
        verdicts.append(
            CaseVerdict(
                c.case_id, c.slice_name, c.is_edge_case, c.fp16_score,
                c.quant_score, drop, mem_s, lat_s, is_div,
            )
        )
    n = len(verdicts)
    mean_drop = round(sum(v.accuracy_drop for v in verdicts) / n, 4)
    naive_ok = mean_drop <= policy.max_mean_accuracy_drop
    gate_ok = naive_ok and divs <= policy.max_allowed_edge_divergence_cases
    return QuantSummary(
        total_cases=n,
        edge_cases_count=len(edge_drops),
        edge_divergence_cases_count=divs,
        mean_fp16_score=round(sum(v.fp16_score for v in verdicts) / n, 4),
        mean_quant_score=round(sum(v.quant_score for v in verdicts) / n, 4),
        mean_accuracy_drop=mean_drop,
        mean_edge_accuracy_drop=round(sum(edge_drops) / len(edge_drops), 4) if edge_drops else 0.0,
        mean_bulk_accuracy_drop=round(sum(bulk_drops) / len(bulk_drops), 4) if bulk_drops else 0.0,
        mean_mem_saved_pct=round(sum(v.mem_saved_pct for v in verdicts) / n, 2),
        mean_latency_reduction_pct=round(sum(v.latency_cut_pct for v in verdicts) / n, 2),
        resolution_floor_pct=round((1.0 / n) * 100.0, 2),
        naive_mean_gate_would_pass=naive_ok,
        gate_passed=gate_ok,
        cases=verdicts,
    )
