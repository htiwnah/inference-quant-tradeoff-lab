"""Explicit policy configuration for quantization accuracy-drop and edge-divergence gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantPolicy:
    policy_version: str = "1.0"
    max_allowed_edge_divergence_cases: int = 0
    edge_drop_threshold: float = 0.05
    max_mean_accuracy_drop: float = 0.05
    fixture_path: str = "fixtures/quant_eval_cases.jsonl"

    def validate(self) -> None:
        if self.max_allowed_edge_divergence_cases < 0:
            raise ValueError("max_allowed_edge_divergence_cases must be >= 0")
        if not (0.0 <= self.edge_drop_threshold <= 1.0):
            raise ValueError("edge_drop_threshold must lie in [0.0, 1.0]")
        if not (0.0 <= self.max_mean_accuracy_drop <= 1.0):
            raise ValueError("max_mean_accuracy_drop must lie in [0.0, 1.0]")


def load_policy(path: str | Path | None = None) -> QuantPolicy:
    if path is None:
        cfg = QuantPolicy()
        cfg.validate()
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Policy not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Policy must be a JSON object")
    cfg = QuantPolicy(**raw)
    cfg.validate()
    return cfg
