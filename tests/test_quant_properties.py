"""Property and CLI tests for inference-quant-tradeoff-lab."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference_quant_tradeoff_lab.analyzer import QuantCase, analyze_cases, load_cases
from inference_quant_tradeoff_lab.cli import (
    EXIT_ANSWER_NO,
    EXIT_FAILURE,
    EXIT_PASS,
    main as cli_main,
    render_table,
)
from inference_quant_tradeoff_lab.config import QuantPolicy, load_policy

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures" / "quant_eval_cases.jsonl"
POL = ROOT / "policy" / "quant_policy.json"


def _case(cid: str = "C1", edge: bool = False, fp: float = 0.92, q: float = 0.90) -> QuantCase:
    return QuantCase(cid, "numeric_precision", edge, fp, q, 4096.0, 2048.0, 140.0, 95.0)


class TestQuantPolicyAndLoader(unittest.TestCase):
    def test_default_policy_validates(self) -> None:
        QuantPolicy().validate()

    def test_committed_policy_loads(self) -> None:
        self.assertAlmostEqual(load_policy(POL).max_mean_accuracy_drop, 0.05)

    def test_load_none_returns_defaults(self) -> None:
        self.assertEqual(load_policy(None).max_allowed_edge_divergence_cases, 0)

    def test_missing_policy_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_policy(Path("/tmp/missing_q_pol_999.json"))

    def test_non_dict_policy_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "p.json"
            p.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(p)

    def test_invalid_policy_bounds_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QuantPolicy(max_allowed_edge_divergence_cases=-1).validate()
        with self.assertRaises(ValueError):
            QuantPolicy(edge_drop_threshold=1.5).validate()
        with self.assertRaises(ValueError):
            QuantPolicy(max_mean_accuracy_drop=-0.1).validate()

    def test_fixture_loads_forty_cases(self) -> None:
        self.assertEqual(len(load_cases(FIX)), 40)

    def test_loader_rejects_missing_empty_and_malformed_files(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_cases(Path("/tmp/missing_q_fix_999.jsonl"))
        with tempfile.TemporaryDirectory() as tmp:
            e = Path(tmp) / "e.jsonl"
            e.write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(e)
            b = Path(tmp) / "b.jsonl"
            b.write_text("[1]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(b)
            m = Path(tmp) / "m.jsonl"
            m.write_text(json.dumps({"case_id": "C1"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(m)
            z = Path(tmp) / "z.jsonl"
            row = {
                "case_id": "C1", "slice_name": "s", "fp16_score": 0.9,
                "quant_score": 0.8, "fp16_mem_mb": 0, "quant_mem_mb": 100,
            }
            z.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(z)

    def test_war_story_null_quant_score_coerced_to_zero_without_crashing(self) -> None:
        """Pins bug where null quant_score in JSON triggered TypeError in float()."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "null.jsonl"
            row = {
                "case_id": "C0", "slice_name": "s", "is_edge_case": True,
                "fp16_score": 0.9, "quant_score": None, "fp16_mem_mb": 100, "quant_mem_mb": 50,
            }
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            pol = QuantPolicy(max_allowed_edge_divergence_cases=5, max_mean_accuracy_drop=1.0)
            s = analyze_cases(load_cases(p), pol)
            self.assertEqual(s.edge_divergence_cases_count, 1)


class TestQuantAnalyzerAndCli(unittest.TestCase):
    def test_empty_cases_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_cases([], QuantPolicy())

    def test_committed_fixture_passes_naive_mean_gate_with_ten_edge_divergences(self) -> None:
        s = analyze_cases(load_cases(FIX), load_policy(POL))
        self.assertEqual(s.total_cases, 40)
        self.assertEqual(s.edge_cases_count, 10)
        self.assertEqual(s.edge_divergence_cases_count, 10)
        self.assertTrue(s.naive_mean_gate_would_pass)
        self.assertFalse(s.gate_passed)
        self.assertAlmostEqual(s.mean_edge_accuracy_drop, 0.18)
        self.assertAlmostEqual(s.mean_bulk_accuracy_drop, 0.0013)
        self.assertAlmostEqual(s.mean_mem_saved_pct, 50.0)
        self.assertAlmostEqual(s.mean_latency_reduction_pct, 32.14)
        self.assertAlmostEqual(s.resolution_floor_pct, 2.5)

    def test_zero_latency_guard_returns_zero_latency_cut(self) -> None:
        c = QuantCase("C1", "s", False, 0.9, 0.9, 100.0, 50.0, 0.0, 0.0)
        s = analyze_cases([c], QuantPolicy())
        self.assertEqual(s.mean_latency_reduction_pct, 0.0)
        self.assertTrue(s.gate_passed)

    def test_to_dict_serializes_all_forty_cases(self) -> None:
        d = analyze_cases(load_cases(FIX), load_policy(POL)).to_dict()
        self.assertEqual(len(d["cases"]), 40)  # type: ignore[arg-type]

    def test_cli_exit_codes_and_table_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            rc1 = cli_main(
                ["audit", "--fixture", str(FIX), "--policy", str(POL), "--out", str(out)]
            )
            self.assertEqual(rc1, EXIT_ANSWER_NO)
            p = Path(tmp) / "p.json"
            p.write_text(json.dumps({"max_allowed_edge_divergence_cases": 50}), encoding="utf-8")
            rc0 = cli_main(
                [
            "audit", "--fixture", str(FIX), "--policy", str(p),
            "--out", str(out), "--json-only",
        ]
            )
            self.assertEqual(rc0, EXIT_PASS)
        self.assertEqual(cli_main(["audit", "--fixture", "/tmp/missing_q_999.jsonl"]), EXIT_FAILURE)
        self.assertEqual(cli_main(["audit", "--max-rows", "bad"]), EXIT_FAILURE)
        self.assertEqual(cli_main(["--help"]), EXIT_PASS)
        tbl = render_table(analyze_cases(load_cases(FIX), load_policy(POL)), max_rows=8)
        self.assertIn("EDGE_LOSS", tbl)
        self.assertIn("Quantization Trade-Off Audit:", tbl)
