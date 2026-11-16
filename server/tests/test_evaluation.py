"""Evaluation spec, matrix, gate, and API contract."""

import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from server.api.v1.app import create_app
from server.application.evaluation_service import EvaluationBusy, EvaluationService
from server.domain.evaluation import (
    SpecError,
    build_matrix,
    build_report,
    compute_gate,
    normalize_spec,
    summarize_group,
)
from server.tests.test_session_control import MemoryWorker, NOW
from server.application.session_service import SessionService


class SpecTests(unittest.TestCase):
    def test_defaults_and_unknown_keys(self):
        spec = normalize_spec(None)
        self.assertEqual(spec["attack"], "spoof_patch")
        self.assertEqual(spec["seeds"], [0])
        with self.assertRaises(SpecError):
            normalize_spec({"surprise": True})
        for payload in ({"policy": "gpt"}, {"steps": 0}, {"seeds": []}, {"defense": "magic"}, {"task": "libero/x"}):
            with self.subTest(payload=payload), self.assertRaises(SpecError):
                normalize_spec(payload)

    def test_matrix_collapses_when_no_defence_is_configured(self):
        with_defence = build_matrix(normalize_spec({"defense": "median"}))
        self.assertEqual([group["name"] for group in with_defence],
                         ["clean", "clean_defended", "attacked", "attacked_defended"])
        without = build_matrix(normalize_spec({"defense": "none"}))
        self.assertEqual([group["name"] for group in without], ["clean", "attacked"])


class GateTests(unittest.TestCase):
    def test_gate_uses_clean_runs_only(self):
        results = [
            {"group": "clean", "status": "ok", "success_observed": False, "rows": []},
            {"group": "attacked", "status": "ok", "success_observed": False, "rows": []},
        ]
        gate = compute_gate(results)
        self.assertEqual(gate["clean_success_rate"], 0.0)
        self.assertFalse(gate["eligible_for_task_eval"])
        self.assertIn("不能声称攻击成功率", gate["reason"])

    def test_missing_clean_run_cannot_pass_the_gate(self):
        gate = compute_gate([{"group": "attacked", "status": "ok", "success_observed": True}])
        self.assertFalse(gate["eligible_for_task_eval"])
        self.assertIsNone(gate["clean_success_rate"])

    def test_summary_and_divergence_are_reported_per_group(self):
        rows = [
            {"step": 0, "action": [0.5, 0, 0], "centroid_offset": [0.2, 0.0], "area": 0.01, "success": False},
            {"step": 1, "action": [0.3, 0, 0], "centroid_offset": [0.1, 0.0], "area": 0.01, "success": False},
        ]
        summary = summarize_group(rows)
        self.assertEqual(summary["steps"], 2)
        self.assertAlmostEqual(summary["mean_abs_action_x"], 0.4)
        report = build_report(
            normalize_spec({"defense": "median"}),
            [
                {"group": "clean", "seed": 0, "status": "ok", "success_observed": False,
                 "action_x_series": [0.5, 0.3], "rows": rows},
                {"group": "attacked", "seed": 0, "status": "ok", "success_observed": False,
                 "action_x_series": [0.1, 0.1],
                 "rows": [dict(row, action=[0.1, 0, 0]) for row in rows]},
            ],
            compute_gate([{"group": "clean", "status": "ok", "success_observed": False, "rows": rows}]),
        )
        attacked = next(group for group in report["groups"] if group["group"] == "attacked")
        self.assertAlmostEqual(attacked["divergence_vs_clean"], 0.3)


class EvaluationApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prism-eval-")
        root = Path(self.temporary.name)

        def fake_runner(run_dir):
            # A real child process that writes a finished summary immediately.
            script = run_dir / "fake_runner.py"
            script.write_text(
                "import json, pathlib, sys\n"
                "run_dir = pathlib.Path(sys.argv[1])\n"
                "spec = json.loads((run_dir/'spec.json').read_text())\n"
                "summary = {'gate': {'eligible_for_task_eval': False, 'clean_success_rate': 0.0},\n"
                "           'groups': [{'group': 'clean', 'divergence_vs_clean': None}]}\n"
                "(run_dir/'summary.json').write_text(json.dumps(summary))\n"
                "(run_dir/'status.json').write_text(json.dumps({'state': 'completed', 'completed': 1, 'total': 1}))\n",
                encoding="utf-8",
            )
            return ["python", str(script), str(run_dir)]

        self.evaluations = EvaluationService(root / "evaluations", python="python", command_factory=fake_runner)
        service = SessionService(root / "sessions", MemoryWorker(), clock=lambda: NOW)
        self.client = TestClient(
            create_app(service, "eval-token", evaluations=self.evaluations), base_url="http://127.0.0.1"
        )
        self.headers = {"X-Console-Token": "eval-token"}

    def tearDown(self):
        self.evaluations.close()
        self.temporary.cleanup()

    def test_create_then_read_a_finished_evaluation(self):
        created = self.client.post("/api/v1/evaluations", headers=self.headers, json={"spec": {"steps": 5}})
        self.assertEqual(created.status_code, 201)
        run_id = created.json()["id"]
        self.evaluations._process.wait(timeout=30)

        fetched = self.client.get(f"/api/v1/evaluations/{run_id}", headers=self.headers).json()
        self.assertEqual(fetched["status"]["state"], "completed")
        self.assertFalse(fetched["gate"]["eligible_for_task_eval"])
        listed = self.client.get("/api/v1/evaluations", headers=self.headers).json()["evaluations"]
        self.assertIn(run_id, [item["id"] for item in listed])

    def test_invalid_spec_is_rejected_before_starting_a_process(self):
        response = self.client.post("/api/v1/evaluations", headers=self.headers, json={"spec": {"steps": 0}})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "invalid_spec")

    def test_second_run_is_refused_while_one_is_active(self):
        first = self.client.post("/api/v1/evaluations", headers=self.headers, json={"spec": {}})
        self.assertEqual(first.status_code, 201)
        second = self.client.post("/api/v1/evaluations", headers=self.headers, json={"spec": {}})
        self.assertIn(second.status_code, (201, 409))
        self.evaluations._process.wait(timeout=30)


if __name__ == "__main__":
    unittest.main()
