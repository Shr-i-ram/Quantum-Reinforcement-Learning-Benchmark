"""
Automated validation of Quantum RL Arena FastAPI endpoints.
Tests environment loading, simulation session creation, stepping,
quantum circuit execution, benchmark registry, and challenge recording.
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.main import app


class TestArenaAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_list_environments(self):
        resp = self.client.get("/api/environments")
        self.assertEqual(resp.status_code, 200)
        envs = resp.json()
        self.assertGreaterEqual(len(envs), 5)
        env_ids = [e["id"] for e in envs]
        self.assertIn("CartPole-v1", env_ids)
        self.assertIn("LunarLander-v3", env_ids)

    def test_simulation_session_and_step(self):
        # 1. Create session with Demonstration Solved agent
        resp = self.client.post("/api/sim/session", json={
            "env_id": "CartPole-v1",
            "algo": "DQN (Demonstration Solved)",
            "seed": 42
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        session_id = data["session_id"]
        self.assertTrue(session_id)
        self.assertIn("frame_base64", data)
        self.assertTrue(data["frame_base64"].startswith("data:image/jpeg;base64,"))

        # 2. Step session
        step_resp = self.client.post("/api/sim/step", json={
            "session_id": session_id
        })
        self.assertEqual(step_resp.status_code, 200)
        step_data = step_resp.json()
        self.assertEqual(step_data["step_count"], 1)
        self.assertIn("frame_base64", step_data)

        # 3. Reset session
        reset_resp = self.client.post("/api/sim/reset", json={
            "session_id": session_id,
            "seed": 123
        })
        self.assertEqual(reset_resp.status_code, 200)
        self.assertEqual(reset_resp.json()["step_count"], 0)

        # 4. Close session
        close_resp = self.client.post("/api/sim/close", json={"session_id": session_id})
        self.assertEqual(close_resp.status_code, 200)

    def test_quantum_simulation(self):
        resp = self.client.post("/api/quantum/simulate", json={
            "features": [0.5, -0.5, 0.25, -0.25],
            "depth": 2
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        expvals = data["expectation_values"]
        self.assertEqual(len(expvals), 4)
        for ev in expvals:
            self.assertGreaterEqual(ev["expval"], -1.0)
            self.assertLessEqual(ev["expval"], 1.0)
        self.assertEqual(len(data["basis_probabilities"]), 16)

    def test_quantum_scaling(self):
        resp = self.client.get("/api/quantum/scaling?shots=1000&depth=2")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("qubits", data)
        self.assertIn("classical_memory_mb", data)
        self.assertIn("crossover_qubit_estimate", data)

    def test_benchmark_endpoints(self):
        resp = self.client.get("/api/benchmark/registry")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("benchmark_experiments", resp.json())

        pareto_resp = self.client.get("/api/benchmark/pareto")
        self.assertEqual(pareto_resp.status_code, 200)
        self.assertIsInstance(pareto_resp.json(), list)

        report_resp = self.client.get("/api/benchmark/report")
        self.assertEqual(report_resp.status_code, 200)
        self.assertIn("# Research Report", report_resp.text)

    def test_frontend_serving(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("<!doctype html>", resp.text.lower())
        self.assertIn("assets/", resp.text)

    def test_challenges(self):
        resp = self.client.get("/api/challenges")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("presets", data)


if __name__ == "__main__":
    unittest.main()
