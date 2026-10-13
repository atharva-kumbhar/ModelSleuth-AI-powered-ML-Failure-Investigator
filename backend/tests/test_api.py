from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import experiments
from app.services.datasets import demo_frame


class ModelSleuthApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        experiments.DB_PATH = Path(self.temp_dir.name) / "test.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    def upload_small(self):
        response = self.client.post(
            "/api/datasets/validate",
            files={"file": ("evaluation.csv", "truth,prediction,sample_id,region\na,a,s1,east\na,b,s2,west\nb,b,s3,east\nb,b,s4,west\n", "text/csv")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def run_small(self):
        dataset = self.upload_small()
        response = self.client.post("/api/analyze", json={
            "dataset_id": dataset["dataset_id"], "true_column": "truth", "predicted_column": "prediction",
            "sample_id_column": "sample_id", "group_columns": ["region"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_csv_metrics_group_findings_and_sample_filter(self):
        result = self.run_small()
        self.assertEqual(result["metrics"]["accuracy"], .75)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["confusion"]["matrix"], [[1, 1], [0, 2]])
        self.assertTrue(result["groups"])
        rows = self.client.get(f"/api/analysis/{result['analysis_id']}/samples", params={"status": "incorrect"})
        self.assertEqual(rows.json()["total"], 1)
        self.assertEqual(rows.json()["samples"][0]["sample_id"], "s2")

    def test_probabilities_are_validated_and_cleanlab_requires_oos_confirmation(self):
        dataset = self.upload_small()
        frame = demo_frame().head(24)
        # Upload a fixture with class columns to exercise the probability contract.
        csv = frame.to_csv(index=False)
        upload = self.client.post("/api/datasets/validate", files={"file": ("prob.csv", csv, "text/csv")})
        ds = upload.json()
        body = {"dataset_id": ds["dataset_id"], "true_column": "true_label", "predicted_column": "predicted_label",
                "sample_id_column": "sample_id", "probability_columns": {"canopy":"prob_canopy", "water":"prob_water", "urban":"prob_urban"}}
        analysis = self.client.post("/api/analyze", json=body).json()
        self.assertIsNotNone(analysis["uncertainty"])
        self.assertIn("out-of-sample", analysis["data_quality"]["cleanlab"])
        body["predictions_out_of_sample"] = True
        response = self.client.post("/api/analyze", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNotNone(response.json()["uncertainty"])

    def test_bad_probabilities_are_reported_and_analysis_continues(self):
        ds = self.upload_small()
        frame = "truth,prediction,p_a,p_b\na,a,2,-1\na,b,2,-1\nb,b,2,-1\n"
        uploaded = self.client.post("/api/datasets/validate", files={"file": ("bad.csv", frame, "text/csv")}).json()
        result = self.client.post("/api/analyze", json={"dataset_id": uploaded["dataset_id"], "true_column":"truth",
            "predicted_column":"prediction", "probability_columns":{"a":"p_a","b":"p_b"}})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertIsNone(result.json()["uncertainty"])
        self.assertTrue(result.json()["warnings"])

    def test_invalid_upload_and_missing_column_have_useful_errors(self):
        invalid = self.client.post("/api/datasets/validate", files={"file": ("bad.xlsx", b"not a csv", "application/octet-stream")})
        self.assertEqual(invalid.status_code, 415)
        ds = self.upload_small()
        response = self.client.post("/api/analyze", json={"dataset_id":ds["dataset_id"],"true_column":"absent","predicted_column":"prediction"})
        self.assertEqual(response.status_code, 422)

    def test_experiment_history_compares_real_metric_deltas(self):
        result = self.run_small()
        first = self.client.post("/api/experiments", json={"analysis_id":result["analysis_id"],"name":"Baseline","baseline":True}).json()
        second = self.client.post("/api/experiments", json={"analysis_id":result["analysis_id"],"name":"Revision"}).json()
        self.assertEqual(len(self.client.get("/api/experiments").json()), 2)
        comparison = self.client.post("/api/experiments/compare", json={"first_id":first["id"],"second_id":second["id"]}).json()
        self.assertEqual(comparison["metric_delta"]["accuracy"], 0)
        self.assertEqual(comparison["confusion_delta"]["matrix"], [[0,0],[0,0]])

    def test_demo_is_generated_reproducibly_and_ai_uses_template_without_keys(self):
        first, again = demo_frame(), demo_frame()
        self.assertTrue(first.equals(again))
        demo = self.client.post("/api/demo").json()
        finding = demo["findings"][0]
        explanation = self.client.post("/api/explain", json={"analysis_id":demo["analysis_id"],"finding_id":finding["id"]}).json()
        self.assertTrue(explanation["fallback"])
        self.assertTrue(explanation["explanation"]["observed_facts"])
        self.assertEqual(self.client.get("/api/ready").status_code, 200)


if __name__ == "__main__":
    unittest.main()
