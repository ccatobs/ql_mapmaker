import tomllib
import unittest
from pathlib import Path

from ccat_mapmaker.notebooks import benchmark_mapmaker as benchmark


class BenchmarkTests(unittest.TestCase):
    def test_normalizes_by_yielded_samples_not_requested_duration(self):
        metadata = {
            "preparation_processing": {"chunk_samples": [100, 100, 40]},
            "observation": {"sample_rate_hz": 10},
            "n_detectors_used": 20,
        }
        result = benchmark.measure(metadata, 12)
        self.assertEqual(result["duration_s"], 24)
        self.assertEqual(result["ms_per_detector_second"], 25)
        self.assertEqual(result["realtime_factor"], 2)

    def test_rejects_empty_or_invalid_workload(self):
        metadata = {
            "preparation_processing": {"chunk_samples": [10]},
            "observation": {"sample_rate_hz": 0},
            "n_detectors_used": 20,
        }
        with self.assertRaises(ValueError):
            benchmark.measure(metadata, 1)

    def test_feature_cases_are_explicit_and_independent(self):
        cases = {case[0]: tomllib.loads(benchmark.config_text(
            Path("/input"), Path("/output"), case, 30, 10)) for case in benchmark.CASES}
        baseline = cases["baseline"]
        self.assertEqual(baseline["pipeline"]["n_iterations"], 0)
        self.assertFalse(baseline["pipeline"]["compute_time_null"])
        self.assertFalse(baseline["pipeline"]["compute_detsplit_null"])
        self.assertNotIn("cosmic_rays", baseline["pipeline"])
        self.assertTrue(baseline["output"]["save_numpy"])
        self.assertFalse(baseline["output"]["save_psd"])
        self.assertTrue(cases["time_null"]["pipeline"]["compute_time_null"])
        self.assertFalse(cases["time_null"]["pipeline"]["compute_detsplit_null"])
        self.assertTrue(cases["det_null"]["pipeline"]["compute_detsplit_null"])
        self.assertIn("cosmic_rays", cases["cosmic"]["pipeline"])
        self.assertTrue(cases["plots"]["output"]["save_png"])
        self.assertTrue(cases["plots"]["output"]["save_psd"])


if __name__ == "__main__":
    unittest.main()
