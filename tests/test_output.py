import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from ccat_mapmaker.mapmaker import output
from ccat_mapmaker.mapmaker import map as map_module


class ConvergenceDiagnosticsTests(unittest.TestCase):
    def test_empty_maps_produce_nan_metrics_without_warnings(self):
        empty_map = np.full((2, 2), np.nan)
        hits = np.ones((2, 2))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = output.compute_convergence_metrics(
                empty_map, [("cm", empty_map)], hits)

        self.assertEqual(metrics["labels"], ["naive", "cm"])
        for key in ("peak", "off_src_rms", "on_src_rms"):
            self.assertTrue(all(np.isnan(value) for value in metrics[key]))
        self.assertTrue(all(np.isnan(value) for value in metrics["map_diff_rms"]))
        self.assertEqual(caught, [])

    def test_metrics_use_only_finite_pixels(self):
        naive = np.array([[1.0, 2.0], [3.0, np.nan]])
        final = np.array([[2.0, 4.0], [np.nan, 10.0]])
        hits = np.ones((2, 2))

        metrics = output.compute_convergence_metrics(naive, [("cm", final)], hits)

        self.assertEqual(metrics["peak"], [3.0, 10.0])
        self.assertEqual(metrics["off_src_rms"], [1.0, 2.0])
        self.assertAlmostEqual(metrics["map_diff_rms"][1], np.sqrt(2.5))

    def test_all_nan_metrics_still_produce_diagnostics_plot(self):
        metrics = {
            "labels": ["naive"],
            "peak": [np.nan],
            "off_src_rms": [np.nan],
            "on_src_rms": [np.nan],
            "map_diff_rms": [np.nan],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "diagnostics.png"
            output.plot_diagnostics(metrics, [], filepath)

            self.assertTrue(filepath.is_file())
            self.assertGreater(filepath.stat().st_size, 0)

    def test_constant_final_map_handles_empty_off_source_region(self):
        constant_map = np.ones((2, 2))
        hits = np.ones((2, 2))

        metrics = output.compute_convergence_metrics(constant_map, [], hits)

        self.assertTrue(np.isnan(metrics["off_src_rms"][0]))
        self.assertEqual(metrics["on_src_rms"][0], 1.0)


class MapCoverageTests(unittest.TestCase):
    def test_map_with_hits_passes_coverage_check(self):
        hits = np.array([[0.0, 1.0], [0.0, 0.0]])

        map_module.requireMapCoverage(
            hits, np.array([10.0, 11.0, 12.0]),
            np.array([-2.0, -1.0, 0.0]))

    def test_map_without_hits_reports_bounds_and_possible_causes(self):
        hits = np.zeros((2, 2))

        with self.assertRaisesRegex(
                ValueError,
                r"No usable TOD samples landed.*RA 10\.000000 to 12\.000000 "
                r"deg.*Dec -2\.000000 to 0\.000000 deg.*map center and extent"):
            map_module.requireMapCoverage(
                hits, np.array([10.0, 11.0, 12.0]),
                np.array([-2.0, -1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
