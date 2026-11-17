import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


class RunDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = Path(__file__).resolve().parents[1] / "ccat_mapmaker"
        cls.path_patch = mock.patch.object(sys, "path", [str(project), *sys.path])
        cls.path_patch.start()
        try:
            importlib.import_module("spt3g")
            cls.spt_patch = None
        except ModuleNotFoundError:
            stub = types.ModuleType("spt3g")
            stub.core = types.SimpleNamespace()
            cls.spt_patch = mock.patch.dict(sys.modules, {"spt3g": stub})
            cls.spt_patch.start()
        cls.runner = importlib.import_module("run_mapmaker")

    @classmethod
    def tearDownClass(cls):
        if cls.spt_patch:
            cls.spt_patch.stop()
        cls.path_patch.stop()

    def make_config(self, root):
        source = root / "input"
        source.mkdir()
        (source / "scan.h5").touch()
        config = root / "settings.toml"
        config.write_text(
            '[data]\ninput_dirs = ["input"]\nfile_format = "h5"\n'
            'format = "simulation"\n'
            '[map]\nra0_deg = 10.0\ndec0_deg = 20.0\n'
            'xlen_deg = 1.0\nylen_deg = 1.0\nres_arcmin = 15.0\n'
            '[pipeline]\nn_iterations = 1\nchunk_duration_s = 10.0\n'
            'auto_exclude_threshold = 0\nwnf_exclude_sigma = 0\n'
            'exclude_detector_indices = [1]\nweight_cap_factor = 0\n'
            'compute_time_null = false\ncompute_detsplit_null = false\n'
            'clean_steps = []\n[pipeline.cosmic_rays]\nsigma = 100.0\n'
            '[output]\noutput_dir = "results"\nobs_object = "test"\n'
            'save_png = false\nsave_numpy = true\n',
            encoding="utf-8",
        )
        return config, source

    def fake_chunks(self, files, *args, **kwargs):
        for _path in files:
            n = 128
            t = np.arange(n) / 20
            signal = np.column_stack((np.sin(t), np.cos(t), np.sin(2 * t)))
            signal[4, 2] = np.nan
            flags = np.zeros_like(signal, dtype=int)
            flags[0, 0] = 1
            yield self.runner.reader.Chunk(
                kids=["a", "b", "c"], signal=signal, common_mode=None,
                ra=np.full((n, 3), 10.0), dec=np.full((n, 3), 20.0),
                ra_bore=np.full(n, 10.0), dec_bore=np.full(n, 20.0),
                t_start=0.0, t_stop=float(t[-1]), sample_rate=20.0,
                chunk_index=0, flags=flags,
            )

    def test_run_saves_provenance_and_processing_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, source = self.make_config(root)
            with mock.patch.object(sys, "argv", ["run_mapmaker", "--config", str(config)]), \
                 mock.patch.object(self.runner.reader, "_iter_h5_simulation_chunks", self.fake_chunks), \
                 mock.patch.object(self.runner.tod, "cleanTod", wraps=self.runner.tod.cleanTod) as clean:
                self.runner.main()
            run, = (root / "results").iterdir()
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertEqual((run / "config.toml").read_bytes(), config.read_bytes())
            self.assertIn("Run completed", (run / "run.log").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["input_files_opened"], [str((source / "scan.h5").resolve())])
            self.assertIn("python", metadata["software"])
            self.assertIn("git", metadata["software"])
            self.assertEqual(metadata["n_detectors_used"], 2)
            self.assertEqual(metadata["n_detectors_excluded"], 1)
            self.assertEqual(metadata["preparation_processing"]["detector_samples"], 384)
            self.assertEqual(metadata["manual_excluded_detectors"], ["b"])
            self.assertEqual(metadata["pipeline_clean_steps"], ["gap_fill", "cosmic_rays"])
            self.assertEqual(metadata["active_cleaning_steps"][1]["parameters"]["sigma"], 100)
            self.assertEqual(metadata["resolved_settings"]["pipeline"]["wnf_exclude_sigma"], 0)
            stats = metadata["pass_processing"]["naive + it_0"]
            self.assertEqual(stats["chunk_samples"], [128])
            self.assertEqual(stats["detector_samples"], 256)
            self.assertEqual(stats["input_flagged_samples"], 1)
            self.assertEqual(stats["gap_flagged_samples"], 1)
            self.assertEqual(stats["combined_flagged_samples"], 2)
            self.assertTrue(all(call.args[2].shape == (2,) for call in clean.call_args_list))
            self.assertEqual(metadata["map_quality"]["covered_pixels"], 1)
            self.assertEqual(metadata["convergence"]["map_diff_rms"][0], None)
            self.assertIn("preparation", metadata["stage_times_s"])
            self.assertIn("it_1", metadata["pass_processing"])

    def test_failed_preparation_retains_config_log_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, source = self.make_config(root)
            (source / "scan.h5").unlink()
            with mock.patch.object(sys, "argv", ["run_mapmaker", "--config", str(config)]):
                with self.assertRaisesRegex(FileNotFoundError, "No .h5 files found"):
                    self.runner.main()
            run, = (root / "results").iterdir()
            self.assertEqual((run / "config.toml").read_bytes(), config.read_bytes())
            self.assertIn("No .h5 files found", (run / "run.log").read_text())
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"]["type"], "FileNotFoundError")

    def test_detector_noise_exclusion_removes_indices(self):
        result = self.runner.tod.excludedDetectors(
            ["a", "b", "c"], np.array([1., 1., 20.]), None,
            {"auto_exclude_threshold": 5, "wnf_exclude_sigma": 0})
        self.assertEqual(result["keep_idx"].tolist(), [0, 1])
        self.assertEqual(result["noise_excluded"], ["c"])

    def test_hdf5_reader_records_input_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scan.h5"
            focalplane = np.zeros(2, dtype=[("name", "S1"), ("quat", "f8", (4,))])
            focalplane["name"] = [b"a", b"b"]
            focalplane["quat"] = [0, 0, 0, 1]
            with h5py.File(path, "w") as data:
                data["instrument/focalplane"] = focalplane
                data["detdata/signal"] = np.ones((2, 8))
                data["detdata/flags"] = np.zeros((2, 8), dtype=int)
                data["shared/times"] = np.arange(8, dtype=float)
                data["shared/boresight_radec"] = np.full((8, 4), 0.5)
            cfg = {
                "data": {"input_dirs": [temp], "file_format": "h5", "format": "simulation"},
                "pipeline": {"chunk_duration_s": 2.0},
            }
            chunks = list(self.runner.reader.iter_chunks(cfg))
            self.assertEqual(sum(chunk.signal.shape[0] for chunk in chunks), 8)
            self.assertEqual(cfg["_input_files_discovered"], [str(path.resolve())])
            self.assertEqual(cfg["_input_files_opened"], [str(path.resolve())])

    def test_white_noise_exclusion_removes_indices(self):
        result = self.runner.tod.excludedDetectors(
            ["a", "b", "c"], np.ones(3), np.array([1., 1., 100.]),
            {"auto_exclude_threshold": 0, "wnf_exclude_sigma": 1})
        self.assertEqual(result["keep_idx"].tolist(), [0, 1])
        self.assertEqual(result["wnf_excluded"], ["c"])


if __name__ == "__main__":
    unittest.main()
