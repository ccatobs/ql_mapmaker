import importlib
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from scipy.signal import welch


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
            self.assertEqual(metadata["null_products"], {})
            self.assertFalse((run / "null.npy").exists())

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

    def test_weighted_nulls_and_missing_split_coverage(self):
        state = {
            "nx": 3, "ny": 1, "keep_idx": np.arange(2),
            "det_offsets_kept": np.zeros(2), "det_noise": np.ones(2),
            "ra_edges": np.arange(4, dtype=float), "dec_edges": np.array([0., 1.]),
        }
        cfg = {"pipeline": {"compute_time_null": True, "compute_detsplit_null": True}}

        def chunks():
            for index, signal, ra, flags in (
                (0, [[10, 14], [18, 26], [4, 999]],
                 [[.5, .5], [1.5, 1.5], [2.5, 2.5]], [[0, 0], [0, 0], [0, 1]]),
                (1, [[8, 12], [5, 7], [0, 0]],
                 [[.5, .5], [1.5, 1.5], [3.5, 3.5]], [[0, 0], [0, 1], [0, 0]]),
            ):
                yield self.runner.reader.Chunk(
                    kids=["a", "b"], signal=np.array(signal, dtype=float),
                    common_mode=None, ra=np.array(ra), dec=np.full((3, 2), .5),
                    ra_bore=np.full(3, .5), dec_bore=np.full(3, .5),
                    t_start=float(index), t_stop=float(index + 1), sample_rate=3.,
                    chunk_index=index, flags=np.array(flags),
                )

        with mock.patch.object(self.runner.reader, "iter_chunks", side_effect=lambda cfg: chunks()):
            result = self.runner._mapPass(
                cfg, state, common_mode=False, weights=np.array([1., 3.]))
            without_nulls = self.runner._mapPass(
                cfg, state, common_mode=False, weights=np.array([1., 3.]),
                compute_nulls=False)
            detector_only = self.runner._mapPass(
                {"pipeline": {"compute_time_null": False, "compute_detsplit_null": True}},
                state, common_mode=False, weights=np.array([1., 3.]))
        np.testing.assert_allclose(result["null_map"], [[1., 9.5, np.nan]], equal_nan=True)
        sign = 1 if result["detector_split_assignment"][0] == 0 else -1
        np.testing.assert_allclose(
            result["detsplit_null_map"], [[-2. * sign, -7.25 * sign, np.nan]],
            equal_nan=True)
        np.testing.assert_allclose(result["time_split_weights"], [[[4., 4., 1.]], [[4., 1., 0.]]])
        np.testing.assert_allclose(result["combined"], without_nulls["combined"], equal_nan=True)
        self.assertIsNone(without_nulls["null_map"])
        self.assertIsNone(without_nulls["detsplit_null_map"])
        self.assertIsNone(detector_only["null_map"])
        np.testing.assert_allclose(
            detector_only["detsplit_null_map"], result["detsplit_null_map"], equal_nan=True)

        def constant_chunks():
            for chunk in chunks():
                chunk.signal[:] = 10
                yield chunk

        with mock.patch.object(self.runner.reader, "iter_chunks",
                               side_effect=lambda cfg: constant_chunks()):
            constant = self.runner._mapPass(
                cfg, state, common_mode=False, weights=np.array([1., 3.]))
        for key in ("null_map", "detsplit_null_map"):
            np.testing.assert_allclose(constant[key][np.isfinite(constant[key])], 0.)

    def test_no_time_overlap_is_reported(self):
        cfg = {"pipeline": {"compute_time_null": True, "compute_detsplit_null": False}}
        state = {"nx": 1, "ny": 1, "keep_idx": np.arange(2),
                 "det_offsets_kept": np.zeros(2), "det_noise": np.ones(2),
                 "ra_edges": np.array([0., 1.]), "dec_edges": np.array([0., 1.])}
        chunk = self.runner.reader.Chunk(
            kids=["a", "b"], signal=np.ones((2, 2)), common_mode=None,
            ra=np.full((2, 2), .5), dec=np.full((2, 2), .5),
            ra_bore=np.full(2, .5), dec_bore=np.full(2, .5),
            t_start=0., t_stop=1., sample_rate=2., chunk_index=0,
            flags=np.zeros((2, 2), dtype=int),
        )
        with mock.patch.object(self.runner.reader, "iter_chunks", return_value=iter([chunk])), \
             self.assertLogs("ccat_mapmaker", level="WARNING") as logs:
            result = self.runner._mapPass(cfg, state, common_mode=False, weights=np.ones(2))
        self.assertTrue(np.isnan(result["null_map"]).all())
        self.assertIn("no finite overlap", logs.output[0])

    def test_detector_null_requires_two_detectors(self):
        cfg = {"pipeline": {"compute_time_null": False, "compute_detsplit_null": True}}
        state = {"nx": 1, "ny": 1, "det_offsets_kept": np.zeros(1)}
        with self.assertRaisesRegex(ValueError, "at least two retained detectors"):
            self.runner._mapPass(cfg, state)

    def test_psd_comparison_uses_matched_clean_segments(self):
        sample_rate = 128.
        t = np.arange(4096) / sample_rate
        common = np.sin(2 * np.pi * 8 * t)
        differential = 0.3 * np.sin(2 * np.pi * 20 * t)
        before = np.column_stack((common + differential, common - differential))
        after = before - np.mean(before, axis=1, keepdims=True)
        flags = np.ones_like(before)
        flags[0, 0] = np.nan
        acc = self.runner._PsdComparison(["a", "b"])
        acc.add(before, after, flags, sample_rate)
        info, result = acc.result()
        self.assertEqual(info["segment_counts"], [2, 3])
        self.assertEqual(info["nperseg"], 2048)
        self.assertEqual(info["candidate_segments"], 3)
        freq, expected = welch(before[1024:3072, 0], fs=sample_rate, nperseg=2048)
        np.testing.assert_allclose(result["frequency_hz"], freq)
        np.testing.assert_allclose(result["before"][:, 0], expected, atol=1e-12)
        at_8 = np.argmin(abs(freq - 8))
        at_20 = np.argmin(abs(freq - 20))
        self.assertGreater(result["before"][at_8, 0], 1.)
        self.assertLess(result["after"][at_8, 0], 1e-20)
        self.assertAlmostEqual(result["before"][at_20, 0], result["after"][at_20, 0])

    def test_psd_comparison_reports_no_clean_segments(self):
        acc = self.runner._PsdComparison(["a"])
        acc.add(np.ones((8, 1)), np.ones((8, 1)), np.full((8, 1), np.nan), 10.)
        info, result = acc.result()
        self.assertIsNone(result)
        self.assertEqual(info["segment_counts"], [0])
        self.assertIn("unavailable", info["status"])
        acc.add(np.ones((8, 1)), np.ones((8, 1)), np.ones((8, 1)), 11.)
        info, result = acc.result()
        self.assertIsNone(result)
        self.assertEqual(info["skipped_sample_rate_chunks"], 1)

    def test_psd_diagnostics_saved_without_map_pngs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = self.make_config(root)

            def clean_chunks(files, *args, **kwargs):
                for chunk in self.fake_chunks(files, *args, **kwargs):
                    yield replace(chunk, signal=np.nan_to_num(chunk.signal),
                                  flags=np.zeros_like(chunk.flags))

            with mock.patch.object(sys, "argv", ["run_mapmaker", "--config", str(config)]), \
                 mock.patch.object(self.runner.reader, "_iter_h5_simulation_chunks", clean_chunks):
                self.runner.main()
            run, = (root / "results").iterdir()
            metadata = json.loads((run / "metadata.json").read_text())
            info = metadata["psd_diagnostics"]
            self.assertEqual(info["status"], "available")
            self.assertEqual(info["weighting_psd_status"], "available")
            self.assertEqual(info["stage"], "it_1")
            self.assertEqual(info["segment_counts"], [1, 1])
            self.assertTrue((run / "psd.png").is_file())
            self.assertTrue((run / "weighting_psd.png").is_file())
            with np.load(run / "psd.npz") as data:
                self.assertEqual(data["before"].shape, (65, 2))
                self.assertEqual(data["after"].shape, (65, 2))
                self.assertEqual(data["detectors"].tolist(), ["a", "c"])
            with np.load(run / "weighting_psd.npz") as data:
                self.assertEqual(data["white_noise_floor"].shape, (3,))
            self.assertFalse((run / "it_1.png").exists())

            config.write_text(config.read_text() + "save_psd = false\n")
            with mock.patch.object(sys, "argv", ["run_mapmaker", "--config", str(config)]), \
                 mock.patch.object(self.runner.reader, "_iter_h5_simulation_chunks", clean_chunks):
                self.runner.main()
            without_psd, = set((root / "results").iterdir()) - {run}
            other_metadata = json.loads((without_psd / "metadata.json").read_text())
            self.assertEqual(other_metadata["psd_diagnostics"]["status"], "disabled")
            self.assertFalse((without_psd / "psd.png").exists())
            for name in ("naive.npy", "it_0.npy", "it_1.npy"):
                np.testing.assert_array_equal(np.load(run / name), np.load(without_psd / name))

    def test_prep_psd_reports_ignored_chunks_without_changing_estimator(self):
        state = self.runner
        cfg = {"data": {}, "pipeline": {}}
        t = np.arange(16, dtype=float)
        chunks = [
            state.reader.Chunk(
                kids=["a"], signal=(np.sin(t[:n]) * amplitude)[:, None],
                common_mode=None, ra=np.full((n, 1), .5), dec=np.full((n, 1), .5),
                ra_bore=np.full(n, .5), dec_bore=np.full(n, .5),
                t_start=float(i * 16), t_stop=float(i * 16 + n - 1),
                sample_rate=16., chunk_index=i, flags=np.zeros((n, 1), dtype=int),
            )
            for i, (n, amplitude) in enumerate(((16, 1.), (12, 10.)))
        ]
        expected = state._fastPeriodogram(
            chunks[0].signal, np.hanning(16)[:, None], 16 * np.sum(np.hanning(16)**2),
            np.nanmedian(chunks[0].signal, axis=0), 16)
        with mock.patch.object(state.reader, "iter_chunks", return_value=iter(chunks)), \
             self.assertLogs("ccat_mapmaker", level="WARNING") as logs:
            result = state._prepPass(cfg)
        np.testing.assert_allclose(result["psd_avg"], expected)
        self.assertEqual(result["processing_stats"]["psd_time_samples_used"], 16)
        self.assertEqual(result["processing_stats"]["psd_eligible_time_samples"], 28)
        self.assertIn("16 of 28", logs.output[0])

    def test_saved_nulls_come_from_last_pass(self):
        for iterations in (0, 1):
            with self.subTest(iterations=iterations), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config, _ = self.make_config(root)
                text = config.read_text().replace("n_iterations = 1", f"n_iterations = {iterations}")
                text = text.replace("compute_time_null = false", "compute_time_null = true")
                text = text.replace("compute_detsplit_null = false", "compute_detsplit_null = true")
                config.write_text(text)

                def two_chunks(files, *args, **kwargs):
                    for chunk in self.fake_chunks(files, *args, **kwargs):
                        yield chunk
                        yield replace(chunk, signal=chunk.signal * 2, chunk_index=1,
                                      t_start=chunk.t_stop, t_stop=2 * chunk.t_stop)

                calls = []
                original = self.runner._mapPass

                def capture(*args, **kwargs):
                    result = original(*args, **kwargs)
                    calls.append((kwargs["compute_nulls"], result))
                    return result

                with mock.patch.object(sys, "argv", ["run_mapmaker", "--config", str(config)]), \
                     mock.patch.object(self.runner.reader, "_iter_h5_simulation_chunks", two_chunks), \
                     mock.patch.object(self.runner, "_mapPass", side_effect=capture):
                    self.runner.main()
                run, = (root / "results").iterdir()
                metadata = json.loads((run / "metadata.json").read_text())
                self.assertEqual([enabled for enabled, _ in calls], [False] * iterations + [True])
                for filename, key in (("null.npy", "null_map"),
                                      ("detsplit_null.npy", "detsplit_null_map")):
                    actual = np.load(run / filename)
                    self.assertGreater(np.isfinite(actual).sum(), 0)
                    np.testing.assert_allclose(actual, calls[-1][1][key], equal_nan=True)
                np.testing.assert_allclose(
                    np.load(run / "time_split_weight_0.npy"),
                    calls[-1][1]["time_split_weights"][0])
                for kind in ("time", "detector"):
                    self.assertEqual(metadata["null_products"][kind]["stage"], f"it_{iterations}")
                    self.assertGreater(metadata["null_products"][kind]["overlap_finite_pixels"], 0)
                groups = metadata["null_products"]["detector"]["groups"]
                self.assertEqual(sorted(len(group) for group in groups), [1, 1])


if __name__ == "__main__":
    unittest.main()
