import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np


class LabReaderTests(unittest.TestCase):
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
            stub.core = types.SimpleNamespace(G3Units=types.SimpleNamespace(s=1e8))
            cls.spt_patch = mock.patch.dict(sys.modules, {"spt3g": stub})
            cls.spt_patch.start()
        cls.reader = importlib.import_module("mapmaker.reader")
        cls.units_patch = None
        if not hasattr(cls.reader.core, "G3Units"):
            cls.units_patch = mock.patch.object(
                cls.reader.core, "G3Units", types.SimpleNamespace(s=1e8), create=True)
            cls.units_patch.start()

    @classmethod
    def tearDownClass(cls):
        if cls.units_patch:
            cls.units_patch.stop()
        if cls.spt_patch:
            cls.spt_patch.stop()
        cls.path_patch.stop()

    def write_block(self, f, block, *, comb=True):
        frame = f.create_group(f"timestream/{block}/frames/1")
        frame.attrs["frame_type"] = "Scan"
        data = frame.create_group("data")
        # Deliberately reorder the rows to exercise name-based IQ pairing.
        data["names"] = np.array([b"r0001_Q", b"r0000_I", b"r0001_I", b"r0000_Q"])
        data["data"] = np.array([
            [0., 0., 0.], [1., 1.5, 2.], [6., 6.5, 7.], [0., 0., 0.]])
        data["times"] = np.array([0, 100000000, 200000000], dtype=np.int64)
        frequency = np.array([[100., 101., 102., 103.], [200., 201., 202., 203.]])
        iq = np.tile([0., 1., 3., 6.], (2, 1)).astype(complex)
        f.create_dataset(f"targ/{block}/test_targ.npy", data=np.vstack([
            frequency.reshape(-1), iq.reshape(-1)]))
        if comb:
            f.create_dataset(
                f"config/{block}/combs/test_tstream_freq_comb.npy",
                data=np.array([101.25, 203.]))

    def read_frame(self, path):
        return next(self.reader._iter_h5_lab_frames([path]))

    def test_pairs_by_name_uses_probe_comb_and_qualifies_detectors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lab.h5"
            with h5py.File(path, "w") as f:
                self.write_block(f, "B1D1")
                self.write_block(f, "B1D2")
            with self.assertLogs("ccat_mapmaker", level="INFO") as logs:
                chunk = self.read_frame(path)
            self.assertEqual(chunk.kids, [
                "B1D1:r0000", "B1D1:r0001", "B1D2:r0000", "B1D2:r0001"])
            expected = np.array([
                [-0.25 / 101.25, 0.],
                [0., (0.5 / 3.) / 203.],
                [0.25 / 101.25, (1. / 3.) / 203.]])
            np.testing.assert_allclose(chunk.signal[:, :2], expected)
            np.testing.assert_allclose(chunk.signal[:, 2:], expected)
            self.assertEqual(chunk.flags.shape, chunk.signal.shape)
            self.assertTrue(any("tstream_freq_comb" in line for line in logs.output))

    def test_missing_comb_falls_back_to_maximum_sweep_derivative(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lab.h5"
            with h5py.File(path, "w") as f:
                self.write_block(f, "B1D1", comb=False)
            with self.assertLogs("ccat_mapmaker", level="WARNING") as logs:
                chunk = self.read_frame(path)
            self.assertIn("maximum IQ sweep derivative", logs.output[0])
            np.testing.assert_allclose(
                chunk.signal[:, 0], np.array([-2., -1.5, -1.]) / 3. / 102.)

    def test_rejects_bad_pairs_and_calibration(self):
        for defect in ("missing_q", "multiple_sweeps", "bad_tone", "zero_gradient"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "lab.h5"
                with h5py.File(path, "w") as f:
                    self.write_block(f, "B1D1")
                    if defect == "missing_q":
                        f["timestream/B1D1/frames/1/data/names"][0] = b"r0002_Q"
                    elif defect == "multiple_sweeps":
                        f.copy("targ/B1D1/test_targ.npy", "targ/B1D1/other_targ.npy")
                    elif defect == "bad_tone":
                        f["config/B1D1/combs/test_tstream_freq_comb.npy"][0] = 1000.
                    else:
                        sweep = f["targ/B1D1/test_targ.npy"]
                        sweep[1, 2] = sweep[1, 1]
                with self.assertRaises(ValueError):
                    self.read_frame(path)

    def test_hybrid_angle_uses_exact_probe_tone(self):
        freq = np.arange(100., 121.)
        sweep_theta = (freq - 110.) * 0.07
        observed_freq = np.array([110.5, 115., 118.])
        theta = (observed_freq - 110.) * 0.07
        converted, used_angle = self.reader.tod.iq_to_df_hybrid(
            np.cos(theta), np.sin(theta), np.cos(sweep_theta), np.sin(sweep_theta),
            freq, f_tone=110.5)
        np.testing.assert_array_equal(used_angle, [False, True, True])
        np.testing.assert_allclose(
            converted, (observed_freq - 110.5) / 110.5, atol=1e-14)

    def test_tone_between_sweep_points_uses_bracketing_gradient(self):
        frequency = np.array([100., 101., 102., 103.])
        sweep_i = np.array([0., 1., 3., 6.])
        result, _ = self.reader.tod.iq_to_df_hybrid(
            np.array([2.5, 3.]), np.zeros(2), sweep_i, np.zeros(4),
            frequency, f_tone=101.75)
        np.testing.assert_allclose(result, [0., 0.25 / 101.75])

    def test_rejects_changed_detector_list_between_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lab.h5"
            with h5py.File(path, "w") as f:
                self.write_block(f, "B1D1")
                self.write_block(f, "B1D2")
                f.copy("timestream/B1D1/frames/1", "timestream/B1D1/frames/2")
            frames = self.reader._iter_h5_lab_frames([path])
            next(frames)
            with self.assertRaisesRegex(ValueError, "detector list changed"):
                next(frames)

    def test_sample_first_frame_is_converted(self):
        sample_dir = Path(os.environ.get(
            "CCAT_MAPMAKER_LAB_SAMPLE_DIR",
            Path.home() / "notebooks/data/ql_mm_samples/lab/sample_h5"))
        paths = sorted(sample_dir.glob("*.h5")) if sample_dir.is_dir() else []
        if not paths:
            self.skipTest("Lab sample HDF5 is unavailable")
        chunk = self.read_frame(paths[0])
        self.assertEqual(len(chunk.kids), 8000)
        self.assertEqual(len(set(chunk.kids)), 8000)
        self.assertEqual(chunk.signal.shape[1], 8000)
        self.assertTrue(np.all(np.isfinite(chunk.signal)))
        self.assertEqual(chunk.kids[:2], ["B1D1:r0000", "B1D1:r0001"])
        with h5py.File(paths[0], "r") as f:
            frame = f["timestream/B1D1/frames/1/data/data"]
            sweep = next(iter(f["targ/B1D1"].values()))
            tone = next(
                ds[0] for name, ds in f["config/B1D1/combs"].items()
                if "tstream_freq_comb" in name)
            calibration = sweep[:, :sweep.shape[1] // 500]
            expected, _ = self.reader.tod.iq_to_df_hybrid(
                frame[0, :chunk.signal.shape[0]], frame[1, :chunk.signal.shape[0]],
                calibration[1].real, calibration[1].imag, calibration[0].real,
                f_tone=tone)
        np.testing.assert_allclose(chunk.signal[:, 0], expected)


if __name__ == "__main__":
    unittest.main()
