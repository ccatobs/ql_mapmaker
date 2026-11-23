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
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from astropy.utils import iers
import astropy.units as u


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
        cls.iers_download = iers.conf.set_temp("auto_download", False)
        cls.iers_download.__enter__()
        cls.iers_age = iers.conf.set_temp("auto_max_age", None)
        cls.iers_age.__enter__()
        cls.units_patch = None
        if not hasattr(cls.reader.core, "G3Units"):
            cls.units_patch = mock.patch.object(
                cls.reader.core, "G3Units", types.SimpleNamespace(s=1e8), create=True)
            cls.units_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.iers_age.__exit__(None, None, None)
        cls.iers_download.__exit__(None, None, None)
        if cls.units_patch:
            cls.units_patch.stop()
        if cls.spt_patch:
            cls.spt_patch.stop()
        cls.path_patch.stop()

    def write_block(self, f, block, *, comb=True):
        if "hk" not in f:
            frame = f.create_group("hk/frames/0")
            frame.attrs["address"] = "observatory.acu.feeds.acu_udp_stream"
            hk = frame.create_group("blocks/0")
            hk["times"] = np.arange(-1, 8, dtype=np.int64) * 100000000
            hk["Azimuth"] = np.full(9, 20.)
            hk["Elevation"] = np.full(9, 21.)
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
        return next(self.reader._iter_h5_lab_chunks(
            [path], 0, self.reader.coord.FYST_SITE))

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

    def test_filtering_handles_bounded_gaps_and_rejects_unfillable_gaps(self):
        samples = np.tile(np.sin(np.arange(32.) / 4)[:, None], (1, 2))
        samples[10, 0] = np.nan
        filtered, flags = self.reader.tod.cleanTod(
            samples, 16., np.zeros(2), {"highpass": {"cutoff_hz": .5}})
        self.assertTrue(np.all(np.isfinite(filtered)))
        self.assertEqual(flags[10, 0], self.reader.tod.GAP_FLAG)
        samples[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "unfillable gap"):
            self.reader.tod.cleanTod(
                samples, 16., np.zeros(2), {"highpass": {"cutoff_hz": .5}})

    def test_rejects_changed_detector_list_between_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lab.h5"
            with h5py.File(path, "w") as f:
                self.write_block(f, "B1D1")
                f.copy("timestream/B1D1/frames/1", "timestream/B1D1/frames/2")
                f["timestream/B1D1/frames/2/data/names"][0] = b"r0002_Q"
            with self.assertRaisesRegex(ValueError, "IQ channel names changed"):
                self.read_frame(path)

    def test_alignment_crosses_frame_boundaries_and_flags_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lab.h5"
            with h5py.File(path, "w") as f:
                self.write_block(f, "B1D1")
                self.write_block(f, "B1D2")
                for block, frames in (
                    ("B1D1", [(1, [0, 1, 2, 3]), (2, [4, 5, 6])]),
                    ("B1D2", [(1, [.5, 1.5]), (2, [2.5, 4.5, 5.5, 6.5])]),
                ):
                    f.copy(f"timestream/{block}/frames/1",
                           f"timestream/{block}/frames/2")
                    for frame_no, seconds in frames:
                        group = f[f"timestream/{block}/frames/{frame_no}/data"]
                        del group["data"]
                        del group["times"]
                        times = np.asarray(seconds)
                        group["times"] = (times * 100000000).astype(np.int64)
                        group["data"] = np.array([
                            np.zeros(len(times)), 1.5 + .2 * times,
                            np.full(len(times), 6.), np.zeros(len(times))])
            chunks = list(self.reader._iter_h5_lab_chunks(
                [path], 2., self.reader.coord.FYST_SITE))
            self.assertEqual([len(c.signal) for c in chunks], [2, 2, 2])
            self.assertAlmostEqual(chunks[0].t_start, .5)
            self.assertAlmostEqual(chunks[-1].t_stop, 6.5)
            sig = np.concatenate([c.signal for c in chunks])
            flags = np.concatenate([c.flags for c in chunks])
            expected = .1 * np.arange(.5, 6., 1.) / 101.25
            np.testing.assert_allclose(sig[:, 0], expected)
            np.testing.assert_allclose(sig[[0, 1, 2, 4, 5], 2], expected[[0, 1, 2, 4, 5]])
            self.assertTrue(np.isnan(sig[3, 2]))
            self.assertTrue(np.all(flags[3, 2:] == self.reader.tod.GAP_FLAG))
            self.assertFalse(flags[:, :2].any())

    def test_pointing_is_interpolated_and_transformed_to_icrs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lab.h5"
            with h5py.File(path, "w") as f:
                self.write_block(f, "B1D1")
                hk = f["hk/frames/0/blocks/0"]
                hk["Azimuth"][:] = [359., 359., 1., 1., 1., 1., 1., 1., 1.]
                hk["Elevation"][:] = [21., 21., 23., 23., 23., 23., 23., 23., 23.]
                status = f.create_group("hk/frames/1")
                status.attrs["address"] = "observatory.acu.feeds.acu_status"
                block = status.create_group("blocks/0")
                block["times"] = np.arange(-1, 8, dtype=np.int64) * 100000000
                block["Azimuth_current_position"] = np.full(9, 100.)
                block["Elevation_current_position"] = np.full(9, 45.)
                block["Azimuth"] = np.full(9, 100.)
                block["Elevation"] = np.full(9, 45.)
                f["timestream/B1D1/frames/1/data/times"][:] = [0, 50000000, 100000000]
            chunk = self.read_frame(path)
            expected = SkyCoord(
                az=np.array([359., 0., 1.]) * u.deg,
                alt=np.array([21., 22., 23.]) * u.deg,
                frame=AltAz(obstime=Time([0., .5, 1.], format="unix", scale="utc"),
                            location=self.reader.coord.FYST_SITE, pressure=0 * u.hPa),
            ).icrs
            np.testing.assert_allclose(chunk.ra_bore, expected.ra.deg, atol=1e-10)
            np.testing.assert_allclose(chunk.dec_bore, expected.dec.deg, atol=1e-10)

    def test_rejects_missing_or_gapped_pointing(self):
        for defect in ("missing", "wrong_address", "incomplete", "missing_times",
                       "duplicate_times", "gap"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "lab.h5"
                with h5py.File(path, "w") as f:
                    self.write_block(f, "B1D1")
                    if defect == "missing":
                        del f["hk"]
                    elif defect == "wrong_address":
                        f["hk/frames/0"].attrs["address"] = "observatory.acu.feeds.acu_status"
                    elif defect == "incomplete":
                        del f["hk/frames/0/blocks/0/Elevation"]
                    elif defect == "missing_times":
                        del f["hk/frames/0/blocks/0/times"]
                    elif defect == "duplicate_times":
                        hk = f["hk/frames/0/blocks/0"]
                        hk["times"][1] = hk["times"][0]
                    else:
                        hk = f["hk/frames/0/blocks/0"]
                        del hk["times"]
                        del hk["Azimuth"]
                        del hk["Elevation"]
                        hk["times"] = np.array([-1, 0, 3, 4], dtype=np.int64) * 100000000
                        hk["Azimuth"] = np.full(4, 20.)
                        hk["Elevation"] = np.full(4, 21.)
                with self.assertRaisesRegex(ValueError, "housekeeping"):
                    self.read_frame(path)

    def test_udp_pointing_interpolates_across_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lab.h5"
            with h5py.File(path, "w") as f:
                self.write_block(f, "B1D1")
                frame = f["hk/frames/0"]
                block = frame["blocks/0"]
                del block["times"]
                del block["Azimuth"]
                del block["Elevation"]
                block["times"] = np.array([-1, 0], dtype=np.int64) * 100000000
                block["Azimuth"] = [19., 20.]
                block["Elevation"] = [20., 21.]
                f.copy(frame, "hk/frames/1")
                block = f["hk/frames/1/blocks/0"]
                block["times"][:] = [100000000, 200000000]
                block["Azimuth"][:] = [21., 22.]
                block["Elevation"][:] = [22., 23.]
                f["timestream/B1D1/frames/1/data/times"][:] = [0, 50000000, 100000000]
            chunk = self.read_frame(path)
            expected = SkyCoord(
                az=np.array([20., 20.5, 21.]) * u.deg,
                alt=np.array([21., 21.5, 22.]) * u.deg,
                frame=AltAz(obstime=Time([0., .5, 1.], format="unix", scale="utc"),
                            location=self.reader.coord.FYST_SITE, pressure=0 * u.hPa),
            ).icrs
            np.testing.assert_allclose(chunk.ra_bore, expected.ra.deg, atol=1e-10)
            np.testing.assert_allclose(chunk.dec_bore, expected.dec.deg, atol=1e-10)

    def test_lab_requires_explicit_site(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lab.h5"
            with h5py.File(path, "w") as f:
                self.write_block(f, "B1D1")
            cfg = {"data": {"input_dirs": [temp], "file_format": "h5", "format": "lab"},
                   "pipeline": {"chunk_duration_s": 1.}, "map": {}}
            with self.assertRaisesRegex(ValueError, "explicit map.site"):
                next(self.reader.iter_chunks(cfg))

    def test_explicit_site_coordinates_are_used(self):
        cfg = {"map": {"site_lat_deg": 42., "site_lon_deg": -71., "site_height_m": 100.}}
        site = self.reader.coord.resolveSite(cfg)
        self.assertAlmostEqual(site.lat.deg, 42.)
        self.assertAlmostEqual(site.lon.deg, -71.)
        self.assertAlmostEqual(site.height.to_value(u.m), 100.)
        with self.assertRaisesRegex(ValueError, "all three"):
            self.reader.coord.resolveSite({"map": {"site_lat_deg": 42.}})

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
        self.assertTrue(np.all((chunk.ra_bore >= 0) & (chunk.ra_bore < 360)))
        self.assertTrue(np.all((chunk.dec_bore >= -90) & (chunk.dec_bore <= 90)))
        with h5py.File(paths[0], "r") as f:
            origin = max(int(f[f"timestream/{block}/frames/1/data/times"][0])
                         for block in f["timestream"])
            data = f["timestream/B1D1/frames/1/data"]
            times = data["times"][:] - origin
            right = np.searchsorted(times, 0)
            sweep = next(iter(f["targ/B1D1"].values()))
            calibration = sweep[:, :sweep.shape[1] // 500]
            tone = next(ds[0] for name, ds in f["config/B1D1/combs"].items()
                        if "tstream_freq_comb" in name)
            native, _ = self.reader.tod.iq_to_df_hybrid(
                data["data"][0, right-1:right+1], data["data"][1, right-1:right+1],
                calibration[1].real, calibration[1].imag, calibration[0].real,
                f_tone=tone)
            expected = np.interp(0, times[right-1:right+1], native)
        self.assertAlmostEqual(chunk.signal[0, 0], expected, places=14)


if __name__ == "__main__":
    unittest.main()
