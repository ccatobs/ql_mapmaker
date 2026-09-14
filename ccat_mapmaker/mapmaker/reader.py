# ============================================================================ #
# reader.py
#
# James Burgoyne, jburgoyne@phas.ubc.ca
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT August 2026
# ============================================================================ #
"""
Reads .g3 and .h5 files and yields fixed-duration Chunk objects for the
pipeline. Three source formats, three ingestion paths:

  g3 simulation (TOAST output):
    Calibration frame (first): focal-plane detector names + offset quats.
    Scan frames (many):        compressed signal + boresight quaternions.
    G3 is a streamed format with no random access, so frames have variable
    sample counts and get rebuffered into fixed-duration chunks by _rechunk.

  h5 simulation (TOAST output):
    Same content as g3 simulation, but HDF5 datasets support direct index
    slicing so chunks are read straight off disk (_iter_h5_simulation_chunks),
    no frame buffering needed.

  g3 blasttng (real BLAST-TNG data):
    Calibration frame: per-KID target sweeps (+ optional cal-lamp exposure).
    Scan frames: raw I/Q, converted to df against each KID's sweep.
    Also streamed, so also goes through _rechunk.

All arrays are time-first: (n_samps, n_dets).
"""


import io
import glob
import pathlib
from typing import Iterator, Optional

import numpy as np
import h5py
from dataclasses import dataclass
from spt3g import core

from .pointing import precompute_det_directions, boresight_to_radec, det_radec_from_boresight
from .signal import iq_to_df, iq_to_df_hybrid, normalize_tod


# ============================================================================ #
# Chunk
# ============================================================================ #
@dataclass
class Chunk:
    """
    All detector and pointing data for one fixed-duration time window.
    All 2-D arrays are time-first: shape (n_samps, n_dets).

    kids[j] is the name of the detector in column j of signal, ra, dec.
    t_start / t_stop are in seconds since the G3 epoch (Jan 1 2001).
    common_mode starts as None; populated downstream if needed.

    flags is a per-(sample, detector) bitmask: 0 = good, nonzero = flagged.
    Different flagging features OR their own bit into a sample's value, so a
    sample can carry more than one reason at once; consumers that don't care
    which reason should test `flags != 0`, not `flags == 1`.
    """
    kids:         list
    signal:       np.ndarray          # (n_samps, n_dets)
    common_mode:  Optional[np.ndarray]# (n_samps,) or None
    ra:           Optional[np.ndarray]# (n_samps, n_dets) deg; None if offsets not applied
    dec:          Optional[np.ndarray]# (n_samps, n_dets) deg; None if offsets not applied
    ra_bore:      np.ndarray          # (n_samps,) deg
    dec_bore:     np.ndarray          # (n_samps,) deg
    t_start:      float
    t_stop:       float
    sample_rate:  float               # Hz
    chunk_index:  int
    flags:        np.ndarray          # (n_samps, n_dets) int bitmask; 0 = good
    boresight_q:  Optional[np.ndarray] = None  # (n_samps, 4) scipy (x,y,z,w); for pointing reconstruction
    det_dirs:     Optional[dict] = None  # {name: (3,) array}; offset directions, for on-demand offset application


# ── Frame buffering (shared by streamed g3 formats) ─────────────────────────
#
# Both g3 ingestion paths (simulation and blasttng) below stream through this
# to turn variable-length frames into fixed-duration Chunks. The h5
# simulation path doesn't use it -- HDF5 datasets are randomly indexable, so
# chunks are sliced directly from disk instead of buffered in memory.

# ============================================================================ #
# _rechunk
# ============================================================================ #
def _rechunk(frame_iter: Iterator[Chunk], chunk_duration_s: float) -> Iterator[Chunk]:
    """
    Rebuffer variable-length frame-level Chunks into fixed-duration Chunks.

    Accumulates frames until chunk_n samples are ready, then yields and
    advances the buffer. The final chunk may be shorter than chunk_n.
    If chunk_duration_s <= 0, yields each frame unchanged (frame-aligned mode).
    """
    buf_sig = buf_ra = buf_dec = buf_rb = buf_db = buf_bq = buf_flags = None
    kids = sample_rate = t_start = det_dirs = None
    chunk_n     = None
    chunk_index = 0

    for frame in frame_iter:
        if kids is None:
            kids        = frame.kids
            sample_rate = frame.sample_rate
            t_start     = frame.t_start
            det_dirs    = frame.det_dirs
            if chunk_duration_s > 0:
                chunk_n = max(1, int(chunk_duration_s * sample_rate))
        if chunk_n is None:
            yield Chunk(
                kids=frame.kids, signal=frame.signal, common_mode=None,
                ra=frame.ra, dec=frame.dec,
                ra_bore=frame.ra_bore, dec_bore=frame.dec_bore,
                boresight_q=frame.boresight_q,
                det_dirs=frame.det_dirs,
                t_start=frame.t_start, t_stop=frame.t_stop, flags=frame.flags,
                sample_rate=frame.sample_rate, chunk_index=chunk_index,
            )
            chunk_index += 1
            continue

        buf_sig = frame.signal      if buf_sig is None else np.concatenate([buf_sig, frame.signal],      axis=0)
        if frame.ra is not None:
            buf_ra  = frame.ra  if buf_ra  is None else np.concatenate([buf_ra,  frame.ra],  axis=0)
            buf_dec = frame.dec if buf_dec is None else np.concatenate([buf_dec, frame.dec], axis=0)
        buf_rb  = frame.ra_bore     if buf_rb  is None else np.concatenate([buf_rb,  frame.ra_bore])
        buf_db  = frame.dec_bore    if buf_db  is None else np.concatenate([buf_db,  frame.dec_bore])
        buf_bq  = frame.boresight_q if buf_bq  is None else np.concatenate([buf_bq,  frame.boresight_q], axis=0)
        buf_flags = frame.flags     if buf_flags is None else np.concatenate([buf_flags, frame.flags], axis=0)

        assert kids is not None and sample_rate is not None and t_start is not None
        while buf_sig.shape[0] >= chunk_n:
            t_stop = t_start + chunk_n / sample_rate
            yield Chunk(
                kids=kids, signal=buf_sig[:chunk_n], common_mode=None,
                ra=buf_ra[:chunk_n] if buf_ra is not None else None,
                dec=buf_dec[:chunk_n] if buf_dec is not None else None,
                ra_bore=buf_rb[:chunk_n], dec_bore=buf_db[:chunk_n],
                boresight_q=buf_bq[:chunk_n] if buf_bq is not None else None ,
                det_dirs=det_dirs,
                t_start=t_start, t_stop=t_stop, flags=buf_flags[:chunk_n],
                sample_rate=sample_rate, chunk_index=chunk_index,
            )
            chunk_index += 1
            t_start = t_stop
            buf_sig = buf_sig[chunk_n:]
            buf_ra  = buf_ra[chunk_n:]  if buf_ra  is not None else None
            buf_dec = buf_dec[chunk_n:] if buf_dec is not None else None
            buf_rb  = buf_rb[chunk_n:]
            buf_db  = buf_db[chunk_n:]
            buf_bq  = buf_bq[chunk_n:]  if buf_bq  is not None else None
            buf_flags = buf_flags[chunk_n:]

    if buf_sig is not None and buf_sig.shape[0] > 0:
        assert kids is not None and sample_rate is not None and t_start is not None
        n      = buf_sig.shape[0]
        t_stop = t_start + n / sample_rate
        yield Chunk(
            kids=kids, signal=buf_sig, common_mode=None,
            ra=buf_ra, dec=buf_dec,
            ra_bore=buf_rb, dec_bore=buf_db,
            boresight_q=buf_bq,
            det_dirs=det_dirs,
            t_start=t_start, t_stop=t_stop,flags=buf_flags,
            sample_rate=sample_rate, chunk_index=chunk_index,
        )


# ── G3 simulation format helpers ─────────────────────────────────────────────

# ============================================================================ #
# _load_g3_focalplane
# ============================================================================ #
def _load_g3_focalplane(frame):
    """
    Extract detector names and precomputed pointing directions from a
    g3 calibration frame.

    The focalplane HDF5 table is embedded as a raw byte buffer, BytesIO lets
    h5py open it in memory without writing to disk.

    NOTE: This will need to be modified when the true offsets are known.

    Returns
    -------
    det_names : list of str
    det_quats : dict {name: (4,) array} — offset quaternions, vector-first (x,y,z,w)
    det_dirs  : dict {name: (3,) array} — precomputed focal-plane directions
    """
    fp_buffer = io.BytesIO(bytes(frame["focalplane"]))
    with h5py.File(fp_buffer, "r") as f:
        det_names = [n.decode("utf-8") for n in f["focalplane"]["name"][:]]
        quats     = f["focalplane"]["quat"][:]
    det_quats = {name: quat for name, quat in zip(det_names, quats)}
    dirs      = precompute_det_directions(quats)  # (n_dets, 3), computed once
    det_dirs  = {name: dirs[i] for i, name in enumerate(det_names)}
    return det_names, det_quats, det_dirs


# ============================================================================ #
# _g3_simulation_scan_to_chunk
# ============================================================================ #
def _g3_simulation_scan_to_chunk(frame, det_names, det_dirs, sample_rate_ref, apply_offsets: bool = True):
    """
    Convert one g3 simulation scan frame into a frame-level Chunk.

    Signal is stored as compressed integers: true_signal = raw / gain + offset.
    boresight_r is built once per frame and reused for all detectors.

    sample_rate_ref : list[float | None] — single-element list to cache sample
                      rate after the first frame (shared across all frames).
    """
    raw     = frame["signal"]
    kids    = [k for k in det_names if k in raw.keys()]
    n_dets  = len(kids)
    n_samps = len(raw[kids[0]])

    boresight_q = np.asarray(frame["shared_boresight_radec"])

    sig     = np.zeros((n_samps, n_dets), dtype=float)
    det_ra  = np.zeros((n_samps, n_dets), dtype=float) if apply_offsets else None
    det_dec = np.zeros((n_samps, n_dets), dtype=float) if apply_offsets else None

    ra_bore, dec_bore, boresight_r = boresight_to_radec(boresight_q)
    boresight_q_scipy = boresight_q[:, [1, 2, 3, 0]]  # reorder w,x,y,z -> x,y,z,w (scipy)

    for i, kid in enumerate(kids):
        y_raw      = np.asarray(raw[kid], dtype=float)
        gain_key   = f"compress_signal_{kid}_gain"
        offset_key = f"compress_signal_{kid}_offset"
        if gain_key in frame and offset_key in frame:
            sig[:, i] = y_raw / float(frame[gain_key]) + float(frame[offset_key])
        else:
            sig[:, i] = y_raw

        if apply_offsets:
            det_ra[:, i], det_dec[:, i] = det_radec_from_boresight(boresight_r, det_dirs[kid])

    ts      = raw[kids[0]]
    t_start = ts.start.time / core.G3Units.s
    t_stop  = ts.stop.time  / core.G3Units.s

    if sample_rate_ref[0] is None:
        sample_rate_ref[0] = n_samps / (t_stop - t_start)

    return Chunk(
        kids=kids, signal=sig, common_mode=None,
        ra=det_ra, dec=det_dec,
        ra_bore=ra_bore, dec_bore=dec_bore,
        boresight_q=boresight_q_scipy,
        det_dirs=det_dirs,
        t_start=t_start, t_stop=t_stop,
        sample_rate=sample_rate_ref[0], flags=np.zeros((n_samps, n_dets), dtype=int),  # TODO: real flagging once available
        chunk_index=-1,  # assigned by _rechunk
    )


# ============================================================================ #
# _iter_g3_simulation_chunks
# ============================================================================ #
def _iter_g3_simulation_chunks(files: list, apply_offsets: bool = True) -> Iterator[Chunk]:
    """Yield one frame-level Chunk per scan frame from CCAT g3 simulation files."""
    det_names       = None
    det_dirs        = None
    sample_rate_ref = [None]
    for path in files:
        for frame in core.G3File(str(path)):
            if frame.type == core.G3FrameType.Calibration:
                det_names, _, det_dirs = _load_g3_focalplane(frame)
            elif frame.type == core.G3FrameType.Scan:
                if det_names is None:
                    raise RuntimeError(
                        "Scan frame encountered before calibration frame. "
                        "Check that the first .g3 file contains a calibration frame."
                    )
                yield _g3_simulation_scan_to_chunk(
                    frame, det_names, det_dirs, sample_rate_ref, apply_offsets
                )


# ── HDF5 simulation format helpers ───────────────────────────────────────────

# ============================================================================ #
# _load_h5_focalplane
# ============================================================================ #
def _load_h5_focalplane(path):
    """
    Extract detector names and precomputed pointing directions from an HDF5
    simulation file's instrument/focalplane table.

    NOTE: This will need to be modified when the true offsets are known.

    Returns
    -------
    det_names : array of str
    det_quats : dict {name: (4,) array} — offset quaternions, vector-first (x,y,z,w)
    det_dirs  : dict {name: (3,) array} — precomputed focal-plane directions
    """
    with h5py.File(path, "r") as h5_file:
        det_names = h5_file['instrument/focalplane']['name'].astype(str)
        quats     = h5_file['instrument/focalplane']['quat']
    det_quats = {name: quat for name, quat in zip(det_names, quats)}
    dirs      = precompute_det_directions(quats)  # (n_dets, 3), computed once
    det_dirs  = {name: dirs[i] for i, name in enumerate(det_names)}
    return det_names, det_quats, det_dirs


# ============================================================================ #
# _iter_h5_simulation_chunks
# ============================================================================ #
def _iter_h5_simulation_chunks(files: list, chunk_duration_s: float, apply_offsets: bool = True) -> Iterator[Chunk]:
    """
    Yield fixed-duration Chunks directly from HDF5 simulation files.

    detdata/signal and detdata/flags are (n_dets, n_samps) on-disk datasets;
    each chunk slices only its own [start:stop] sample window out of them via
    h5py, so at most one chunk's worth of (n_dets, chunk_n) data is ever
    materialized in memory. times/boresight_radec are (n_samps,)-sized
    (detector-independent) so those are read in full per file -- cheap
    regardless of chunk_duration_s since they don't scale with n_dets.

    sample_rate is established once, from the first file, and reused for
    every subsequent file's chunk_n (matches the previous _rechunk-based
    behaviour, which also assumed one constant rate across the whole
    observation). Chunks don't span file boundaries -- unlike _rechunk's
    frame buffering, a file's leftover tail becomes its own short chunk
    rather than being spliced onto the next file's head, since HDF5 files
    aren't guaranteed contiguous in time the way consecutive g3 frames are.
    """
    det_names       = None
    det_dirs        = None
    sample_rate_ref = [None]
    chunk_index     = 0

    for path in files:
        if det_names is None:
            det_names, _, det_dirs = _load_h5_focalplane(path)

        with h5py.File(path, 'r') as h5_file:
            raw      = h5_file["detdata/signal"]    # (n_dets, n_samps), on-disk
            flags_ds = h5_file["detdata/flags"]      # (n_dets, n_samps), on-disk
            times    = np.asarray(h5_file["shared/times"])
            n_samps  = len(times)

            boresight_q = np.roll(np.asarray(h5_file["shared/boresight_radec"]), 1)  # raw (w,x,y,z), shape (n,4)
            ra_bore, dec_bore, boresight_r = boresight_to_radec(boresight_q)

            if sample_rate_ref[0] is None:
                sample_rate_ref[0] = n_samps / (times[-1] - times[0])
            sample_rate = sample_rate_ref[0]
            chunk_n = max(1, int(chunk_duration_s * sample_rate)) if chunk_duration_s > 0 else n_samps

            for start in range(0, n_samps, chunk_n):
                stop = min(start + chunk_n, n_samps)

                sig = np.asarray(raw[:, start:stop], dtype=float).T   # (n, n_dets)
                flg = np.asarray(flags_ds[:, start:stop], dtype=int).T   # (n, n_dets)

                det_ra = det_dec = None
                if apply_offsets:
                    det_ra  = np.zeros((stop - start, len(det_names)), dtype=float)
                    det_dec = np.zeros((stop - start, len(det_names)), dtype=float)
                    for i, kid in enumerate(det_names):
                        det_ra[:, i], det_dec[:, i] = det_radec_from_boresight(
                            boresight_r[start:stop], det_dirs[kid]
                        )

                yield Chunk(
                    kids=det_names, signal=sig, common_mode=None,
                    ra=det_ra, dec=det_dec,
                    ra_bore=ra_bore[start:stop], dec_bore=dec_bore[start:stop],
                    boresight_q=boresight_q[start:stop],
                    det_dirs=det_dirs,
                    t_start=times[start], t_stop=times[stop - 1],
                    sample_rate=sample_rate, flags=flg,
                    chunk_index=chunk_index,
                )
                chunk_index += 1


# ── Real BLAST-TNG format helpers ────────────────────────────────────────────

# ============================================================================ #
# _interp_to_length
# ============================================================================ #
def _interp_to_length(a: np.ndarray, n_new: int) -> np.ndarray:
    """
    Index-based linear interpolation, stretching/compressing `a` to `n_new`
    samples.

    Pointing telemetry and KID readout are commonly sampled at different
    native rates within the same frame (confirmed on real BLAST-TNG data:
    ra/dec at 1428 samples/frame vs I/Q at 1510). This assumes both span the
    same real time range uniformly within the frame same convention as
    the original notebook's alignMasterAndRoachTods.
    """
    if len(a) == n_new:
        return a
    x_old = np.arange(len(a))
    x_new = np.linspace(0, len(a) - 1, n_new)
    return np.interp(x_new, x_old, a)


# ============================================================================ #
# _load_blasttng_calibration
# ============================================================================ #
def _load_blasttng_calibration(frame, target_sweeps_key: str = "target_sweeps"):
    """
    Extract detector names and calibration sweep data from a calibration frame.

    Target sweeps are stored as a G3TimestreamMap with keys "<kid>_I",
    "<kid>_Q", "<kid>_F" for each detector's calibration sweep -- same
    convention as external/blasttng-to-g3's g3_utils/signal.py.

    Also reads pre-baked per-detector "ra_shifts"/"dec_shifts" from the
    calibration frame which were done by Jonah, if present (as written by g3_packager, see
    external/blasttng-to-g3/g3_packager/frame_generators.py's
    get_kid_shifts, sourced from a one-off empirical shift table).

    NOTE : These
    aren't guaranteed to exist for every dataset; callers should fall back
    to computing shifts themselves (e.g. via the per-detector pass) when
    baked_shifts comes back None.
    """
    target_sweeps_map = frame[target_sweeps_key]
    kids = sorted({name[:-2] for name in target_sweeps_map.keys()})

    target_sweeps = {}
    for kid in kids:
        If = np.array(target_sweeps_map[f"{kid}_I"])
        Qf = np.array(target_sweeps_map[f"{kid}_Q"])
        Ff = np.array(target_sweeps_map[f"{kid}_F"])
        target_sweeps[kid] = (If, Qf, Ff)

    baked_shifts = None
    if "ra_shifts" in frame and "dec_shifts" in frame:
        ra_shifts_map  = frame["ra_shifts"]
        dec_shifts_map = frame["dec_shifts"]
        baked_shifts = {
            kid: (ra_shifts_map[kid] / core.G3Units.deg, dec_shifts_map[kid] / core.G3Units.deg)
            for kid in kids if kid in ra_shifts_map and kid in dec_shifts_map
        }

    return kids, target_sweeps, baked_shifts


# ============================================================================ #
# get_blasttng_baked_shifts
# ============================================================================ #
def get_blasttng_baked_shifts(cfg: dict) -> Optional[dict]:
    """
    Peek at the first blasttng .g3 file's calibration frame for pre-baked
    per-detector ra_shifts/dec_shifts, without iterating the whole file.
    """
    if cfg["data"]["format"] != "blasttng":
        return None

    files = []
    for pattern in cfg["data"]["input_dirs"]:
        for d in sorted(glob.glob(pattern)):
            files.extend(sorted(pathlib.Path(d).rglob("*.g3")))
    if not files:
        return None

    for frame in core.G3File(str(files[0])):
        if frame.type == core.G3FrameType.Calibration:
            _, _, baked_shifts = _load_blasttng_calibration(frame)
            return baked_shifts
    return None


# ============================================================================ #
# get_blasttng_site
# ============================================================================ #
def get_blasttng_site(cfg: dict):
    """
    Peek at the first blasttng .g3 file's first scan frame for the gondola's
    lat/lon/alt telemetry (same keys blasttng-to-g3's coords.add_radec_so3g
    reads), and return the middle-of-frame position as an astropy
    EarthLocation. Returns None if no lat/lon/alt telemetry is present.
    """
    if cfg["data"]["format"] != "blasttng":
        return None

    from astropy.coordinates import EarthLocation
    import astropy.units as u

    files = []
    for pattern in cfg["data"]["input_dirs"]:
        for d in sorted(glob.glob(pattern)):
            files.extend(sorted(pathlib.Path(d).rglob("*.g3")))
    if not files:
        return None

    for frame in core.G3File(str(files[0])):
        if frame.type == core.G3FrameType.Scan and all(k in frame for k in ("lat", "lon", "alt")):
            mid     = len(frame["lat"]) // 2
            lat_deg = frame["lat"][mid] / core.G3Units.deg
            lon_deg = frame["lon"][mid] / core.G3Units.deg
            alt_m   = frame["alt"][mid] / core.G3Units.m
            return EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    return None


# ============================================================================ #
# _load_blasttng_cal_lamp_df
# ============================================================================ #
def _load_blasttng_cal_lamp_df(frame, kids, target_sweeps, iq_key: str = "cal_lamp_data",
                               df_method: str = "hybrid", threshold_frac: float = 0.05):
    """
    Compute each detector's df timestream during the calibration-lamp
    exposure, for use as the reference in signal.normalize_tod.
    """
    if iq_key not in frame:
        return None

    super_ts = frame[iq_key]
    names    = np.asarray(super_ts.names)

    cal_lamp_df = {}
    for kid in kids:
        i_matches = np.where(names == f"{kid}_I")[0]
        q_matches = np.where(names == f"{kid}_Q")[0]
        if len(i_matches) == 0 or len(q_matches) == 0:
            continue  # this kid isn't in the cal-lamp exposure; skip it
        I = np.asarray(super_ts.data[i_matches[0]], dtype=float)
        Q = np.asarray(super_ts.data[q_matches[0]], dtype=float)
        If, Qf, Ff = target_sweeps[kid]

        if df_method == "hybrid":
            df, _ = iq_to_df_hybrid(I, Q, If, Qf, Ff, threshold_frac=threshold_frac)
        else:
            df = iq_to_df(I, Q, If, Qf, Ff)
        cal_lamp_df[kid] = np.nan_to_num(df, nan=0.0)

    return cal_lamp_df


# ============================================================================ #
# _blasttng_scan_to_chunk
# ============================================================================ #
def _blasttng_scan_to_chunk(frame, kids, target_sweeps, sample_rate_ref,
                            iq_key: str = "data", df_method: str = "hybrid",
                            threshold_frac: float = 0.05, cal_lamp_df: dict = None):
    """
    Convert one real-data scan frame into a frame-level Chunk.

    Raw I/Q is stored as a G3SuperTimestream with names "<kid>_I", "<kid>_Q"
    (same convention as the calibration sweep). Each detector's I/Q is
    converted to fractional frequency shift (df) against its own
    calibration sweep.

    """
    super_ts = frame[iq_key]
    names    = np.asarray(super_ts.names)
    n_dets   = len(kids)
    n_samps  = super_ts.data.shape[1]

    sig = np.zeros((n_samps, n_dets), dtype=float)

    for i, kid in enumerate(kids):
        i_idx = int(np.where(names == f"{kid}_I")[0][0])
        q_idx = int(np.where(names == f"{kid}_Q")[0][0])
        I = np.asarray(super_ts.data[i_idx], dtype=float)
        Q = np.asarray(super_ts.data[q_idx], dtype=float)
        If, Qf, Ff = target_sweeps[kid]

        if df_method == "hybrid":
            df, _used_fallback = iq_to_df_hybrid(I, Q, If, Qf, Ff, threshold_frac=threshold_frac)
        else:
            df = iq_to_df(I, Q, If, Qf, Ff)
        sig[:, i] = df

    # Real BLAST-TNG readout has brief dropouts where every channel's raw I/Q
    # goes NaN simultaneously (confirmed on roach1_pass3.g3: ~5% of samples,
    # present in every single frame ) iq_to_df_hybrid correctly propagates that NaN
    # through, so it needs handling here before signal reaches anything else.

    sig = np.nan_to_num(sig, nan=0.0)

    # Normalize each detector against its own cal-lamp exposure (median-zero,
    # cal-lamp-peak-scaled), matches g3_utils.signal.NormalizeDF
    # (Without this, every
    # detector's raw sensitivity differs (depends on that resonator's own
    # coupling/quality factor)
    if cal_lamp_df is not None:
        for i, kid in enumerate(kids):
            if kid in cal_lamp_df:
                sig[:, i] = normalize_tod(sig[:, i], cal_lamp_df[kid])

    ra_bore  = np.array(frame["ra"])  / core.G3Units.deg
    dec_bore = np.array(frame["dec"]) / core.G3Units.deg
    ra_bore  = _interp_to_length(ra_bore, n_samps)
    dec_bore = _interp_to_length(dec_bore, n_samps)

    t_start = super_ts.times[0].time  / core.G3Units.s
    t_stop  = super_ts.times[-1].time / core.G3Units.s

    if sample_rate_ref[0] is None:
        sample_rate_ref[0] = n_samps / (t_stop - t_start)

    flags = np.zeros((n_samps, n_dets), dtype=int)  # TODO: real flagging once available

    return Chunk(
        kids=kids, signal=sig, common_mode=None,
        ra=None, dec=None,  # per-detector offsets not yet available for real data
        ra_bore=ra_bore, dec_bore=dec_bore,
        boresight_q=None, det_dirs=None,
        t_start=t_start, t_stop=t_stop,
        sample_rate=sample_rate_ref[0], flags=flags,
        chunk_index=-1,  # assigned by _rechunk
    )


# ============================================================================ #
# _iter_blasttng_chunks
# ============================================================================ #
def _iter_blasttng_chunks(files: list, iq_key: str = "data", target_sweeps_key: str = "target_sweeps",
                          cal_lamp_key: str = "cal_lamp_data",
                          df_method: str = "hybrid", threshold_frac: float = 0.05) -> Iterator[Chunk]:
    """Yield one frame-level Chunk per scan frame from real BLAST-TNG-format .g3 files."""
    kids = None
    target_sweeps = None
    cal_lamp_df = None
    sample_rate_ref = [None]

    for path in files:
        for frame in core.G3File(str(path)):
            if frame.type == core.G3FrameType.Calibration:
                kids, target_sweeps, _baked_shifts = _load_blasttng_calibration(frame, target_sweeps_key)
                cal_lamp_df = _load_blasttng_cal_lamp_df(
                    frame, kids, target_sweeps, iq_key=cal_lamp_key,
                    df_method=df_method, threshold_frac=threshold_frac,
                )
            elif frame.type == core.G3FrameType.Scan:
                if kids is None:
                    raise RuntimeError(
                        "Scan frame encountered before calibration frame. "
                        "Check that the first .g3 file contains a calibration frame."
                    )
                yield _blasttng_scan_to_chunk(
                    frame, kids, target_sweeps, sample_rate_ref,
                    iq_key=iq_key, df_method=df_method, threshold_frac=threshold_frac,
                    cal_lamp_df=cal_lamp_df,
                )


# ── Public interface ───────────────────────────────────────────────────────────

# ============================================================================ #
# iter_chunks
# ============================================================================ #
def iter_chunks(cfg: dict, apply_offsets: bool = True) -> Iterator[Chunk]:
    """
    Yield fixed-duration Chunks from the files specified in cfg.
    """
    files = []
    file_fmt = cfg["data"]["file_format"]
    for pattern in cfg["data"]["input_dirs"]:
        for d in sorted(glob.glob(pattern)):
            files.extend(sorted(pathlib.Path(d).rglob(f"*.{file_fmt}")))

    if not files:
        raise FileNotFoundError(
            f"No .{file_fmt} files found for patterns: {cfg['data']['input_dirs']}"
        )
    fmt = cfg["data"]["format"]
    chunk_duration_s = cfg["pipeline"].get("chunk_duration_s", 1.0)
    start_offset_s   = cfg["pipeline"].get("start_offset_s", 0.0)
    max_duration_s   = cfg["pipeline"].get("max_duration_s", None)

    if fmt == "simulation" and file_fmt == "h5":
        # HDF5 datasets support direct index slicing, so chunks are read
        # straight off disk -- no frame buffering/_rechunk needed here.
        chunked_iter = _iter_h5_simulation_chunks(files, chunk_duration_s, apply_offsets)
    elif fmt == "simulation":
        frame_iter = _iter_g3_simulation_chunks(files, apply_offsets)
        chunked_iter = _rechunk(frame_iter, chunk_duration_s)
    elif fmt == "blasttng":
        blasttng_cfg = cfg.get("blasttng", {})
        frame_iter = _iter_blasttng_chunks(
            files,
            iq_key=blasttng_cfg.get("iq_key", "data"),
            target_sweeps_key=blasttng_cfg.get("target_sweeps_key", "target_sweeps"),
            cal_lamp_key=blasttng_cfg.get("cal_lamp_key", "cal_lamp_data"),
            df_method=blasttng_cfg.get("df_method", "hybrid"),
            threshold_frac=blasttng_cfg.get("threshold_frac", 0.05),
        )
        chunked_iter = _rechunk(frame_iter, chunk_duration_s)
    else:
        raise NotImplementedError(
            f"Data format '{fmt}' is not yet implemented. "
            f"Currently supported: 'simulation', 'blasttng'."
        )

    t_obs_start = None
    for chunk in chunked_iter:
        if t_obs_start is None:
            t_obs_start = chunk.t_start
        elapsed = chunk.t_stop - t_obs_start
        if elapsed <= start_offset_s:
            continue
        if max_duration_s is not None and (chunk.t_start - t_obs_start) >= start_offset_s + max_duration_s:
            break
        yield chunk
