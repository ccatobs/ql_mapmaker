# ============================================================================ #
# reader.py
#
# Reads .g3 files and yields fixed-duration Chunk objects for the pipeline.
#
# G3 file structure (simulation format):
#   Calibration frame (first): focal-plane detector names + offset quaternions.
#   Scan frames (many):        compressed signal + boresight quaternions.
#
# G3 frames have variable sample counts. iter_g3_chunks rebuffers them into
# fixed-duration chunks (chunk_duration_s) so downstream code always gets a
# predictable amount of data. All arrays are time-first: (n_samps, n_dets).
# ============================================================================ #

import io
import glob
import pathlib
from typing import Iterator, Optional

import numpy as np
import h5py
import h5py as h5
from dataclasses import dataclass
from spt3g import core

from .pointing import precompute_det_directions, boresight_to_radec, det_radec_from_boresight


@dataclass
class Chunk:
    """
    All detector and pointing data for one fixed-duration time window.
    All 2-D arrays are time-first: shape (n_samps, n_dets).

    kids[j] is the name of the detector in column j of signal, ra, dec.
    t_start / t_stop are in seconds since the G3 epoch (Jan 1 2001).
    common_mode and flags start as None; populated downstream if needed.
    """
    kids:         list
    signal:       np.ndarray          # (n_samps, n_dets)
    common_mode:  Optional[np.ndarray]# (n_samps,) or None
    ra:           np.ndarray          # (n_samps, n_dets) deg
    dec:          np.ndarray          # (n_samps, n_dets) deg
    ra_bore:      np.ndarray          # (n_samps,) deg
    dec_bore:     np.ndarray          # (n_samps,) deg
    t_start:      float
    t_stop:       float
    sample_rate:  float               # Hz
    chunk_index:  int
    flags:        np.ndarray          # (n_samps, n_dets) bool
    boresight_q:  Optional[np.ndarray] =W None  # (n_samps, 4) scipy (x,y,z,w); for pointing reconstruction


# ── Simulation format helpers ──────────────────────────────────────────────────

# Adapted from Bonnie Slocombe, https://github.com/bonnieslocombe/g3_mapmaking, mapmaker/g3mapmaker.py, QuickMapMaker.Process
def _load_simulation_focalplane(file, file_fmt: str):
    """
    Extract detector names and precomputed pointing directions from a calibration frame.

    The focalplane HDF5 table is embedded as a raw byte buffer, BytesIO lets
    h5py open it in memory without writing to disk.

    Returns
    -------
    det_names : list of str
    det_quats : dict {name: (4,) array} — offset quaternions, vector-first (x,y,z,w)
    det_dirs  : dict {name: (3,) array} — precomputed focal-plane directions
    """
    if file_fmt == 'g3':
        fp_buffer = io.BytesIO(bytes(file["focalplane"]))
        with h5py.File(fp_buffer, "r") as f:
            det_names = [n.decode("utf-8") for n in f["focalplane"]["name"][:]]
            quats     = f["focalplane"]["quat"][:]
        det_quats = {name: quat for name, quat in zip(det_names, quats)}
        dirs      = precompute_det_directions(quats)  # (n_dets, 3), computed once
        det_dirs  = {name: dirs[i] for i, name in enumerate(det_names)}
    elif file_fmt == 'h5':
        with h5py.File(file, "r") as h5_file:
            det_names = h5_file['instrument/focalplane']['name'].astype(str)
            quats     = h5_file['instrument/focalplane']['quat']
        det_quats = {name: quat for name, quat in zip(det_names, quats)}
        dirs      = precompute_det_directions(quats)  # (n_dets, 3), computed once
        det_dirs  = {name: dirs[i] for i, name in enumerate(det_names)}
    return det_names, det_quats, det_dirs


# Adapted from Bonnie Slocombe, https://github.com/bonnieslocombe/g3_mapmaking, mapmaker/g3mapmaker.py, QuickMapMaker.Process
def _simulation_scan_to_chunk(path, det_names, det_dirs, sample_rate_ref, file_fmt):
    """
    Convert one simulation scan frame into a frame-level Chunk.

    Signal is stored as compressed integers: true_signal = raw / gain + offset.
    boresight_r is built once per frame and reused for all detectors.

    sample_rate_ref : list[float | None] — single-element list to cache sample
                      rate after the first frame (shared across all frames).
    """
    if file_fmt == 'g3':
        frame = path
        raw     = frame["signal"]
        kids    = [k for k in det_names if k in raw.keys()]
        n_dets  = len(kids)
        n_samps = len(raw[kids[0]])
    
        boresight_q = np.asarray(frame["shared_boresight_radec"])
    
        sig     = np.zeros((n_samps, n_dets), dtype=float)
        det_ra  = np.zeros((n_samps, n_dets), dtype=float)
        det_dec = np.zeros((n_samps, n_dets), dtype=float)
    
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
            t_start=t_start, t_stop=t_stop,
            sample_rate=sample_rate_ref[0], flags = flags.T,
            chunk_index=-1,  # assigned by _rechunk
        )

    elif file_fmt == 'h5':
        with h5.File(path, 'r') as h5_file:
            raw     = h5_file["detdata/signal"]
            flags   = np.copy(h5_file["detdata/flags"])
            # flags[:,0:100000] += 1
            # flags[0:20,:] += 1
            n_dets  = len(raw)
            n_samps = len(raw[0])
            boresight_q = np.roll(np.asarray(h5_file["shared/boresight_radec"]), 1)
        
            sig     = np.zeros((n_samps, n_dets), dtype=float)
            det_ra  = np.zeros((n_samps, n_dets), dtype=float)
            det_dec = np.zeros((n_samps, n_dets), dtype=float)
        
            ra_bore, dec_bore, boresight_r = boresight_to_radec(boresight_q)
        
            for i, kid in enumerate(det_names):
                y_raw      = np.asarray(raw[i], dtype=float)
                gain_key   = f"compress_signal_{kid}_gain"
                offset_key = f"compress_signal_{kid}_offset"
                sig[:, i] = y_raw
                det_ra[:, i], det_dec[:, i] = det_radec_from_boresight(boresight_r, det_dirs[kid])

            ts      = h5_file["shared/times"]
            t_start = ts[0]
            t_stop  = ts[-1]
        
            if sample_rate_ref[0] is None:
                sample_rate_ref[0] = n_samps / (t_stop - t_start)
            chunk_returned = Chunk(
                kids=det_names, signal=sig, common_mode=None,
                ra=det_ra, dec=det_dec,
                ra_bore=ra_bore, dec_bore=dec_bore,
                t_start=t_start, t_stop=t_stop,
                sample_rate=sample_rate_ref[0], flags = flags.T,
                chunk_index=-1,  # assigned by _rechunk
            )
            return chunk_returned

def _rechunk(frame_iter: Iterator[Chunk], chunk_duration_s: float, file_fmt: str) -> Iterator[Chunk]:
    """
    Rebuffer variable-length frame-level Chunks into fixed-duration Chunks.

    Accumulates frames until chunk_n samples are ready, then yields and
    advances the buffer. The final chunk may be shorter than chunk_n.
    If chunk_duration_s <= 0, yields each frame unchanged (frame-aligned mode).
    """
    buf_sig = buf_ra = buf_dec = buf_rb = buf_db = buf_bq = buf_flags = None
    kids = sample_rate = t_start = None
    chunk_n     = None
    chunk_index = 0

    for frame in frame_iter:
        if kids is None:
            kids        = frame.kids
            sample_rate = frame.sample_rate
            t_start     = frame.t_start
            if chunk_duration_s > 0:
                chunk_n = max(1, int(chunk_duration_s * sample_rate))
        if chunk_n is None:
            yield Chunk(
                kids=frame.kids, signal=frame.signal, common_mode=None,
                ra=frame.ra, dec=frame.dec,
                ra_bore=frame.ra_bore, dec_bore=frame.dec_bore,
                boresight_q=frame.boresight_q,
                t_start=frame.t_start, t_stop=frame.t_stop, flags=frame.flags,
                sample_rate=frame.sample_rate, chunk_index=chunk_index,
            )
            chunk_index += 1
            continue

        buf_sig = frame.signal      if buf_sig is None else np.concatenate([buf_sig, frame.signal],      axis=0)
        buf_ra  = frame.ra          if buf_ra  is None else np.concatenate([buf_ra,  frame.ra],           axis=0)
        buf_dec = frame.dec         if buf_dec is None else np.concatenate([buf_dec, frame.dec],          axis=0)
        buf_rb  = frame.ra_bore     if buf_rb  is None else np.concatenate([buf_rb,  frame.ra_bore])
        buf_db  = frame.dec_bore    if buf_db  is None else np.concatenate([buf_db,  frame.dec_bore])
        buf_bq  = frame.boresight_q if buf_bq  is None else np.concatenate([buf_bq,  frame.boresight_q], axis=0)
        buf_flags = frame.flags     if buf_flags is None else np.concatenate([buf_flags, frame.flags], axis=0)

        assert kids is not None and sample_rate is not None and t_start is not None
        while buf_sig.shape[0] >= chunk_n:
            t_stop = t_start + chunk_n / sample_rate
            yield Chunk(
                kids=kids, signal=buf_sig[:chunk_n], common_mode=None,
                ra=buf_ra[:chunk_n], dec=buf_dec[:chunk_n],
                ra_bore=buf_rb[:chunk_n], dec_bore=buf_db[:chunk_n],
                boresight_q=buf_bq[:chunk_n],
                t_start=t_start, t_stop=t_stop, flags=buf_flags[:chunk_n],
                sample_rate=sample_rate, chunk_index=chunk_index,
            )
            chunk_index += 1
            t_start = t_stop
            buf_sig = buf_sig[chunk_n:]
            buf_ra  = buf_ra[chunk_n:]
            buf_dec = buf_dec[chunk_n:]
            buf_rb  = buf_rb[chunk_n:]
            buf_db  = buf_db[chunk_n:]
            buf_bq  = buf_bq[chunk_n:]
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
            t_start=t_start, t_stop=t_stop, flags=buf_flags,
            sample_rate=sample_rate, chunk_index=chunk_index,
        )


def _iter_simulation_chunks(files: list, file_fmt: str) -> Iterator[Chunk]:
    """Yield one frame-level Chunk per scan frame from CCAT simulation files."""
    det_names       = None
    det_dirs        = None
    sample_rate_ref = [None]
    if file_fmt == 'g3':
        for path in files:
            for frame in core.G3File(str(path)):
                if frame.type == core.G3FrameType.Calibration:
                    det_names, _, det_dirs = _load_simulation_focalplane(frame, file_fmt)
                elif frame.type == core.G3FrameType.Scan:
                    if det_names is None:
                        raise RuntimeError(
                            "Scan frame encountered before calibration frame. "
                            "Check that the first .g3 file contains a calibration frame."
                        )
                    yield _simulation_scan_to_chunk(
                        frame, det_names, det_dirs, sample_rate_ref, file_fmt
                    )
    elif file_fmt == 'h5':
        for i, path in enumerate(files):
            # with h5.File(path, 'r') as h5_file:
            # print(f"h5_file: {h5_file}")
            if i == 0:
                try:
                    det_names, _, det_dirs = _load_simulation_focalplane(path, file_fmt)
                except RuntimeError:
                    print("Not able to extract focalplane information.")
            if det_names is None:
                raise RuntimeError(
                    "No detector information (detector names) found."
                )
            yield _simulation_scan_to_chunk(
                path, det_names, det_dirs, sample_rate_ref, file_fmt
            )

# ── Public interface ───────────────────────────────────────────────────────────

def iter_chunks(cfg: dict) -> Iterator[Chunk]:
    """
    Yield fixed-duration Chunks from the files specified in cfg.

    Time windowing — all three from config.toml [pipeline]:
      chunk_duration_s  seconds of data per Chunk (default 1.0 s)
      start_offset_s    skip this many seconds at the start of the observation
      max_duration_s    stop after this many seconds (measured after the offset)

    Example: start_offset_s=100, max_duration_s=200 → process seconds 100–300.
    """
    files = []
    file_fmt = cfg["data"]["file_format"]
    if file_fmt == 'g3':
        for pattern in cfg["data"]["input_dirs"]:
            for d in sorted(glob.glob(pattern)):
                files.extend(sorted(pathlib.Path(d).rglob(f"*.{file_fmt}")))
                
    elif file_fmt == 'h5':
        for pattern in cfg["data"]["input_dirs"]:
            files.extend(sorted(glob.glob(pattern)))
    
    if not files:
        raise FileNotFoundError(
            f"No .{file_fmt} files found for patterns: {cfg['data']['input_dirs']}"
        )
    fmt = cfg["data"]["format"]
    if fmt == "simulation":
        frame_iter = _iter_simulation_chunks(files, file_fmt)
    else:
        raise NotImplementedError(
            f"Data format '{fmt}' is not yet implemented. "
            f"Currently supported: 'simulation'."
        )

    chunk_duration_s = cfg["pipeline"].get("chunk_duration_s", 1.0)
    start_offset_s   = cfg["pipeline"].get("start_offset_s", 0.0)
    max_duration_s   = cfg["pipeline"].get("max_duration_s", None)
    
    t_obs_start = None
    for chunk in _rechunk(frame_iter, chunk_duration_s, file_fmt):
        flags = chunk.flags
        if t_obs_start is None:
            t_obs_start = chunk.t_start
        elapsed = chunk.t_stop - t_obs_start
        if elapsed <= start_offset_s:
            continue
        if max_duration_s is not None and (chunk.t_start - t_obs_start) >= start_offset_s + max_duration_s:
            break
        yield chunk
