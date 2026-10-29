# ============================================================================ #
# reader.py
#
# James Burgoyne, jamesrburgoyne@icloud.com
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT 2026
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

  h5 lab
    Most recent h5 files in draft CCAT format.

All arrays are time-first: (n_samps, n_dets).
"""


# TODO: chunk_duration_s requires knowing how many seconds you want
# it would be better to be adaptable to memory allowances.
# For example, determine available memory (with some buffer),
# and have a max from config file, then choose smaller of two.
# Then determine how many seconds of chunk that is.
# Need to consider what the allowable buffer would be from testing,
# since we will have overhead from other things in memory too.



import io
import glob
import pathlib
from typing import Iterator, Optional

import numpy as np
import h5py
from dataclasses import dataclass
from spt3g import core # type: ignore

from mapmaker import tod
# from mapmaker import map
from mapmaker import coord
# from mapmaker import output
# from mapmaker import reader


# ============================================================================ #
# Class: Chunk
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
# g3 ingestion paths stream through this to turn variable-length frames into fixed-duration Chunks. The h5 simulation path doesn't use it -- HDF5 datasets are randomly indexable, so chunks are sliced directly from disk instead of buffered in memory.

# ============================================================================ #
# _rechunk
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




# ============================================================================ #
# H5 LAB
# ============================================================================ #


# ============================================================================ #
# _iter_h5_lab_frames
def _iter_h5_lab_frames(files: list, apply_offsets: bool = True) -> Iterator[Chunk]:
    """Yield one frame-level Chunk per scan frame from HDF5 lab files."""

    sample_rate_ref = [None]

    for path in files:
        with h5py.File(path, 'r') as f:
            if "timestream" not in f:
                continue
            timestream_grp = f["timestream"]
            blocks = list(timestream_grp.keys())
            if not blocks:
                continue

            # Determine available frame keys from the first block
            first_block = blocks[0]
            if "frames" not in timestream_grp[first_block]:
                continue
            frames_grp = timestream_grp[first_block]["frames"]
            frame_keys = sorted([k for k in frames_grp.keys() if k.isdigit()], key=int)

            for fk in frame_keys:
                sample_block = frames_grp[fk]
                frame_type = sample_block.attrs.get("frame_type", b"")
                if isinstance(frame_type, bytes):
                    frame_type = frame_type.decode("utf-8")

                if frame_type != "Scan":
                    continue

                block_data_list = []
                min_samps = None

                # Gather data from all detector blocks for this frame
                for block in blocks:
                    block_frame_path = f"timestream/{block}/frames/{fk}"
                    if block_frame_path not in f:
                        continue
                    b_frame = f[block_frame_path]
                    
                    if "data" not in b_frame or "data" not in b_frame["data"] or "times" not in b_frame["data"]:
                        continue

                    names_ds = b_frame["data"]["names"][:]
                    names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names_ds]
                    
                    sig_ds = b_frame["data"]["data"][:]      # (n_channels, n_samps)
                    times_ds = b_frame["data"]["times"][:]   # (n_samps,)

                    n_samps = len(times_ds)
                    if min_samps is None or n_samps < min_samps:
                        min_samps = n_samps

                    block_data_list.append((names, sig_ds, times_ds))

                if not block_data_list or min_samps is None or min_samps == 0:
                    continue

                # Align and concatenate across blocks up to min_samps
                kids_list = []
                sig_parts = []
                reference_times = None

                for names, sig_ds, times_ds in block_data_list:
                    kids_list.extend(names)
                    sig_trimmed = sig_ds[:, :min_samps].T.astype(float)  # (n_samps, n_channels)
                    sig_parts.append(sig_trimmed)
                    if reference_times is None:
                        reference_times = times_ds[:min_samps]

                kids = kids_list
                sig = np.concatenate(sig_parts, axis=1)  # (n_samps, n_total_dets)
                n_samps = min_samps
                n_dets = len(kids)

                t_start = reference_times[0] / core.G3Units.s
                t_stop = reference_times[-1] / core.G3Units.s

                if sample_rate_ref[0] is None and t_stop > t_start:
                    sample_rate_ref[0] = n_samps / (t_stop - t_start)
                sample_rate = sample_rate_ref[0] if sample_rate_ref[0] is not None else 1.0

                ra_bore = reference_times / core.G3Units.s
                dec_bore = np.zeros_like(ra_bore)
                flags = np.zeros((n_samps, n_dets), dtype=int)

                yield Chunk(
                    kids=kids, signal=sig, common_mode=None,
                    ra=None, dec=None,
                    ra_bore=ra_bore, dec_bore=dec_bore,
                    boresight_q=None, det_dirs=None,
                    t_start=t_start, t_stop=t_stop,
                    sample_rate=sample_rate, flags=flags,
                    chunk_index=-1,
                )


# ============================================================================ #
# _iter_h5_lab_chunks
def _iter_h5_lab_chunks(files: list, chunk_duration_s: float, apply_offsets: bool = True) -> Iterator[Chunk]:
    """Yield fixed-duration Chunks directly from HDF5 lab files."""
    frame_iter = _iter_h5_lab_frames(files, apply_offsets)
    return _rechunk(frame_iter, chunk_duration_s)



# ============================================================================ #
# _precalcChunkTotal
def _precalcChunkTotal(cfg):
    """Estimate the number of chunks without loading the detector timestreams.
    """

    if cfg["data"]["format"] != "lab":
        return {"total_chunks": None}

    files = []
    file_format = cfg["data"]["file_format"]
    for pattern in cfg["data"]["input_dirs"]:
        for directory in sorted(glob.glob(pattern)):
            files.extend(sorted(pathlib.Path(directory).rglob(f"*.{file_format}")))

    if not files:
        return {"total_chunks": None}

    total_samples = 0
    scan_frames = 0
    sample_rate_ref = None
    chunk_sample_rate = None

    for path in files:
        with h5py.File(path, "r") as f:
            if "timestream" not in f:
                continue
            timestream = f["timestream"]
            blocks = list(timestream.keys())
            if not blocks:
                continue

            first_block_frames = timestream[blocks[0]].get("frames")
            if first_block_frames is None:
                continue
            frame_keys = sorted(
                (key for key in first_block_frames.keys() if key.isdigit()),
                key=int,
            )

            for frame_key in frame_keys:
                sample_frame = first_block_frames[frame_key]
                frame_type = sample_frame.attrs.get("frame_type", b"")
                if isinstance(frame_type, bytes):
                    frame_type = frame_type.decode("utf-8")
                if frame_type != "Scan":
                    continue

                min_samples = None
                reference_times = None
                for block in blocks:
                    frame_path = f"timestream/{block}/frames/{frame_key}"
                    if frame_path not in f:
                        continue
                    block_frame = f[frame_path]
                    if (
                        "data" not in block_frame
                        or "data" not in block_frame["data"]
                        or "times" not in block_frame["data"]
                    ):
                        continue

                    times = block_frame["data"]["times"]
                    n_samples = times.shape[0]
                    if min_samples is None or n_samples < min_samples:
                        min_samples = n_samples
                    if reference_times is None:
                        reference_times = times

                if not min_samples:
                    continue

                scan_frames += 1
                total_samples += min_samples

                if sample_rate_ref is None and reference_times is not None:
                    t_start = reference_times[0] / core.G3Units.s
                    t_stop = reference_times[min_samples - 1] / core.G3Units.s
                    if t_stop > t_start:
                        sample_rate_ref = min_samples / (t_stop - t_start)
                if chunk_sample_rate is None:
                    chunk_sample_rate = sample_rate_ref or 1.0

    chunk_duration_s = cfg["pipeline"].get("chunk_duration_s", 1.0)
    if chunk_duration_s <= 0:
        total_chunks = scan_frames
    elif total_samples == 0:
        total_chunks = 0
    else:
        chunk_size = max(1, int(chunk_duration_s * (chunk_sample_rate or 1.0)))
        total_chunks = (total_samples + chunk_size - 1) // chunk_size

    return {"total_chunks": total_chunks}




# ============================================================================ #
# G3 SIMULATION FORMAT
# ============================================================================ #


# ============================================================================ #
# _packObsInfo
def packObsInfo(t_obs_start, t_obs_stop, n_dets, sample_rate, n_chunks):
    """G3 expected obs_info.
    """

    return {
        't_start_g3s' :t_obs_start,
        't_stop_g3s'  :t_obs_stop,
        'duration_s'  :t_obs_stop - t_obs_start,
        'n_detectors' :n_dets,
        'sample_rate_hz' :sample_rate,
        'n_chunks'    :n_chunks,
    }


# ============================================================================ #
# _load_g3_focalplane
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
    dirs      = coord.precompute_det_directions(quats)  # (n_dets, 3), computed once
    det_dirs  = {name: dirs[i] for i, name in enumerate(det_names)}
    return det_names, det_quats, det_dirs


# ============================================================================ #
# _g3_simulation_scan_to_chunk
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

    ra_bore, dec_bore, boresight_r = coord.boresight_to_radec(boresight_q)
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
            det_ra[:, i], det_dec[:, i] = coord.det_radec_from_boresight(boresight_r, det_dirs[kid])

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




# ============================================================================ #
# HDF5 SIMULATION FORMAT
# ============================================================================ #


# ============================================================================ #
# _load_h5_focalplane
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
    dirs      = coord.precompute_det_directions(quats)  # (n_dets, 3), computed once
    det_dirs  = {name: dirs[i] for i, name in enumerate(det_names)}
    return det_names, det_quats, det_dirs


# ============================================================================ #
# _iter_h5_simulation_chunks
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
            ra_bore, dec_bore, boresight_r = coord.boresight_to_radec(boresight_q)

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
                        det_ra[:, i], det_dec[:, i] = coord.det_radec_from_boresight(
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




# ============================================================================ #
# API
# ============================================================================ #


# ============================================================================ #
# iter_chunks
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

    if fmt == "lab":
        chunked_iter = _iter_h5_lab_chunks(files, chunk_duration_s, apply_offsets)

    elif fmt == "simulation" and file_fmt == "h5":
        # HDF5 datasets support direct index slicing, so chunks are read
        # straight off disk -- no frame buffering/_rechunk needed here.
        chunked_iter = _iter_h5_simulation_chunks(files, chunk_duration_s, apply_offsets)

    elif fmt == "simulation":
        frame_iter = _iter_g3_simulation_chunks(files, apply_offsets)
        chunked_iter = _rechunk(frame_iter, chunk_duration_s)
    
    else:
        raise NotImplementedError(
            f"Data format '{fmt}' is not yet implemented. "
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
