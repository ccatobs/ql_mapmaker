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


import io
import glob
import logging
import math
import pathlib
import re
from collections import deque
from typing import Iterator, Optional

import numpy as np
import h5py
from dataclasses import dataclass, replace
from spt3g import core # type: ignore

from mapmaker import tod
# from mapmaker import map
from mapmaker import coord
# from mapmaker import output
# from mapmaker import reader

LOGGER = logging.getLogger("ccat_mapmaker")


# ============================================================================ #
# Class: Chunk
# ============================================================================ #
@dataclass
class Chunk:
    """
    All detector and pointing data for one fixed-duration time window.
    All 2-D arrays are time-first: shape (n_samps, n_dets).

    kids[j] is the name of the detector in column j of signal, ra, dec.
    t_start / t_stop are Unix seconds.
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
def _auto_chunk_samples(
        n_dets: int, sample_rate: float, pipeline_cfg: dict
        ) -> int:
    """Choose a conservative chunk length from the current available memory."""
    if n_dets < 1 or not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Automatic chunk sizing requires detectors and a positive sample rate")

    memory_fraction = pipeline_cfg.get("chunk_memory_fraction", 0.5)
    memory_limit_gb = pipeline_cfg.get("chunk_memory_gb", 0.0)
    if (
        not isinstance(memory_fraction, (int, float))
        or not math.isfinite(memory_fraction)
        or not 0 < memory_fraction <= 1
    ):
        raise ValueError("pipeline.chunk_memory_fraction must be in (0, 1]")
    if (
        not isinstance(memory_limit_gb, (int, float))
        or not math.isfinite(memory_limit_gb)
        or memory_limit_gb < 0
    ):
        raise ValueError("pipeline.chunk_memory_gb must be finite and non-negative")

    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError(
            "Automatic chunk sizing requires psutil; install the project environment"
        ) from exc

    available_bytes = psutil.virtual_memory().available
    target_bytes = available_bytes * memory_fraction
    if memory_limit_gb > 0:
        target_bytes = min(target_bytes, memory_limit_gb * 1024**3)

    # Account conservatively for input arrays, cleaning temporaries, and binning.
    bytes_per_sample = n_dets * 256
    samples = int(target_bytes // bytes_per_sample)
    if samples < 1:
        raise MemoryError(
            "Available chunk-memory budget is too small for even one sample "
            f"across {n_dets} detectors"
        )
    pipeline_cfg.setdefault("_resolved_chunk_durations_s", []).append(
        samples / sample_rate)
    return samples


def _rechunk(
        frame_iter: Iterator[Chunk], chunk_duration_s: float | str,
        pipeline_cfg: dict | None = None
        ) -> Iterator[Chunk]:
    """
    Rebuffer variable-length frame-level Chunks into fixed-duration Chunks.

    Accumulates frames until chunk_n samples are ready, then yields and
    advances the buffer. The final chunk may be shorter than chunk_n.
    If chunk_duration_s <= 0, yields each frame unchanged (frame-aligned mode).
    """
    kids = sample_rate = t_start = det_dirs = None
    chunk_n = None
    chunk_index = 0
    pending: deque[Chunk] = deque()
    pending_offset = 0
    pending_samples = 0

    def take_samples(n_samples: int) -> dict[str, np.ndarray | None]:
        nonlocal pending_offset, pending_samples
        parts: dict[str, list[np.ndarray]] = {
            "signal": [],
            "ra": [],
            "dec": [],
            "ra_bore": [],
            "dec_bore": [],
            "boresight_q": [],
            "flags": [],
        }
        remaining = n_samples
        while remaining:
            frame = pending[0]
            take = min(remaining, frame.signal.shape[0] - pending_offset)
            end = pending_offset + take
            for name in parts:
                arr = getattr(frame, name)
                if arr is not None:
                    parts[name].append(arr[pending_offset:end])
            pending_offset = end
            remaining -= take
            pending_samples -= take
            if pending_offset == frame.signal.shape[0]:
                pending.popleft()
                pending_offset = 0

        return {
            name: (
                values[0] if len(values) == 1
                else np.concatenate(values, axis=0) if values
                else None
            )
            for name, values in parts.items()
        }

    for frame in frame_iter:
        if kids is None:
            kids        = frame.kids
            sample_rate = frame.sample_rate
            t_start = frame.t_start
            det_dirs    = frame.det_dirs
            if chunk_duration_s == "auto":
                if pipeline_cfg is None:
                    raise ValueError("Automatic chunk sizing requires pipeline configuration")
                chunk_n = _auto_chunk_samples(
                    len(kids), sample_rate, pipeline_cfg)
                print(
                    f" | Auto chunk size: {chunk_n} samples "
                    f"({chunk_n / sample_rate:.2f} s)"
                )
            elif chunk_duration_s > 0:
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

        if frame.signal.shape[0] == 0:
            continue

        pending.append(frame)
        pending_samples += frame.signal.shape[0]

        assert kids is not None and sample_rate is not None and t_start is not None
        while pending_samples >= chunk_n:
            arrays = take_samples(chunk_n)
            t_stop = t_start + chunk_n / sample_rate
            yield Chunk(
                kids=kids, signal=arrays["signal"], common_mode=None,
                ra=arrays["ra"], dec=arrays["dec"],
                ra_bore=arrays["ra_bore"], dec_bore=arrays["dec_bore"],
                boresight_q=arrays["boresight_q"],
                det_dirs=det_dirs,
                t_start=t_start, t_stop=t_stop, flags=arrays["flags"],
                sample_rate=sample_rate, chunk_index=chunk_index,
            )
            chunk_index += 1
            t_start = t_stop

    if pending_samples > 0:
        assert kids is not None and sample_rate is not None and t_start is not None
        n = pending_samples
        arrays = take_samples(n)
        t_stop = t_start + n / sample_rate
        yield Chunk(
            kids=kids, signal=arrays["signal"], common_mode=None,
            ra=arrays["ra"], dec=arrays["dec"],
            ra_bore=arrays["ra_bore"], dec_bore=arrays["dec_bore"],
            boresight_q=arrays["boresight_q"],
            det_dirs=det_dirs,
            t_start=t_start, t_stop=t_stop, flags=arrays["flags"],
            sample_rate=sample_rate, chunk_index=chunk_index,
        )




# ============================================================================ #
# H5 LAB
# ============================================================================ #


def _lab_channel_pairs(names: list[str], block: str) -> list[tuple[str, int, int]]:
    """Match the block-local resonator IDs to their I and Q channel rows."""
    pairs: dict[int, dict[str, int]] = {}
    for row, name in enumerate(names):
        match = re.fullmatch(r"r(\d+)_([IQ])", name)
        if match is None:
            raise ValueError(f"Lab block {block}: unrecognized IQ channel {name!r}")
        index, component = int(match.group(1)), match.group(2)
        pair = pairs.setdefault(index, {})
        if component in pair:
            raise ValueError(f"Lab block {block}: duplicate {component} channel for r{index:04d}")
        pair[component] = row

    if not pairs or sorted(pairs) != list(range(len(pairs))):
        raise ValueError(f"Lab block {block}: resonator IDs must be contiguous starting at r0000")
    if any(set(pair) != {"I", "Q"} for pair in pairs.values()):
        raise ValueError(f"Lab block {block}: every resonator must have both I and Q channels")
    return [
        (f"{block}:r{index:04d}", pairs[index]["I"], pairs[index]["Q"])
        for index in range(len(pairs))
    ]


def _load_lab_calibration(f: h5py.File, block: str, n_dets: int):
    """Load one target sweep and optional timestream frequency comb for a block."""
    target_path = f"targ/{block}"
    if target_path not in f or not isinstance(f[target_path], h5py.Group):
        raise ValueError(f"Lab block {block}: missing target sweep group {target_path}")
    targets = [
        obj for name, obj in f[target_path].items()
        if "targ" in name and isinstance(obj, h5py.Dataset)
    ]
    if len(targets) != 1:
        raise ValueError(f"Lab block {block}: expected exactly one target sweep, found {len(targets)}")
    sweep = np.asarray(targets[0])
    if (
        sweep.ndim != 2 or sweep.shape[0] != 2
        or sweep.shape[1] % n_dets != 0 or sweep.shape[1] // n_dets < 2
        or not np.iscomplexobj(sweep) or not np.all(np.isfinite(sweep))
        or np.any(sweep[0].imag != 0)
    ):
        raise ValueError(f"Lab block {block}: invalid target sweep shape or values")
    frequencies = sweep[0].real.reshape(n_dets, -1)
    iq = sweep[1].reshape(n_dets, -1)
    if np.any(frequencies <= 0) or np.any(np.diff(frequencies, axis=1) <= 0):
        raise ValueError(f"Lab block {block}: target sweep frequencies must be positive and increasing")

    comb_path = f"config/{block}/combs"
    combs = []
    if comb_path in f:
        if not isinstance(f[comb_path], h5py.Group):
            raise ValueError(f"Lab block {block}: {comb_path} is not a group")
        combs = [
            obj for name, obj in f[comb_path].items()
            if "tstream_freq_comb" in name and isinstance(obj, h5py.Dataset)
        ]
    if len(combs) > 1:
        raise ValueError(f"Lab block {block}: multiple timestream frequency combs")
    if combs:
        tones = np.asarray(combs[0])
        if (
            tones.shape != (n_dets,) or np.iscomplexobj(tones)
            or not np.all(np.isfinite(tones))
            or np.any(tones < frequencies[:, 0]) or np.any(tones > frequencies[:, -1])
        ):
            raise ValueError(f"Lab block {block}: invalid or out-of-sweep probe-tone frequencies")
        LOGGER.info("Lab block %s probe tones: %s", block, combs[0].name)
    else:
        indices = [tod.find_i_dvmax(z.real, z.imag) for z in iq]
        tones = frequencies[np.arange(n_dets), indices]
        LOGGER.warning(
            "Lab block %s has no timestream frequency comb; using maximum IQ "
            "sweep derivative as the probe-tone estimate", block)

    indices = np.argmin(np.abs(frequencies - tones[:, None]), axis=1)
    offsets = tones - frequencies[np.arange(n_dets), indices]
    neighbors = np.where(
        offsets < 0, indices - 1,
        np.where(indices + 1 < frequencies.shape[1], indices + 1, indices - 1))
    if np.any(iq[np.arange(n_dets), indices] == iq[np.arange(n_dets), neighbors]):
        raise ValueError(f"Lab block {block}: zero IQ sweep gradient at a probe tone")
    return frequencies, iq, tones


_LAB_TICKS_PER_SECOND = 100_000_000
_LAB_GAP_FACTOR = 1.5


@dataclass
class _LabStream:
    block: str
    pairs: list
    calibration: tuple
    times: np.ndarray
    frames: list
    cadence: float


def _index_lab_stream(f: h5py.File, block: str) -> _LabStream:
    frames_path = f"timestream/{block}/frames"
    if frames_path not in f:
        raise ValueError(f"Lab block {block}: missing frames")
    group = f[frames_path]
    times_parts, frames = [], []
    names = pairs = None
    count = 0
    for key in sorted((k for k in group if k.isdigit()), key=int):
        frame = group[key]
        frame_type = frame.attrs.get("frame_type", "")
        if isinstance(frame_type, bytes):
            frame_type = frame_type.decode("utf-8")
        if frame_type != "Scan":
            continue
        if "data" not in frame or any(k not in frame["data"] for k in ("names", "times", "data")):
            raise ValueError(f"Lab block {block} frame {key}: missing IQ data, names or times")
        data = frame["data"]
        frame_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in data["names"][:]]
        timestamps = np.asarray(data["times"])
        if timestamps.ndim != 1 or timestamps.dtype.kind not in "iu" or not len(timestamps):
            raise ValueError(f"Lab block {block} frame {key}: invalid timestamps")
        timestamps = timestamps.astype(np.int64)
        if data["data"].shape != (len(frame_names), len(timestamps)):
            raise ValueError(f"Lab block {block} frame {key}: IQ data shape does not match names/times")
        if names is None:
            names = frame_names
            pairs = _lab_channel_pairs(names, block)
        elif names != frame_names:
            raise ValueError(f"Lab block {block} frame {key}: IQ channel names changed")
        frames.append((count, count + len(timestamps), data["data"].name))
        count += len(timestamps)
        times_parts.append(timestamps)
    if not times_parts or pairs is None:
        raise ValueError(f"Lab block {block}: no scan frames")
    timestamps = np.concatenate(times_parts)
    differences = np.diff(timestamps)
    if len(differences) == 0 or np.any(differences <= 0):
        raise ValueError(f"Lab block {block}: timestamps must be strictly increasing")
    return _LabStream(
        block, pairs, _load_lab_calibration(f, block, len(pairs)), timestamps,
        frames, float(np.median(differences)))


def _load_lab_pointing(f: h5py.File):
    if "hk/frames" not in f:
        raise ValueError("Lab file is missing housekeeping frames")
    samples = []
    for frame in f["hk/frames"].values():
        if "blocks" not in frame:
            continue
        for block in frame["blocks"].values():
            fields = ("times", "Azimuth_current_position", "Elevation_current_position")
            if not all(name in block for name in fields):
                continue
            times = np.asarray(block["times"])
            az = np.asarray(block[fields[1]], dtype=float)
            el = np.asarray(block[fields[2]], dtype=float)
            if (times.ndim != 1 or times.dtype.kind not in "iu" or not len(times)
                    or az.shape != times.shape or el.shape != times.shape
                    or not np.all(np.isfinite(az)) or not np.all(np.isfinite(el))
                    or np.any((az < 0) | (az >= 360)) or np.any((el < -90) | (el > 90))):
                raise ValueError(f"Invalid lab housekeeping pointing in {block.name}")
            samples.append((times.astype(np.int64), az, el))
    if not samples:
        raise ValueError("Lab file has no housekeeping azimuth/elevation samples")
    times = np.concatenate([s[0] for s in samples])
    az = np.concatenate([s[1] for s in samples])
    el = np.concatenate([s[2] for s in samples])
    order = np.argsort(times)
    times, az, el = times[order], az[order], el[order]
    differences = np.diff(times)
    if len(differences) == 0 or np.any(differences <= 0):
        raise ValueError("Lab housekeeping pointing timestamps must be strictly increasing")
    return times, np.unwrap(np.deg2rad(az)), el, float(np.median(differences))


def _lab_brackets(times: np.ndarray, targets: np.ndarray, cadence: float):
    """Find valid interpolation brackets without spanning missing samples."""
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    exact = np.abs(times[right] - targets) < 0.01
    left = np.where(exact, right, right - 1)
    valid = (left >= 0) & (targets >= times[0]) & (targets <= times[-1])
    left = np.maximum(left, 0)
    valid &= exact | ((times[right] - times[left]) <= _LAB_GAP_FACTOR * cadence)
    return left, right, valid


def _read_lab_aligned(f: h5py.File, stream: _LabStream, origin: int,
                      targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = (stream.times - origin).astype(float)
    left, right, valid = _lab_brackets(times, targets, stream.cadence)
    start, stop = int(left.min()), int(right.max()) + 1
    parts = [
        np.asarray(f[path][:, max(start, lo) - lo:min(stop, hi) - lo])
        for lo, hi, path in stream.frames if lo < stop and hi > start
    ]
    raw = np.concatenate(parts, axis=1)
    source_times = times[start:stop]
    frequencies, iq, tones = stream.calibration
    signal = np.empty((len(targets), len(stream.pairs)), dtype=float)
    for j, (name, i_row, q_row) in enumerate(stream.pairs):
        converted, _ = tod.iq_to_df_hybrid(
            raw[i_row], raw[q_row], iq[j].real, iq[j].imag,
            frequencies[j], f_tone=tones[j])
        if not np.all(np.isfinite(converted)):
            raise ValueError(f"Lab block {stream.block}: non-finite df for {name}")
        signal[:, j] = np.interp(targets, source_times, converted)
    signal[~valid, :] = np.nan
    flags = np.zeros(signal.shape, dtype=int)
    flags[~valid, :] = tod.GAP_FLAG
    return signal, flags


def _iter_h5_lab_chunks(
        files: list, chunk_duration_s: float | str, site,
        pipeline_cfg: dict | None = None) -> Iterator[Chunk]:
    """Yield lab chunks on a shared timestamp grid with housekeeping pointing."""
    expected_kids = None
    sample_rate_ref = None
    last_stop = None
    chunk_index = 0
    for path in files:
        with h5py.File(path, "r") as f:
            if "timestream" not in f or not len(f["timestream"]):
                raise ValueError(f"Lab file {path}: missing timestream blocks")
            streams = [_index_lab_stream(f, block) for block in f["timestream"]]
            kids = [name for stream in streams for name, _, _ in stream.pairs]
            if expected_kids is None:
                expected_kids = kids
            elif kids != expected_kids:
                raise ValueError(f"Lab file {path}: detector list changed")
            origin = max(int(stream.times[0]) for stream in streams)
            end = min(int(stream.times[-1]) for stream in streams)
            if end <= origin:
                raise ValueError(f"Lab file {path}: detector blocks have no common time interval")
            cadence = streams[0].cadence
            sample_rate = _LAB_TICKS_PER_SECOND / cadence
            if sample_rate_ref is not None and not np.isclose(sample_rate, sample_rate_ref, rtol=1e-4):
                raise ValueError(f"Lab file {path}: sample rate changed between files")
            sample_rate_ref = sample_rate
            if last_stop is not None and origin <= last_stop:
                raise ValueError(f"Lab file {path}: observations overlap or are out of time order")
            n_samples = int(np.floor((end - origin) / cadence)) + 1
            hk_times, hk_az, hk_el, hk_cadence = _load_lab_pointing(f)
            hk_times = (hk_times - origin).astype(float)
            if chunk_duration_s == "auto":
                if pipeline_cfg is None:
                    raise ValueError("Automatic chunk sizing requires pipeline configuration")
                chunk_n = _auto_chunk_samples(len(kids), sample_rate, pipeline_cfg)
            elif chunk_duration_s > 0:
                chunk_n = max(1, int(chunk_duration_s * sample_rate))
            else:
                chunk_n = None
            if chunk_n is None:
                grid = np.arange(n_samples, dtype=float) * cadence
                boundaries = [0]
                for _, hi, _ in streams[0].frames:
                    frame_end = float(streams[0].times[hi - 1] - origin)
                    boundary = int(np.searchsorted(grid, frame_end, side="right"))
                    if boundary > boundaries[-1]:
                        boundaries.append(boundary)
                if boundaries[-1] < n_samples:
                    boundaries.append(n_samples)
                windows = zip(boundaries[:-1], boundaries[1:])
            else:
                windows = ((start, min(start + chunk_n, n_samples))
                           for start in range(0, n_samples, chunk_n))
            for start, stop in windows:
                targets = np.arange(start, stop, dtype=float) * cadence
                _, _, pointing_valid = _lab_brackets(hk_times, targets, hk_cadence)
                if not np.all(pointing_valid):
                    raise ValueError(f"Lab file {path}: housekeeping pointing does not cover the detector timestamps")
                az = np.rad2deg(np.interp(targets, hk_times, hk_az)) % 360
                el = np.interp(targets, hk_times, hk_el)
                unix_times = origin / _LAB_TICKS_PER_SECOND + targets / _LAB_TICKS_PER_SECOND
                ra, dec = coord.altaz_to_radec(az, el, unix_times, site)
                signals, flags = zip(*[
                    _read_lab_aligned(f, stream, origin, targets) for stream in streams])
                t_start = origin / _LAB_TICKS_PER_SECOND + start / sample_rate
                t_stop = origin / _LAB_TICKS_PER_SECOND + stop / sample_rate
                yield Chunk(
                    kids=kids, signal=np.concatenate(signals, axis=1), common_mode=None,
                    ra=None, dec=None, ra_bore=ra, dec_bore=dec,
                    boresight_q=None, det_dirs=None, t_start=t_start, t_stop=t_stop,
                    sample_rate=sample_rate, flags=np.concatenate(flags, axis=1),
                    chunk_index=chunk_index)
                chunk_index += 1
            last_stop = origin + int(round((n_samples - 1) * cadence))


def _precalcChunkTotal(cfg):
    """Chunk counts are not known until input timestamps have been indexed."""
    return {"total_chunks": None}




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
def _iter_h5_simulation_chunks(
        files: list, chunk_duration_s: float | str,
        apply_offsets: bool = True, pipeline_cfg: dict | None = None
        ) -> Iterator[Chunk]:
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
            if chunk_duration_s == "auto":
                if pipeline_cfg is None:
                    raise ValueError("Automatic chunk sizing requires pipeline configuration")
                chunk_n = _auto_chunk_samples(
                    len(det_names), sample_rate, pipeline_cfg)
                if chunk_index == 0:
                    print(
                        f" | Auto chunk size: {chunk_n} samples "
                        f"({chunk_n / sample_rate:.2f} s)"
                    )
            else:
                chunk_n = (
                    max(1, int(chunk_duration_s * sample_rate))
                    if chunk_duration_s > 0 else n_samps
                )

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
    cfg["_input_files_discovered"] = [str(path.resolve()) for path in files]

    def track_opened_files(paths):
        opened = cfg.setdefault("_input_files_opened", [])
        for path in paths:
            resolved = str(path.resolve())
            if resolved not in opened:
                opened.append(resolved)
            yield path

    files = track_opened_files(files)
    fmt = cfg["data"]["format"]
    chunk_duration_s = cfg["pipeline"].get("chunk_duration_s", 1.0)
    if isinstance(chunk_duration_s, str) and chunk_duration_s != "auto":
        raise ValueError("pipeline.chunk_duration_s must be seconds or 'auto'")
    pipeline_cfg = cfg["pipeline"]
    sample_step = pipeline_cfg.get("sample_step", 1)
    if isinstance(sample_step, bool) or not isinstance(sample_step, int) or sample_step < 1:
        raise ValueError("pipeline.sample_step must be a positive integer")
    start_offset_s   = cfg["pipeline"].get("start_offset_s", 0.0)
    max_duration_s   = cfg["pipeline"].get("max_duration_s", None)

    if fmt == "lab":
        if file_fmt != "h5":
            raise ValueError("Lab data currently requires HDF5 input")
        map_cfg = cfg.get("map", {})
        if "site" not in map_cfg and not any(
                key in map_cfg for key in ("site_lat_deg", "site_lon_deg", "site_height_m")):
            raise ValueError("Lab pointing requires an explicit map.site or site latitude, longitude and height")
        site = coord.resolveSite(cfg)
        if site is None:
            raise ValueError(f"Unknown lab observing site {map_cfg.get('site')!r}")
        LOGGER.warning("Lab detector pointing offsets are unavailable; using boresight pointing")
        chunked_iter = _iter_h5_lab_chunks(
            files, chunk_duration_s, site, pipeline_cfg)

    elif fmt == "simulation" and file_fmt == "h5":
        # HDF5 datasets support direct index slicing, so chunks are read
        # straight off disk -- no frame buffering/_rechunk needed here.
        chunked_iter = _iter_h5_simulation_chunks(
            files, chunk_duration_s, apply_offsets, pipeline_cfg)

    elif fmt == "simulation":
        frame_iter = _iter_g3_simulation_chunks(files, apply_offsets)
        chunked_iter = _rechunk(frame_iter, chunk_duration_s, pipeline_cfg)
    
    else:
        raise NotImplementedError(
            f"Data format '{fmt}' is not yet implemented. "
        )

    t_obs_start = None
    selected_samples = 0
    for chunk in chunked_iter:
        if t_obs_start is None:
            t_obs_start = chunk.t_start
        elapsed = chunk.t_stop - t_obs_start
        if elapsed <= start_offset_s:
            continue
        if max_duration_s is not None and (chunk.t_start - t_obs_start) >= start_offset_s + max_duration_s:
            break
        if sample_step > 1:
            offset = (-selected_samples) % sample_step
            selected_samples += chunk.signal.shape[0]
            if offset >= chunk.signal.shape[0]:
                continue
            sample_slice = slice(offset, None, sample_step)
            arrays = {}
            for name in ("signal", "flags", "ra", "dec", "ra_bore", "dec_bore",
                         "boresight_q", "common_mode"):
                value = getattr(chunk, name)
                arrays[name] = value[sample_slice] if value is not None else None
            chunk = replace(chunk, **arrays, sample_rate=chunk.sample_rate / sample_step)
        yield chunk
