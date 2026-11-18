#!/usr/bin/env python
# ============================================================================ #
# run_mapmaker.py
#
# James Burgoyne, jamesrburgoyne@icloud.com
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT 2026
#
# CCAT Quick-Look Mapmaker -- main entry point.

# use config.toml: python run_mapmaker.py
# use other config: python run_mapmaker.py --config my_config.toml
# print timing bottlenecks: python run_mapmaker.py --profile
#
# Pipeline:
# 1. First pass -- compute per-detector baselines and map centre
# 2. Naive map -- bin cleaned signal with no common-mode subtraction
# 3. Initial CM -- subtract naive mean across detectors, rebin
# 4. Iterate -- subtract sky-informed common mode, rebin (n_iterations times)
# 5. Save -- write maps and metadata to disk
# ============================================================================ #

import os
import io
import sys
import time
import math
import logging
import pstats
import pathlib
import argparse
import cProfile
import pprint
import copy
import importlib.metadata
import inspect
import platform
import subprocess
import numpy as np
from scipy.signal import welch
from scipy.ndimage import gaussian_filter
import tomllib as toml
from datetime import datetime, timedelta, timezone

from spt3g import core # type: ignore

from mapmaker import tod
from mapmaker import map
from mapmaker import coord
from mapmaker import output
from mapmaker import reader

LOGGER = logging.getLogger("ccat_mapmaker")


# ============================================================================ #
# _resolvePaths
def _resolvePaths(cfg: dict, config_path: pathlib.Path) -> dict:
    cfg_dir = config_path.parent.resolve()

    def resolve(path_str: str) -> str:
        p = pathlib.Path(path_str)
        if not p.is_absolute():
            p = (cfg_dir / p).resolve()
        return str(p)

    cfg["data"]["input_dirs"] = [resolve(d) for d in cfg["data"]["input_dirs"]]
    cfg["output"]["output_dir"] = resolve(cfg["output"]["output_dir"])

    return cfg


# ============================================================================ #
# _loadConfiguration
def _loadConfiguration(config_path_str: str) -> tuple[dict, bytes]:
    """Load the TOML configuration file and retain its original bytes.

    config_path_str: Absolute or relative path string.
    """

    config_path = pathlib.Path(config_path_str)
    config_bytes = config_path.read_bytes()
    cfg = toml.loads(config_bytes.decode("utf-8"))
    cfg = _resolvePaths(cfg, config_path)

    print(f" Config : {config_path}")
    for d in cfg['data']['input_dirs']:
        print(f" Input  : {d}")
    print(f" Format : {cfg['data']['format']}")
    print(f" Output : {cfg['output']['output_dir']}")

    return cfg, config_bytes


# ============================================================================ #
# _setupArgParser
def _setupArgParser():
    """Setup CLI arguments.
    """

    parser = argparse.ArgumentParser(description="CCAT Quick-Look Mapmaker")
    parser.add_argument("--config", default="config.toml", help="Path to config file (default: config.toml)")
    parser.add_argument("--profile", action="store_true", help="Print timing bottlenecks on exit")
    args = parser.parse_args()

    return args


# ============================================================================ #
# _setupProfiler
def _setupProfiler(args):
    """Setup code timing profiling.
    """

    profiler = cProfile.Profile()

    if args.profile:
        profiler.enable()
        
    return profiler


def _setupRunLogger(log_path: pathlib.Path):
    """Write detailed run diagnostics to a file, without logging per chunk."""
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    LOGGER.addHandler(handler)
    return handler


def _softwareProvenance():
    versions = {}
    for package in ("numpy", "scipy", "astropy", "matplotlib", "h5py",
                    "spt3g", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    provenance = {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "packages": versions,
    }
    try:
        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(pathlib.Path(__file__).resolve().parent), *args],
                text=True, stderr=subprocess.PIPE, timeout=10).strip()

        provenance["git"] = {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        provenance["git"] = {"error": str(exc)}
    return provenance


def _effectiveSettings(cfg):
    """Record defaults used by the main map passes, without changing behavior."""
    settings = copy.deepcopy(cfg)
    pipe = settings["pipeline"]
    for key, value in {
        "start_offset_s": 0.0, "max_duration_s": None,
        "chunk_duration_s": 1.0, "chunk_memory_fraction": 0.5,
        "chunk_memory_gb": 0.0, "auto_exclude_threshold": 5.0,
        "wnf_exclude_sigma": 3.0, "weight_cap_factor": 5.0,
        "compute_time_null": True, "compute_detsplit_null": True,
        "exclude_detectors": [], "exclude_detector_indices": [],
    }.items():
        pipe.setdefault(key, value)
    for key, value in {"save_numpy": True, "save_png": True, "save_psd": True}.items():
        settings["output"].setdefault(key, value)
    return settings


def _activeCleaningSteps(pipe):
    steps = [{"name": "gap_fill", "parameters": {"min_cubic_gap": 4}}]
    for name, function in (("cosmic_rays", tod._removeCosmicRays),
                           ("highpass", tod._highpassFilter),
                           ("notch", tod._notchFilter)):
        if pipe.get(name) is not None:
            defaults = {
                key: param.default for key, param in inspect.signature(function).parameters.items()
                if param.default is not inspect.Parameter.empty
            }
            defaults.update(pipe[name])
            steps.append({"name": name, "parameters": defaults})
    return steps




# ============================================================================ #
# == TOOLS ==
# ============================================================================ #


# ============================================================================ #
# _printLoadingBar
def _printLoadingBar(iteration=1, total=None, length=40, clear=False):
    """Print progress, using a spinner when the total chunk count is unknown.

    Args:
        iteration (int): Current step or iteration index.
        total (int | None): Total steps, or None when unknown.
        length (int, optional): Character length of the progress bar. Defaults to 40.
    """

    if clear:
        line_width = 80
        sys.stdout.write(f'\r{" " * line_width}\r')
        sys.stdout.flush()
        return

    if total is None or total <= 0:
        spinner = "|/-\\"[iteration % 4]
        status = f"{spinner} Processed {iteration} chunks"
    else:
        percent = min(iteration / total, 1.0) * 100
        filled_length = int(length * min(iteration, total) // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        status = f'|{bar}| {percent:.1f}% Complete'
    sys.stdout.write(f'\r{status:<79}')
    sys.stdout.flush()


# ============================================================================ #
# _fastPeriodogram
def _fastPeriodogram(signal, window, window_scale, med, length):
    """This is used to replace the slow Welch.
    """

    # We need to remove the baseline to avoid PSD spike
    signal_centered = signal - med

    windowed_signal = signal_centered * window
    fft_res = np.fft.rfft(windowed_signal, axis=0)
    
    # Compute power spectral density
    p = (np.abs(fft_res)**2) / window_scale
    if length % 2 == 0:
        p[1:-1, :] *= 2
    else:
        p[1:, :] *= 2

    return p


class _PsdComparison:
    """Accumulate matched, clean Welch segments for a small detector sample."""

    def __init__(self, names):
        self.indices = np.linspace(0, len(names) - 1, min(16, len(names)), dtype=int)
        self.names = [str(names[i]) for i in self.indices]
        self.nperseg = None
        self.freqs = None
        self.before_sum = None
        self.after_sum = None
        self.counts = np.zeros(len(self.indices), dtype=int)
        self.candidate_segments = 0
        self.sample_rate = None
        self.nonfinite_spectra = 0
        self.skipped_sample_rate_chunks = 0

    def add(self, before, after, flag_mask, sample_rate):
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            self.skipped_sample_rate_chunks += 1
            return
        if self.nperseg is None:
            if before.shape[0] < 4:
                return
            self.nperseg = min(2048, before.shape[0])
            self.sample_rate = sample_rate
            self.freqs = np.fft.rfftfreq(self.nperseg, 1 / sample_rate)
            shape = (len(self.freqs), len(self.indices))
            self.before_sum = np.zeros(shape)
            self.after_sum = np.zeros(shape)
        elif not np.isclose(sample_rate, self.sample_rate, rtol=1e-8, atol=0):
            self.skipped_sample_rate_chunks += 1
            return

        step = max(1, self.nperseg // 2)
        for start in range(0, before.shape[0] - self.nperseg + 1, step):
            self.candidate_segments += 1
            stop = start + self.nperseg
            raw = before[start:stop]
            cleaned = after[start:stop, self.indices]
            flags = flag_mask[start:stop, self.indices]
            valid = np.all(np.isfinite(raw) & np.isfinite(cleaned) & np.isfinite(flags), axis=0)
            if not np.any(valid):
                continue
            _, p_before = welch(raw[:, valid], fs=sample_rate, window="hann",
                                nperseg=self.nperseg, noverlap=0, axis=0)
            _, p_after = welch(cleaned[:, valid], fs=sample_rate, window="hann",
                               nperseg=self.nperseg, noverlap=0, axis=0)
            finite = np.all(np.isfinite(p_before) & np.isfinite(p_after), axis=0)
            self.nonfinite_spectra += int(np.count_nonzero(~finite))
            indices = np.flatnonzero(valid)[finite]
            self.before_sum[:, indices] += p_before[:, finite]
            self.after_sum[:, indices] += p_after[:, finite]
            self.counts[indices] += 1

    def result(self):
        info = {
            "detectors": self.names, "segment_counts": self.counts.tolist(),
            "candidate_segments": self.candidate_segments,
            "nonfinite_spectral_segments": self.nonfinite_spectra,
            "skipped_sample_rate_chunks": self.skipped_sample_rate_chunks,
            "nperseg": self.nperseg, "window": "Hann (periodic)",
            "overlap_samples": self.nperseg // 2 if self.nperseg else None,
            "detrend": "constant",
        }
        if self.freqs is None or not np.any(self.counts):
            info["status"] = "unavailable: no valid matched segments"
            return info, None
        valid = self.counts > 0
        info["status"] = "available"
        info["frequency_resolution_hz"] = float(self.freqs[1] - self.freqs[0])
        return info, {
            "frequency_hz": self.freqs,
            "before": self.before_sum[:, valid] / self.counts[valid],
            "after": self.after_sum[:, valid] / self.counts[valid],
            "detectors": np.asarray(self.names)[valid],
            "segment_counts": self.counts[valid],
        }


# ============================================================================ #
# _resolve_det_selection
def _resolve_det_selection(all_kids: list, pd_cfg: dict) -> tuple[list, np.ndarray]:
    n_total = len(all_kids)
    name_to_idx = {k: i for i, k in enumerate(all_kids)}

    names = pd_cfg.get("detectors", [])
    if names:
        valid = [(n, name_to_idx[n]) for n in names if n in name_to_idx]
        missing = [n for n in names if n not in name_to_idx]
        if missing:
            print(f"  Warning: {len(missing)} detector name(s) not found and skipped: {missing[:5]}")
        kids_sel = [n for n, _ in valid]
        sel_idx = np.array([i for _, i in valid], dtype=int)
        return kids_sel, sel_idx

    indices = pd_cfg.get("detector_indices", [])
    if indices:
        valid = [i for i in indices if 0 <= i < n_total]
        if len(valid) < len(indices):
            print(f"  Warning: {len(indices)-len(valid)} index/indices out of range and skipped")
        sel_idx = np.array(valid, dtype=int)
        kids_sel = [all_kids[i] for i in sel_idx]
        return kids_sel, sel_idx

    max_det = pd_cfg.get("max_detectors", 0)
    if max_det > 0 and max_det < n_total:
        sel_idx = np.linspace(0, n_total - 1, max_det, dtype=int)
    else:
        sel_idx = np.arange(n_total)
    kids_sel = [all_kids[i] for i in sel_idx]
    return kids_sel, sel_idx


# ============================================================================ #
# _stridedMedian
def _stridedMedian(sig, min_samples=100, axis=0):
    """Compute a strided median along a specified axis for fast estimation.
    Nan safe.

    sig (array of floats): Input array containing the data stream.
    min_samples (int): Minimum samples to use in median calculation.
    axis (int): Axis along which the median is computed. Default is 0.

    Returns:
        (array of floats): Median array computed on the strided data.
    """

    # Total number of samples in the signal.
    num_samples = sig.shape[axis]

    # Fast stride calculation
    def fastPowerOfTwo(n, min_samples):
        if n <= min_samples:
            return 1
        return 1 << math.floor(math.log2(n/min_samples))
    stride_size = fastPowerOfTwo(num_samples, min_samples)

    # Dynamic stride slice along given axis
    slices = [slice(None)] * sig.ndim
    slices[axis] = slice(None, None, stride_size)

    # Nan aware median calculation
    med = np.nanmedian(sig[tuple(slices)], axis=axis)

    return med




# ============================================================================ #
# == PASSES ==
# ============================================================================ #


# ============================================================================ #
# _prepPass
def _prepPass(cfg: dict): 
    """Streaming loop over all data chunks to compute:
    - Boresight pointing averages (for map center fallback)
    - Per-detector medians and noise estimates (via temporal differences)
    - Average Power Spectral Density (PSD) and white noise floor characteristics
    """

    # Stream through chunks of time-ordered data (TOD)
    init = False
    stats = {"chunk_samples": [], "detector_samples": 0,
             "input_flagged_samples": 0, "fully_flagged_chunks": 0,
             "psd_eligible_time_samples": 0}
    for i,chunk in enumerate(reader.iter_chunks(cfg)): # chunk_id=chunk.chunk_index

        _printLoadingBar(i + 1, cfg.get('total_chunks'), length=40)
        stats["chunk_samples"].append(int(chunk.signal.shape[0]))
        stats["detector_samples"] += chunk.flags.size
        stats["input_flagged_samples"] += int(np.count_nonzero(chunk.flags))

        # Skip chunks where all detector data is flagged out
        if np.all(chunk.flags != 0):
            stats["fully_flagged_chunks"] += 1
            continue 
        stats["psd_eligible_time_samples"] += int(chunk.signal.shape[0])

        # Initialize on first valid chunk
        # ------------------------------------------------------ #
        if not init:

            t_obs_start       = chunk.t_start
            n_dets            = chunk.signal.shape[1]
            sample_rate       = chunk.sample_rate
            kids              = chunk.kids
            nominal_chunk_len = chunk.signal.shape[0]

            ra_sum = dec_sum = n_bore = n_chunks = psd_count = 0

            det_median_sum = np.zeros(n_dets)
            det_diff_sum = np.zeros(n_dets)
            det_diff_sum_sq = np.zeros(n_dets)
            det_diff_count = np.zeros(n_dets, dtype=int)

            # Percalcuations for the FFT periodgram
            window = np.hanning(nominal_chunk_len)[:, None]
            window_scale = sample_rate * np.sum(window**2)
            freqs = np.fft.rfftfreq(nominal_chunk_len, d=1.0/sample_rate)

            psd_sum = None # will be initialized in Sxx section

            init = True
        # ------------------------------------------------------ #

        # Accumulate only differences whose two endpoint samples are usable.
        diffs = np.subtract(chunk.signal[1:], chunk.signal[:-1], dtype=float)
        valid_mask = (
            np.isfinite(diffs)
            & (chunk.flags[1:] == 0)
            & (chunk.flags[:-1] == 0)
        )

        # Advance accumulations (t=0.53 s)
        ra_sum          += chunk.ra_bore.sum()
        dec_sum         += chunk.dec_bore.sum()
        n_bore          += len(chunk.ra_bore)
        det_diff_sum += np.sum(diffs, axis=0, where=valid_mask)
        np.square(diffs, out=diffs)
        det_diff_sum_sq += np.sum(diffs, axis=0, where=valid_mask)
        det_diff_count  += np.sum(valid_mask, axis=0)
        n_chunks        += 1

        # We are using a strided median to reduce computation.
        # Dynamically determines stride size for statistical robustness.
        # det_median = np.nanmedian(chunk.signal[::4, :], axis=0)
        det_median = _stridedMedian(chunk.signal, axis=0)
        det_median_sum  += det_median

        # Sxx, adding per loop.
        # The Welch is the second bottle neck (t=0.16 s).
        # _fastPeriodogram is slightly faster (t=0.13 s). 
        if chunk.signal.shape[0] == nominal_chunk_len:
            # f, p = welch(chunk.signal, fs=sample_rate, window="hann", axis=0)
            p = _fastPeriodogram(
                chunk.signal, window, window_scale, 
                det_median, nominal_chunk_len)
            if psd_sum is None:
                psd_sum = np.zeros_like(p)
                psd_freqs = freqs
            psd_sum += p
            psd_count += 1

        # stop time is from last chunk
        t_obs_stop = chunk.t_stop

    _printLoadingBar(clear=True)

    # Observation telemetry
    obs_info = reader.packObsInfo(
        t_obs_start, t_obs_stop, n_dets, sample_rate, n_chunks)

    # Calculate detector noise variance from sample differences.
    # Scaled by 1/sqrt{2} to correct for the variance doubling introduced by first-order differencing.
    det_noise = tod.findNoise(
        det_diff_sum, det_diff_sum_sq, det_diff_count)/np.sqrt(2)

    # Average Sxx across chunks
    psd_avg = psd_sum/psd_count if psd_count>0 else None
    stats.update({
        "psd_chunk_samples": nominal_chunk_len,
        "psd_chunks_used": psd_count,
        "psd_time_samples_used": psd_count * nominal_chunk_len,
        "psd_frequency_resolution_hz": float(sample_rate / nominal_chunk_len),
        "psd_input_flags_applied": False,
        "psd_nonfinite_detectors": (
            int(np.count_nonzero(~np.all(np.isfinite(psd_avg), axis=0)))
            if psd_avg is not None else None),
    })
    if stats["psd_time_samples_used"] < stats["psd_eligible_time_samples"]:
        LOGGER.warning(
            "Weighting PSD used %d of %d eligible time samples; only chunks of %d samples contribute",
            stats["psd_time_samples_used"], stats["psd_eligible_time_samples"], nominal_chunk_len)
    if stats["input_flagged_samples"]:
        LOGGER.warning("Weighting PSD does not mask input flags; flagged samples may affect detector weights")
    if stats["psd_nonfinite_detectors"]:
        LOGGER.warning("Weighting PSD is non-finite for %d detectors", stats["psd_nonfinite_detectors"])

    # Extract the white noise floor
    white_noise_floor = tod.findWhiteNoiseFloor(
        psd_avg, psd_freqs, sample_rate)

    state = {
        "ra0_auto": ra_sum/n_bore, 
        "dec0_auto": dec_sum/n_bore, 
        "det_offsets": det_median_sum/n_chunks, 
        "det_noise": det_noise, 
        "all_kids": kids, 
        "obs_info": obs_info, 
        "white_noise_floor": white_noise_floor, 
        "psd_avg": psd_avg, 
        "psd_freqs": psd_freqs,
        "processing_stats": stats,
    }
    return state


    # cfg: dict, 
    # pipe_cfg: dict,
    # ra_edges: np.ndarray, 
    # dec_edges: np.ndarray,
    # det_offsets: np.ndarray,
    # current_map: np.ndarray = None,
    # common_mode: bool = True,
    # return_sample: bool = False,
    # keep_idx: np.ndarray = None,
    # boresight_only: bool = False,
    # collect_tod_rms: bool = False,
    # compute_time_null: bool = True,
    # compute_detsplit_null: bool = False,
    # weights: np.ndarray = None):

# ============================================================================ #
# _mapPass
def _mapPass(
    cfgs: dict,
    state_1: dict,
    current_map: np.ndarray = None,
    common_mode: bool = True,
    weights: np.ndarray = None,
    also_naive: bool = False,
    compute_nulls: bool = True,
    collect_psd: bool = False,
    ):

    pipe_cfg = cfgs['pipeline']
    compute_time_null = compute_nulls and pipe_cfg.get("compute_time_null", True)
    compute_detsplit_null = compute_nulls and pipe_cfg.get("compute_detsplit_null", True)
    boresight_only = False
    nx, ny = state_1['nx'], state_1['ny']

    # Excluded detectors
    idx = state_1.get("keep_idx") 
    if idx is None:
        idx = slice(None)
    n_dets_kept = len(state_1["det_offsets_kept"])

    def zerosMap():
        # float32 should be okay for ql maps?
        return np.zeros((ny, nx), dtype=np.float32)
    def zerosMapList(num=2):
        return [zerosMap() for i in range(num)]

    # Initialize tracking variables
    total_data = zerosMap()
    total_hits = zerosMap()
    total_sumsq = zerosMap()
    naive_data = zerosMap() if also_naive else None
    naive_hits = zerosMap() if also_naive else None
    null_data = zerosMapList() if compute_time_null else None
    null_hits = zerosMapList() if compute_time_null else None
    detsplit_data = zerosMapList() if compute_detsplit_null else None
    detsplit_hits = zerosMapList() if compute_detsplit_null else None
    det_half = None
    if compute_detsplit_null:
        if n_dets_kept < 2:
            raise ValueError("Detector-split null requires at least two retained detectors")
        order = np.random.default_rng(0).permutation(n_dets_kept)
        det_half = np.ones(n_dets_kept, dtype=int)
        det_half[order[:n_dets_kept // 2]] = 0
    psd = _PsdComparison(state_1["kids_kept"]) if collect_psd else None
    tod_rms_data: list = []
    stats = {
        "chunk_samples": [], "fully_flagged_chunks": 0,
        "detector_samples": 0, "input_flagged_samples": 0,
        "gap_flagged_samples": 0, "cosmic_ray_flagged_samples": 0,
        "combined_flagged_samples": 0,
    }

    init = False
    for i,chunk in enumerate(reader.iter_chunks(cfgs)):
        chunk_id = chunk.chunk_index
        stats["chunk_samples"].append(int(chunk.signal.shape[0]))

        _printLoadingBar(i + 1, cfgs.get('total_chunks'), length=40)

        # Skip chunks where all detector data is flagged out
        if np.all(chunk.flags != 0):
            stats["fully_flagged_chunks"] += 1
            continue

        # Initialize on first valid chunk
        # ------------------------------------------------------ #
        if not init:

            n_dets = len(chunk.kids)
            sample_rate = chunk.sample_rate

            init = True
        # ------------------------------------------------------ #

        # Timestreams (w/out excluded detectors)
        sig = chunk.signal[:, idx]
        ra = chunk.ra[:, idx] if chunk.ra is not None else None
        dec = chunk.dec[:, idx] if chunk.dec is not None else None
        flags = chunk.flags[:, idx]
        stats["detector_samples"] += flags.size
        stats["input_flagged_samples"] += int(np.count_nonzero(flags))
        
        # Subtract signal bias
        sig -= state_1["det_offsets_kept"][np.newaxis, :]

        # Use boresight w/out unique detector shifts
        if boresight_only or ra is None:
            ra = chunk.ra_bore[:, np.newaxis] + np.zeros(sig.shape[1])
            dec = chunk.dec_bore[:, np.newaxis] + np.zeros(sig.shape[1])

        # Build flag mask: 0->nan
        flag_mask = np.where(flags!=0, np.nan, 1.0)

        # Clean the timestreams (cosmic rays, gaps, etc.) (0.29 s)
        # Not using the flag_mask here 
        # because I don't know what the flags are.
        sig, clean_flags = tod.cleanTod(
            sig, chunk.sample_rate, state_1['det_noise'][idx],
            steps=cfgs.get('pipeline'))
        stats["gap_flagged_samples"] += int(np.count_nonzero(clean_flags & tod.GAP_FLAG))
        stats["cosmic_ray_flagged_samples"] += int(np.count_nonzero(clean_flags & tod.FLAG_COSMIC_RAY))
        stats["combined_flagged_samples"] += int(np.count_nonzero((flags != 0) | (clean_flags != 0)))
        clean_flag_mask = np.where(clean_flags!=0, np.nan, flag_mask)
        psd_before = sig[:, psd.indices].copy() if psd is not None else None

        # print(f"1: {time.perf_counter() - t}")
        # t = time.perf_counter()

        naive_sig = sig if also_naive else None

        # Common mode subtraction
        if common_mode:
            sig = tod.subtractCommonMode(
                sig, flag_mask, 
                ra, dec, state_1["ra_edges"], state_1["dec_edges"],
                current_map)
        if psd is not None:
            psd.add(psd_before, sig, clean_flag_mask, chunk.sample_rate)

        # Make map for this chunk (0.2 s)
        if also_naive:
            naive_d, naive_h, d, h, sq = map.binChunkPair(
                naive_sig, sig, clean_flag_mask, ra, dec,
                state_1["ra_edges"], state_1["dec_edges"],
                weights=weights,
            )
            naive_data += naive_d
            naive_hits += naive_h
        else:
            d, h, sq = map.binChunk(
                sig, clean_flag_mask, ra, dec,
                state_1["ra_edges"], state_1["dec_edges"],
                weights=weights)

        # Add chunk map to running total maps
        total_data += d
        total_hits += h
        total_sumsq += sq

        # Null maps
        if compute_time_null:
            half = chunk_id % 2
            null_data[half] += d
            null_hits[half] += h
        if compute_detsplit_null:
            for dhalf in (0, 1):
                m = det_half == dhalf
                w_half = weights[m] if weights is not None else None
                d_ds, h_ds, _ = map.binChunk(
                    sig[:, m], clean_flag_mask[:, m], ra[:, m], dec[:, m], state_1["ra_edges"], state_1["dec_edges"], 
                    weights=w_half)
                detsplit_data[dhalf] += d_ds
                detsplit_hits[dhalf] += h_ds

    _printLoadingBar(clear=True)

    # 
    map.requireMapCoverage(total_hits, state_1["ra_edges"], state_1["dec_edges"])

    # Create the total maps
    with np.errstate(invalid='ignore', divide='ignore'):
        combined = np.where(total_hits>0, total_data/total_hits, np.nan)
        mean_sq = np.where(total_hits>0, total_sumsq/total_hits, np.nan)
        variance = np.clip(mean_sq - combined ** 2, 0, None)
        noise_map = np.sqrt(variance / total_hits)

        def compute_null(hits, data):
            a = np.where(hits[0] > 0, data[0] / hits[0], np.nan)
            b = np.where(hits[1] > 0, data[1] / hits[1], np.nan)
            return (a - b) / 2.0
        null_map = compute_null(null_hits, null_data) if compute_time_null else None
        detsplit_null_map = compute_null(detsplit_hits, detsplit_data) if compute_detsplit_null else None

        naive = None
        if also_naive:
            naive = np.where(naive_hits>0, naive_data/naive_hits, np.nan)

    for label, null in (("time", null_map), ("detector", detsplit_null_map)):
        if null is not None and not np.isfinite(null).any():
            LOGGER.warning("%s null has no finite overlap between split maps", label)

    return {
        'combined': combined,
        'naive': naive,
        'total_hits': total_hits,
        'noise_map': noise_map,
        'processing_stats': stats,
        'null_map': null_map,
        'detsplit_null_map': detsplit_null_map,
        'time_split_weights': null_hits,
        'detector_split_weights': detsplit_hits,
        'detector_split_assignment': det_half,
        'psd_comparison': psd.result() if psd is not None else None,
        'n_dets': n_dets,
        'sample_rate': sample_rate,
        'tod_rms_data': tod_rms_data
    }


# ============================================================================ #
# _per_detector_pass
def _per_detector_pass(cfg: dict, pipe_cfg: dict, pd_cfg: dict,
                       ra_edges: np.ndarray, dec_edges: np.ndarray,
                       det_offsets: np.ndarray):
    ny = len(dec_edges) - 1
    nx = len(ra_edges) - 1

    apply_offsets = pd_cfg.get("apply_offsets", True)
    kids_sel = sel_idx = None
    det_data = det_hits = None

    for chunk in reader.iter_chunks(cfg, apply_offsets=apply_offsets):
        if kids_sel is None:
            kids_sel, sel_idx = _resolve_det_selection(chunk.kids, pd_cfg)
            n_sel = len(kids_sel)
            det_data = np.zeros((n_sel, ny, nx), dtype=np.float32)
            det_hits = np.zeros((n_sel, ny, nx), dtype=np.float32)

        flags = chunk.flags
        flag_mask = np.ones(np.shape(flags))
        flag_mask[flags != 0] = np.nan
        sig = chunk.signal - det_offsets[np.newaxis, :]

        sig, new_flags = tod.clean_tod(
            sig, flag_mask, chunk.sample_rate, 
            steps=pipe_cfg.get("clean_steps", []), 
            step_params={
                "cosmic_rays": pipe_cfg.get("cosmic_rays", {}),
                "highpass": pipe_cfg.get("highpass", {}),
                "notch": pipe_cfg.get("notch", {}),
            })

        bin_flag_mask = np.where(new_flags != 0, np.nan, flag_mask)

        for j, i in enumerate(sel_idx):
            if apply_offsets:
                ra, dec = chunk.ra[:, i], chunk.dec[:, i]
            else:
                ra, dec = chunk.ra_bore, chunk.dec_bore
            d, h = map.bin_detector(sig[:, i], bin_flag_mask[:, i], ra, dec, ra_edges, dec_edges)
            det_data[j] += d
            det_hits[j] += h

    return kids_sel, det_data, det_hits




# ============================================================================ #
# == STEPS ==
# ============================================================================ #


# ============================================================================ #
# _mapPrep
def _mapPrep(cfg: dict) -> dict:
    """Orchestrates Step 1 of the mapmaking pipeline:
    1. Executes the first data pass to gather baselines, noise, and PSD.
    2. Determine excluded detectors.
    3. Computes inverse-variance detector weights.
    4. Resolves the map center (pinned config vs. target lookup vs. auto-boresight).
    5. Sets up the spatial pixel grid edges.
    """

    t0 = time.perf_counter()
    t = t0
    
    # Run the initial data streaming pass:
    # - Boresight pointing averages (for map center fallback).
    # - Per-detector medians and noise estimates (via temporal differences).
    # - Average PSD and white noise floor characteristics.
    s_fp = _prepPass(cfg) # state var

    print(f" | First pass done. [{time.perf_counter()-t:.1f} s]")
    t = time.perf_counter()

    # Decide which detectors to excluded from maps, based on:
    # - Manual lists.
    # - Automated metrics (noise/std, white noise floor).
    s_de = tod.excludedDetectors( # detector state
        s_fp['all_kids'], s_fp['det_noise'], 
        s_fp['white_noise_floor'], cfg["pipeline"])

    print(f" | Using {len(s_de['keep_idx'])}/{len(s_fp['all_kids'])} detectors")
    if len(s_de["keep_idx"]) == 0:
        raise ValueError("All detectors were excluded; check detector selection and thresholds")
    if cfg["pipeline"].get("compute_detsplit_null", True) and len(s_de["keep_idx"]) < 2:
        raise ValueError("Detector-split null requires at least two retained detectors")
    manual_names = set(cfg["pipeline"].get("exclude_detectors", []))
    manual_indices = set(cfg["pipeline"].get("exclude_detector_indices", []))
    manual_excluded = [
        name for i, name in enumerate(s_fp["all_kids"])
        if name in manual_names or i in manual_indices
    ]

    # Find detector weights.
    # This uses a static weight per detector:
    # The inverse of the white noise floor (0.4 to 0.9 nyqist of PSD).
    # TODO: this uses a median calc
    det_weights = tod.findWeights(
        s_fp['white_noise_floor'], len(s_fp['all_kids']), cfg["pipeline"])

    # Format observation time metadata for logging
    _G3_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
    obs_utc = _G3_EPOCH + timedelta(seconds=s_fp['obs_info']["t_start_g3s"])
    print(f" | Obs start: {obs_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f" | Duration: {s_fp['obs_info']['duration_s']:.1f} s")
    print(f" | Detectors: {s_fp['obs_info']['n_detectors']} @ {s_fp['obs_info']['sample_rate_hz']:.1f} Hz")

    # Determine the central RA/Dec coordinate for the map grid
    s_ma = coord.resolveMapCenter( # map state
        cfg, s_fp['obs_info'], s_fp['ra0_auto'], s_fp['dec0_auto'])

    # Construct coord pixel grid boundaries (edges)
    s_ma |= map.makeMapEdges( # add to map state dict
        ra0_deg=s_ma['ra0_deg'], dec0_deg=s_ma['dec0_deg'],
        xlen_deg=cfg["map"]["xlen_deg"], ylen_deg=cfg["map"]["ylen_deg"],
        res_deg=cfg["map"]["res_arcmin"]/60.0,
    )

    print(f" | Map grid: {s_ma['ny']} x {s_ma['nx']} pixels", end="")
    print(f" ({cfg['map']['res_arcmin']:.1f} arcmin/pixel)", end="")
    print(f", centre = ({s_ma['ra0_deg']:.4f}, {s_ma['dec0_deg']:.4f}) [{s_ma['centre_source']}]")

    # Return a dictionary containing all computed parameters for subsequent pipeline steps
    return {
        "det_offsets": s_fp['det_offsets'],
        "det_noise": s_fp['det_noise'],
        "all_kids": s_fp['all_kids'],
        "obs_info": s_fp['obs_info'],
        "obs_utc": obs_utc,
        "white_noise_floor": s_fp['white_noise_floor'],
        "psd_avg": s_fp['psd_avg'],
        "psd_freqs": s_fp['psd_freqs'],
        "keep_idx": s_de['keep_idx'],
        "exclude_set": s_de['exclude_set'],
        "auto_excluded": s_de['noise_excluded'],
        "manual_excluded": manual_excluded,
        "wnf_excluded": s_de['wnf_excluded'],
        "wnf_cutoff": s_de['wnf_cutoff'],
        "preparation_stats": s_fp["processing_stats"],
        "det_weights": det_weights,
        "ra0_deg": s_ma['ra0_deg'],
        "dec0_deg": s_ma['dec0_deg'],
        "ra_edges": s_ma['ra_edges'],
        "dec_edges": s_ma['dec_edges'],
        "ny": s_ma['ny'],
        "nx": s_ma['nx'],
        "kids_kept": [s_fp['all_kids'][i] for i in s_de['keep_idx']],
        "det_weights_kept": det_weights[s_de['keep_idx']],
        "det_offsets_kept": s_fp['det_offsets'][s_de['keep_idx']],
    }


# ============================================================================ #
# _makeMaps
def _makeMaps(cfg: dict, state_1: dict) -> dict:
    """
    1. Make a 'naive' first map
    2. Make a more robust map:
    - Common mode removed (1 pass).
    - Hits map.
    - Noise map.
    - Null map (difference between two random samples of kids).
    """

    pipe_cfg = cfg["pipeline"]
    chunk_s = pipe_cfg.get("chunk_duration_s", 1.0)

    # Share the input read and cleaning work between the naive and CM maps.
    t = time.perf_counter()
    s_sp = _mapPass(
        cfg, state_1, weights=state_1["det_weights_kept"], also_naive=True,
        compute_nulls=pipe_cfg["n_iterations"] == 0,
        collect_psd=cfg["output"].get("save_psd", True) and pipe_cfg["n_iterations"] == 0)
    initial_pass_time = time.perf_counter() - t

    print(f" | Naive + initial common-mode maps (chunk_duration_s={chunk_s})")
    print(f" | {s_sp['n_dets']} detectors"\
          f", {s_sp['sample_rate']:.1f} Hz, [{initial_pass_time:.1f}s]")

    return {
        "naive": s_sp['naive'],
        "combined_map": s_sp['combined'],
        "hits": s_sp['total_hits'],
        "noise_map": s_sp['noise_map'],
        "null_map": s_sp['null_map'],
        "detsplit_null_map": s_sp['detsplit_null_map'],
        "time_split_weights": s_sp["time_split_weights"],
        "detector_split_weights": s_sp["detector_split_weights"],
        "detector_split_assignment": s_sp["detector_split_assignment"],
        "psd_comparison": s_sp["psd_comparison"],
        "sr": s_sp['sample_rate'],
        "tod_rms_data": s_sp['tod_rms_data'],
        "t_initial": initial_pass_time,
        "processing_stats": s_sp["processing_stats"],
    }


# ============================================================================ #
# _makeIterativeMaps
def _makeIterativeMaps(
        cfg: dict, state_1: dict, state_2: dict
        ) -> tuple[np.ndarray, list, list]:
    """
    """

    pipe_cfg = cfg["pipeline"]
    n_iters = pipe_cfg["n_iterations"]
    combined_map = state_2["combined_map"]

    cm_maps = [("it_0", combined_map.copy())]
    pass_times = [("naive + it_0", state_2["t_initial"])]
    pass_stats = {"naive + it_0": state_2["processing_stats"]}
    null_result = state_2

    if n_iters > 0:

        for i in range(1, n_iters + 1):
            LOGGER.info("Iteration %d/%d started", i, n_iters)
            t = time.perf_counter()

            s_sp = _mapPass(
                cfg, state_1, weights=state_1["det_weights_kept"],
                current_map=combined_map,
                also_naive=True, compute_nulls=i == n_iters,
                collect_psd=cfg["output"].get("save_psd", True) and i == n_iters)
            combined_map = s_sp['combined']
            if i == n_iters:
                null_result = s_sp

            t_iter = time.perf_counter() - t
            print(f" | Iteration {i}/{n_iters} [{t_iter:.1f}s]")
            LOGGER.info("Iteration %d/%d complete in %.1f s", i, n_iters, t_iter)

            cm_maps.append((f"it_{i}", combined_map.copy()))
            pass_times.append((f"it_{i}", t_iter))
            pass_stats[f"it_{i}"] = s_sp["processing_stats"]

    else:
        print("Step 3: Skipping (n_iterations = 0).")

    return {
        'combined_map': combined_map,
        'cm_maps': cm_maps,
        'pass_times': pass_times,
        'pass_stats': pass_stats,
        'null_map': null_result['null_map'],
        'detsplit_null_map': null_result['detsplit_null_map'],
        'time_split_weights': null_result['time_split_weights'],
        'detector_split_weights': null_result['detector_split_weights'],
        'detector_split_assignment': null_result['detector_split_assignment'],
        'psd_comparison': null_result['psd_comparison'],
    }


# ============================================================================ #
# _buildMetadata
def _buildMetadata(
        cfg: dict, step1_res: dict, cm_maps: list, 
        pass_times: list, metrics: dict, timestamp: str, elapsed: float
        ) -> dict:
    """
    """
    
    pipe_cfg = cfg["pipeline"]
    map_cfg = cfg["map"]
    obs_info = step1_res["obs_info"]
    metadata = {
        "run_timestamp": timestamp,
        "data_format": cfg["data"]["format"],
        "obs_t_start_g3s": obs_info["t_start_g3s"],
        "obs_t_stop_g3s": obs_info["t_stop_g3s"],
        "obs_t_start_utc": step1_res["obs_utc"].isoformat(),
        "obs_duration_s": obs_info["duration_s"],
        "n_detectors": obs_info["n_detectors"],
        "sample_rate_hz": obs_info["sample_rate_hz"],
        "n_chunks": obs_info["n_chunks"],
        "map_ny": step1_res["ny"],
        "map_nx": step1_res["nx"],
        "map_ra0_deg": step1_res["ra0_deg"],
        "map_dec0_deg": step1_res["dec0_deg"],
        "map_res_arcmin": map_cfg["res_arcmin"],
        "map_xlen_deg": map_cfg["xlen_deg"],
        "map_ylen_deg": map_cfg["ylen_deg"],
        "pipeline_n_iterations": pipe_cfg["n_iterations"],
        "pipeline_start_offset_s": pipe_cfg.get("start_offset_s", 0.0),
        "pipeline_max_duration_s": pipe_cfg.get("max_duration_s", None),
        "pipeline_chunk_duration_s": pipe_cfg.get("chunk_duration_s", 1.0),
        "pipeline_chunk_memory_fraction": pipe_cfg.get("chunk_memory_fraction", 0.5),
        "pipeline_chunk_memory_gb": pipe_cfg.get("chunk_memory_gb", 0.0),
        "pipeline_resolved_chunk_durations_s": pipe_cfg.get(
            "_resolved_chunk_durations_s", []),
        "pipeline_clean_steps": [step["name"] for step in _activeCleaningSteps(pipe_cfg)],
        "pipeline_cosmic_rays": pipe_cfg.get("cosmic_rays", {}),
        "pipeline_highpass": pipe_cfg.get("highpass", {}),
        "pipeline_notch": pipe_cfg.get("notch", {}),
        "map_weighting": (
            "inverse_variance_white_noise_floor"
            if pipe_cfg.get("weight_cap_factor", 5.0) > 0
            and step1_res["white_noise_floor"] is not None
            and np.any(np.isfinite(step1_res["white_noise_floor"])
                       & (step1_res["white_noise_floor"] > 0))
            else "uniform"),
        "weight_cap_factor": pipe_cfg.get("weight_cap_factor", 5.0),
        "n_detectors_used": len(step1_res["keep_idx"]),
        "n_detectors_excluded": len(step1_res["exclude_set"]),
        "auto_excluded_detectors": step1_res["auto_excluded"],
        "wnf_excluded_detectors": step1_res["wnf_excluded"],
        "wnf_exclusion_cutoff": step1_res["wnf_cutoff"],
        "manual_excluded_detectors": step1_res["manual_excluded"],
        "total_runtime_s": round(elapsed, 2),
        "convergence": {
            "labels": metrics["labels"],
            "peak": metrics["peak"],
            "off_src_rms": metrics["off_src_rms"],
            "map_diff_rms": metrics["map_diff_rms"],
            "pass_times_s": [t for _, t in pass_times],
        },
        "map_pass_times_s": dict(pass_times),
    }

    return metadata


def _mapQuality(final_map, hits, ra_edges, dec_edges):
    valid = (hits > 0) & np.isfinite(final_map)
    quality = {
        "covered_pixels": int(np.count_nonzero(hits > 0)),
        "total_pixels": int(hits.size),
        "coverage_fraction": float(np.mean(hits > 0)),
        "finite_covered_pixels": int(np.count_nonzero(valid)),
        "hits_meaning": "sum of detector weights, not raw sample count",
    }
    if valid.any():
        values = final_map[valid]
        peak_index = np.argmax(np.where(valid, final_map, -np.inf))
        iy, ix = np.unravel_index(peak_index, final_map.shape)
        quality.update({
            "peak": float(final_map[iy, ix]),
            "peak_ra_deg": float((ra_edges[ix] + ra_edges[ix + 1]) / 2),
            "peak_dec_deg": float((dec_edges[iy] + dec_edges[iy + 1]) / 2),
            "covered_median": float(np.median(values)),
            "covered_p05": float(np.percentile(values, 5)),
            "covered_p95": float(np.percentile(values, 95)),
        })
    return quality


def _nullMetadata(state_1, state_3):
    products = {}
    for label, map_key, weight_key in (
            ("time", "null_map", "time_split_weights"),
            ("detector", "detsplit_null_map", "detector_split_weights")):
        null = state_3[map_key]
        if null is None:
            continue
        weights = state_3[weight_key]
        products[label] = {
            "stage": state_3["cm_maps"][-1][0],
            "definition": "(split_0 - split_1) / 2",
            "shared_processing": True,
            "split_covered_pixels": [int(np.count_nonzero(w > 0)) for w in weights],
            "overlap_finite_pixels": int(np.count_nonzero(np.isfinite(null))),
        }
    if "time" in products:
        products["time"]["split"] = "chunk index parity"
    if "detector" in products:
        assignment = state_3["detector_split_assignment"]
        products["detector"]["split"] = "balanced random detector groups"
        products["detector"]["seed"] = 0
        products["detector"]["groups"] = [
            [name for name, half in zip(state_1["kids_kept"], assignment) if half == i]
            for i in (0, 1)
        ]
    return products


def _savePsdDiagnostics(state_1, state_3, out_dir, out_cfg):
    if not out_cfg.get("save_psd", True):
        return {"status": "disabled"}

    LOGGER.info("Saving PSD diagnostics for the final map pass and preparation")
    info, comparison = state_3["psd_comparison"]
    info["stage"] = state_3["cm_maps"][-1][0]
    info["before"] = "after baseline removal and configured cleaning, before common mode"
    info["after"] = "after common-mode subtraction"
    info["segment_selection"] = "matched segments with no input or cleaning flags and finite signals"
    if info["nonfinite_spectral_segments"]:
        LOGGER.warning("Excluded %d matched PSD segments with non-finite spectra",
                       info["nonfinite_spectral_segments"])
    if info["skipped_sample_rate_chunks"]:
        LOGGER.warning("Skipped %d PSD comparison chunks with invalid or differing sample rates",
                       info["skipped_sample_rate_chunks"])
    if comparison is None:
        LOGGER.warning("PSD comparison unavailable: no valid matched segments")
    else:
        output.plot_psd(comparison["frequency_hz"], comparison["before"],
                        comparison["after"], out_dir / "psd.png", info["stage"])
        if out_cfg.get("save_numpy", True):
            np.savez(out_dir / "psd.npz", **comparison)

    psd = state_1["psd_avg"]
    if psd is None:
        info["weighting_psd_status"] = "unavailable: no preparation PSD"
        LOGGER.warning("Weighting PSD diagnostic unavailable: no preparation PSD")
    else:
        finite_detectors = np.all(np.isfinite(psd), axis=0)
        if not np.any(finite_detectors):
            info["weighting_psd_status"] = "unavailable: no finite detector spectra"
            LOGGER.warning("Weighting PSD diagnostic unavailable: no finite detector spectra")
        else:
            spectra = psd if finite_detectors.all() else psd[:, finite_detectors]
            freqs = state_1["psd_freqs"]
            summary = np.percentile(spectra, [25, 50, 75], axis=1)
            info["weighting_psd_status"] = "available"
            info["weighting_psd_finite_detectors"] = int(np.count_nonzero(finite_detectors))
            output.plot_weighting_psd(freqs, summary, state_1["obs_info"]["sample_rate_hz"],
                                      out_dir / "weighting_psd.png")
            if out_cfg.get("save_numpy", True):
                np.savez(
                    out_dir / "weighting_psd.npz", frequency_hz=freqs,
                    p25=summary[0], median=summary[1], p75=summary[2],
                    detector_names=np.asarray(state_1["all_kids"], dtype=str),
                    white_noise_floor=state_1["white_noise_floor"],
                )
    return info


# ============================================================================ #
# _saveOutputs
def _saveOutputs(
        cfg:dict, state_1:dict, state_2:dict, state_3:dict, t_total:float,
        timestamp: str, out_dir: pathlib.Path,
        ) -> tuple[pathlib.Path, dict]:
    """
    """

    LOGGER.info("Writing map arrays and images to %s", out_dir)
    output.saveMaps(
        state_2["naive"], state_3['cm_maps'], 
        state_2["hits"], state_2["noise_map"], state_3["null_map"],
        state_1["ra_edges"], state_1["dec_edges"],
        out_dir, cfg["output"], detsplit_null=state_3["detsplit_null_map"],
        time_split_weights=state_3["time_split_weights"],
        detector_split_weights=state_3["detector_split_weights"],
    )

    print(" | Saved maps")

    # output.plot_wnf_diagnostic(
    #     step1_res["white_noise_floor"], step1_res["wnf_cutoff"],
    #     cfg["pipeline"].get("wnf_exclude_sigma", 3.0),
    #     os.path.join(out_dir, "wnf_diagnostic.png")
    # )
    # print(" | Saved wnf_diagnostic.png")

    metrics = output.compute_convergence_metrics(
        state_2["naive"], state_3['cm_maps'], state_2["hits"])
    LOGGER.info("Computing convergence metrics and diagnostics plot")
    output.plot_diagnostics(
        metrics, state_3['pass_times'], 
        os.path.join(out_dir, "diagnostics.png"))
    
    print(" | Saved diagnostics")

    # output.plot_tod_rms(step2_res["tod_rms_data"], os.path.join(out_dir, "tod_rms.png"))
    # print("    Saved tod_rms.png")

    # print("  Convergence summary:")
    # for label, peak, rms, diff in zip(
    #     metrics["labels"], metrics["peak"],
    #     metrics["off_src_rms"], metrics["map_diff_rms"]
    # ):
    #     diff_str = f"  map_diff={diff:.4f}" if not np.isnan(diff) else ""
    #     print(f"    {label:12s}  peak={peak:.4f}  off_src_rms={rms:.4f}{diff_str}")

    # Metadata
    elapsed = time.perf_counter() - t_total
    metadata = _buildMetadata(cfg, state_1, state_3['cm_maps'], state_3['pass_times'], metrics, timestamp, elapsed)
    metadata["psd_diagnostics"] = _savePsdDiagnostics(state_1, state_3, out_dir, cfg["output"])
    return out_dir, metadata




# ============================================================================ #
# STEPS 5 & 6: OPTIONAL PASSES & PROFILING
# ============================================================================ #


# ============================================================================ #
#   step_5_boresight_comparison
def step_5_boresight_comparison(cfg: dict, step1_res: dict, combined_map: np.ndarray, out_dir: str) -> None:
    if cfg["map"].get("compare_boresight", False):
        print("\nStep 5: Boresight-only comparison pass...")
        pipe_cfg = cfg["pipeline"]
        t = time.perf_counter()
        bore_map, _, _, _, _, _, _, _, _ = _streaming_pass(
            cfg, pipe_cfg, step1_res["ra_edges"], step1_res["dec_edges"],
            step1_res["det_offsets_kept"], common_mode=False, boresight_only=True,
            keep_idx=step1_res["keep_idx"], weights=step1_res["det_weights_kept"],
            compute_time_null=pipe_cfg.get("compute_time_null", True)
        )
        print(f"  [{time.perf_counter() - t:.1f}s]")
        output.plot_boresight_comparison(
            combined_map, bore_map, step1_res["ra_edges"], step1_res["dec_edges"],
            pathlib.Path(out_dir) / "boresight_comparison.png"
        )
        print("  Saved boresight_comparison.png")


# ============================================================================ #
#   step_6_per_detector_maps
def step_6_per_detector_maps(cfg: dict, step1_res: dict, pd_results: tuple | None,
                             out_dir: str, metadata: dict) -> None:
    pd_cfg = cfg.get("per_detector", {})
    pipe_cfg = cfg["pipeline"]

    if pd_cfg.get("enabled", False):
        print("\nStep 6: Per-detector maps...")
        if pd_results is not None:
            kids_sel, det_data, det_hits = pd_results
        else:
            t = time.perf_counter()
            kids_sel, det_data, det_hits = _per_detector_pass(
                cfg, pipe_cfg, pd_cfg, step1_res["ra_edges"],
                step1_res["dec_edges"], step1_res["det_offsets"]
            )
            print(f"  Pass complete [{time.perf_counter() - t:.1f}s]")

        flagged = output.save_per_detector_maps(
            kids_sel, det_data, det_hits,
            step1_res["ra_edges"], step1_res["dec_edges"],
            os.path.join(out_dir, "per_detector"), pd_cfg,
            psd_avg=step1_res["psd_avg"], psd_freqs=step1_res["psd_freqs"],
            psd_all_kids=step1_res["all_kids"]
        )
        metadata["detectors_to_check"] = list(flagged.keys())
        metadata["n_detectors_flagged"] = len(flagged)
        output.save_metadata(metadata, out_dir)
    else:
        print("\nStep 6: Per-detector maps disabled (set per_detector.enabled = true to enable).")


# ============================================================================ #
#   finish_run
def finish_run(args, profiler: cProfile.Profile, out_dir: str,
               elapsed: float, log_path: pathlib.Path) -> None:
    print()
    print(f"Total time : {elapsed:.1f}s")
    print(f"Maps saved : {out_dir}/")
    print(f"Log saved  : {log_path}")
    print("=" * 60)

    if args.profile:
        profiler.disable()
        buf = io.StringIO()
        pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(30)
        print("\n" + "=" * 60)
        print("PROFILE -- top 30 functions by cumulative time")
        print("=" * 60)
        print(buf.getvalue())
        prof_path = os.path.join(out_dir, "profile.prof")
        profiler.dump_stats(str(prof_path))
        print(f"Full profile : {prof_path}")
        print(f"Visualise    : snakeviz {prof_path}")




# ============================================================================ #
# MAIN ENTRY POINT
# ============================================================================ #


# ============================================================================ #
#   main
def main():

    # TODO: audit streamer
    # TODO: audit iter_chunks

    print(f"{'='*60}\nCCAT Quick-Look Mapmaker\n{'='*60}")

    # Setup
    t_total = time.perf_counter()
    args = _setupArgParser()
    prof = _setupProfiler(args)
    cfgs, config_bytes = _loadConfiguration(args.config)
    software = _softwareProvenance()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    output_root = pathlib.Path(cfgs["output"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    out_dir = output_root / f'{cfgs["output"]["obs_object"]}_{timestamp}'
    out_dir.mkdir()
    log_path = out_dir / "run.log"
    log_handler = _setupRunLogger(log_path)
    LOGGER.info("Starting mapmaking run")
    LOGGER.info("Configuration file: %s", pathlib.Path(args.config).resolve())
    LOGGER.info("Run log: %s", log_path)
    if "error" in software["git"]:
        LOGGER.warning("Could not determine Git provenance: %s", software["git"]["error"])

    metadata = {
        "run_timestamp": timestamp,
        "status": "running",
        "config_source": str(pathlib.Path(args.config).resolve()),
        "stage_times_s": {},
    }

    def save_run_metadata():
        metadata["input_files_discovered"] = cfgs.get("_input_files_discovered", [])
        metadata["input_files_opened"] = cfgs.get("_input_files_opened", [])
        metadata["pipeline_resolved_chunk_durations_s"] = cfgs["pipeline"].get(
            "_resolved_chunk_durations_s", [])
        metadata["total_runtime_s"] = round(time.perf_counter() - t_total, 2)
        output.save_metadata(metadata, out_dir)

    try:
        (out_dir / "config.toml").write_bytes(config_bytes)
        metadata["resolved_settings"] = _effectiveSettings(cfgs)
        metadata["active_cleaning_steps"] = _activeCleaningSteps(cfgs["pipeline"])
        if "clean_steps" in cfgs["pipeline"]:
            LOGGER.warning(
                "pipeline.clean_steps does not select main map-pass cleaning; "
                "configured step blocks determine which steps are invoked"
            )
        metadata["software"] = software
        save_run_metadata()

        # Precalculate chunk size
        LOGGER.info("Precalculating total chunks where supported")
        step_start = time.perf_counter()
        cfgs |= reader._precalcChunkTotal(cfgs) # if possible ("lab" only right now)
        LOGGER.info(
            "Chunk-count preparation complete in %.1f s; total_chunks=%s",
            time.perf_counter() - step_start, cfgs.get("total_chunks", "unknown"),
        )
        LOGGER.debug("Run configuration:\n%s", pprint.pformat(cfgs, sort_dicts=True))

        # Map preparation data pass
        print()
        print("Step 1 (preparation)")
        LOGGER.info("Step 1/4: preparation started")
        step_start = time.perf_counter()
        state_1 = _mapPrep(cfgs)
        step_elapsed = time.perf_counter() - step_start
        n_dets = state_1["obs_info"]["n_detectors"]
        metadata["stage_times_s"]["preparation"] = step_elapsed
        metadata.update({
            "n_detectors": n_dets,
            "n_detectors_used": len(state_1["keep_idx"]),
            "n_detectors_excluded": len(state_1["exclude_set"]),
            "auto_excluded_detectors": state_1["auto_excluded"],
            "manual_excluded_detectors": state_1["manual_excluded"],
            "wnf_excluded_detectors": state_1["wnf_excluded"],
            "wnf_exclusion_cutoff": state_1["wnf_cutoff"],
            "observation": state_1["obs_info"],
            "preparation_processing": state_1["preparation_stats"],
            "detector_weight_summary": {
                "min": float(np.min(state_1["det_weights_kept"])),
                "median": float(np.median(state_1["det_weights_kept"])),
                "max": float(np.max(state_1["det_weights_kept"])),
            },
        })
        save_run_metadata()
        LOGGER.info(
            "Step 1/4: preparation complete in %.1f s; %d/%d detectors kept, "
            "%d excluded (auto-noise flags=%d, white-noise-floor flags=%d); "
            "observation duration %.1f s; sample rate %.1f Hz; map %d x %d "
            "pixels centered at (%.6f, %.6f)",
            step_elapsed, len(state_1["keep_idx"]), len(state_1["all_kids"]),
            len(state_1["exclude_set"]), len(state_1["auto_excluded"]),
            len(state_1["wnf_excluded"]),
            state_1["obs_info"]["duration_s"],
            state_1["obs_info"]["sample_rate_hz"], state_1["ny"], state_1["nx"],
            state_1["ra0_deg"], state_1["dec0_deg"],
        )

        # Map making data pass
        print()
        print("Step 2 (map making)")
        LOGGER.info("Step 2/4: initial map pass started")
        step_start = time.perf_counter()
        state_2 = _makeMaps(cfgs, state_1)
        metadata["stage_times_s"]["initial_maps"] = time.perf_counter() - step_start
        metadata["pass_processing"] = {"naive + it_0": state_2["processing_stats"]}
        save_run_metadata()
        LOGGER.info("Step 2/4: initial maps complete in %.1f s", metadata["stage_times_s"]["initial_maps"])

        # Map making common mode data passes
        print()
        n_iterations = cfgs["pipeline"]["n_iterations"]
        print(f"Step 3 ({n_iterations} iteration(s))")
        LOGGER.info("Step 3/4: %d iterative pass(es) starting", n_iterations)
        step_start = time.perf_counter()
        state_3 = _makeIterativeMaps(cfgs, state_1, state_2)
        metadata["stage_times_s"]["iterations"] = time.perf_counter() - step_start
        metadata["pass_processing"] = state_3["pass_stats"]
        save_run_metadata()
        LOGGER.info("Step 3/4: iterative maps complete in %.1f s", metadata["stage_times_s"]["iterations"])

        # Save outputs
        print()
        print("Step 4 (save outputs)")
        LOGGER.info("Step 4/4: saving maps and diagnostics")
        step_start = time.perf_counter()
        out_dir, map_metadata = _saveOutputs(
            cfgs, state_1, state_2, state_3, t_total, timestamp, out_dir)
        metadata.update(map_metadata)
        metadata["null_products"] = _nullMetadata(state_1, state_3)
        metadata["map_quality"] = _mapQuality(
            state_3["combined_map"], state_2["hits"],
            state_1["ra_edges"], state_1["dec_edges"])
        metadata["stage_times_s"]["save_outputs"] = time.perf_counter() - step_start
        metadata["status"] = "completed"
        save_run_metadata()
        LOGGER.info("Step 4/4: outputs saved in %.1f s", metadata["stage_times_s"]["save_outputs"])

        print()
        finish_run(args, prof, str(out_dir), time.perf_counter() - t_total, log_path)
        LOGGER.info("Run completed in %.1f s; outputs: %s",
                    time.perf_counter() - t_total, out_dir)
    except (Exception, KeyboardInterrupt) as exc:
        LOGGER.exception("Run failed; partial output files, if any, are in %s", out_dir)
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_run_metadata()
        raise
    finally:
        LOGGER.removeHandler(log_handler)
        log_handler.close()


if __name__ == "__main__":
    main()