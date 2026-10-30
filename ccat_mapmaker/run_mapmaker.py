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
import pstats
import pathlib
import argparse
import cProfile
import numpy as np
from scipy.signal import welch
from scipy.ndimage import gaussian_filter
import tomli as toml # tomllib after python 3.11
from datetime import datetime, timedelta, timezone

from spt3g import core # type: ignore

from mapmaker import tod
from mapmaker import map
from mapmaker import coord
from mapmaker import output
from mapmaker import reader


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
def _loadConfiguration(config_path_str: str) -> tuple[dict, pathlib.Path]:
    """Load the yaml configuration file.

    config_path_str: Absolute or relative path string.
    """

    config_path = pathlib.Path(config_path_str)
    with open(config_path, "rb") as f:
        cfg = toml.load(f)
    cfg = _resolvePaths(cfg, config_path)

    print(f"Config : {config_path}")
    for d in cfg['data']['input_dirs']:
        print(f" Input  : {d}")
    print(f" Format : {cfg['data']['format']}")
    print(f" Output : {cfg['output']['output_dir']}\n")

    return cfg


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



# ============================================================================ #
# _printLoadingBar
def _printLoadingBar(iteration=1, total=1, length=40, clear=False):
    """Prints an in-place updating loading bar to stdout, clearing the line upon completion.

    Args:
        iteration (int): Current step or iteration index.
        total (int): Total number of iterations for completion.
        length (int, optional): Character length of the progress bar. Defaults to 40.
    """

    if clear:
        line_width = length + 18
        sys.stdout.write(f'\r{" " * line_width}\r')
        sys.stdout.flush()
        return

    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.1f}% Complete')
    sys.stdout.flush()


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
    for i,chunk in enumerate(reader.iter_chunks(cfg)): # chunk_id=chunk.chunk_index

        if cfg['total_chunks'] is not None:
            _printLoadingBar(i, cfg['total_chunks'], length=40)

        # Skip chunks where all detector data is flagged out
        if np.all(chunk.flags != 0):
            continue 

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

            psd_sum = None # will be initialized in Sxx section

            init = True
        # ------------------------------------------------------ #

        # Create a copy of the signal with flagged points set to NaN
        signal_masked = chunk.signal.astype(float, copy=True)
        signal_masked[chunk.flags != 0] = np.nan

        # Temporal differences and validity mask
        diffs = signal_masked[1:] - signal_masked[:-1]
        valid_mask = np.isfinite(diffs)

        t = time.perf_counter()

        # Advance accumulations (t=0.53 s)
        ra_sum          += chunk.ra_bore.sum()
        dec_sum         += chunk.dec_bore.sum()
        n_bore          += len(chunk.ra_bore)
        # det_median_sum  += np.nanmedian(chunk.signal, axis=0)
        det_diff_sum    += np.nansum(diffs, axis=0)
        det_diff_sum_sq += np.nansum((diffs ** 2), axis=0)
        det_diff_count  += np.sum(valid_mask, axis=0)
        n_chunks        += 1

        print("5: ", time.perf_counter() - t)
        t = time.perf_counter()

        # Sxx, adding per loop (t=0.16 s)
        if chunk.signal.shape[0] == nominal_chunk_len:
            f, p = welch(chunk.signal, fs=sample_rate, window="hann", axis=0)
            if psd_sum is None:
                psd_sum = np.zeros_like(p)
                psd_freqs = f
            psd_sum += p
            psd_count += 1

        print("6: ", time.perf_counter() - t)
        t = time.perf_counter()

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
        "psd_freqs": psd_freqs
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
    return_sample: bool = False,
    collect_tod_rms: bool = False,
    weights: np.ndarray = None
    ):

    pipe_cfg = cfgs['pipeline']
    compute_time_null = pipe_cfg.get("compute_time_null", True)
    compute_detsplit_null = pipe_cfg.get("compute_detsplit_null", True)
    boresight_only = False
    nx, ny = state_1['nx'], state_1['ny']

    total_data = np.zeros((ny, nx), dtype=float)
    total_hits = np.zeros((ny, nx), dtype=float)
    total_sumsq = np.zeros((ny, nx), dtype=float)

    null_map = None
    if compute_time_null:
        null_data = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]
        null_hits = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]

    detsplit_null_map = None
    if compute_detsplit_null:
        n_dets_kept = len(state_1["det_offsets_kept"])
        det_half = np.random.default_rng(0).integers(0, 2, size=n_dets_kept)
        detsplit_data = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]
        detsplit_hits = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]

    n_dets = sample_rate = None
    psd_raw = psd_cm = None
    tod_rms_data: list = []

    # Configure cleaning steps used by clean_tod
    clean_steps = pipe_cfg.get("clean_steps", [])
    step_params = {
        "cosmic_rays": pipe_cfg.get("cosmic_rays", {}),
        "highpass":    pipe_cfg.get("highpass", {}),
        "notch":       pipe_cfg.get("notch", {}),
    }

    for i,chunk in enumerate(reader.iter_chunks(cfgs)):

        if cfgs['total_chunks'] is not None:
            _printLoadingBar(i, cfgs['total_chunks'], length=40)

        flags = chunk.flags
        chunk_id = chunk.chunk_index
        if np.all(flags != 0):
            continue
        if n_dets is None:
            n_dets = len(chunk.kids)
            sample_rate = chunk.sample_rate

        ra = chunk.ra
        dec = chunk.dec
        flags = chunk.flags
        no_per_detector_offsets = ra is None

        if state_1["keep_idx"] is not None:
            sig = chunk.signal[:, state_1["keep_idx"]] - state_1["det_offsets_kept"][np.newaxis, :]
            flags = flags[:, state_1["keep_idx"]]
            if ra is not None:
                ra = ra[:, state_1["keep_idx"]]
                dec = dec[:, state_1["keep_idx"]]
        else:
            sig = chunk.signal - state_1["det_offsets_kept"][np.newaxis, :]

        flag_mask = np.ones(np.shape(flags))
        flag_mask[flags != 0] = np.nan

        if boresight_only or ra is None:
            ra = np.repeat(chunk.ra_bore[:, np.newaxis], sig.shape[1], axis=1)
            dec = np.repeat(chunk.dec_bore[:, np.newaxis], sig.shape[1], axis=1)
            # ra = np.broadcast_to(chunk.ra_bore[:, np.newaxis], sig.shape) # maybe slightly faster
            # dec = np.broadcast_to(chunk.dec_bore[:, np.newaxis], sig.shape)

        if clean_steps:
            sig, new_flags = tod.clean_tod(sig, flag_mask, chunk.sample_rate,
                                       steps=clean_steps,
                                       step_params=step_params)
            if new_flags is None:
                bin_flag_mask = flag_mask
            else:
                new_flags = np.asarray(new_flags)
                if np.any(new_flags != 0):
                    bin_flag_mask = flag_mask.copy()
                    bin_flag_mask[new_flags != 0] = np.nan
                else:
                    bin_flag_mask = flag_mask
        else:
            new_flags = None
            bin_flag_mask = flag_mask

        if return_sample and psd_raw is None:
            psd_raw = sig.copy()

        if collect_tod_rms:
            rms_raw = float(np.median(np.sqrt(np.mean(sig ** 2, axis=0))))

        if common_mode:
            if current_map is not None:
                sig = tod.iterate_common_mode(sig, flag_mask, ra, dec,
                                          current_map, state_1["ra_edges"], state_1["dec_edges"])
            else:
                sig = tod.subtract_common_mode(sig, tod.estimate_common_mode(sig, flag_mask))

        if return_sample and psd_cm is None:
            psd_cm = sig.copy() * flag_mask

        if collect_tod_rms:
            rms_cm = float(np.median(np.sqrt(np.mean(sig ** 2, axis=0)))) if common_mode else rms_raw
            tod_rms_data.append((chunk.t_start, rms_raw, rms_cm))

        d, h, sq = map.bin_chunk(sig, bin_flag_mask, ra, dec, state_1["ra_edges"], state_1["dec_edges"], weights=weights)
        total_data += d
        total_hits += h
        total_sumsq += sq

        if compute_time_null:
            half = chunk_id % 2
            null_data[half] += d
            null_hits[half] += h

        if compute_detsplit_null:
            for dhalf in (0, 1):
                m = det_half == dhalf
                w_half = weights[m] if weights is not None else None
                d_ds, h_ds, _ = map.bin_chunk(sig[:, m], bin_flag_mask[:, m], ra[:, m], dec[:, m],
                                          state_1["ra_edges"], state_1["dec_edges"], weights=w_half)
                detsplit_data[dhalf] += d_ds
                detsplit_hits[dhalf] += h_ds

    _printLoadingBar(clear=True)

    with np.errstate(invalid='ignore', divide='ignore'):
        combined = np.where(total_hits > 0, total_data / total_hits, np.nan)
        mean_sq = np.where(total_hits > 0, total_sumsq / total_hits, np.nan)
        variance = np.clip(mean_sq - combined ** 2, 0, None)
        noise_map = np.sqrt(variance / total_hits)

        if compute_time_null:
            map_a = np.where(null_hits[0] > 0, null_data[0] / null_hits[0], np.nan)
            map_b = np.where(null_hits[1] > 0, null_data[1] / null_hits[1], np.nan)
            null_map = (map_a - map_b) / 2.0

        if compute_detsplit_null:
            map_a_ds = np.where(detsplit_hits[0] > 0, detsplit_data[0] / detsplit_hits[0], np.nan)
            map_b_ds = np.where(detsplit_hits[1] > 0, detsplit_data[1] / detsplit_hits[1], np.nan)
            detsplit_null_map = (map_a_ds - map_b_ds) / 2.0

    sample = {"raw": psd_raw, "cm": psd_cm} if return_sample else None

    return {
        'combined': combined,
        'total_hits': total_hits,
        'noise_map': noise_map,
        'null_map': null_map,
        'detsplit_null_map': detsplit_null_map,
        'n_dets': n_dets,
        'sample_rate': sample_rate,
        'sample': sample,
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

        sig, new_flags = tod.clean_tod(sig, flag_mask, chunk.sample_rate,
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

    print(f"Step 1 (preparation)")

    t0 = time.perf_counter()
    t = t0
    
    # Run the initial data streaming pass:
    # - Boresight pointing averages (for map center fallback).
    # - Per-detector medians and noise estimates (via temporal differences).
    # - Average PSD and white noise floor characteristics.
    s_fp = _prepPass(cfg) # state var

    print(f"  First pass done. [{time.perf_counter()-t:.1f} s]")
    t = time.perf_counter()

    # Decide which detectors to excluded from maps, based on:
    # - Manual lists.
    # - Automated metrics (noise/std, white noise floor).
    s_de = tod.excludedDetectors( # detector state
        s_fp['all_kids'], s_fp['det_noise'], 
        s_fp['white_noise_floor'], cfg["pipeline"])

    print(f"  Detector exclusions done. [{time.perf_counter()-t:.1f} s]")
    t = time.perf_counter()

    # Find detector weights.
    # This uses a static weight per detector:
    # The inverse of the white noise floor (0.4 to 0.9 nyqist of PSD).
    det_weights = tod.findWeights(
        s_fp['white_noise_floor'], len(s_fp['all_kids']), cfg["pipeline"])

    # Format observation time metadata for logging
    _G3_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
    obs_utc = _G3_EPOCH + timedelta(seconds=s_fp['obs_info']["t_start_g3s"])
    print(f"  Obs start: {obs_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Duration: {s_fp['obs_info']['duration_s']:.1f} s")
    print(f"  Detectors: {s_fp['obs_info']['n_detectors']} @ {s_fp['obs_info']['sample_rate_hz']:.1f} Hz")

    # Determine the central RA/Dec coordinate for the map grid
    s_ma = coord.resolveMapCenter( # map state
        cfg, s_fp['obs_info'], s_fp['ra0_auto'], s_fp['dec0_auto'])

    # Construct coord pixel grid boundaries (edges)
    s_ma |= map.makeMapEdges( # add to map state dict
        ra0_deg=s_ma['ra0_deg'], dec0_deg=s_ma['dec0_deg'],
        xlen_deg=cfg["map"]["xlen_deg"], ylen_deg=cfg["map"]["ylen_deg"],
        res_deg=cfg["map"]["res_arcmin"]/60.0,
    )

    print(f"  Map grid: {s_ma['ny']} x {s_ma['nx']} pixels", end="")
    print(f" ({cfg['map']['res_arcmin']:.1f} arcmin/pixel)", end="")
    print(f", centre = ({s_ma['ra0_deg']:.4f}, {s_ma['dec0_deg']:.4f}) [{s_ma['centre_source']}]")

    # Return a dictionary containing all computed parameters for subsequent pipeline steps
    return {
        "det_offsets": s_fp['det_offsets'],
        "all_kids": s_fp['all_kids'],
        "obs_info": s_fp['obs_info'],
        "obs_utc": obs_utc,
        "white_noise_floor": s_fp['white_noise_floor'],
        "psd_avg": s_fp['psd_avg'],
        "psd_freqs": s_fp['psd_freqs'],
        "keep_idx": s_de['keep_idx'],
        "exclude_set": s_de['exclude_set'],
        "auto_excluded": s_de['noise_excluded'],
        "wnf_excluded": s_de['wnf_excluded'],
        "wnf_cutoff": s_de['wnf_cutoff'],
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

    print(f"Step 2 (map making)")

    pipe_cfg = cfg["pipeline"]
    chunk_s = pipe_cfg.get("chunk_duration_s", 1.0)

    # Make a 'naive' first map
    t = time.perf_counter()
    s_sp1 = _mapPass( # state var
        cfg, state_1,
        common_mode=False, return_sample=True,
    )
    t1 = time.perf_counter() - t

    print(f"  Naive map (chunk_duration_s={chunk_s})")
    print(f"    {s_sp1['n_dets']} detectors"\
          f", {s_sp1['sample_rate']:.1f} Hz, [{t1:.1f}s]")

    # Make a more robust map:
    # - Common mode removed (1 pass).
    # - Hits map.
    # - Noise map.
    # - Null map (difference between two random samples of kids).
    # TODO: null map needs to be separated from streaming pass and improved.
    t = time.perf_counter()
    s_sp2 = _mapPass(
        cfg, state_1, 
        return_sample=True, collect_tod_rms=True,
        weights=state_1["det_weights_kept"]
    )
    t2 = time.perf_counter() - t

    print(f"  Initial common-mode pass [{t2:.1f}s]")

    return {
        "naive": s_sp1['combined'],
        "combined_map": s_sp2['combined'],
        "hits": s_sp2['total_hits'],
        "noise_map": s_sp2['noise_map'],
        "null_map": s_sp2['null_map'],
        "detsplit_null_map": s_sp2['detsplit_null_map'],
        "sr": s_sp1['sample_rate'],
        "raw_sample": s_sp1['sample'],
        "cm_sample": s_sp2['sample'],
        "tod_rms_data": s_sp2['tod_rms_data'],
        "t_naive": t1,
        "t_it0": t2,
    }


# ============================================================================ #
#   step_3_iterations
def step_3_iterations(cfg: dict, step1_res: dict, step2_res: dict, kid_shifts: dict | None) -> tuple[np.ndarray, list, list]:
    pipe_cfg = cfg["pipeline"]
    n_iters = pipe_cfg["n_iterations"]
    combined_map = step2_res["combined_map"]

    cm_maps = [("it_0", combined_map.copy())]
    pass_times = [("naive", step2_res["t_naive"]), ("it_0", step2_res["t_it0"])]

    if n_iters > 0:
        print(f"Step 3: {n_iters} iteration(s)...")
        for i in range(1, n_iters + 1):
            t = time.perf_counter()
            print(f"  Iteration {i}/{n_iters}...", end=" ", flush=True)
            combined_map, _, _, _, _, _, _, _, _ = _streaming_pass(
                cfg, pipe_cfg, step1_res["ra_edges"], step1_res["dec_edges"],
                step1_res["det_offsets_kept"], current_map=combined_map,
                keep_idx=step1_res["keep_idx"], kids_kept=step1_res["kids_kept"],
                kid_shifts=kid_shifts, weights=step1_res["det_weights_kept"],
                compute_time_null=pipe_cfg.get("compute_time_null", True),
            )
            t_iter = time.perf_counter() - t
            cm_maps.append((f"it_{i}", combined_map.copy()))
            pass_times.append((f"it_{i}", t_iter))
            print(f"[{t_iter:.1f}s]")
    else:
        print("Step 3: Skipping (n_iterations = 0).")

    return combined_map, cm_maps, pass_times




# ============================================================================ #
# STEP 4: OUTPUT SAVING & METADATA
# ============================================================================ #


# ============================================================================ #
#   _build_metadata
def _build_metadata(cfg: dict, step1_res: dict, cm_maps: list, pass_times: list,
                    metrics: dict, timestamp: str, elapsed: float) -> dict:
    pipe_cfg = cfg["pipeline"]
    map_cfg = cfg["map"]
    obs_info = step1_res["obs_info"]
    all_kids = step1_res["all_kids"]
    n_total = len(all_kids)

    manual_names = pipe_cfg.get("exclude_detectors", [])
    manual_indices = pipe_cfg.get("exclude_detector_indices", [])

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
        "pipeline_clean_steps": pipe_cfg.get("clean_steps", []),
        "pipeline_cosmic_rays": pipe_cfg.get("cosmic_rays", {}),
        "pipeline_highpass": pipe_cfg.get("highpass", {}),
        "pipeline_notch": pipe_cfg.get("notch", {}),
        "map_weighting": ("inverse_variance_white_noise_floor"
                          if pipe_cfg.get("weight_cap_factor", 5.0) > 0 else "uniform"),
        "weight_cap_factor": pipe_cfg.get("weight_cap_factor", 5.0),
        "n_detectors_used": len(step1_res["keep_idx"]),
        "n_detectors_excluded": len(step1_res["exclude_set"]),
        "auto_excluded_detectors": step1_res["auto_excluded"],
        "manual_excluded_detectors": manual_names + [all_kids[i] for i in manual_indices if 0 <= i < n_total],
        "total_runtime_s": round(elapsed, 2),
        "convergence": {
            "labels": metrics["labels"],
            "peak": metrics["peak"],
            "off_src_rms": metrics["off_src_rms"],
            "map_diff_rms": metrics["map_diff_rms"],
            "pass_times_s": [t for _, t in pass_times],
        },
    }

    return metadata


# ============================================================================ #
#   step_4_save_outputs
def step_4_save_outputs(
        cfg: dict, step1_res: dict, step2_res: dict,
        cm_maps: list, pass_times: list, t_total: float
        ) -> tuple[str, dict]:
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    n_dets = step1_res["obs_info"]["n_detectors"]
    out_dir = os.path.join(pathlib.Path(cfg["output"]["output_dir"]),
                           f'{cfg["output"]["obs_object"]}_d{n_dets}_{timestamp}')

    output.save_iteration_maps(
        step2_res["naive"], cm_maps, step2_res["hits"], step2_res["noise_map"],
        step2_res["null_map"], step1_res["ra_edges"], step1_res["dec_edges"],
        out_dir, cfg["output"], detsplit_null=step2_res["detsplit_null_map"]
    )

    output.plot_wnf_diagnostic(
        step1_res["white_noise_floor"], step1_res["wnf_cutoff"],
        cfg["pipeline"].get("wnf_exclude_sigma", 3.0),
        os.path.join(out_dir, "wnf_diagnostic.png")
    )
    print("    Saved wnf_diagnostic.png")

    metrics = output.compute_convergence_metrics(step2_res["naive"], cm_maps, step2_res["hits"])
    output.plot_diagnostics(metrics, pass_times, os.path.join(out_dir, "diagnostics.png"))
    print("    Saved diagnostics.png")

    output.plot_psd(
        step2_res["raw_sample"]["raw"], step2_res["cm_sample"]["cm"],
        step2_res["sr"], os.path.join(out_dir, "psd.png")
    )
    print("    Saved psd.png")

    output.plot_tod_rms(step2_res["tod_rms_data"], os.path.join(out_dir, "tod_rms.png"))
    print("    Saved tod_rms.png")

    print("  Convergence summary:")
    for label, peak, rms, diff in zip(
        metrics["labels"], metrics["peak"],
        metrics["off_src_rms"], metrics["map_diff_rms"]
    ):
        diff_str = f"  map_diff={diff:.4f}" if not np.isnan(diff) else ""
        print(f"    {label:12s}  peak={peak:.4f}  off_src_rms={rms:.4f}{diff_str}")

    elapsed = time.perf_counter() - t_total
    metadata = _build_metadata(cfg, step1_res, cm_maps, pass_times, metrics, timestamp, elapsed)
    output.save_metadata(metadata, out_dir)

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
def finish_run(args, profiler: cProfile.Profile, out_dir: str, elapsed: float) -> None:
    print()
    print(f"Total time : {elapsed:.1f}s")
    print(f"Maps saved : {out_dir}/")
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

    print(f"{'='*60}\nCCAT Quick-Look Mapmaker\n{'='*60}")

    # SETUP
    t_total = time.perf_counter()
    args = _setupArgParser()
    prof = _setupProfiler(args)
    cfgs = _loadConfiguration(args.config)
    cfgs |= reader._precalcChunkTotal(cfgs) # if possible ("lab" only right now)

    # STEP 1: A single data pass to prepare for map making.
    # - Executes the first data pass to gather baselines, noise, and PSD.
    # - Determine excluded detectors.
    # - Computes inverse-variance detector weights.
    # - Resolves the map center.
    # - Sets up the spatial pixel grid edges.
    state_1 = _mapPrep(cfgs)

    # STEP 2: Two data passes to build out the maps.
    # - 'naive' first map.
    # - Common mode removed map (1 pass).
    # - Hits map.
    # - Noise map.
    # - Null map (difference between two random samples of kids).
    state_2 = _makeMaps(cfgs, state_1)

    # TODO: audit streamer
    # TODO: audit from here

    combined_map, cm_maps, pass_times = step_3_iterations(
        cfgs, state_1, state_2, kid_shifts)

    out_dir, metadata = step_4_save_outputs(
        cfgs, state_1, state_2, cm_maps, pass_times, t_total)

    step_5_boresight_comparison(cfgs, state_1, combined_map, out_dir)

    step_6_per_detector_maps(cfgs, state_1, pd_results, out_dir, metadata)

    finish_run(args, prof, out_dir, time.perf_counter() - t_total)


if __name__ == "__main__":
    main()