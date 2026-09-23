#!/usr/bin/env python
# ============================================================================ #
# run_mapmaker.py
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

import sys
import pathlib
import argparse
import time
import cProfile
import pstats
import io
from datetime import datetime, timedelta, timezone
import os

try:
    import tomllib  # 3.11 onwards
except ImportWarning:
    import tomli as tomllib

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import welch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from mapmaker.reader import iter_chunks, get_blasttng_baked_shifts, get_blasttng_site
from mapmaker.cleaning import clean_tod, find_psd_anomalies
from mapmaker.binning import make_map_edges, bin_chunk, bin_detector
from mapmaker.common_mode import estimate_common_mode, subtract_common_mode, iterate_common_mode
from mapmaker import output
from mapmaker import target


# ============================================================================ #
# PATH RESOLUTION & CONFIGURATION
# ============================================================================ #
def resolve_paths(cfg: dict, config_path: pathlib.Path) -> dict:
    cfg_dir = config_path.parent.resolve()

    def resolve(path_str: str) -> str:
        p = pathlib.Path(path_str)
        if not p.is_absolute():
            p = (cfg_dir / p).resolve()
        return str(p)

    cfg["data"]["input_dirs"] = [resolve(d) for d in cfg["data"]["input_dirs"]]
    cfg["output"]["output_dir"] = resolve(cfg["output"]["output_dir"])
    return cfg


def load_configuration(config_path_str: str) -> tuple[dict, pathlib.Path]:
    config_path = pathlib.Path(config_path_str)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    cfg = resolve_paths(cfg, config_path)
    return cfg, config_path


# ============================================================================ #
# STEP 0: PROBE MEDIANS
# ============================================================================ #
def _compute_blasttng_probe_medians(cfg: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "blasttng_probe_medians.npz")

    if os.path.exists(save_path):
        print(f"  Found cached probe medians: {save_path}")
        with np.load(save_path) as data:
            kids = data["kids"]
            medians = data["medians"]
            return dict(zip(kids, medians))

    first_chunk = next(iter_chunks(cfg))
    kids = first_chunk.kids
    n_dets = len(kids)

    det_medians = np.zeros(n_dets, dtype=float)

    print(f"  Computing probe tone medians across all chunks for {n_dets} detectors...")
    for i in range(n_dets):
        det_tod_list = []
        for chunk in iter_chunks(cfg):
            flags = chunk.flags[:, i] if chunk.flags.ndim > 1 else chunk.flags
            sig_i = chunk.signal[:, i].copy()
            sig_i[flags != 0] = np.nan
            det_tod_list.append(sig_i)

        full_det_tod = np.concatenate(det_tod_list)
        det_medians[i] = np.nanmedian(full_det_tod)

        del det_tod_list, full_det_tod

    np.savez(save_path, kids=np.array(kids), medians=det_medians)
    print(f"  Saved probe medians to: {save_path}")
    return dict(zip(kids, det_medians))


def step_0_probe_medians(cfg: dict) -> None:
    if cfg["data"]["format"] == "blasttng":
        print("Step 0: Calculating BLAST-TNG probe tone medians...")
        t = time.perf_counter()
        base_out_dir = pathlib.Path(cfg["output"]["output_dir"])
        _compute_blasttng_probe_medians(cfg, str(base_out_dir))
        print(f"  Probe tones found. [{time.perf_counter()-t:.1f}s]")


# ============================================================================ #
# STEP 1: FIRST PASS & DETECTOR SELECTION
# ============================================================================ #
def _first_pass(cfg: dict):
    ra_sum = dec_sum = n_bore = 0
    det_median_sum = None
    det_diff_sum = None
    det_diff_sum_sq = None
    det_diff_count = None

    n_chunks = 0
    t_obs_start = t_obs_stop = None
    n_dets = sample_rate = kids = None

    psd_sum = None
    psd_count = 0
    psd_freqs = None
    nominal_chunk_len = None

    for chunk in iter_chunks(cfg):
        flags = chunk.flags
        chunk_id = chunk.chunk_index

        if np.all(flags != 0):
            print(f"No good data in chunk with index {chunk_id}. Skipping to next chunk")
            continue

        flag_mask = np.ones(np.shape(flags))
        flag_mask[flags != 0] = np.nan

        if t_obs_start is None:
            t_obs_start = chunk.t_start
            n_dets = chunk.signal.shape[1]
            sample_rate = chunk.sample_rate
            kids = chunk.kids
            nominal_chunk_len = chunk.signal.shape[0]

        t_obs_stop = chunk.t_stop

        ra_sum += chunk.ra_bore.sum()
        dec_sum += chunk.dec_bore.sum()
        n_bore += len(chunk.ra_bore)

        if det_median_sum is None:
            det_median_sum = np.zeros(n_dets)
            det_diff_sum = np.zeros(n_dets)
            det_diff_sum_sq = np.zeros(n_dets)
            det_diff_count = np.zeros(n_dets, dtype=int)

        det_median_sum += np.nanmedian(chunk.signal, axis=0)

        diffs = chunk.signal[1:] * flag_mask[1:] - chunk.signal[:-1] * flag_mask[:-1]

        valid_mask = np.isfinite(diffs)
        det_diff_sum += np.nansum(diffs, axis=0)
        det_diff_sum_sq += np.nansum((diffs ** 2), axis=0)
        det_diff_count += np.sum(valid_mask, axis=0)
        n_chunks += 1

        if chunk.signal.shape[0] == nominal_chunk_len:
            f, p = welch(chunk.signal, fs=sample_rate, window="hann", axis=0)
            if psd_sum is None:
                psd_sum = np.zeros_like(p)
                psd_freqs = f
            psd_sum += p
            psd_count += 1

    obs_info = dict(
        t_start_g3s=t_obs_start,
        t_stop_g3s=t_obs_stop,
        duration_s=t_obs_stop - t_obs_start,
        n_detectors=n_dets,
        sample_rate_hz=sample_rate,
        n_chunks=n_chunks,
    )

    var_diffs = (det_diff_sum_sq / det_diff_count) - (det_diff_sum / det_diff_count) ** 2
    det_noise = np.sqrt(np.clip(var_diffs, 0, None)) / np.sqrt(2)

    psd_avg = psd_sum / psd_count if psd_count > 0 else None
    white_noise_floor = None

    if psd_avg is not None:
        nyquist = sample_rate / 2
        white_band = (psd_freqs > 0.4 * nyquist) & (psd_freqs < 0.9 * nyquist)
        white_noise_floor = np.median(psd_avg[white_band, :], axis=0)

    return (ra_sum / n_bore, dec_sum / n_bore, det_median_sum / n_chunks,
            det_noise, kids, obs_info, white_noise_floor, psd_avg, psd_freqs)


def _check_psd_anomalies(psd_avg: np.ndarray, psd_freqs: np.ndarray, pipe_cfg: dict) -> list:
    psd_anomalies = []
    if psd_avg is not None:
        anomaly_sigma = pipe_cfg.get("psd_anomaly_sigma", 5.0)
        if anomaly_sigma > 0:
            psd_anomalies = find_psd_anomalies(psd_avg, psd_freqs, sigma=anomaly_sigma)
            if psd_anomalies:
                summary = ", ".join(f"{a['freq_hz']:.2f} Hz ({a['power_ratio_db']:.1f} dB)"
                                    for a in psd_anomalies[:5])
                print(f"  PSD anomalies flagged: {len(psd_anomalies)} ({summary}"
                      f"{', ...' if len(psd_anomalies) > 5 else ''})")
    return psd_anomalies


def _filter_detectors(all_kids: list, det_noise: np.ndarray, white_noise_floor: np.ndarray,
                      pipe_cfg: dict) -> tuple[np.ndarray, np.ndarray, set, list, list, float]:
    n_total = len(all_kids)
    name_to_i = {k: i for i, k in enumerate(all_kids)}
    exclude_set = set()

    manual_names = pipe_cfg.get("exclude_detectors", [])
    manual_indices = pipe_cfg.get("exclude_detector_indices", [])
    for nm in manual_names:
        if nm in name_to_i:
            exclude_set.add(name_to_i[nm])
    for idx in manual_indices:
        if 0 <= idx < n_total:
            exclude_set.add(idx)

    auto_thresh = pipe_cfg.get("auto_exclude_threshold", 5.0)
    auto_excluded = []
    if auto_thresh > 0:
        median_noise = np.median(det_noise)
        cutoff_noise = auto_thresh * median_noise
        auto_bad = np.where(det_noise > cutoff_noise)[0]
        auto_excluded = [all_kids[i] for i in auto_bad]
        exclude_set.update(auto_bad.tolist())
        print(f"  Auto-exclusion: noise median={median_noise:.4f}  "
              f"cutoff={cutoff_noise:.4f} ({auto_thresh}x)  "
              f"excluded={len(auto_bad)}/{n_total}")

    if manual_names or manual_indices:
        print(f"  Manual exclusion: {len(manual_names) + len(manual_indices)} detector(s)")

    raw_keep_idx = np.array([i for i in range(n_total) if i not in exclude_set], dtype=int)

    wnf_sigma = pipe_cfg.get("wnf_exclude_sigma", 3.0)
    wnf_excluded = []
    wnf_cutoff = None
    if wnf_sigma > 0 and white_noise_floor is not None:
        finite_wnf = np.isfinite(white_noise_floor) & (white_noise_floor > 0)
        log_wnf = np.log10(white_noise_floor[finite_wnf])
        wnf_cutoff = 10 ** (np.median(log_wnf) + wnf_sigma * np.std(log_wnf))
        wnf_bad = np.where(finite_wnf & (white_noise_floor > wnf_cutoff))[0]
        wnf_excluded = [all_kids[i] for i in wnf_bad]
        exclude_set.update(wnf_bad.tolist())
        print(f"  WNF-exclusion: cutoff={wnf_cutoff:.4g} (log-median+{wnf_sigma}sigma)  "
              f"excluded={len(wnf_bad)}/{n_total}")

    keep_idx = np.array([i for i in range(n_total) if i not in exclude_set], dtype=int)
    print(f"  Using {len(keep_idx)}/{n_total} detectors")

    return raw_keep_idx, keep_idx, exclude_set, auto_excluded, wnf_excluded, wnf_cutoff


def _compute_detector_weights(white_noise_floor: np.ndarray, n_total: int, pipe_cfg: dict) -> np.ndarray:
    weight_cap_factor = pipe_cfg.get("weight_cap_factor", 5.0)
    det_weights = np.ones(n_total)
    if weight_cap_factor > 0 and white_noise_floor is not None:
        finite_wnf = np.isfinite(white_noise_floor) & (white_noise_floor > 0)
        if np.any(finite_wnf):
            det_weights[finite_wnf] = 1.0 / white_noise_floor[finite_wnf]
            median_w = np.median(det_weights[finite_wnf])
            det_weights = np.clip(det_weights, 0, weight_cap_factor * median_w)
            det_weights[~finite_wnf] = median_w
    return det_weights


def _resolve_map_center(cfg: dict, obs_info: dict, ra0_auto: float, dec0_auto: float) -> tuple[float, float, str]:
    map_cfg = cfg["map"]
    centre_pinned = "ra0_deg" in map_cfg and "dec0_deg" in map_cfg
    target_name = map_cfg.get("target")

    if centre_pinned:
        return map_cfg["ra0_deg"], map_cfg["dec0_deg"], "config"
    elif target_name:
        site = None
        if target.is_solar_system_body(target_name):
            site = (get_blasttng_site(cfg) if cfg["data"]["format"] == "blasttng"
                    else target.FYST_SITE)
            if site is None:
                raise RuntimeError(
                    f"Can't resolve target '{target_name}': no lat/lon/alt telemetry "
                    f"found in the data to compute its ephemeris position."
                )
        ra0_deg, dec0_deg = target.resolve_target(target_name, obs_info["t_start_g3s"], site=site)
        return ra0_deg, dec0_deg, f"target lookup ({target_name})"
    else:
        return ra0_auto, dec0_auto, "auto (mean boresight)"


def step_1_first_pass(cfg: dict) -> dict:
    map_cfg = cfg["map"]
    pipe_cfg = cfg["pipeline"]
    centre_pinned = "ra0_deg" in map_cfg and "dec0_deg" in map_cfg

    print(f"Step 1: First pass ({'baselines only' if centre_pinned else 'baselines + map centre'})...")
    t = time.perf_counter()
    (ra0_auto, dec0_auto, det_offsets, det_noise, all_kids, obs_info,
     white_noise_floor, psd_avg, psd_freqs) = _first_pass(cfg)
    print(f"  Baselines: {det_offsets.min():.4f} - {det_offsets.max():.4f}  [{time.perf_counter()-t:.1f}s]")

    psd_anomalies = _check_psd_anomalies(psd_avg, psd_freqs, pipe_cfg)

    raw_keep_idx, keep_idx, exclude_set, auto_excluded, wnf_excluded, wnf_cutoff = _filter_detectors(
        all_kids, det_noise, white_noise_floor, pipe_cfg
    )

    n_total = len(all_kids)
    det_weights = _compute_detector_weights(white_noise_floor, n_total, pipe_cfg)

    _G3_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
    obs_utc = _G3_EPOCH + timedelta(seconds=obs_info["t_start_g3s"])
    print(f"  Obs start  : {obs_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Duration   : {obs_info['duration_s']:.1f} s")
    print(f"  Detectors  : {obs_info['n_detectors']} @ {obs_info['sample_rate_hz']:.1f} Hz")

    ra0_deg, dec0_deg, centre_source = _resolve_map_center(cfg, obs_info, ra0_auto, dec0_auto)

    res_deg = map_cfg["res_arcmin"] / 60.0
    ra_edges, dec_edges = make_map_edges(
        ra0_deg=ra0_deg, dec0_deg=dec0_deg,
        xlen_deg=map_cfg["xlen_deg"], ylen_deg=map_cfg["ylen_deg"],
        res_deg=res_deg,
    )
    ny, nx = len(dec_edges) - 1, len(ra_edges) - 1
    print(f"  Map grid   : {ny} x {nx} pixels ({map_cfg['res_arcmin']:.1f} arcmin/pixel), "
          f"centre = ({ra0_deg:.4f}, {dec0_deg:.4f}) [{centre_source}]")

    return {
        "det_offsets": det_offsets,
        "all_kids": all_kids,
        "obs_info": obs_info,
        "obs_utc": obs_utc,
        "white_noise_floor": white_noise_floor,
        "psd_avg": psd_avg,
        "psd_freqs": psd_freqs,
        "psd_anomalies": psd_anomalies,
        "raw_keep_idx": raw_keep_idx,
        "keep_idx": keep_idx,
        "exclude_set": exclude_set,
        "auto_excluded": auto_excluded,
        "wnf_excluded": wnf_excluded,
        "wnf_cutoff": wnf_cutoff,
        "det_weights": det_weights,
        "ra0_deg": ra0_deg,
        "dec0_deg": dec0_deg,
        "ra_edges": ra_edges,
        "dec_edges": dec_edges,
        "ny": ny,
        "nx": nx,
        "kids_kept": [all_kids[i] for i in keep_idx],
        "raw_det_offsets_kept": det_offsets[raw_keep_idx],
        "raw_kids_kept": [all_kids[i] for i in raw_keep_idx],
        "det_weights_kept": det_weights[keep_idx],
        "raw_det_weights_kept": det_weights[raw_keep_idx],
        "det_offsets_kept": det_offsets[keep_idx],
    }


# ============================================================================ #
# PER-DETECTOR PASS & SHIFTS
# ============================================================================ #
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


def _per_detector_pass(cfg: dict, pipe_cfg: dict, pd_cfg: dict,
                       ra_edges: np.ndarray, dec_edges: np.ndarray,
                       det_offsets: np.ndarray):
    ny = len(dec_edges) - 1
    nx = len(ra_edges) - 1

    apply_offsets = pd_cfg.get("apply_offsets", True)
    kids_sel = sel_idx = None
    det_data = det_hits = None

    for chunk in iter_chunks(cfg, apply_offsets=apply_offsets):
        if kids_sel is None:
            kids_sel, sel_idx = _resolve_det_selection(chunk.kids, pd_cfg)
            n_sel = len(kids_sel)
            det_data = np.zeros((n_sel, ny, nx), dtype=np.float32)
            det_hits = np.zeros((n_sel, ny, nx), dtype=np.float32)

        flags = chunk.flags
        flag_mask = np.ones(np.shape(flags))
        flag_mask[flags != 0] = np.nan
        sig = chunk.signal - det_offsets[np.newaxis, :]

        sig, new_flags = clean_tod(sig, flag_mask, chunk.sample_rate,
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
            d, h = bin_detector(sig[:, i], bin_flag_mask[:, i], ra, dec, ra_edges, dec_edges)
            det_data[j] += d
            det_hits[j] += h

    return kids_sel, det_data, det_hits


def _compute_shifts_from_pd_results(pd_results: tuple, ra_edges: np.ndarray, dec_edges: np.ndarray,
                                     ra0_deg: float, dec0_deg: float, pd_cfg: dict) -> dict:
    kids_sel, det_data, det_hits = pd_results
    ra_centres = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec_centres = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    min_snr = pd_cfg.get("shift_min_snr", 3.0)
    kid_shifts = {}

    for j, kid in enumerate(kids_sel):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(det_hits[j] > 0, det_data[j] / det_hits[j], np.nan)
        if not np.any(np.isfinite(m)):
            continue

        filled = np.where(np.isfinite(m), m, np.nanmedian(m))
        smoothed = gaussian_filter(filled, sigma=2)
        iy, ix = np.unravel_index(np.argmax(smoothed), smoothed.shape)

        off_mask = np.isfinite(m) & (m < np.nanpercentile(m[np.isfinite(m)], 50))
        noise = np.sqrt(np.nanmean(m[off_mask] ** 2)) if off_mask.any() else np.nan
        peak_val = m[iy, ix]
        peak_snr = peak_val / noise if (np.isfinite(noise) and noise > 0 and np.isfinite(peak_val)) else np.nan

        if np.isfinite(peak_snr) and peak_snr >= min_snr:
            kid_shifts[kid] = (ra0_deg - ra_centres[ix], dec0_deg - dec_centres[iy])

    print(f"  Computed shifts for {len(kid_shifts)}/{len(kids_sel)} detectors (S/N >= {min_snr})")
    return kid_shifts


def step_1b_shift_correction(cfg: dict, step1_res: dict) -> tuple[dict | None, tuple | None]:
    pipe_cfg = cfg["pipeline"]
    pd_cfg = cfg.get("per_detector", {})
    ra_edges, dec_edges = step1_res["ra_edges"], step1_res["dec_edges"]
    det_offsets = step1_res["det_offsets"]

    kid_shifts = None
    pd_results = None

    if cfg["data"]["format"] == "blasttng":
        baked_shifts = get_blasttng_baked_shifts(cfg)
        if baked_shifts:
            kid_shifts = baked_shifts
            print(f"\nStep 1b: Using pre-baked per-detector shifts from the calibration "
                  f"frame ({len(kid_shifts)} detectors) -- skipping empirical computation.")

    if cfg["data"]["format"] == "blasttng" and pd_cfg.get("enabled", False):
        label = "Per-detector maps" if kid_shifts is not None else "Per-detector maps (for shift correction)"
        print(f"\nStep 1b: {label}...")
        t = time.perf_counter()
        kids_sel, det_data, det_hits = _per_detector_pass(
            cfg, pipe_cfg, pd_cfg, ra_edges, dec_edges, det_offsets,
        )
        print(f"  Pass complete [{time.perf_counter() - t:.1f}s]")
        pd_results = (kids_sel, det_data, det_hits)

    if kid_shifts is None and pd_results is not None:
        kid_shifts = _compute_shifts_from_pd_results(
            pd_results, ra_edges, dec_edges, step1_res["ra0_deg"], step1_res["dec0_deg"], pd_cfg
        )

    return kid_shifts, pd_results


# ============================================================================ #
# STREAMING PASS DRIVER
# ============================================================================ #
def _streaming_pass(cfg: dict, pipe_cfg: dict,
                    ra_edges: np.ndarray, dec_edges: np.ndarray,
                    det_offsets: np.ndarray,
                    current_map: np.ndarray = None,
                    common_mode: bool = True,
                    return_sample: bool = False,
                    keep_idx: np.ndarray = None,
                    boresight_only: bool = False,
                    collect_tod_rms: bool = False,
                    kids_kept: list = None,
                    kid_shifts: dict = None,
                    compute_time_null: bool = True,
                    compute_detsplit_null: bool = False,
                    weights: np.ndarray = None):
    ny = len(dec_edges) - 1
    nx = len(ra_edges) - 1
    total_data = np.zeros((ny, nx), dtype=float)
    total_hits = np.zeros((ny, nx), dtype=float)
    total_sumsq = np.zeros((ny, nx), dtype=float)

    null_map = None
    if compute_time_null:
        null_data = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]
        null_hits = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]

    detsplit_null_map = None
    if compute_detsplit_null:
        n_dets_kept = len(det_offsets)
        det_half = np.random.default_rng(0).integers(0, 2, size=n_dets_kept)
        detsplit_data = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]
        detsplit_hits = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]

    n_dets = sample_rate = None
    psd_raw = psd_cm = None
    tod_rms_data: list = []

    for chunk in iter_chunks(cfg):
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

        if keep_idx is not None:
            sig = chunk.signal[:, keep_idx] - det_offsets[np.newaxis, :]
            flags = flags[:, keep_idx]
            if ra is not None:
                ra = ra[:, keep_idx]
                dec = dec[:, keep_idx]
        else:
            sig = chunk.signal - det_offsets[np.newaxis, :]

        flag_mask = np.ones(np.shape(flags))
        flag_mask[flags != 0] = np.nan

        if boresight_only or ra is None:
            ra = np.repeat(chunk.ra_bore[:, np.newaxis], sig.shape[1], axis=1)
            dec = np.repeat(chunk.dec_bore[:, np.newaxis], sig.shape[1], axis=1)

            if no_per_detector_offsets and not boresight_only and kid_shifts and kids_kept is not None:
                shift_ra = np.array([kid_shifts.get(k, (0.0, 0.0))[0] for k in kids_kept])
                shift_dec = np.array([kid_shifts.get(k, (0.0, 0.0))[1] for k in kids_kept])
                ra = ra + shift_ra[np.newaxis, :]
                dec = dec + shift_dec[np.newaxis, :]

        sig, new_flags = clean_tod(sig, flag_mask, chunk.sample_rate,
                                   steps=pipe_cfg.get("clean_steps", []),
                                   step_params={
                                       "cosmic_rays": pipe_cfg.get("cosmic_rays", {}),
                                       "highpass": pipe_cfg.get("highpass", {}),
                                       "notch": pipe_cfg.get("notch", {}),
                                   })

        bin_flag_mask = np.where(new_flags != 0, np.nan, flag_mask)

        if return_sample and psd_raw is None:
            psd_raw = sig.copy()

        if collect_tod_rms:
            rms_raw = float(np.median(np.sqrt(np.mean(sig ** 2, axis=0))))

        if common_mode:
            if current_map is not None:
                sig = iterate_common_mode(sig, flag_mask, ra, dec,
                                          current_map, ra_edges, dec_edges)
            else:
                sig = subtract_common_mode(sig, estimate_common_mode(sig, flag_mask))

        if return_sample and psd_cm is None:
            psd_cm = sig.copy() * flag_mask

        if collect_tod_rms:
            rms_cm = float(np.median(np.sqrt(np.mean(sig ** 2, axis=0)))) if common_mode else rms_raw
            tod_rms_data.append((chunk.t_start, rms_raw, rms_cm))

        d, h, sq = bin_chunk(sig, bin_flag_mask, ra, dec, ra_edges, dec_edges, weights=weights)
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
                d_ds, h_ds, _ = bin_chunk(sig[:, m], bin_flag_mask[:, m], ra[:, m], dec[:, m],
                                          ra_edges, dec_edges, weights=w_half)
                detsplit_data[dhalf] += d_ds
                detsplit_hits[dhalf] += h_ds

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
    return (combined, total_hits, noise_map, null_map, detsplit_null_map,
            n_dets, sample_rate, sample, tod_rms_data)


# ============================================================================ #
# STEPS 1c, 2, & 3: MAPMAKING PASSES
# ============================================================================ #
def step_1c_raw_map(cfg: dict, step1_res: dict, kid_shifts: dict | None) -> np.ndarray:
    pipe_cfg = cfg["pipeline"]
    t = time.perf_counter()
    n_total = len(step1_res["all_kids"])
    print(f"\nStep 1c: Raw map ({len(step1_res['raw_keep_idx'])}/{n_total} detectors, "
          f"manual+auto exclusion only)...")

    raw_map, _, _, _, _, _, _, _, _ = _streaming_pass(
        cfg, pipe_cfg, step1_res["ra_edges"], step1_res["dec_edges"],
        step1_res["raw_det_offsets_kept"], common_mode=False,
        keep_idx=step1_res["raw_keep_idx"], kids_kept=step1_res["raw_kids_kept"],
        kid_shifts=kid_shifts, weights=step1_res["raw_det_weights_kept"],
        compute_time_null=pipe_cfg.get("compute_time_null", True)
    )
    print(f"  [{time.perf_counter() - t:.1f}s]")
    return raw_map


def step_2_naive_and_initial_cm(cfg: dict, step1_res: dict, kid_shifts: dict | None) -> dict:
    pipe_cfg = cfg["pipeline"]
    chunk_s = pipe_cfg.get("chunk_duration_s", 1.0)

    t = time.perf_counter()
    print(f"Step 2: Naive map (chunk_duration_s={chunk_s})...")
    naive, _, _, _, _, n_dets, sr, raw_sample, _ = _streaming_pass(
        cfg, pipe_cfg, step1_res["ra_edges"], step1_res["dec_edges"],
        step1_res["det_offsets_kept"], common_mode=False, return_sample=True,
        keep_idx=step1_res["keep_idx"], kids_kept=step1_res["kids_kept"],
        kid_shifts=kid_shifts, weights=step1_res["det_weights_kept"],
        compute_time_null=pipe_cfg.get("compute_time_null", True)
    )
    t_naive = time.perf_counter() - t
    print(f"  {n_dets} detectors, {sr:.1f} Hz  [{t_naive:.1f}s]")

    t = time.perf_counter()
    print("Step 2: Initial common-mode pass...")
    (combined_map, hits, noise_map, null_map, detsplit_null_map,
     _, _, cm_sample, tod_rms_data) = _streaming_pass(
        cfg, pipe_cfg, step1_res["ra_edges"], step1_res["dec_edges"],
        step1_res["det_offsets_kept"], return_sample=True,
        keep_idx=step1_res["keep_idx"], collect_tod_rms=True,
        kids_kept=step1_res["kids_kept"], kid_shifts=kid_shifts,
        compute_time_null=pipe_cfg.get("compute_time_null", True),
        compute_detsplit_null=pipe_cfg.get("compute_detsplit_null", True),
        weights=step1_res["det_weights_kept"]
    )
    t_it0 = time.perf_counter() - t
    print(f"  [{t_it0:.1f}s]")

    return {
        "naive": naive,
        "combined_map": combined_map,
        "hits": hits,
        "noise_map": noise_map,
        "null_map": null_map,
        "detsplit_null_map": detsplit_null_map,
        "sr": sr,
        "raw_sample": raw_sample,
        "cm_sample": cm_sample,
        "tod_rms_data": tod_rms_data,
        "t_naive": t_naive,
        "t_it0": t_it0,
    }


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
        "psd_anomalies": step1_res["psd_anomalies"],
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

    if cfg["data"]["format"] == "blasttng":
        blasttng_cfg = cfg.get("blasttng", {})
        metadata["df_method"] = blasttng_cfg.get("df_method", "hybrid")
        metadata["threshold_frac"] = blasttng_cfg.get("threshold_frac", 0.05)

    return metadata


def step_4_save_outputs(cfg: dict, step1_res: dict, step2_res: dict, raw_map: np.ndarray,
                        cm_maps: list, pass_times: list, t_total: float) -> tuple[str, dict]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    n_dets = step1_res["obs_info"]["n_detectors"]
    out_dir = os.path.join(pathlib.Path(cfg["output"]["output_dir"]),
                           f'{cfg["output"]["obs_object"]}_d{n_dets}_{timestamp}')

    output.save_iteration_maps(
        step2_res["naive"], cm_maps, step2_res["hits"], step2_res["noise_map"],
        step2_res["null_map"], step1_res["ra_edges"], step1_res["dec_edges"],
        out_dir, cfg["output"], raw_map=raw_map, detsplit_null=step2_res["detsplit_null_map"]
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
def main():
    parser = argparse.ArgumentParser(description="CCAT Quick-Look Mapmaker")
    parser.add_argument("--config", default="config.toml", help="Path to config file (default: config.toml)")
    parser.add_argument("--profile", action="store_true", help="Print timing bottlenecks on exit")
    args = parser.parse_args()

    profiler = cProfile.Profile()
    if args.profile:
        profiler.enable()

    cfg, config_path = load_configuration(args.config)
    t_total = time.perf_counter()

    print("=" * 60)
    print("CCAT Quick-Look Mapmaker")
    print("=" * 60)
    print(f"Config : {config_path}")
    for d in cfg['data']['input_dirs']:
        print(f"Input  : {d}")
    print(f"Format : {cfg['data']['format']}")
    print(f"Output : {cfg['output']['output_dir']}\n")

    step_0_probe_medians(cfg)

    step1_res = step_1_first_pass(cfg)

    kid_shifts, pd_results = step_1b_shift_correction(cfg, step1_res)

    raw_map = step_1c_raw_map(cfg, step1_res, kid_shifts)

    step2_res = step_2_naive_and_initial_cm(cfg, step1_res, kid_shifts)

    combined_map, cm_maps, pass_times = step_3_iterations(cfg, step1_res, step2_res, kid_shifts)

    out_dir, metadata = step_4_save_outputs(cfg, step1_res, step2_res, raw_map, cm_maps, pass_times, t_total)

    step_5_boresight_comparison(cfg, step1_res, combined_map, out_dir)

    step_6_per_detector_maps(cfg, step1_res, pd_results, out_dir, metadata)

    finish_run(args, profiler, out_dir, time.perf_counter() - t_total)


if __name__ == "__main__":
    main()