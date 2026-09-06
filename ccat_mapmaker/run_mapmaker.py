#!/usr/bin/env python
# ============================================================================ #
# run_mapmaker.py
#
# CCAT Quick-Look Mapmaker -- main entry point.
#
# Usage:
#   python run_mapmaker.py                          # uses config.toml
#   python run_mapmaker.py --config my_config.toml
#   python run_mapmaker.py --profile                # print timing bottlenecks
#
# Pipeline:
#   1. First pass    -- compute per-detector baselines and map centre
#   2. Naive map     -- bin cleaned signal with no common-mode subtraction
#   3. Initial CM    -- subtract naive mean across detectors, rebin
#   4. Iterate       -- subtract sky-informed common mode, rebin (n_iterations times)
#   5. Save          -- write maps and metadata to disk
# ============================================================================ #

import sys
import pathlib
import argparse
import tomllib
import time
import json
import cProfile
import pstats
import io
from datetime import datetime, timedelta, timezone
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import periodogram

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from mapmaker.reader      import iter_chunks, get_blasttng_baked_shifts, get_blasttng_site
from mapmaker.cleaning    import clean_tod, find_psd_anomalies
from mapmaker.binning     import make_map_edges, bin_chunk, bin_detector
from mapmaker.common_mode import estimate_common_mode, subtract_common_mode, iterate_common_mode
from mapmaker               import output
from mapmaker               import target


def resolve_paths(cfg: dict, config_path: pathlib.Path) -> dict:
    cfg_dir = config_path.parent.resolve()

    def resolve(path_str: str) -> str:
        p = pathlib.Path(path_str)
        if not p.is_absolute():
            p = (cfg_dir / p).resolve()
        return str(p)

    cfg["data"]["input_dirs"]   = [resolve(d) for d in cfg["data"]["input_dirs"]]
    cfg["output"]["output_dir"] = resolve(cfg["output"]["output_dir"])
    return cfg


def _first_pass(cfg: dict):
    """
    Single streaming pass to compute per-detector baselines, noise, mean boresight,
    and white-noise-floor PSD.

    Uses the median of each chunk's signal (averaged across chunks) for the
    baseline. Computes per-detector noise as the true global std of first-differences by accumulating sum and sum-of-squares
    across all chunks before computing the final statistic.

    PSD is accumulated as an average of per-chunk periodograms, equivalent to
    Welch's method using each chunk as one segment, so the full timestream never
    needs to be held in memory (contrast with computing this in a notebook via
    scipy.signal.welch on the concatenated array). Only full-length chunks are
    included, since a truncated final chunk has a different frequency axis and
    can't be averaged with the others; with hundreds of chunks per observation,
    dropping one is negligible.
    white_noise_floor is the median PSD in a high-frequency band (0.4-0.9x
    Nyquist), same convention as blasttng_flagging_characterization.ipynb.
    """
    ra_sum = dec_sum = n_bore = 0
    det_median_sum = det_diff_sum = det_diff_sum_sq = None
    det_diff_count = 0
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
            n_dets      = chunk.signal.shape[1]
            sample_rate = chunk.sample_rate
            kids        = chunk.kids
            nominal_chunk_len = chunk.signal.shape[0]

        t_obs_stop = chunk.t_stop

        ra_sum  += chunk.ra_bore.sum()
        dec_sum += chunk.dec_bore.sum()
        n_bore  += len(chunk.ra_bore)

        if det_median_sum is None:
            det_median_sum   = np.zeros(n_dets)
            det_diff_sum     = np.zeros(n_dets)
            det_diff_sum_sq  = np.zeros(n_dets)
        det_median_sum  += np.nanmedian(chunk.signal, axis=0)
        diffs            = chunk.signal[1:]*flag_mask[1:] - chunk.signal[:-1]*flag_mask[:-1]
        det_diff_sum    += np.nansum(diffs, axis=0)
        det_diff_sum_sq += np.nansum((diffs ** 2), axis=0)
        det_diff_count  += len(diffs)
        n_chunks += 1

        if chunk.signal.shape[0] == nominal_chunk_len:
            f, p = periodogram(chunk.signal, fs=sample_rate, axis=0)
            if psd_sum is None:
                psd_sum   = np.zeros_like(p)
                psd_freqs = f
            psd_sum   += p
            psd_count += 1

    obs_info = dict(
        t_start_g3s    = t_obs_start,
        t_stop_g3s     = t_obs_stop,
        duration_s     = t_obs_stop - t_obs_start,
        n_detectors    = n_dets,
        sample_rate_hz = sample_rate,
        n_chunks       = n_chunks,
    )
    det_noise = np.sqrt(np.clip(
        det_diff_sum_sq / det_diff_count - (det_diff_sum / det_diff_count) ** 2,
        0, None
    ))

    psd_avg = psd_sum / psd_count if psd_count > 0 else None
    white_noise_floor = None
    if psd_avg is not None:
        nyquist    = sample_rate / 2
        white_band = (psd_freqs > 0.4 * nyquist) & (psd_freqs < 0.9 * nyquist)
        # psd_avg is (n_freq, n_dets) -- median over freq axis for each detector
        white_noise_floor = np.median(psd_avg[white_band, :], axis=0)

    return (ra_sum / n_bore, dec_sum / n_bore, det_median_sum / n_chunks, det_noise,
            kids, obs_info, white_noise_floor, psd_avg, psd_freqs)


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
    """
    One streaming pass over all chunks: baseline subtract, clean, optionally
    common-mode subtract, then bin into the map accumulator.

    current_map: if provided, uses sky-informed common-mode (iterate_common_mode);
                 otherwise uses naive mean across detectors.
    common_mode: if False, skips common-mode subtraction entirely (naive map).
    weights: optional (n_dets,) per-detector map weight, same order as
             det_offsets/keep_idx (e.g. inverse-variance from white_noise_floor).
             Passed straight through to bin_chunk; None means uniform weighting.
    return_sample: if True, captures the first chunk's signal before and after
                   CM subtraction for PSD diagnostics.
    collect_tod_rms: if True, records median detector RMS per chunk before and
                     after CM subtraction as a list of (t_start, rms_raw, rms_cm).
    kids_kept: names of the kept detectors, same order as det_offsets/keep_idx.
               Only used to look up kid_shifts; ignored otherwise.
    kid_shifts: optional {kid_name: (ra_shift_deg, dec_shift_deg)}, applied on
                top of shared boresight pointing when per-detector offsets
                aren't known (chunk.ra is None) -- e.g. real data before focal
                plane calibration, where each detector's own peak (found by
                the per-detector pass) is used to correct for its position on
                the array, same mechanism blasttng-to-g3/g3_utils and mmi both
                use. Never applied when boresight_only=True (that mode means
                "ignore any offset, show pure boresight" by design), and has
                no effect when chunk.ra is already populated (simulation with
                apply_offsets=True), so this leaves simulation behaviour
                unchanged.
    compute_time_null: if False, skips the chunk-parity (even/odd) null/
                jackknife map -- numerically free (reuses the main bin_chunk
                result), so this only saves the final map-arithmetic and
                whatever output.py would've written for it. Returns
                null_map=None when off.
    compute_detsplit_null: if True, also builds a null map from a random 50/50
                *detector*-identity split (fixed seed, reproducible) rather than
                the chunk-parity time split above -- tests whether one random
                half of the array agrees with the other, independent of any
                time-domain noise correlation. Costs two extra bin_chunk calls
                per chunk (unlike the chunk-parity null map), so off by
                default -- only worth enabling once per run, not on every pass.
    """
    ny = len(dec_edges) - 1
    nx = len(ra_edges)  - 1
    total_data  = np.zeros((ny, nx), dtype=float)
    total_hits  = np.zeros((ny, nx), dtype=float)
    total_sumsq = np.zeros((ny, nx), dtype=float)

    # Null-test split: whole chunks alternate between two independent halves
    # (chunk_index parity) so both halves get matched sky coverage over the
    # observation. (data_a, hits_a), (data_b, hits_b). Only touched when enabled.
    null_map = None
    if compute_time_null:
        null_data = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]
        null_hits = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]

    # Detector-split null test: random 50/50 split by detector identity, fixed
    # over the whole pass. Separate accumulators, only touched when enabled.
    detsplit_null_map = None
    if compute_detsplit_null:
        n_dets_kept   = len(det_offsets)
        det_half      = np.random.default_rng(0).integers(0, 2, size=n_dets_kept)
        detsplit_data = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]
        detsplit_hits = [np.zeros((ny, nx), dtype=float), np.zeros((ny, nx), dtype=float)]

    n_dets = sample_rate = None
    psd_raw = psd_cm = None
    tod_rms_data: list = []

    for chunk in iter_chunks(cfg):
        flags = chunk.flags
        chunk_id = chunk.chunk_index
        if np.all(flags != 0):
            # print(f"No good data in chunk with index {chunk_id}. Skipping to next chunk")
            continue
        if n_dets is None:
            n_dets      = len(chunk.kids)
            sample_rate = chunk.sample_rate

        ra    = chunk.ra
        dec   = chunk.dec
        flags = chunk.flags
        no_per_detector_offsets = ra is None

        if keep_idx is not None:
            sig   = chunk.signal[:, keep_idx] - det_offsets[np.newaxis, :]
            flags = flags[:, keep_idx]
            if ra is not None:
                ra  = ra[:,  keep_idx]
                dec = dec[:, keep_idx]
        else:
            sig = chunk.signal - det_offsets[np.newaxis, :]

        flag_mask = np.ones(np.shape(flags))
        flag_mask[flags != 0] = np.nan

        # ra is None whenever per-detector focal-plane offsets aren't available
        # (e.g. real data before calibration, or apply_offsets=False) -- fall
        # back to shared boresight pointing for every kept detector, same as
        # the explicit boresight_only comparison pass.
        if boresight_only or ra is None:
            ra  = np.repeat(chunk.ra_bore[:, np.newaxis], sig.shape[1], axis=1)
            dec = np.repeat(chunk.dec_bore[:, np.newaxis], sig.shape[1], axis=1)

            # Per-detector shift correction -- only when offsets are truly
            # unknown (not for an explicit boresight_only comparison, which
            # means "show me it with no offset applied" by definition).
            if no_per_detector_offsets and not boresight_only and kid_shifts and kids_kept is not None:
                shift_ra  = np.array([kid_shifts.get(k, (0.0, 0.0))[0] for k in kids_kept])
                shift_dec = np.array([kid_shifts.get(k, (0.0, 0.0))[1] for k in kids_kept])
                ra  = ra  + shift_ra[np.newaxis, :]
                dec = dec + shift_dec[np.newaxis, :]

        sig, new_flags = clean_tod(sig, flag_mask, chunk.sample_rate,
                        steps=pipe_cfg.get("clean_steps", []),
                        step_params={
                            "cosmic_rays": pipe_cfg.get("cosmic_rays", {}),
                            "highpass":    pipe_cfg.get("highpass", {}),
                            "notch":       pipe_cfg.get("notch", {}),
                        })

        # Newly-detected reasons (e.g. cosmic-ray hits) only affect what
        # lands in the map, they're excluded here, but flag_mask itself
        # (used above for cleaning/common-mode) is left untouched, since
        # those steps already handle scattered gaps fine on their own.
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
        # print(f"d: {d}")
        total_data  += d
        total_hits  += h
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

        # RMS noise map: standard error of the per-pixel mean, from the actual
        # scatter of samples landing in each pixel (E[x^2] - E[x]^2 / N),
        # rather than assuming unit variance per sample (see 1/sqrt(hits)).
        mean_sq   = np.where(total_hits > 0, total_sumsq / total_hits, np.nan)
        variance  = np.clip(mean_sq - combined ** 2, 0, None)
        noise_map = np.sqrt(variance / total_hits)

        # Null/jackknife map: half-difference of two independent chunk-parity
        # splits. Should be consistent with noise (no residual structure) if
        # the combined map's apparent features are real signal, not noise.
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


def _resolve_det_selection(all_kids: list, pd_cfg: dict) -> tuple[list, np.ndarray]:
    """
    Resolve which detectors to map from per_detector config.

    Priority:
      1. detectors        -- list of detector names
      2. detector_indices -- list of integer indices
      3. max_detectors    -- 0 = all, N = evenly sample N
    Returns (selected_names, index_array).
    """
    n_total = len(all_kids)
    name_to_idx = {k: i for i, k in enumerate(all_kids)}

    names = pd_cfg.get("detectors", [])
    if names:
        valid = [(n, name_to_idx[n]) for n in names if n in name_to_idx]
        missing = [n for n in names if n not in name_to_idx]
        if missing:
            print(f"  Warning: {len(missing)} detector name(s) not found and skipped: {missing[:5]}")
        kids_sel = [n for n, _ in valid]
        sel_idx  = np.array([i for _, i in valid], dtype=int)
        return kids_sel, sel_idx

    indices = pd_cfg.get("detector_indices", [])
    if indices:
        valid = [i for i in indices if 0 <= i < n_total]
        if len(valid) < len(indices):
            print(f"  Warning: {len(indices)-len(valid)} index/indices out of range and skipped")
        sel_idx  = np.array(valid, dtype=int)
        kids_sel = [all_kids[i] for i in sel_idx]
        return kids_sel, sel_idx

    max_det = pd_cfg.get("max_detectors", 0)
    if max_det > 0 and max_det < n_total:
        sel_idx  = np.linspace(0, n_total - 1, max_det, dtype=int)
    else:
        sel_idx  = np.arange(n_total)
    kids_sel = [all_kids[i] for i in sel_idx]
    return kids_sel, sel_idx


def _per_detector_pass(cfg: dict, pipe_cfg: dict, pd_cfg: dict,
                       ra_edges: np.ndarray, dec_edges: np.ndarray,
                       det_offsets: np.ndarray):
    """
    One streaming pass accumulating a separate signal map for each detector.

    No common-mode subtraction here: per-detector maps are used for pointing
    reconstruction, PSDs, and flagging, which need each detector's own raw
    noise characteristics rather than a common-mode-cleaned residual.

    If pd_cfg["apply_offsets"] is False, detector offsets are not applied
    (see iter_g3_chunks); every detector is instead binned against the shared
    boresight sky position (chunk.ra_bore/dec_bore), since offsets aren't
    known/trusted yet. Interpreting those centroids as focal-plane offsets
    is done offline, not here.

    Detector maps are accumulated as float32 to limit memory usage.

    Returns (kids, det_data, det_hits) where det_data/hits are (n_sel, ny, nx).
    """
    ny = len(dec_edges) - 1
    nx = len(ra_edges)  - 1

    apply_offsets = pd_cfg.get("apply_offsets", True)
    kids_sel = sel_idx = None
    det_data = det_hits = None

    for chunk in iter_chunks(cfg, apply_offsets=apply_offsets):
        if kids_sel is None:
            kids_sel, sel_idx = _resolve_det_selection(chunk.kids, pd_cfg)
            n_sel    = len(kids_sel)
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
                            "highpass":    pipe_cfg.get("highpass", {}),
                            "notch":       pipe_cfg.get("notch", {}),
                        })

        # See _streaming_pass: newly-detected reasons only affect what lands
        # in the map, not the flag_mask cleaning already ran with.
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


def main():
    parser = argparse.ArgumentParser(description="CCAT Quick-Look Mapmaker")
    parser.add_argument("--config",  default="config.toml",
                        help="Path to config file (default: config.toml)")
    parser.add_argument("--profile", action="store_true",
                        help="Print timing bottlenecks on exit")
    args = parser.parse_args()

    profiler = cProfile.Profile()
    if args.profile:
        profiler.enable()

    config_path = pathlib.Path(args.config)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    cfg = resolve_paths(cfg, config_path)

    t_total = time.perf_counter()

    print("=" * 60)
    print("CCAT Quick-Look Mapmaker")
    print("=" * 60)
    print(f"Config : {config_path}")
    for d in cfg['data']['input_dirs']:
        print(f"Input  : {d}")
    print(f"Format : {cfg['data']['format']}")
    print(f"Output : {cfg['output']['output_dir']}")
    print()

    # ------------------------------------------------------------------ #
    # STEP 1: First pass -- baselines and map grid
    # ------------------------------------------------------------------ #
    map_cfg  = cfg["map"]
    pipe_cfg = cfg["pipeline"]
    res_deg  = map_cfg["res_arcmin"] / 60.0

    centre_pinned = "ra0_deg" in map_cfg and "dec0_deg" in map_cfg
    print(f"Step 1: First pass ({'baselines only' if centre_pinned else 'baselines + map centre'})...")
    t = time.perf_counter()
    (ra0_auto, dec0_auto, det_offsets, det_noise, all_kids, obs_info,
     white_noise_floor, psd_avg, psd_freqs) = _first_pass(cfg)
    print(f"  Baselines: {det_offsets.min():.4f} - {det_offsets.max():.4f}  [{time.perf_counter()-t:.1f}s]")

    # Narrowband contaminants (e.g. unexplained lines like AMKID's ~0.3 Hz --
    # Reyes et al. 2026), flagged in metadata for follow-up, not removed.
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

    # Resolve detector exclusion (manual + optional auto)
    n_total   = len(all_kids)
    name_to_i = {k: i for i, k in enumerate(all_kids)}
    exclude_set = set()

    manual_names   = pipe_cfg.get("exclude_detectors", [])
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
        auto_bad     = np.where(det_noise > cutoff_noise)[0]
        auto_excluded = [all_kids[i] for i in auto_bad]
        exclude_set.update(auto_bad.tolist())
        print(f"  Auto-exclusion: noise median={median_noise:.4f}  "
              f"cutoff={cutoff_noise:.4f} ({auto_thresh}x)  "
              f"excluded={len(auto_bad)}/{n_total}")

    if manual_names or manual_indices:
        print(f"  Manual exclusion: {len(manual_names) + len(manual_indices)} detector(s)")

    # "raw" keep set: manual + auto only, NOT white-noise-floor-filtered -- used
    # for the raw comparison map below (no common-mode applied to it).
    raw_keep_idx = np.array([i for i in range(n_total) if i not in exclude_set], dtype=int)

    # White-noise-floor exclusion, on top of manual/auto, this becomes the
    # detector set used for the actual (common-mode-iterated) science map.
    # NOT a linear N*median cutoff: white_noise_floor is extremely heavy-tailed
    # (75th percentile can already be ~7x the median), so a linear multiplier
    # excludes a large, arbitrary chunk of the array rather than catching real
    # outliers, see blasttng_flagging_characterization.ipynb Section 8 for the
    # distribution plots that motivated this. Cutoff is done in log-space
    # instead: median + N*sigma of log10(white_noise_floor), matching the
    # log-normal-ish shape of this metric (same technique as Chapin et al. 2013,
    # SCUBA-2/SMURF Section 3.1.6, "outliers from the centre of the logarithm...
    # of the bolometer noise distribution").
    wnf_sigma    = pipe_cfg.get("wnf_exclude_sigma", 3.0)
    wnf_excluded = []
    wnf_cutoff   = None
    if wnf_sigma > 0 and white_noise_floor is not None:
        finite_wnf = np.isfinite(white_noise_floor) & (white_noise_floor > 0)
        log_wnf    = np.log10(white_noise_floor[finite_wnf])
        wnf_cutoff = 10 ** (np.median(log_wnf) + wnf_sigma * np.std(log_wnf))
        wnf_bad    = np.where(finite_wnf & (white_noise_floor > wnf_cutoff))[0]
        wnf_excluded = [all_kids[i] for i in wnf_bad]
        exclude_set.update(wnf_bad.tolist())
        print(f"  WNF-exclusion: cutoff={wnf_cutoff:.4g} (log-median+{wnf_sigma}sigma)  "
              f"excluded={len(wnf_bad)}/{n_total}")

    keep_idx = np.array([i for i in range(n_total) if i not in exclude_set], dtype=int)
    det_offsets_kept = det_offsets[keep_idx]
    print(f"  Using {len(keep_idx)}/{n_total} detectors")

    # Per-detector map weight = inverse-variance (1/white_noise_floor), from
    # the whole-observation estimate above, not recomputed per chunk, since
    # chunk_duration_s is too short for a stable per-chunk PSD estimate.
    # Capped at weight_cap_factor x the median weight so one anomalously-quiet
    # detector (a WNF near zero that still slipped past the exclusion cut
    # above) can't dominate the map; detectors with no usable WNF estimate
    # fall back to the median weight rather than being silently up- or
    # down-weighted. weight_cap_factor <= 0 disables weighting entirely
    # (uniform weights, same as before this feature existed).
    weight_cap_factor = pipe_cfg.get("weight_cap_factor", 5.0)
    det_weights = np.ones(n_total)
    if weight_cap_factor > 0 and white_noise_floor is not None:
        finite_wnf = np.isfinite(white_noise_floor) & (white_noise_floor > 0)
        if np.any(finite_wnf):
            det_weights[finite_wnf] = 1.0 / white_noise_floor[finite_wnf]
            median_w = np.median(det_weights[finite_wnf])
            det_weights = np.clip(det_weights, 0, weight_cap_factor * median_w)
            det_weights[~finite_wnf] = median_w

    _G3_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
    obs_utc   = _G3_EPOCH + timedelta(seconds=obs_info["t_start_g3s"])
    print(f"  Obs start  : {obs_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Duration   : {obs_info['duration_s']:.1f} s")
    print(f"  Detectors  : {obs_info['n_detectors']} @ {obs_info['sample_rate_hz']:.1f} Hz")

    target_name = map_cfg.get("target")
    if centre_pinned:
        ra0_deg, dec0_deg = map_cfg["ra0_deg"], map_cfg["dec0_deg"]
        centre_source = "config"
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
        centre_source = f"target lookup ({target_name})"
    else:
        ra0_deg, dec0_deg = ra0_auto, dec0_auto
        centre_source = "auto (mean boresight)"

    ra_edges, dec_edges = make_map_edges(
        ra0_deg=ra0_deg, dec0_deg=dec0_deg,
        xlen_deg=map_cfg["xlen_deg"], ylen_deg=map_cfg["ylen_deg"],
        res_deg=res_deg,
    )
    ny, nx = len(dec_edges) - 1, len(ra_edges) - 1
    print(f"  Map grid   : {ny} x {nx} pixels ({map_cfg['res_arcmin']:.1f} arcmin/pixel), "
          f"centre = ({ra0_deg:.4f}, {dec0_deg:.4f}) [{centre_source}]")

    kids_kept = [all_kids[i] for i in keep_idx]
    raw_det_offsets_kept = det_offsets[raw_keep_idx]
    raw_kids_kept        = [all_kids[i] for i in raw_keep_idx]
    det_weights_kept     = det_weights[keep_idx]
    raw_det_weights_kept = det_weights[raw_keep_idx]

    # ------------------------------------------------------------------ #
    # STEP 1b: Per-detector shift correction (real data only) 
    # Real per-detector focal-plane offsets aren't known yet (see reader.py),
    # so every detector currently gets binned at the same shared boresight
    # position,  this collapses the whole array onto one line instead of
    # the wide swath a real spread-out focal plane traces out, leaving large
    # gaps between scan legs in the combined map. Fix: run the per-detector
    # pass now, and use each detector's own peak (found against boresight
    # alone) to correct for its position on the array before the combined
    # map bins it, same mechanism blasttng-to-g3/g3_utils (roach1_shifts_radec.npy,
    # baked into the calibration frame) and mmi (per-detector sourceCoords)
    # both actually use.
    # Simulation already has real per-detector offsets via apply_offsets, so
    # this is skipped entirely there and nothing about its behaviour changes.
    # ------------------------------------------------------------------ #
    pd_cfg = cfg.get("per_detector", {})
    kid_shifts = None
    pd_results = None  # (kids_sel, det_data, det_hits) -- reused at Step 6 if already computed here

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
        # No pre-baked shifts available -- fall back to computing our own,
        # empirically, from the per-detector pass just run above.
        kids_sel, det_data, det_hits = pd_results
        ra_centres  = 0.5 * (ra_edges[:-1]  + ra_edges[1:])
        dec_centres = 0.5 * (dec_edges[:-1] + dec_edges[1:])
        min_snr = pd_cfg.get("shift_min_snr", 3.0)
        kid_shifts = {}
        for j, kid in enumerate(kids_sel):
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.where(det_hits[j] > 0, det_data[j] / det_hits[j], np.nan)
            if not np.any(np.isfinite(m)):
                continue

            # Smooth before peak-finding: a single detector's own map has a
            # low hit count per pixel, so raw argmax mostly just finds the
            # brightest noise spike rather than a real source. Same fix
            # blasttng-to-g3's g3_utils.maps.SingleMapBinner.source_coords
            # uses (gaussian_filter before argmax).
            filled   = np.where(np.isfinite(m), m, np.nanmedian(m))
            smoothed = gaussian_filter(filled, sigma=2)
            iy, ix   = np.unravel_index(np.argmax(smoothed), smoothed.shape)

            # Only trust detectors with a real detection: same peak-vs-
            # off-source-noise convention as output.py's centroids.json.
            # A low-S/N "peak" is noise; shifting by it would scatter that
            # detector's contribution essentially randomly instead of
            # correcting its position, which is worse than not shifting it.
            off_mask = np.isfinite(m) & (m < np.nanpercentile(m[np.isfinite(m)], 50))
            noise    = np.sqrt(np.nanmean(m[off_mask] ** 2)) if off_mask.any() else np.nan
            peak_val = m[iy, ix]
            peak_snr = peak_val / noise if (np.isfinite(noise) and noise > 0 and np.isfinite(peak_val)) else np.nan

            if np.isfinite(peak_snr) and peak_snr >= min_snr:
                kid_shifts[kid] = (ra0_deg - ra_centres[ix], dec0_deg - dec_centres[iy])
        print(f"  Computed shifts for {len(kid_shifts)}/{len(kids_sel)} detectors (S/N >= {min_snr})")

    # ------------------------------------------------------------------ #
    # STEP 1c: Raw map : manual+auto exclusion only, no white-noise-floor
    # filtering and no common-mode subtraction. Comparison reference against
    # quiet_map/combined_map below, which additionally excludes WNF outliers
    # and gets the full common-mode treatment.
    # ------------------------------------------------------------------ #
    t = time.perf_counter()
    print(f"\nStep 1c: Raw map ({len(raw_keep_idx)}/{n_total} detectors, "
          f"manual+auto exclusion only)...")
    raw_map, _, _, _, _, _, _, _, _ = _streaming_pass(
        cfg, pipe_cfg, ra_edges, dec_edges, raw_det_offsets_kept,
        common_mode=False, keep_idx=raw_keep_idx,
        kids_kept=raw_kids_kept, kid_shifts=kid_shifts,
        weights=raw_det_weights_kept,
        compute_time_null=pipe_cfg.get("compute_time_null", True))
    print(f"  [{time.perf_counter() - t:.1f}s]")

    # ------------------------------------------------------------------ #
    # STEP 2: Naive map + initial common-mode pass
    # ------------------------------------------------------------------ #
    chunk_s     = pipe_cfg.get("chunk_duration_s", 1.0)
    t = time.perf_counter()
    print(f"Step 2: Naive map (chunk_duration_s={chunk_s})...")
    naive, _, _, _, _, n_dets, sr, raw_sample, _ = _streaming_pass(
        cfg, pipe_cfg, ra_edges, dec_edges, det_offsets_kept,
        common_mode=False, return_sample=True, keep_idx=keep_idx,
        kids_kept=kids_kept, kid_shifts=kid_shifts,
        weights=det_weights_kept,
        compute_time_null=pipe_cfg.get("compute_time_null", True))
    t_naive = time.perf_counter() - t
    print(f"  {n_dets} detectors, {sr:.1f} Hz  [{t_naive:.1f}s]")

    t = time.perf_counter()
    print("Step 2: Initial common-mode pass...")
    (combined_map, hits, noise_map, null_map, detsplit_null_map,
     _, _, cm_sample, tod_rms_data) = _streaming_pass(
        cfg, pipe_cfg, ra_edges, dec_edges, det_offsets_kept,
        return_sample=True, keep_idx=keep_idx, collect_tod_rms=True,
        kids_kept=kids_kept, kid_shifts=kid_shifts,
        compute_time_null=pipe_cfg.get("compute_time_null", True),
        compute_detsplit_null=pipe_cfg.get("compute_detsplit_null", True),
        weights=det_weights_kept)
    t_it0 = time.perf_counter() - t
    print(f"  [{t_it0:.1f}s]")

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir    = os.path.join(pathlib.Path(cfg["output"]["output_dir"]), f'{cfg["output"]["obs_object"]}_d{n_dets}_{timestamp}')
    cm_maps    = [("it_0", combined_map.copy())]
    pass_times = [("naive", t_naive), ("it_0", t_it0)]

    # ------------------------------------------------------------------ #
    # STEP 3: Iterative common-mode passes
    # ------------------------------------------------------------------ #
    n_iters = pipe_cfg["n_iterations"]
    if n_iters > 0:
        print(f"Step 3: {n_iters} iteration(s)...")
        for i in range(1, n_iters + 1):
            t = time.perf_counter()
            print(f"  Iteration {i}/{n_iters}...", end=" ", flush=True)
            combined_map, hits, noise_map, null_map, _, _, _, _, _ = _streaming_pass(
                cfg, pipe_cfg, ra_edges, dec_edges, det_offsets_kept,
                current_map=combined_map, keep_idx=keep_idx,
                kids_kept=kids_kept, kid_shifts=kid_shifts,
                weights=det_weights_kept,
                compute_time_null=pipe_cfg.get("compute_time_null", True),
            )
            t_iter = time.perf_counter() - t
            cm_maps.append((f"it_{i}", combined_map.copy()))
            pass_times.append((f"it_{i}", t_iter))
            print(f"[{t_iter:.1f}s]")
    else:
        print("Step 3: Skipping (n_iterations = 0).")

    # ------------------------------------------------------------------ #
    # STEP 4: Save outputs
    # ------------------------------------------------------------------ #
    output.save_iteration_maps(naive, cm_maps, hits, noise_map, null_map,
                               ra_edges, dec_edges, out_dir, cfg["output"],
                               raw_map=raw_map, detsplit_null=detsplit_null_map)

    output.plot_wnf_diagnostic(white_noise_floor, wnf_cutoff, wnf_sigma,
                               os.path.join(out_dir, "wnf_diagnostic.png"))
    print("    Saved wnf_diagnostic.png")

    metrics = output.compute_convergence_metrics(naive, cm_maps, hits)
    output.plot_diagnostics(metrics, pass_times, os.path.join(out_dir, "diagnostics.png"))
    print("    Saved diagnostics.png")

    output.plot_psd(raw_sample["raw"], cm_sample["cm"], sr, os.path.join(out_dir, "psd.png"))
    print("    Saved psd.png")

    output.plot_tod_rms(tod_rms_data, os.path.join(out_dir, "tod_rms.png"))
    print("    Saved tod_rms.png")

    print("  Convergence summary:")
    for label, peak, rms, diff in zip(
        metrics["labels"], metrics["peak"],
        metrics["off_src_rms"], metrics["map_diff_rms"]
    ):
        diff_str = f"  map_diff={diff:.4f}" if not np.isnan(diff) else ""
        print(f"    {label:12s}  peak={peak:.4f}  off_src_rms={rms:.4f}{diff_str}")

    elapsed  = time.perf_counter() - t_total
    metadata = {
        "run_timestamp"              : timestamp,
        "data_format"                : cfg["data"]["format"],
        "obs_t_start_g3s"            : obs_info["t_start_g3s"],
        "obs_t_stop_g3s"             : obs_info["t_stop_g3s"],
        "obs_t_start_utc"            : obs_utc.isoformat(),
        "obs_duration_s"             : obs_info["duration_s"],
        "n_detectors"                : obs_info["n_detectors"],
        "sample_rate_hz"             : obs_info["sample_rate_hz"],
        "n_chunks"                   : obs_info["n_chunks"],
        "map_ny"                     : ny,
        "map_nx"                     : nx,
        "map_ra0_deg"                : ra0_deg,
        "map_dec0_deg"               : dec0_deg,
        "map_res_arcmin"             : map_cfg["res_arcmin"],
        "map_xlen_deg"               : map_cfg["xlen_deg"],
        "map_ylen_deg"               : map_cfg["ylen_deg"],
        "pipeline_n_iterations"      : pipe_cfg["n_iterations"],
        "pipeline_start_offset_s"    : pipe_cfg.get("start_offset_s", 0.0),
        "pipeline_max_duration_s"    : pipe_cfg.get("max_duration_s", None),
        "pipeline_chunk_duration_s"  : pipe_cfg.get("chunk_duration_s", 1.0),
        "pipeline_clean_steps"       : pipe_cfg.get("clean_steps", []),
        "pipeline_cosmic_rays"       : pipe_cfg.get("cosmic_rays", {}),
        "pipeline_highpass"          : pipe_cfg.get("highpass", {}),
        "pipeline_notch"             : pipe_cfg.get("notch", {}),
        "map_weighting"              : ("inverse_variance_white_noise_floor"
                                        if weight_cap_factor > 0 else "uniform"),
        "weight_cap_factor"          : weight_cap_factor,
        "psd_anomalies"              : psd_anomalies,
        "n_detectors_used"           : len(keep_idx),
        "n_detectors_excluded"       : len(exclude_set),
        "auto_excluded_detectors"    : auto_excluded,
        "manual_excluded_detectors"  : manual_names + [all_kids[i] for i in manual_indices if 0 <= i < n_total],
        "total_runtime_s"            : round(elapsed, 2),
        "convergence"                : {
            "labels"      : metrics["labels"],
            "peak"        : metrics["peak"],
            "off_src_rms" : metrics["off_src_rms"],
            "map_diff_rms": metrics["map_diff_rms"],
            "pass_times_s": [t for _, t in pass_times],
        },
    }
    if cfg["data"]["format"] == "blasttng":
        blasttng_cfg = cfg.get("blasttng", {})
        metadata["df_method"]      = blasttng_cfg.get("df_method", "hybrid")
        metadata["threshold_frac"] = blasttng_cfg.get("threshold_frac", 0.05)
    output.save_metadata(metadata, out_dir)

    # ------------------------------------------------------------------ #
    # STEP 5: Boresight comparison map (optional)
    # ------------------------------------------------------------------ #
    if cfg["map"].get("compare_boresight", False):
        print("\nStep 5: Boresight-only comparison pass...")
        t = time.perf_counter()
        bore_map, _, _, _, _, _, _, _, _ = _streaming_pass(
            cfg, pipe_cfg, ra_edges, dec_edges, det_offsets_kept,
            common_mode=False, boresight_only=True, keep_idx=keep_idx,
            weights=det_weights_kept,
            compute_time_null=pipe_cfg.get("compute_time_null", True))
        print(f"  [{time.perf_counter() - t:.1f}s]")
        output.plot_boresight_comparison(
            combined_map, bore_map, ra_edges, dec_edges,
            out_dir / "boresight_comparison.png")
        print("  Saved boresight_comparison.png")

    # ------------------------------------------------------------------ #
    # STEP 6: Per-detector maps (optional)
    # ------------------------------------------------------------------ #
    if pd_cfg.get("enabled", False):
        print(f"\nStep 5: Per-detector maps...")
        if pd_results is not None:
            kids_sel, det_data, det_hits = pd_results  # already computed at Step 1b
        else:
            t = time.perf_counter()
            kids_sel, det_data, det_hits = _per_detector_pass(
                cfg, pipe_cfg, pd_cfg, ra_edges, dec_edges, det_offsets,
            )
            print(f"  Pass complete [{time.perf_counter() - t:.1f}s]")
        flagged = output.save_per_detector_maps(
            kids_sel, det_data, det_hits,
            ra_edges, dec_edges,
            os.path.join(out_dir, "per_detector"),
            pd_cfg,
            psd_avg=psd_avg, psd_freqs=psd_freqs, psd_all_kids=all_kids,
        )
        metadata["detectors_to_check"] = list(flagged.keys())
        metadata["n_detectors_flagged"] = len(flagged)
        output.save_metadata(metadata, out_dir)
    else:
        print("\nStep 5: Per-detector maps disabled (set per_detector.enabled = true to enable).")

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


if __name__ == "__main__":
    main()
