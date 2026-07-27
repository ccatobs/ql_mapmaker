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

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from mapmaker.reader      import iter_chunks
from mapmaker.cleaning    import clean_tod
from mapmaker.binning     import make_map_edges, bin_chunk, bin_detector
from mapmaker.common_mode import estimate_common_mode, subtract_common_mode, iterate_common_mode
from mapmaker               import output


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
    Single streaming pass to compute per-detector baselines, noise, and mean boresight.

    Uses the median of each chunk's signal (averaged across chunks) for the
    baseline -- robust to cosmic ray tails. Computes per-detector noise as the
    true global std of first-differences by accumulating sum and sum-of-squares
    across all chunks before computing the final statistic.
    """
    ra_sum = dec_sum = n_bore = 0
    det_median_sum = det_diff_sum = det_diff_sum_sq = None
    det_diff_count = 0
    n_chunks = 0
    t_obs_start = t_obs_stop = None
    n_dets = sample_rate = kids = None

    for chunk in iter_chunks(cfg):
        flags = chunk.flags
        chunk_id = chunk.chunk_index
        if np.mean(flags) == 1:
            print(f"No good data in chunk with index {chunk_id}. Skipping to next chunk")
            continue
        flag_mask = np.ones(np.shape(flags))
        flag_mask[flags == 1] = np.nan
        if t_obs_start is None:
            t_obs_start = chunk.t_start
            n_dets      = chunk.signal.shape[1]
            sample_rate = chunk.sample_rate
            kids        = chunk.kids

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
    return ra_sum / n_bore, dec_sum / n_bore, det_median_sum / n_chunks, det_noise, kids, obs_info


def _streaming_pass(cfg: dict, pipe_cfg: dict,
                    ra_edges: np.ndarray, dec_edges: np.ndarray,
                    det_offsets: np.ndarray,
                    current_map: np.ndarray = None,
                    common_mode: bool = True,
                    return_sample: bool = False,
                    keep_idx: np.ndarray = None,
                    boresight_only: bool = False,
                    collect_tod_rms: bool = False):
    """
    One streaming pass over all chunks: baseline subtract, clean, optionally
    common-mode subtract, then bin into the map accumulator.

    current_map: if provided, uses sky-informed common-mode (iterate_common_mode);
                 otherwise uses naive mean across detectors.
    common_mode: if False, skips common-mode subtraction entirely (naive map).
    return_sample: if True, captures the first chunk's signal before and after
                   CM subtraction for PSD diagnostics.
    collect_tod_rms: if True, records median detector RMS per chunk before and
                     after CM subtraction as a list of (t_start, rms_raw, rms_cm).
    """
    ny = len(dec_edges) - 1
    nx = len(ra_edges)  - 1
    total_data = np.zeros((ny, nx), dtype=float)
    total_hits = np.zeros((ny, nx), dtype=float)

    n_dets = sample_rate = None
    psd_raw = psd_cm = None
    tod_rms_data: list = []

    for chunk in iter_chunks(cfg):
        flags = chunk.flags
        chunk_id = chunk.chunk_index
        if np.mean(flags) == 1:
            # print(f"No good data in chunk with index {chunk_id}. Skipping to next chunk")
            continue
        if n_dets is None:
            n_dets      = len(chunk.kids)
            sample_rate = chunk.sample_rate

        ra  = chunk.ra
        dec = chunk.dec
        flags = chunk.flags
        flag_mask = np.ones(np.shape(flags))
        flag_mask[flags == 1] = np.nan
        
        if keep_idx is not None:
            sig = chunk.signal[:, keep_idx] - det_offsets[np.newaxis, :]
            ra  = ra[:,  keep_idx]
            dec = dec[:, keep_idx]
        else:
            sig = chunk.signal - det_offsets[np.newaxis, :]

        if boresight_only:
            ra  = np.repeat(chunk.ra_bore[:, np.newaxis], sig.shape[1], axis=1)
            dec = np.repeat(chunk.dec_bore[:, np.newaxis], sig.shape[1], axis=1)

        sig = clean_tod(sig, flag_mask, chunk.sample_rate,
                        cosmic_rays=pipe_cfg["clean_cosmic_rays"],
                        highpass_hz=pipe_cfg["highpass_cutoff_hz"])

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
            
        d, h = bin_chunk(sig, flag_mask, ra, dec, ra_edges, dec_edges)
        # print(f"d: {d}")
        total_data += d
        total_hits += h

    with np.errstate(invalid='ignore', divide='ignore'):
        combined = np.where(total_hits > 0, total_data / total_hits, np.nan)

    sample = {"raw": psd_raw, "cm": psd_cm} if return_sample else None
    return combined, total_hits, n_dets, sample_rate, sample, tod_rms_data


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
        flag_mask[flags == 1] = np.nan
        sig = chunk.signal - det_offsets[np.newaxis, :]
        
        sig = clean_tod(sig, flag_mask, chunk.sample_rate,
                        cosmic_rays=pipe_cfg["clean_cosmic_rays"],
                        highpass_hz=pipe_cfg["highpass_cutoff_hz"])

        for j, i in enumerate(sel_idx):
            if apply_offsets:
                ra, dec = chunk.ra[:, i], chunk.dec[:, i]
            else:
                ra, dec = chunk.ra_bore, chunk.dec_bore
            d, h = bin_detector(sig[:, i], flag_mask[:, i], ra, dec, ra_edges, dec_edges)
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
    ra0_auto, dec0_auto, det_offsets, det_noise, all_kids, obs_info = _first_pass(cfg)
    print(f"  Baselines: {det_offsets.min():.4f} - {det_offsets.max():.4f}  [{time.perf_counter()-t:.1f}s]")

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

    keep_idx = np.array([i for i in range(n_total) if i not in exclude_set], dtype=int)
    det_offsets_kept = det_offsets[keep_idx]
    print(f"  Using {len(keep_idx)}/{n_total} detectors")

    _G3_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
    obs_utc   = _G3_EPOCH + timedelta(seconds=obs_info["t_start_g3s"])
    print(f"  Obs start  : {obs_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Duration   : {obs_info['duration_s']:.1f} s")
    print(f"  Detectors  : {obs_info['n_detectors']} @ {obs_info['sample_rate_hz']:.1f} Hz")

    ra0_deg  = map_cfg["ra0_deg"]  if centre_pinned else ra0_auto
    dec0_deg = map_cfg["dec0_deg"] if centre_pinned else dec0_auto
    centre_source = "config" if centre_pinned else "auto (mean boresight)"

    ra_edges, dec_edges = make_map_edges(
        ra0_deg=ra0_deg, dec0_deg=dec0_deg,
        xlen_deg=map_cfg["xlen_deg"], ylen_deg=map_cfg["ylen_deg"],
        res_deg=res_deg,
    )
    ny, nx = len(dec_edges) - 1, len(ra_edges) - 1
    print(f"  Map grid   : {ny} x {nx} pixels ({map_cfg['res_arcmin']:.1f} arcmin/pixel), "
          f"centre = ({ra0_deg:.4f}, {dec0_deg:.4f}) [{centre_source}]")

    # ------------------------------------------------------------------ #
    # STEP 2: Naive map + initial common-mode pass
    # ------------------------------------------------------------------ #
    chunk_s     = pipe_cfg.get("chunk_duration_s", 1.0)
    t = time.perf_counter()
    print(f"Step 2: Naive map (chunk_duration_s={chunk_s})...")
    naive, _, n_dets, sr, raw_sample, _ = _streaming_pass(
        cfg, pipe_cfg, ra_edges, dec_edges, det_offsets_kept,
        common_mode=False, return_sample=True, keep_idx=keep_idx)
    t_naive = time.perf_counter() - t
    print(f"  {n_dets} detectors, {sr:.1f} Hz  [{t_naive:.1f}s]")

    t = time.perf_counter()
    print("Step 2: Initial common-mode pass...")
    combined_map, hits, _, _, cm_sample, tod_rms_data = _streaming_pass(
        cfg, pipe_cfg, ra_edges, dec_edges, det_offsets_kept,
        return_sample=True, keep_idx=keep_idx, collect_tod_rms=True)
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
            combined_map, hits, _, _, _, _ = _streaming_pass(
                cfg, pipe_cfg, ra_edges, dec_edges, det_offsets_kept,
                current_map=combined_map, keep_idx=keep_idx,
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
    output.save_iteration_maps(naive, cm_maps, hits,
                               ra_edges, dec_edges, out_dir, cfg["output"])

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
        "pipeline_clean_cosmic_rays" : pipe_cfg["clean_cosmic_rays"],
        "pipeline_highpass_cutoff_hz": pipe_cfg.get("highpass_cutoff_hz", 0.0),
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
    output.save_metadata(metadata, out_dir)

    # ------------------------------------------------------------------ #
    # STEP 5: Boresight comparison map (optional)
    # ------------------------------------------------------------------ #
    if cfg["map"].get("compare_boresight", False):
        print("\nStep 5: Boresight-only comparison pass...")
        t = time.perf_counter()
        bore_map, _, _, _, _, _ = _streaming_pass(
            cfg, pipe_cfg, ra_edges, dec_edges, det_offsets_kept,
            common_mode=False, boresight_only=True, keep_idx=keep_idx)
        print(f"  [{time.perf_counter() - t:.1f}s]")
        output.plot_boresight_comparison(
            combined_map, bore_map, ra_edges, dec_edges,
            out_dir / "boresight_comparison.png")
        print("  Saved boresight_comparison.png")

    # ------------------------------------------------------------------ #
    # STEP 6: Per-detector maps (optional)
    # ------------------------------------------------------------------ #
    pd_cfg = cfg.get("per_detector", {})
    if pd_cfg.get("enabled", False):
        print(f"\nStep 5: Per-detector maps...")
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
