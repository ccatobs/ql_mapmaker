# ============================================================================ #
# output.py
#
# Audrey Yang, audyang@student.ubc.ca
# Vlad Grecu, vlad.grecu07@gmail.com
# CCAT August 2026
#
# Save map outputs to disk.
# ============================================================================ #

import json
import math
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FONT_TITLE  = 13
FONT_LABEL  = 11
FONT_TICK   = 9
CMAP_SIGNAL = "plasma"
CMAP_HITS   = "viridis"
CMAP_NULL   = "RdBu_r"
DPI         = 200


def save_per_detector_maps(kids: list, det_data: np.ndarray, det_hits: np.ndarray,
                           ra_edges: np.ndarray, dec_edges: np.ndarray,
                           out_dir: pathlib.Path, pd_cfg: dict,
                           psd_avg: np.ndarray = None, psd_freqs: np.ndarray = None,
                           psd_all_kids: list = None):
    """
    Save per-detector signal maps and centroids for pointing model reconstruction.

    For each detector:
      - Computes the signal map (data / hits)
      - Finds the peak pixel and records its RA/Dec as the centroid
      - Optionally saves a PNG and/or .npy file
      - If psd_avg/psd_freqs/psd_all_kids are given (from run_mapmaker._first_pass),
        also saves a per-detector PSD plot looked up by name via psd_all_kids
        since that list's ordering isn't guaranteed to match `kids` (which may be
        a filtered/reordered subset from _per_detector_pass).

    Centroids for all detectors are saved to centroids.json regardless of
    save_png/save_numpy settings.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_png     = pd_cfg.get("save_png",   True)
    save_numpy   = pd_cfg.get("save_numpy", False)
    save_psd_png = pd_cfg.get("save_psd_png", True)
    psd_kid_to_i = {k: i for i, k in enumerate(psd_all_kids)} if psd_all_kids is not None else {}

    ra_centres  = 0.5 * (ra_edges[:-1]  + ra_edges[1:])
    dec_centres = 0.5 * (dec_edges[:-1] + dec_edges[1:])

    centroids = {}
    n_dets = len(kids)

    for j, name in enumerate(kids):
        data = det_data[j].astype(float)
        hits = det_hits[j].astype(float)

        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(hits > 0, data / hits, np.nan)
        if np.isnan(np.nanmean(m)) == True:
            continue
        peak_snr = float("nan")
        peak_ra  = float("nan")
        peak_dec = float("nan")
        noise    = float("nan")
        if np.any(np.isfinite(m)):
            flat_idx        = np.nanargmax(m)
            iy, ix          = np.unravel_index(flat_idx, m.shape)
            peak_ra         = float(ra_centres[ix])
            peak_dec        = float(dec_centres[iy])
            off_mask        = np.isfinite(m) & (m < np.nanpercentile(m[np.isfinite(m)], 50))
            noise           = float(np.sqrt(np.nanmean(m[off_mask] ** 2))) if off_mask.any() else float("nan")
            peak_snr        = float(np.nanmax(m)) / noise if noise > 0 else float("nan")

        centroids[name] = dict(peak_ra_deg=peak_ra, peak_dec_deg=peak_dec,
                               peak_snr=peak_snr, off_src_rms=noise)

        safe_name = name.replace("/", "_").replace(" ", "_")

        if save_numpy:
            np.save(out_dir / f"{safe_name}.npy", m)

        if save_png:
            vmin, vmax = _percentile_scale(m)
            _plot_map(m, ra_edges, dec_edges,
                      filepath=out_dir / f"{safe_name}.png",
                      title=f"Detector: {name}",
                      cmap=CMAP_SIGNAL, vmin=vmin, vmax=vmax,
                      cbar_label="Signal")

        if save_psd_png and psd_avg is not None and name in psd_kid_to_i:
            psd_idx = psd_kid_to_i[name]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.loglog(psd_freqs[1:], psd_avg[1:, psd_idx], color="#2980b9", lw=1.1)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("PSD  [signal$^2$ Hz$^{-1}$]")
            ax.set_title(f"Detector: {name}")
            ax.grid(alpha=0.2, which="both")
            plt.tight_layout()
            plt.savefig(out_dir / f"{safe_name}_psd.png", dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        if (j + 1) % 50 == 0 or (j + 1) == n_dets:
            print(f"    {j + 1}/{n_dets} detectors saved")

    with open(out_dir / "centroids.json", "w") as f:
        json.dump(centroids, f, indent=2)
    print(f"    Saved centroids.json ({n_dets} detectors)")

    # Flag noisy detectors
    threshold = pd_cfg.get("noise_threshold", 3.0)
    rms_vals  = [v["off_src_rms"] for v in centroids.values()
                 if np.isfinite(v["off_src_rms"])]
    flagged   = {}
    if rms_vals:
        median_rms = float(np.median(rms_vals))
        cutoff     = threshold * median_rms
        flagged    = {name: {"off_src_rms": v["off_src_rms"],
                             "ratio_to_median": v["off_src_rms"] / median_rms}
                     for name, v in centroids.items()
                     if np.isfinite(v["off_src_rms"]) and v["off_src_rms"] > cutoff}
        with open(out_dir / "flagged_detectors.json", "w") as f:
            json.dump({"median_rms": median_rms,
                       "threshold_factor": threshold,
                       "cutoff_rms": cutoff,
                       "n_flagged": len(flagged),
                       "detectors": flagged}, f, indent=2)
        print(f"    Noise median={median_rms:.4f}  cutoff={cutoff:.4f} "
              f"({threshold}x)  flagged={len(flagged)}/{n_dets}")
        print(f"    Saved flagged_detectors.json")

    return flagged


def save_iteration_maps(naive: np.ndarray,
                        cm_maps: list[tuple[str, np.ndarray]],
                        hits: np.ndarray,
                        noise: np.ndarray,
                        null: np.ndarray,
                        ra_edges: np.ndarray, dec_edges: np.ndarray,
                        output_dir: pathlib.Path,
                        out_cfg: dict,
                        raw_map: np.ndarray = None,
                        detsplit_null: np.ndarray = None):
    """
    Save naive map, CM iteration maps, hits, noise, and null to disk.

    noise: per-pixel RMS of the mean, from the actual sample scatter
           (E[x^2] - E[x]^2) / hits, see run_mapmaker._streaming_pass.
    time_null:  half-difference of two chunk-split independent maps: should
           show no residual structure if the combined map's features are
           real signal rather than noise (a jackknife/null test). None if
           run_mapmaker._streaming_pass(compute_time_null=False) -- skipped.
    raw_map: optional naive-style map (no common-mode) using manual+auto
             exclusion only, comparison reference against `naive`/cm_maps,
             which additionally exclude white-noise-floor outliers.
    detsplit_null: optional null map from a random 50/50 *detector*-identity
             split (vs. `time_null`'s chunk- time split) -- see
             run_mapmaker._streaming_pass(compute_detsplit_null=True).

    Writes .npy and .png for each map, plus an overview grid (overview.png).
    Grid layout: naive + all CM iterations in the top rows, hits + noise +
    null below.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_maps = [("naive", naive)] + cm_maps + [("hits", hits), ("noise", noise)]
    if null is not None:
        all_maps.append(("null", null))
    if raw_map is not None:
        all_maps.append(("raw", raw_map))
    if detsplit_null is not None:
        all_maps.append(("detsplit_null", detsplit_null))

    if out_cfg.get("save_numpy", True):
        for label, m in all_maps:
            np.save(output_dir / f"{label}.npy", m)
        np.save(output_dir / "ra_edges.npy",  ra_edges)
        np.save(output_dir / "dec_edges.npy", dec_edges)
        print(f"    Saved {len(all_maps)} .npy files")

    if out_cfg.get("save_png", True):
        signal_maps = [("naive", naive)] + cm_maps
        sig_vmin, sig_vmax   = _global_scale(signal_maps)
        hits_vmin, hits_vmax = _percentile_scale(hits)
        noise_vmin, noise_vmax = _percentile_scale(noise)

        for label, m in signal_maps:
            _plot_map(m, ra_edges, dec_edges,
                      filepath=output_dir / f"{label}.png",
                      title=_pretty_title(label),
                      cmap=CMAP_SIGNAL, vmin=sig_vmin, vmax=sig_vmax,
                      cbar_label="Signal")
        if raw_map is not None:
            _plot_map(raw_map, ra_edges, dec_edges,
                      filepath=output_dir / "raw.png",
                      title="Raw Map (manual+auto exclusion only)",
                      cmap=CMAP_SIGNAL, vmin=sig_vmin, vmax=sig_vmax,
                      cbar_label="Signal")
        if null is not None:
            null_vmin, null_vmax = _global_scale([("null", null)])
            _plot_map(null, ra_edges, dec_edges,
                      filepath=output_dir / "time_null.png",
                      title="Null Map (time chunk half-difference)",
                      cmap=CMAP_NULL, vmin=null_vmin, vmax=null_vmax,
                      cbar_label="Signal")
        if detsplit_null is not None:
            ds_vmin, ds_vmax = _global_scale([("detsplit_null", detsplit_null)])
            _plot_map(detsplit_null, ra_edges, dec_edges,
                      filepath=output_dir / "detsplit_null.png",
                      title="Null Map (random detector-split half-difference)",
                      cmap=CMAP_NULL, vmin=ds_vmin, vmax=ds_vmax,
                      cbar_label="Signal")
        _plot_map(hits, ra_edges, dec_edges,
                  filepath=output_dir / "hits.png",
                  title="Hit Map",
                  cmap=CMAP_HITS, vmin=hits_vmin, vmax=hits_vmax,
                  cbar_label="Samples per pixel")
        _plot_map(noise, ra_edges, dec_edges,
                  filepath=output_dir / "noise.png",
                  title="Noise Map",
                  cmap=CMAP_HITS, vmin=noise_vmin, vmax=noise_vmax,
                  cbar_label="RMS of pixel mean")

        print(f"    Saved {len(all_maps)} individual PNGs")
        print(f"      Signal scale : {sig_vmin:.3f} - {sig_vmax:.3f}")

        _plot_overview_grid(naive, cm_maps, hits, noise,
                            ra_edges, dec_edges,
                            filepath=output_dir / "overview.png",
                            sig_vmin=sig_vmin, sig_vmax=sig_vmax,
                            hits_vmin=hits_vmin, hits_vmax=hits_vmax,
                            noise_vmin=noise_vmin, noise_vmax=noise_vmax)
        print("    Saved overview.png")


def compute_convergence_metrics(naive: np.ndarray,
                                cm_maps: list[tuple[str, np.ndarray]],
                                hits: np.ndarray) -> dict:
    """
    Compute per-iteration diagnostics to track convergence.

    Off-source mask is defined once from the final map: valid pixels (hits > 0)
    where signal is below the 50th percentile. This works for both point sources
    and extended emission as long as less than half the map is source-dominated.

    Returns a dict with lists (one entry per map stage, starting from naive):
      labels       : stage name
      peak         : peak signal value
      off_src_rms  : RMS in the off-source region
      map_diff_rms : RMS difference from the previous stage (NaN for naive)
    """
    all_maps = [("naive", naive)] + cm_maps
    final_map = cm_maps[-1][1] if cm_maps else naive

    valid      = hits > 0
    threshold  = np.nanpercentile(final_map[valid], 50)
    off_source = valid & (final_map < threshold)
    on_source = valid & (final_map >= threshold)

    labels, peaks, rms_vals_off, rms_vals_on, diff_rms = [], [], [], [], []
    prev = None
    for label, m in all_maps:
        labels.append(label)
        peaks.append(float(np.nanmax(m)))
        rms_vals_off.append(float(np.sqrt(np.nanmean(m[off_source] ** 2))))
        rms_vals_on.append(float(np.sqrt(np.nanmean(m[on_source] ** 2))))
        if prev is None:
            diff_rms.append(float("nan"))
        else:
            diff = m - prev
            diff_rms.append(float(np.sqrt(np.nanmean(diff[valid] ** 2))))
        prev = m

    return dict(labels=labels, peak=peaks, off_src_rms=rms_vals_off, on_src_rms=rms_vals_on, map_diff_rms=diff_rms)


def plot_tod_rms(tod_rms_data: list, filepath: pathlib.Path):
    """
    Plot median detector RMS per chunk vs elapsed time through the observation.

    Shows whether noise is stationary. Each point is one chunk; two lines show
    the noise level before and after common-mode subtraction.
    tod_rms_data : list of (t_start, rms_raw, rms_cm) from _streaming_pass.
    """
    if not tod_rms_data:
        return

    t0      = tod_rms_data[0][0]
    t_vals  = [d[0] - t0 for d in tod_rms_data]
    rms_raw = [d[1] for d in tod_rms_data]
    rms_cm  = [d[2] for d in tod_rms_data]

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(t_vals, rms_raw, color="#e74c3c", linewidth=1.2, label="Before CM subtraction")
    ax.plot(t_vals, rms_cm,  color="#2980b9", linewidth=1.2, label="After CM subtraction")
    ax.set_xlabel("Time from obs start (s)", fontsize=FONT_LABEL)
    ax.set_ylabel("Median detector RMS", fontsize=FONT_LABEL)
    ax.set_title("TOD RMS vs Time", fontsize=FONT_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_TICK)
    ax.grid(alpha=0.3)
    fig.suptitle("CCAT Prime-Cam Quick-Look Diagnostic",
                 fontsize=12, fontweight="bold", color="#444444", y=1.02)
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def plot_psd(raw_tod: np.ndarray, cleaned_tod: np.ndarray,
             sample_rate: float, filepath: pathlib.Path,
             n_det_sample: int = 10):
    """
    PSD diagnostic comparing raw timestreams vs common-mode-cleaned timestreams.

    Computes Welch PSD for up to n_det_sample evenly-spaced detectors and plots
    median + 25–75th percentile band on a log-log scale.
    """
    from scipy.signal import welch

    n_dets  = raw_tod.shape[1]
    det_idx = np.linspace(0, n_dets - 1, min(n_det_sample, n_dets), dtype=int)
    nperseg = min(256, raw_tod.shape[0])

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for tod, label, color in [
        (raw_tod,     "Raw (before CM)",    "#e74c3c"),
        (cleaned_tod, "Cleaned (after CM)", "#2980b9"),
    ]:
        psds = []
        for i in det_idx:
            f, p = welch(tod[:, i], fs=sample_rate, nperseg=nperseg)
            psds.append(p)
        psds = np.array(psds)
        med  = np.median(psds, axis=0)
        lo   = np.percentile(psds, 25, axis=0)
        hi   = np.percentile(psds, 75, axis=0)
        ax.loglog(f[1:], med[1:], color=color, linewidth=1.5, label=label)
        ax.fill_between(f[1:], lo[1:], hi[1:], color=color, alpha=0.2)

    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("Power Spectral Density", fontsize=FONT_LABEL)
    ax.set_title("Timestream PSD: Before vs After Common-Mode Subtraction",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_TICK)
    ax.grid(True, which="both", alpha=0.2)
    fig.suptitle("CCAT Prime-Cam Quick-Look Diagnostic",
                 fontsize=12, fontweight="bold", color="#444444", y=1.02)

    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(metrics: dict, pass_times: list[tuple[str, float]],
                     filepath: pathlib.Path):
    """
    Four-panel diagnostics figure: peak signal, off-source RMS, convergence
    (map diff RMS), and runtime per pass.
    """
    labels = metrics["labels"]
    x      = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("CCAT Prime-Cam Iteration Diagnostics",
                 fontsize=14, fontweight="bold")

    def _line(ax, y, title, ylabel, color):
        finite = [(i, v) for i, v in enumerate(y) if not (isinstance(v, float) and np.isnan(v))]
        xi, yi = zip(*finite)
        ax.plot(xi, yi, "o-", color=color, linewidth=2, markersize=7)
        for i, v in finite:
            ax.annotate(f"{v:.3g}", (i, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=FONT_TICK)
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_title(l) for l in labels],
                           rotation=20, ha="right", fontsize=FONT_TICK)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(axis="y", alpha=0.3)

    _line(axes[0, 0], metrics["peak"],        "Peak Signal",        "Signal",       "#c0392b")
    _line(axes[0, 1], metrics["off_src_rms"], "Off-Source RMS",     "RMS",          "#2980b9")
    _line(axes[1, 0], metrics["map_diff_rms"],"Convergence (Map Diff RMS)", "RMS",  "#27ae60")
    _line(axes[1, 1], metrics["on_src_rms"],  "On-Source RMS",      "RMS",          "#e67e22")

    # Runtime bar chart
    # ax = axes[1, 1]
    # pt_labels = [p[0] for p in pass_times]
    # pt_vals   = [p[1] for p in pass_times]
    # bars = ax.bar(pt_labels, pt_vals, color="#8e44ad", alpha=0.85)
    # for bar, val in zip(bars, pt_vals):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
    #             f"{val:.1f}s", ha="center", va="bottom", fontsize=FONT_TICK)
    # ax.set_title("Runtime per Pass", fontsize=FONT_TITLE, fontweight="bold")
   # ax.set_ylabel("Time (s)", fontsize=FONT_LABEL)
    # ax.tick_params(axis="x", rotation=20, labelsize=FONT_TICK)
   # ax.tick_params(axis="y", labelsize=FONT_TICK)
   # ax.grid(axis="y", alpha=0.3)

    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_wnf_diagnostic(white_noise_floor: np.ndarray, cutoff: float, sigma: float,
                        filepath: pathlib.Path):
    """
    Two-panel white-noise-floor distribution diagnostic: the raw distribution
    (log-x, heavy-tailed) with the applied cutoff marked, and log10(white_noise_floor)
    on a linear axis (roughly bell-shaped) with the median and sigma lines
    """
    wnf = white_noise_floor[np.isfinite(white_noise_floor) & (white_noise_floor > 0)]
    if len(wnf) == 0:
        return
    median_wnf = np.median(wnf)
    log_wnf    = np.log10(wnf)
    med_log    = np.median(log_wnf)
    std_log    = np.std(log_wnf)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bins = np.logspace(np.log10(wnf.min()), np.log10(wnf.max()), 60)
    axes[0].hist(wnf, bins=bins, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].axvline(median_wnf, color="k", lw=1.5, label=f"median ({median_wnf:.2e})")
    if cutoff is not None:
        n_excl = int((wnf > cutoff).sum())
        axes[0].axvline(cutoff, color="#e74c3c", ls="--", lw=2,
                        label=f"cutoff ({cutoff:.2e}) -- excludes {n_excl}/{len(wnf)}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("white_noise_floor")
    axes[0].set_ylabel("# detectors")
    axes[0].set_title("Raw distribution (heavy-tailed)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2, which="both")

    axes[1].hist(log_wnf, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
    axes[1].axvline(med_log, color="k", lw=1.5, label=f"median = {med_log:.2f}")
    if cutoff is not None:
        axes[1].axvline(med_log + sigma * std_log, color="#e74c3c", ls="--", lw=2,
                        label=f"median + {sigma}sigma = {med_log + sigma*std_log:.2f}")
    axes[1].set_xlabel("log10(white_noise_floor)")
    axes[1].set_ylabel("# detectors")
    axes[1].set_title("log10(white_noise_floor)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_boresight_comparison(with_offsets: np.ndarray, boresight_only: np.ndarray,
                              ra_edges: np.ndarray, dec_edges: np.ndarray,
                              filepath: pathlib.Path):
    """
    Side-by-side comparison of the final map (with focal plane offsets applied)
    vs a naive map made using boresight-only pointing (no offsets).

    Both panels share the same colour scale so the smearing is visually obvious.
    """
    all_vals = np.concatenate([with_offsets[np.isfinite(with_offsets)].ravel(),
                               boresight_only[np.isfinite(boresight_only)].ravel()])
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99.5)
    if vmax == vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    fig.suptitle("CCAT Prime-Cam: Effect of Focal Plane Offsets",
                 fontsize=14, fontweight="bold")

    _draw_panel(axes[0], with_offsets,  "With Focal Plane Offsets",
                ra_edges, dec_edges, CMAP_SIGNAL, vmin, vmax, "Signal")
    _draw_panel(axes[1], boresight_only, "Boresight-Only (no offsets)",
                ra_edges, dec_edges, CMAP_SIGNAL, vmin, vmax, "Signal")

    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def save_metadata(metadata: dict, output_dir: pathlib.Path):
    """Write run metadata to metadata.json."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("    Saved metadata.json")


def _pretty_title(label: str) -> str:
    if label == "naive":
        return "Naive Map"
    if label == "it_0":
        return "Common-Mode Pass (Initial)"
    if label.startswith("it_"):
        n = label.split("_")[1]
        return f"Common-Mode Pass (Iteration {n})"
    return label.replace("_", " ").title()


def _percentile_scale(m: np.ndarray):
    vals = m[np.isfinite(m)]
    vmin = np.percentile(vals, 1)
    vmax = np.percentile(vals, 99.5)
    if vmax == vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _global_scale(maps: list[tuple[str, np.ndarray]]):
    all_vals = np.concatenate([m[np.isfinite(m)].ravel() for _, m in maps])
    half = max(abs(np.percentile(all_vals, 1)), abs(np.percentile(all_vals, 99.5)))
    if half == 0:
        half = 1.0
    return -half, half


def _draw_panel(ax, m, title, ra_edges, dec_edges, cmap, vmin, vmax, cbar_label):
    extent = [ra_edges[0], ra_edges[-1], dec_edges[0], dec_edges[-1]]
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#e0e0e0")

    im = ax.imshow(m, origin="lower", extent=extent, aspect="equal",
                   cmap=cmap_obj, vmin=vmin, vmax=vmax)

    valid = np.isfinite(m)
    if valid.any():
        rows = np.where(valid.any(axis=1))[0]
        cols = np.where(valid.any(axis=0))[0]
        ra_lo  = ra_edges[cols[0]]
        ra_hi  = ra_edges[min(cols[-1] + 1, len(ra_edges) - 1)]
        dec_lo = dec_edges[rows[0]]
        dec_hi = dec_edges[min(rows[-1] + 1, len(dec_edges) - 1)]
        pad_ra  = (ra_hi  - ra_lo)  * 0.05
        pad_dec = (dec_hi - dec_lo) * 0.05
        ax.set_xlim(ra_lo - pad_ra,  ra_hi  + pad_ra)
        ax.set_ylim(dec_lo - pad_dec, dec_hi + pad_dec)

    ax.invert_xaxis()
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=6)
    ax.set_xlabel("RA (deg)", fontsize=FONT_LABEL)
    ax.set_ylabel("Dec (deg)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(cbar_label, fontsize=FONT_LABEL)
    cb.ax.tick_params(labelsize=FONT_TICK)
    return im


def _plot_overview_grid(naive: np.ndarray,
                        cm_maps: list[tuple[str, np.ndarray]],
                        hits: np.ndarray, noise: np.ndarray,
                        ra_edges: np.ndarray, dec_edges: np.ndarray,
                        filepath: pathlib.Path,
                        sig_vmin: float, sig_vmax: float,
                        hits_vmin: float, hits_vmax: float,
                        noise_vmin: float, noise_vmax: float):
    signal_maps = [("naive", naive)] + cm_maps
    n_sig      = len(signal_maps)
    ncols      = 4
    n_sig_rows = math.ceil(n_sig / ncols)
    nrows      = n_sig_rows + 1  # signal rows + hits/noise row

    fig = plt.figure(figsize=(7 * ncols, 7 * nrows), constrained_layout=True)
    fig.suptitle("CCAT Prime-Cam Quick-Look Map", fontsize=15, fontweight="bold", y=1.01)
    gs = fig.add_gridspec(nrows, ncols)

    for idx, (label, m) in enumerate(signal_maps):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        _draw_panel(ax, m, _pretty_title(label),
                    ra_edges, dec_edges, CMAP_SIGNAL,
                    sig_vmin, sig_vmax, "Signal")

    ax_hits  = fig.add_subplot(gs[n_sig_rows, 0:1])
    ax_noise = fig.add_subplot(gs[n_sig_rows, 1:2])
    _draw_panel(ax_hits,  hits,  "Hit Map",   ra_edges, dec_edges,
                CMAP_HITS, hits_vmin,  hits_vmax,  "Samples per pixel")
    _draw_panel(ax_noise, noise, "Noise Map", ra_edges, dec_edges,
                CMAP_HITS, noise_vmin, noise_vmax, "RMS of pixel mean")

    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_map(m: np.ndarray,
              ra_edges: np.ndarray, dec_edges: np.ndarray,
              filepath: pathlib.Path,
              title: str = "Quick-Look Map",
              cmap: str = CMAP_SIGNAL,
              vmin: float = None, vmax: float = None,
              cbar_label: str = "Signal"):
    if vmin is None or vmax is None:
        vmin, vmax = _percentile_scale(m)

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    _draw_panel(ax, m, title, ra_edges, dec_edges, cmap, vmin, vmax, cbar_label)
    fig.suptitle("CCAT Prime-Cam Quick-Look Map", fontsize=12,
                 fontweight="bold", color="#444444", y=1.02)
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
