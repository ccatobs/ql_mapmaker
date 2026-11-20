"""Rebuild the figures used in CCAT_SWG_mapmaker.md from local sample data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[2]
JUPITER_CENTER = (141.9077477, 15.6022855)
INK = "#162333"
MUTED = "#657487"
ACCENT = "#e66b3d"


def _quoted(path: Path) -> str:
    return json.dumps(str(path.resolve()))


def _write_config(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    object_name: str,
    map_settings: str,
    pipeline_settings: str,
    save_psd: bool = False,
) -> None:
    path.write_text(
        "[data]\n"
        f"input_dirs = [{_quoted(input_dir)}]\n"
        'file_format = "h5"\n'
        'format = "simulation"\n'
        "[map]\n"
        f"{map_settings}"
        "[pipeline]\n"
        f"{pipeline_settings}"
        "[output]\n"
        f"output_dir = {_quoted(output_dir)}\n"
        f"obs_object = {json.dumps(object_name)}\n"
        "save_numpy = true\n"
        "save_png = false\n"
        f"save_psd = {'true' if save_psd else 'false'}\n",
        encoding="utf-8",
    )


def _run_mapmaker(config: Path, output_dir: Path) -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "ccat_mapmaker" / "run_mapmaker.py"),
         "--config", str(config)],
        cwd=ROOT,
        check=True,
    )
    run_dirs = list(output_dir.glob("*"))
    if len(run_dirs) != 1:
        raise RuntimeError(f"Expected one mapmaker output directory, found {run_dirs}")
    return run_dirs[0]


def _make_jupiter_view(sample: Path, input_dir: Path) -> None:
    view = input_dir / "jupiter_crossing.h5"
    with h5py.File(sample, "r") as original, h5py.File(view, "w") as subset:
        times = original["shared/times"][:]
        start = int(np.searchsorted(times, times[0] + 200))
        stop = int(np.searchsorted(times, times[0] + 220))
        if stop <= start:
            raise ValueError("The Jupiter sample does not contain the expected crossing window")
        original.copy("instrument/focalplane", subset.require_group("instrument"))
        subset["shared/times"] = times[start:stop]
        subset["shared/boresight_radec"] = original["shared/boresight_radec"][start:stop]
        for name in ("signal", "flags"):
            dataset = original[f"detdata/{name}"]
            layout = h5py.VirtualLayout(
                shape=(dataset.shape[0], stop - start), dtype=dataset.dtype
            )
            source = h5py.VirtualSource(
                str(sample.resolve()), dataset.name, shape=dataset.shape
            )
            layout[:, :] = source[:, start:stop]
            subset.create_virtual_dataset(f"detdata/{name}", layout)


def _map_panel(ax, data, ra_edges, dec_edges, title, *, cmap="RdBu_r", limit=None):
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError(f"{title} contains no finite pixels")
    if limit is None:
        limit = float(np.nanpercentile(np.abs(finite), 99.5))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    image = ax.imshow(
        data,
        origin="lower",
        extent=(ra_edges[0], ra_edges[-1], dec_edges[0], dec_edges[-1]),
        aspect="auto",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.invert_xaxis()
    ax.set_title(title, color=INK, fontsize=12, weight="bold")
    ax.set_xlabel("Right ascension (deg)")
    ax.set_ylabel("Declination (deg)")
    ax.tick_params(labelsize=8, colors=MUTED)
    return image


def _peak_center(data, ra_edges, dec_edges):
    y, x = np.unravel_index(np.nanargmax(data), data.shape)
    return (
        (ra_edges[x] + ra_edges[x + 1]) / 2,
        (dec_edges[y] + dec_edges[y + 1]) / 2,
    )


def _save_map_comparison(
    run_dir: Path, output: Path, *, labels, names, title, crop_half_width=None
):
    ra = np.load(run_dir / "ra_edges.npy")
    dec = np.load(run_dir / "dec_edges.npy")
    maps = [np.load(run_dir / f"{name}.npy") for name in names]
    all_finite = np.concatenate([m[np.isfinite(m)] for m in maps])
    limit = float(np.nanpercentile(np.abs(all_finite), 99.5))
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    center = (
        ((ra[0] + ra[-1]) / 2, (dec[0] + dec[-1]) / 2)
        if crop_half_width else None
    )
    for ax, data, label in zip(axes, maps, labels):
        image = _map_panel(ax, data, ra, dec, label, limit=limit)
        if center:
            ax.set_xlim(center[0] + crop_half_width, center[0] - crop_half_width)
            ax.set_ylim(center[1] - crop_half_width, center[1] + crop_half_width)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Simulation signal (uncalibrated)")
    fig.suptitle(title, color=INK, fontsize=16, weight="bold")
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_orion_coverage(run_dir: Path, output: Path):
    ra = np.load(run_dir / "ra_edges.npy")
    dec = np.load(run_dir / "dec_edges.npy")
    hits = np.load(run_dir / "hits.npy")
    positive = hits[np.isfinite(hits) & (hits > 0)]
    if positive.size == 0:
        raise ValueError("The Orion A run produced no covered pixels")
    upper = float(np.nanpercentile(positive, 99.5))
    lower = float(positive.min())
    if upper <= lower:
        upper = lower * 1.01
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_where(hits <= 0, hits),
        origin="lower",
        extent=(ra[0], ra[-1], dec[0], dec[-1]),
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=lower, vmax=upper),
        interpolation="nearest",
    )
    ax.invert_xaxis()
    ax.set_title("Orion A · relative map weight", color=INK, weight="bold")
    ax.set_xlabel("Right ascension (deg)")
    ax.set_ylabel("Declination (deg)")
    fig.colorbar(image, ax=ax, label="Sum of detector weights (log scale)")
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_null_comparison(run_dir: Path, output: Path):
    ra = np.load(run_dir / "ra_edges.npy")
    dec = np.load(run_dir / "dec_edges.npy")
    final_map = np.load(run_dir / "it_3.npy")
    null_map = np.load(run_dir / "detsplit_null.npy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        (final_map, null_map),
        ("Orion A · final map", "Detector-split half-difference"),
    ):
        image = _map_panel(ax, data, ra, dec, title)
        center = ((ra[0] + ra[-1]) / 2, (dec[0] + dec[-1]) / 2)
        ax.set_xlim(center[0] + 0.75, center[0] - 0.75)
        ax.set_ylim(center[1] - 0.75, center[1] + 0.75)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Simulation signal (uncalibrated)")
    fig.suptitle("A map alongside a diagnostic—not a stand-alone noise estimate",
                 color=INK, fontsize=15, weight="bold")
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_psd(run_dir: Path, output: Path):
    with np.load(run_dir / "psd.npz") as spectra:
        frequencies = spectra["frequency_hz"]
        before = spectra["before"]
        after = spectra["after"]
    fig, ax = plt.subplots(figsize=(8.5, 4.7), constrained_layout=True)
    for values, color, label in (
        (before, "#e74c3c", "Before common mode"),
        (after, "#2980b9", "After common mode"),
    ):
        low, median, high = np.percentile(values, [25, 50, 75], axis=1)
        valid = (frequencies > 0) & (median > 0)
        ax.loglog(frequencies[valid], median[valid], color=color, label=label)
        band = valid & (low > 0) & (high > 0)
        ax.fill_between(
            frequencies[band], low[band], high[band], color=color, alpha=0.18
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (simulation signal² / Hz)")
    ax.set_title("Matched detector spectra · final map pass", color=INK, weight="bold")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0e}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_science_entrypoints(jupiter_run: Path, orion_run: Path, output: Path):
    jupiter_ra = np.load(jupiter_run / "ra_edges.npy")
    jupiter_dec = np.load(jupiter_run / "dec_edges.npy")
    jupiter_map = np.load(jupiter_run / "it_1.npy")
    orion_ra = np.load(orion_run / "ra_edges.npy")
    orion_dec = np.load(orion_run / "dec_edges.npy")
    orion_map = np.load(orion_run / "it_3.npy")
    hits = np.load(orion_run / "hits.npy")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    center = _peak_center(jupiter_map, jupiter_ra, jupiter_dec)
    _map_panel(axes[0], jupiter_map, jupiter_ra, jupiter_dec, "Pointing / compact sources")
    axes[0].set_xlim(center[0] + 0.55, center[0] - 0.55)
    axes[0].set_ylim(center[1] - 0.55, center[1] + 0.55)
    center = (
        (orion_ra[0] + orion_ra[-1]) / 2,
        (orion_dec[0] + orion_dec[-1]) / 2,
    )
    _map_panel(axes[1], orion_map, orion_ra, orion_dec, "Extended emission")
    axes[1].set_xlim(center[0] + 0.75, center[0] - 0.75)
    axes[1].set_ylim(center[1] - 0.75, center[1] + 0.75)
    positive = hits[np.isfinite(hits) & (hits > 0)]
    coverage = axes[2].imshow(
        np.ma.masked_where(hits <= 0, hits),
        origin="lower",
        extent=(orion_ra[0], orion_ra[-1], orion_dec[0], orion_dec[-1]),
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=float(positive.min()),
                     vmax=float(np.nanpercentile(positive, 99.5))),
        interpolation="nearest",
    )
    axes[2].invert_xaxis()
    axes[2].set_title("Scan depth", color=INK, fontsize=12, weight="bold")
    axes[2].set_xlabel("Right ascension (deg)")
    axes[2].set_ylabel("Declination (deg)")
    axes[2].tick_params(labelsize=8, colors=MUTED)
    fig.colorbar(coverage, ax=axes[2], shrink=0.78, label="Relative map weight")
    fig.suptitle("One quick-look workflow · different entry points",
                 color=INK, fontsize=15, weight="bold")
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_pipeline(output: Path):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    fig.patch.set_facecolor("#f4f7fa")
    ax.set_facecolor("#f4f7fa")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    blocks = [
        (0.25, "Detector\nsamples", "signals · flags\npointing"),
        (2.65, "Prepare", "baselines · noise\nmap grid"),
        (5.05, "Make a map", "bin samples\ninto pixels"),
        (7.45, "Iterate", "estimate shared\ncommon mode"),
        (9.85, "Inspect", "maps · coverage\nnulls · PSD"),
    ]
    for x, title, subtitle in blocks:
        ax.text(x + 0.83, 1.95, title, ha="center", va="center",
                color="white", weight="bold", fontsize=13,
                bbox={"boxstyle": "round,pad=0.8", "facecolor": INK, "edgecolor": "none"})
        ax.text(x + 0.83, 0.8, subtitle, ha="center", va="center",
                color=MUTED, fontsize=10)
    for x in (2.15, 4.55, 6.95, 9.35):
        ax.annotate("", xy=(x + 0.35, 1.95), xytext=(x - 0.15, 1.95),
                    arrowprops={"arrowstyle": "->", "lw": 2, "color": ACCENT})
    ax.text(6, 0.15, "A quick-look reconstruction for checking the observation and guiding the next question",
            ha="center", color=INK, fontsize=11, weight="bold")
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def build_figures(sample_root: Path, output_dir: Path) -> None:
    jupiter = sample_root / "sim/planet_ATMdata_d3426/sim_PCAM280_h5_Jupiter_d3426/Jupiter-0-0.h5"
    orion_dir = sample_root / "sim/orionA_ATMdata_d100_primecam_f280/sim_PCAM280_h5_ORIONA_2026_07_15_d100"
    orion_files = sorted(orion_dir.glob("*.h5"))
    if not jupiter.is_file():
        raise FileNotFoundError(f"Jupiter sample not found: {jupiter}")
    if not orion_files:
        raise FileNotFoundError(f"No Orion A HDF5 samples found in {orion_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ccat-swg-") as temp:
        work = Path(temp)
        jupiter_input = work / "jupiter_input"
        jupiter_input.mkdir()
        _make_jupiter_view(jupiter, jupiter_input)
        jupiter_config = work / "jupiter.toml"
        _write_config(
            jupiter_config,
            jupiter_input,
            work / "jupiter_results",
            object_name="jupiter_swg_demo",
            map_settings=(
                f"ra0_deg = {JUPITER_CENTER[0]}\n"
                f"dec0_deg = {JUPITER_CENTER[1]}\n"
                "xlen_deg = 1.5\nylen_deg = 1.5\nres_arcmin = 0.5\n"
            ),
            pipeline_settings=(
                "chunk_duration_s = 10\nsample_step = 10\nn_iterations = 1\n"
                "auto_exclude_threshold = 0\nwnf_exclude_sigma = 0\n"
                "weight_cap_factor = 0\ncompute_time_null = false\n"
                "compute_detsplit_null = false\n"
            ),
        )
        jupiter_run = _run_mapmaker(jupiter_config, work / "jupiter_results")

        orion_config = work / "orion.toml"
        _write_config(
            orion_config,
            orion_dir,
            work / "orion_results",
            object_name="orion_a_swg_demo",
            map_settings="res_arcmin = 0.5\n",
            pipeline_settings=(
                "chunk_duration_s = 30\nsample_step = 2\nn_iterations = 3\n"
                "compute_time_null = true\ncompute_detsplit_null = true\n"
            ),
            save_psd=True,
        )
        orion_run = _run_mapmaker(orion_config, work / "orion_results")

        _save_map_comparison(
            jupiter_run,
            output_dir / "jupiter_maps.png",
            labels=("Naive map", "After common-mode iteration"),
            names=("naive", "it_1"),
            title="Jupiter · a compact-source / pointing check",
        )
        _save_map_comparison(
            orion_run,
            output_dir / "orion_a_maps.png",
            labels=("Naive map", "After 3 iterations"),
            names=("naive", "it_3"),
            title="Orion A · extended emission in a simulation",
            crop_half_width=0.75,
        )
        _save_orion_coverage(orion_run, output_dir / "orion_a_coverage.png")
        _save_null_comparison(orion_run, output_dir / "orion_a_null.png")
        _save_psd(orion_run, output_dir / "orion_a_psd.png")
        _save_science_entrypoints(
            jupiter_run, orion_run, output_dir / "science_entrypoints.png"
        )
        for name, run_dir in (("jupiter", jupiter_run), ("orion_a", orion_run)):
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            print(
                f"{name}: {metadata['n_detectors_used']}/{metadata['n_detectors']} "
                f"detectors, {metadata['total_runtime_s']:.1f} s, "
                f"{metadata['map_quality']['covered_pixels']:,} covered pixels"
            )

    _save_pipeline(output_dir / "pipeline.png")
    print(f"Presentation figures saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path.home() / "notebooks/data/ql_mm_samples",
        help="Root directory containing sim/planet_* and sim/orionA_* samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "presentation_assets",
        help="Directory for generated PNG figures",
    )
    args = parser.parse_args()
    build_figures(args.sample_root.expanduser(), args.output_dir.expanduser())


if __name__ == "__main__":
    main()
