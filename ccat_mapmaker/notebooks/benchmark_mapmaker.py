"""Benchmark mapmaker options on a fixed window of the Orion A simulation."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    Path.home() / "notebooks/data/ql_mm_samples/sim/"
    "orionA_ATMdata_d100_primecam_f280/sim_PCAM280_h5_ORIONA_2026_07_15_d100"
)
CASES = (
    ("baseline", "Minimum: 0 iterations", 0, False, False, False, False),
    ("iter1", "1 iteration", 1, False, False, False, False),
    ("iter2", "2 iterations", 2, False, False, False, False),
    ("iter3", "3 iterations", 3, False, False, False, False),
    ("iter5", "5 iterations", 5, False, False, False, False),
    ("time_null", "3 iterations + time null", 3, True, False, False, False),
    ("det_null", "3 iterations + detector null", 3, False, True, False, False),
    ("both_nulls", "3 iterations + both nulls", 3, True, True, False, False),
    ("cosmic", "3 iterations + cosmic rays", 3, False, False, True, False),
    ("plots", "3 iterations + PNG and PSD", 3, False, False, False, True),
)


def config_text(input_dir: Path, output_dir: Path, case: tuple, duration: float,
                chunk_duration: float) -> str:
    _, _, iterations, time_null, det_null, cosmic, plots = case
    boolean = lambda value: str(value).lower()
    text = (
        "[data]\n"
        f"input_dirs = [{json.dumps(str(input_dir.resolve()))}]\n"
        'file_format = "h5"\nformat = "simulation"\n'
        "[map]\nres_arcmin = 0.5\n"
        "[pipeline]\n"
        f"max_duration_s = {duration}\nchunk_duration_s = {chunk_duration}\n"
        f"sample_step = 1\nn_iterations = {iterations}\n"
        "auto_exclude_threshold = 0\nwnf_exclude_sigma = 0\n"
        "weight_cap_factor = 0\n"
        f"compute_time_null = {boolean(time_null)}\n"
        f"compute_detsplit_null = {boolean(det_null)}\n"
    )
    if cosmic:
        text += "[pipeline.cosmic_rays]\nsigma = 3.5\nn = 2\nmax_width = 5\n"
    return text + (
        "[output]\n"
        f"output_dir = {json.dumps(str(output_dir.resolve()))}\n"
        'obs_object = "benchmark"\nsave_numpy = true\n'
        f"save_png = {boolean(plots)}\nsave_psd = {boolean(plots)}\n"
        "[per_detector]\nenabled = false\n"
    )


def measure(metadata: dict, wall_s: float) -> dict:
    """Normalize by actually yielded samples, not the requested time limit."""
    samples = sum(metadata["preparation_processing"]["chunk_samples"])
    sample_rate = metadata["observation"]["sample_rate_hz"]
    detectors = metadata["n_detectors_used"]
    if (samples <= 0 or not math.isfinite(sample_rate) or sample_rate <= 0
            or detectors <= 0 or not math.isfinite(wall_s) or wall_s <= 0):
        raise ValueError("Benchmark requires positive duration, detector count and runtime")
    duration = samples / sample_rate
    return {
        "wall_s": wall_s,
        "duration_s": duration,
        "time_samples": samples,
        "detectors_used": detectors,
        "ms_per_detector_second": 1000 * wall_s / (detectors * duration),
        "realtime_factor": duration / wall_s,
        "metadata": metadata,
    }


def plot_results(results: dict, path: Path) -> None:
    records = results["runs"]
    workloads = {
        (r["time_samples"], r["metadata"]["observation"]["sample_rate_hz"],
         r["detectors_used"], r["metadata"]["map_nx"], r["metadata"]["map_ny"],
         tuple(r["metadata"]["input_files_opened"]))
        for r in records
    }
    if len(workloads) != 1:
        raise ValueError("Runs processed different workloads; inspect results before comparison")
    fig, (ax, stages_ax) = plt.subplots(
        1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.35, 1]},
        layout="constrained",
    )
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    labels = [case[1] for case in CASES]
    ys = list(range(len(CASES)))
    stage_names = ("preparation", "initial_maps", "iterations", "save_outputs")
    stage_labels = ("Preparation", "Initial maps", "Iterations", "Save outputs")
    colors = ("#4c78a8", "#72b7b2", "#f58518", "#b279a2")

    for y, case in enumerate(CASES):
        runs = [r for r in records if r["case"] == case[0]]
        if not runs:
            raise ValueError(f"No measurements for {case[0]}")
        values = [r["ms_per_detector_second"] for r in runs]
        med = statistics.median(values)
        ax.barh(y, med, color="#4c78a8" if y < 5 else "#e89a52", height=0.65)
        ax.errorbar(med, y, xerr=[[med - min(values)], [max(values) - med]],
                    fmt="none", ecolor="#263238", capsize=3)
        wall = statistics.median(r["wall_s"] for r in runs)
        realtime = statistics.median(r["realtime_factor"] for r in runs)
        ax.annotate(f"  {wall:.1f} s  |  {realtime:.2f}× real time",
                    (max(values), y), va="center", fontsize=8)
        left = 0
        for name, label, color in zip(stage_names, stage_labels, colors):
            value = statistics.median(
                r["metadata"]["stage_times_s"][name] for r in runs)
            stages_ax.barh(y, value, left=left, color=color, height=0.65,
                           label=label if y == 0 else None)
            left += value

    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    stages_ax.set_yticks(ys, [])
    stages_ax.invert_yaxis()
    ax.set_xlabel("Wall time / detector / observation second (ms)")
    stages_ax.set_xlabel("Median in-process stage time (s)")
    ax.set_title("End-to-end cost", loc="left", weight="bold")
    stages_ax.set_title("Where time goes", loc="left", weight="bold")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.65)
    handles, legend_labels = stages_ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.06),
               ncol=4, frameon=False, fontsize=8)
    for axis in (ax, stages_ax):
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    first = records[0]
    fig.suptitle("Quick-look mapmaker: local Orion A benchmark", weight="bold", fontsize=15)
    fig.text(
        0.01, 0.005,
        f"{results['machine']['processor']} / {results['machine']['platform']}  |  "
        f"{first['detectors_used']} detectors  |  "
        f"{first['duration_s']:.1f} s of data at full sample rate  |  "
        f"{results['repeats']} runs/case; bars = median, whiskers = min–max\n"
        "Wall time includes process startup and output; stages exclude startup and other overhead. "
        "Warm-cache simulation; variants change map products/quality.",
        fontsize=8, color="#444444",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Directory containing the Orion A simulation HDF5 files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ccat_mapmaker/notebooks/benchmark_results",
                        help="Parent directory for a new timestamped result directory")
    parser.add_argument("--duration", type=float, default=30,
                        help="Requested observation seconds (rounded up to chunk boundaries)")
    parser.add_argument("--chunk-duration", type=float, default=10,
                        help="Fixed chunk duration in seconds")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1 or args.duration <= 0 or args.chunk_duration <= 0:
        parser.error("repeats, duration and chunk duration must be positive")
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir() or not list(input_dir.rglob("*.h5")):
        parser.error(f"No HDF5 files found in {input_dir}")

    result_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir.mkdir(parents=True)
    processor = platform.processor()
    if sys.platform == "darwin":
        processor = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    results = {
        "description": "Full-rate Orion A simulation, fixed chunking; warm-up then randomized repeat blocks",
        "input_dir": str(input_dir),
        "requested_duration_s": args.duration,
        "chunk_duration_s": args.chunk_duration,
        "repeats": args.repeats,
        "seed": args.seed,
        "machine": {
            "platform": platform.platform(),
            "processor": processor,
            "logical_cpus": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "runs": [],
    }
    results_path = result_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    rng = random.Random(args.seed)

    with tempfile.TemporaryDirectory(prefix="ccat-benchmark-") as temp:
        work = Path(temp)

        def run(case: tuple, repeat: int, warmup: bool = False) -> None:
            name = "warmup" if warmup else f"{case[0]}_{repeat}"
            output = work / name
            config = result_dir / f"{name}.toml"
            config.write_text(config_text(input_dir, output, case, args.duration,
                                          args.chunk_duration))
            log = result_dir / f"{name}.log"
            print(f"Running {name}...", flush=True)
            with log.open("w") as stream:
                start = time.perf_counter()
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "ccat_mapmaker/run_mapmaker.py"),
                     "--config", str(config)],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    check=False,
                )
                wall_s = time.perf_counter() - start
            if completed.returncode:
                raise RuntimeError(f"{name} failed (exit {completed.returncode}); see {log}")
            run_dirs = [p for p in output.iterdir() if p.is_dir()]
            if len(run_dirs) != 1:
                raise RuntimeError(f"Expected one output directory for {name}: {run_dirs}")
            metadata = json.loads((run_dirs[0] / "metadata.json").read_text())
            if metadata["status"] != "completed":
                raise RuntimeError(f"{name} did not complete successfully; see {log}")
            if not warmup:
                measurement = measure(metadata, wall_s)
                measurement.update(case=case[0], repeat=repeat)
                results["runs"].append(measurement)
                results_path.write_text(json.dumps(results, indent=2) + "\n")
            print(f"  {wall_s:.2f} s", flush=True)

        run(CASES[0], 0, warmup=True)
        for repeat in range(1, args.repeats + 1):
            cases = list(CASES)
            rng.shuffle(cases)
            for case in cases:
                run(case, repeat)

    plot_results(results, result_dir / "benchmark.png")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
