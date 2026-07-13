# CCAT Quick-Look Mapmaker

Standalone command-line mapmaker for CCAT Prime Cam. Reads `.g3` detector data files, runs iterative common-mode subtraction, and saves sky maps to disk.

## Prerequisites

- [Anaconda](https://www.anaconda.com/download) (or Miniconda)
- `.g3` simulation or observation data files

## Setup

### First time on a new machine

1. Clone this repo (or copy the `ccat_mapmaker/` directory).

2. Create the conda environment from the repo root:
   ```bash
   conda env create -f environment.yml
   ```
   This installs Python 3.11, numpy, scipy, astropy, matplotlib, spt3g, h5py, and jupyter.

3. Activate it:
   ```bash
   conda activate ccat
   ```

### Every time you work on this project

Activate the environment before doing anything — `spt3g` is not available outside it:

```bash
conda activate ccat
```

Your prompt will change from `(base)` to `(ccat)`. Deactivate when done:

```bash
conda deactivate
```

## Quick start

### 1. Point the config at your data

Open `ccat_mapmaker/config.toml` and set `input_dirs` to the directory containing your `.g3` files:

```toml
[data]
input_dirs = ["/path/to/your/g3/files"]
format = "simulation"   # or "blasttng" for BLAST-TNG files
```

`input_dirs` accepts glob patterns and is expanded recursively for `*.g3` files.

### 2. Set the map centre and size (optional)

If you know where your source is, pin the map centre. Otherwise omit these lines and the mapmaker will auto-detect the centre from the boresight:

```toml
[map]
ra0_deg  = 142.0
dec0_deg = 15.6
xlen_deg   = 1.0   # map width in degrees
ylen_deg   = 1.0   # map height in degrees
res_arcmin = 0.25  # pixel size in arcminutes
```

### 3. Run

From the `CCAT/` directory (one level above `ccat_mapmaker/`):

```bash
python ccat_mapmaker/run_mapmaker.py --config ccat_mapmaker/config.toml
```

Or from inside `ccat_mapmaker/`:

```bash
python run_mapmaker.py
```

The default config path is `config.toml` relative to the working directory.

### 4. Find your outputs

Each run creates a timestamped subdirectory under `ccat_mapmaker/output/`, e.g.:

```
ccat_mapmaker/output/2024-07-08_14-32-01/
```

Open `overview.png` first — it shows all pipeline maps side by side.

## What you'll see when it runs

```
============================================================
CCAT Quick-Look Mapmaker
============================================================
Config : ccat_mapmaker/config.toml
Input  : /path/to/data
Format : simulation
Output : ccat_mapmaker/output

Step 1: First pass (baselines + map centre)...
  Baselines: -0.0123 - 0.0456  [2.3s]
  Auto-exclusion: noise median=0.0021  cutoff=0.0105 (5.0x)  excluded=3/280
  Using 277/280 detectors
  Obs start  : 2024-01-15 10:32:00 UTC
  Duration   : 100.0 s
  Detectors  : 280 @ 200.0 Hz
  Map grid   : 240 x 240 pixels (0.25 arcmin/pixel), centre = (142.0000, 15.6000) [config]

Step 2: Naive map (chunk_duration_s=1.0)...
Step 2: Initial common-mode pass...
Step 3: 4 iteration(s)...
  Iteration 1/4... [1.8s]
  ...

Total time : 18.3s
Maps saved : ccat_mapmaker/output/2024-07-08_14-32-01/
============================================================
```

## Configuring a run

All settings live in `config.toml`. The most commonly changed ones:

| Setting | Section | What it controls |
|---|---|---|
| `input_dirs` | `[data]` | Paths to `.g3` files (glob patterns OK) |
| `format` | `[data]` | `"simulation"` or `"blasttng"` |
| `ra0_deg`, `dec0_deg` | `[map]` | Map centre (omit to auto-detect) |
| `res_arcmin` | `[map]` | Pixel size in arcminutes |
| `n_iterations` | `[pipeline]` | Common-mode subtraction iterations (3–5 is typical) |
| `start_offset_s` | `[pipeline]` | Skip this many seconds at the start |
| `max_duration_s` | `[pipeline]` | Process at most this many seconds (comment out for full obs) |
| `highpass_cutoff_hz` | `[pipeline]` | High-pass filter; set to `0.0` for simulation data |
| `auto_exclude_threshold` | `[pipeline]` | Exclude detectors with noise > N × median noise (`0` to disable) |
| `output_dir` | `[output]` | Where to write outputs |

## Optional features

### Boresight comparison

Adds a side-by-side plot comparing the map made with focal plane offsets applied vs. the boresight-only (no offset) map. Useful for verifying that pointing offsets are being applied correctly.

```toml
[map]
compare_boresight = true
```

Output: `boresight_comparison.png`

### Per-detector maps

Makes a separate map for each detector and saves the peak RA/Dec centroid of each. The main use case is pointing model reconstruction from planet scans.

```toml
[per_detector]
enabled         = true
save_png        = true    # save a .png per detector
save_numpy      = false
noise_threshold = 3.0     # flag detectors with off-source RMS > N * median RMS

# Choose ONE detector selection method:
max_detectors = 10            # evenly sample N detectors (0 = all)
# detectors = ["PC_f280_A_001"]   # specific detectors by name
# detector_indices = [0, 5, 100]  # specific detectors by index
```

Output: `per_detector/` directory with `centroids.json` and optionally per-detector maps.

### Profiling

Print the top 30 slowest functions and save a `.prof` file:

```bash
python run_mapmaker.py --profile
```

Visualise with [snakeviz](https://jiffyclub.github.io/snakeviz/):

```bash
pip install snakeviz
snakeviz output/<timestamp>/profile.prof
```

## Pipeline overview

Each run does the following:

1. **First pass** — scans all data to compute per-detector baselines and (optionally) the map centre.
2. **Naive map** — bins cleaned signal directly into a map with no common-mode subtraction.
3. **Initial common-mode pass** — estimates atmospheric noise as the mean across detectors, subtracts it, then rebins.
4. **Iterative passes** — repeats common-mode subtraction using the previous map as a sky signal prediction, refining the atmosphere estimate each time.
5. **Save** — writes all maps, diagnostics, and metadata to a timestamped output directory.

## Output files

| File | Contents |
|---|---|
| `overview.png` | All maps in one grid — **start here** |
| `naive.npy` / `.png` | Map with no common-mode subtraction |
| `it_0.npy` / `.png` | After first common-mode pass |
| `it_N.npy` / `.png` | Map after N common-mode iterations |
| `hits.npy` / `.png` | Detector samples per pixel |
| `noise.npy` / `.png` | Noise estimate (1/√hits) |
| `ra_edges.npy`, `dec_edges.npy` | Pixel grid edges (for loading maps in numpy) |
| `diagnostics.png` | Peak signal, off-source RMS, convergence, and runtime per pass |
| `psd.png` | Timestream PSD before vs. after common-mode subtraction |
| `metadata.json` | Full run parameters, observation info, and convergence metrics |
| `boresight_comparison.png` | Focal plane offset comparison (if `compare_boresight = true`) |
| `per_detector/centroids.json` | Per-detector peak RA/Dec centroids (if per-detector maps enabled) |

### Loading a map in Python

```python
import numpy as np

data     = np.load("output/<timestamp>/it_4.npy")
hits     = np.load("output/<timestamp>/hits.npy")
ra_edges = np.load("output/<timestamp>/ra_edges.npy")
dec_edges = np.load("output/<timestamp>/dec_edges.npy")
```

## Code structure

```
run_mapmaker.py     entry point, drives the pipeline
config.toml         settings (edit this)
environment.yml     conda environment specification
mapmaker/
    reader.py       reads .g3 files and rebuffers into fixed-duration chunks
    pointing.py     converts quaternion pointing to RA/Dec
    signal.py       I/Q to frequency shift conversion (real data only)
    cleaning.py     cosmic ray removal and high-pass filtering
    common_mode.py  common-mode noise subtraction
    binning.py      bins detector samples into sky pixels
    output.py       saves maps and diagnostics to disk
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'spt3g'`**
The `ccat` conda environment is not active. Run `conda activate ccat`.

**`FileNotFoundError` on startup**
Check that `input_dirs` in `config.toml` points to a directory that contains `.g3` files. The path is relative to the location of `config.toml`.

**Map looks empty or has very few hits**
Try omitting `ra0_deg` / `dec0_deg` so the mapmaker auto-detects the centre, or increase `xlen_deg` / `ylen_deg`.

**Run is slow**
Reduce `max_duration_s` to process a shorter time window, or reduce `n_iterations`. For memory issues with large arrays, set `time_chunk_size` to ~10000.
