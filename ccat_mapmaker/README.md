# CCAT Quick-Look Mapmaker

Standalone command-line mapmaker for CCAT Prime Cam. Reads `.g3` or `.h5` detector data files (TOAST simulations, or real BLAST-TNG data), cleans and bins the timestreams, runs iterative common-mode subtraction, and saves sky maps to disk.

## Prerequisites

- [Anaconda](https://www.anaconda.com/download) (or Miniconda)
- `.g3`/`.h5` simulation files, or real BLAST-TNG `.g3` files

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

Activate the environment before doing anything, `spt3g` is not available outside it:

```bash
conda activate ccat
```

Your prompt will change from `(base)` to `(ccat)`. Deactivate when done:

```bash
conda deactivate
```

## Quick start

### 1. Point the config at your data

Open `ccat_mapmaker/config.toml` and set `input_dirs` to the directory containing your data files:

```toml
[data]
input_dirs  = ["/path/to/your/data"]
file_format = "g3"            # or "h5"
format      = "simulation"    # or "blasttng" for real BLAST-TNG data
```

`input_dirs` accepts glob patterns and is expanded recursively for files matching `file_format`.

`file_format` and `format` are independent: `file_format` says what kind of file to look for on disk, `format` says how to interpret its contents. TOAST simulations can be either `.g3` or `.h5`; real BLAST-TNG data is always `.g3`.

If `format = "blasttng"`, also fill in the `[blasttng]` section with the G3 keys for the raw I/Q data, calibration sweeps, and (optionally) a cal-lamp exposure used to normalize each detector before it's converted to a frequency shift.

### 2. Set the map centre and size (optional)

If you know where your source is, pin the map centre with `ra0_deg`/`dec0_deg`, or point at it by name with `target` (planets and other solar-system bodies are resolved from an ephemeris at the observation time; anything else is looked up on SIMBAD). Omit all three and the mapmaker auto-detects the centre from the boresight instead.

```toml
[map]
ra0_deg  = 142.0
dec0_deg = 15.6
# target = "Jupiter"   # alternative to ra0_deg/dec0_deg

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

Open `overview.png` first, it shows all pipeline maps side by side.

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
  PSD anomalies flagged: 1 (0.31 Hz (6.2 dB))
  Auto-exclusion: noise median=0.0021  cutoff=0.0105 (5.0x)  excluded=3/280
  WNF-exclusion: cutoff=0.0043 (log-median+3.0sigma)  excluded=5/277
  Using 272/280 detectors
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

The `PSD anomalies flagged` and `WNF-exclusion` lines only appear if `psd_anomaly_sigma` / `wnf_exclude_sigma` are enabled and find something (see below).

## Configuring a run

All settings live in `config.toml`. The most commonly changed ones:

| Setting | Section | What it controls |
|---|---|---|
| `input_dirs` | `[data]` | Paths to data files (glob patterns OK) |
| `file_format` | `[data]` | `"g3"` or `"h5"` |
| `format` | `[data]` | `"simulation"` or `"blasttng"` |
| `ra0_deg`, `dec0_deg`, `target` | `[map]` | Map centre (omit all three to auto-detect) |
| `res_arcmin` | `[map]` | Pixel size in arcminutes |
| `n_iterations` | `[pipeline]` | Common-mode subtraction iterations (3-5 is typical) |
| `start_offset_s` | `[pipeline]` | Skip this many seconds at the start |
| `max_duration_s` | `[pipeline]` | Process at most this many seconds (comment out for full obs) |
| `clean_steps` | `[pipeline]` | Ordered list of cleaning steps to run, see below |
| `auto_exclude_threshold` | `[pipeline]` | Exclude detectors with noise > N x median noise (`0` to disable) |
| `wnf_exclude_sigma` | `[pipeline]` | Exclude detectors by white-noise-floor outlier, on top of `auto_exclude_threshold` (`0` to disable) |
| `weight_cap_factor` | `[pipeline]` | Cap each detector's inverse-variance map weight at N x the median weight (`0` for uniform weighting) |
| `psd_anomaly_sigma` | `[pipeline]` | Flag narrowband PSD contaminants for follow-up, doesn't remove anything (`0` to disable) |
| `output_dir` | `[output]` | Where to write outputs |

## Cleaning steps

Timestream cleaning runs as an ordered list of self-contained steps:

```toml
[pipeline]
clean_steps = ["highpass"]   # any of: "cosmic_rays", "highpass", "notch"

[pipeline.cosmic_rays]
sigma = 3.5   # spike threshold, in units of sigma * std(tod)
n     = 2     # samples masked on each side of a detected spike

[pipeline.highpass]
cutoff_hz = 0.0   # 0.0 disables it; keep it off for simulation data (no atmosphere to remove)

[pipeline.notch]
freq_hz      = 0     # 0 auto-detects the mains frequency (50 or 60 Hz) per chunk
n_harmonics  = 5
bandwidth_hz = 1.0
```

Add, remove, or reorder steps freely, each one has its own `[pipeline.<step>]` block. `cosmic_rays` removes sudden spikes by interpolation, `highpass` removes slow atmospheric drift, and `notch` removes mains-line pickup and its harmonics.

Separately, `psd_anomaly_sigma` looks for narrowband features in the timestream PSD that don't fit either category (not at a mains frequency, not atmospheric drift). These aren't removed, just flagged in `metadata.json` for follow-up, since an unexplained line could be a real instrumental effect worth investigating rather than something safe to filter out.

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
apply_offsets   = false   # false = bin against the shared boresight, not per-detector
                           # offsets (use this while focal-plane offsets aren't trusted yet)

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

1. **First pass**: scans all data to compute per-detector baselines, white-noise-floor, and (optionally) the map centre.
2. **Naive map**: bins cleaned signal directly into a map with no common-mode subtraction.
3. **Initial common-mode pass**: estimates atmospheric noise as the mean across detectors, subtracts it, then rebins.
4. **Iterative passes**: repeats common-mode subtraction using the previous map as a sky signal prediction, refining the atmosphere estimate each time.
5. **Save**: writes all maps, diagnostics, and metadata to a timestamped output directory.

## Output files

| File | Contents |
|---|---|
| `overview.png` | All maps in one grid, start here |
| `naive.npy` / `.png` | Map with no common-mode subtraction |
| `it_0.npy` / `.png` | After first common-mode pass |
| `it_N.npy` / `.png` | Map after N common-mode iterations |
| `hits.npy` / `.png` | Detector samples per pixel |
| `noise.npy` / `.png` | Per-pixel noise estimate from sample variance |
| `time_null.npy` / `.png` | Chunk-parity (even/odd) half-difference null map, if `compute_time_null = true` |
| `detsplit_null.npy` / `.png` | Random 50/50 detector-split half-difference null map, if `compute_detsplit_null = true` |
| `ra_edges.npy`, `dec_edges.npy` | Pixel grid edges (for loading maps in numpy) |
| `diagnostics.png` | Peak signal, off-source RMS, convergence, and runtime per pass |
| `psd.png` | Timestream PSD before vs. after common-mode subtraction |
| `metadata.json` | Full run parameters, observation info, convergence metrics, and any flagged `psd_anomalies` |
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
    reader.py       reads .g3/.h5 files (TOAST simulation or real BLAST-TNG) into fixed-duration chunks
    pointing.py     converts quaternion pointing to RA/Dec
    signal.py       I/Q to frequency shift conversion and cal-lamp normalization (real data only)
    target.py       resolves a map centre from a target name (ephemeris or SIMBAD)
    cleaning.py     cosmic ray removal, high-pass and notch filtering, PSD anomaly flagging
    common_mode.py  common-mode noise subtraction
    binning.py      bins detector samples into sky pixels
    output.py       saves maps and diagnostics to disk
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'spt3g'`**
The `ccat` conda environment is not active. Run `conda activate ccat`.

**`FileNotFoundError` on startup**
Check that `input_dirs` in `config.toml` points to a directory that contains files matching `file_format`. The path is relative to the location of `config.toml`.

**Map looks empty or has very few hits**
Try omitting `ra0_deg` / `dec0_deg` / `target` so the mapmaker auto-detects the centre, or increase `xlen_deg` / `ylen_deg`.

**Run is slow**
Reduce `max_duration_s` to process a shorter time window, or reduce `n_iterations`. For memory issues with large arrays, set `time_chunk_size` to ~10000.
