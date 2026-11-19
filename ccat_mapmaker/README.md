# CCAT Quick-Look Mapmaker

Standalone command-line mapmaker for CCAT Prime Cam. Reads `.g3` or `.h5` detector data files (TOAST simulations, or real lab data), cleans and bins the timestreams, runs iterative common-mode subtraction, and saves sky maps to disk.

## Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) (recommended) or
  [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). The full
  Anaconda distribution is not required.
- `.g3`/`.h5` simulation or lab format files.

## Setup

### First time on a new machine

1. Clone the full repository. The environment file and tests are at the
   repository root, alongside `ccat_mapmaker/`.

2. From the repository root, create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate ccat
   ```

   This installs Python 3.11 and the mapmaker's runtime dependencies, including
   `spt3g`. The application imports `spt3g` at startup, so it is required for
   all runs, including HDF5 simulations.

### Every time you work on this project

Activate the environment before running the mapmaker:

```bash
conda activate ccat
```

Deactivate when done if you like:

```bash
conda deactivate
```

### Optional: Jupyter notebooks

Jupyter is not needed to run the command-line mapmaker. To use the notebooks
in `ccat_mapmaker/notebooks/`, install JupyterLab into the active environment:

```bash
conda install -c conda-forge jupyterlab
```

## Tests

From the repository root, run `python -m unittest discover -s tests -v`.
The sample-backed tests are skipped when their data is unavailable. By default
they look under `~/notebooks/data/ql_mm_samples/`; set
`CCAT_MAPMAKER_LAB_SAMPLE_DIR`, `CCAT_MAPMAKER_SIM_SAMPLE_DIR`, or
`CCAT_MAPMAKER_G3_SAMPLE_DIR` to the corresponding sample directory to override
those locations. The lab and G3 integration tests need `spt3g`; see Setup for
why it is included in the environment for every run.

The simulation test uses `Jupiter-0-0.h5` and creates a temporary HDF5 virtual
view of a 20-second Jupiter crossing, without copying the multi-GB detector
arrays. It downsamples that window and checks the naive and iterative maps for
a bright peak near Jupiter. It is a quick integration check, not a validation
of Jupiter's resolved size, atmospheric modelling, or a full observation.
The G3 simulation test runs `frames-0006.g3` through the same map pipeline,
using a temporary symlink instead of copying the file, and checks for a bright
peak near Jupiter in both the naive and iterative maps.

## Quick start

### 1. Point the config at your data

Open `ccat_mapmaker/config.toml` and set `input_dirs` to the directory containing your data files:

```toml
[data]
input_dirs  = ["/path/to/your/data"]
file_format = "g3"            # or "h5"
format      = "simulation"    # or "lab"
```

`input_dirs` accepts glob patterns and is expanded recursively for files matching `file_format`.

`file_format` and `format` are independent: `file_format` says what kind of file to look for on disk, `format` says how to interpret its contents. TOAST simulations can be either `.g3` or `.h5`; real BLAST-TNG data is always `.g3`.

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

To size an axis automatically, omit its length: leave out `xlen_deg` for the
width, `ylen_deg` for the height, or both. Each axis is determined
independently, so one can be fixed while the other is automatic. The preparation pass measures finite, input-unflagged detector and boresight
pointing for retained detectors over the selected time window. The map stays
centred at the configured, target, or automatic centre, and extends at least
one pixel beyond the measured bounds. Later cleaning can exclude additional
samples. Time-window limits are applied at chunk boundaries. An extent cannot
be determined if no retained detector has usable pointing. Per-detector
diagnostic maps use this same grid; samples from excluded detectors selected
for those diagnostics are not included in the extent calculation.

RA is currently binned in its raw 0–360 degree coordinates without wraparound.
Scans crossing 0 degrees, or a centre far from the data, can therefore produce
very large maps; check the reported grid size and available memory.

### 3. Run

From the repository root (the directory containing `environment.yml`):

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
ccat_mapmaker/output/RCW92_2024-07-08_14-32-01-123456/
```

When PNG output is enabled, open `overview.png` first; it shows the maps side
by side. Each run directory also contains the original `config.toml`, a
`run.log`, and `metadata.json` with settings and diagnostics. The directory
is created before preparation, so failed or interrupted runs retain their log
and partial metadata. Failures before the configuration can be loaded cannot
be recorded there. Chunk-level progress remains on the console only.

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
  WNF-exclusion: cutoff=0.0043 (log-median+3.0sigma)  excluded=5/277
  Using 272/280 detectors
  Obs start  : 2024-01-15 10:32:00 UTC
  Duration   : 100.0 s
  Detectors  : 280 @ 200.0 Hz
  Map grid   : 240 x 240 pixels (0.25 arcmin/pixel), centre = (142.0000, 15.6000) [config]

Step 2: Naive + initial common-mode maps (chunk_duration_s=auto)...
  Auto chunk size: 4000 samples (20.00 s)
Step 3: 4 iteration(s)...
  Iteration 1/4... [1.8s]
  ...

Total time : 18.3s
Maps saved : ccat_mapmaker/output/RCW92_2024-07-08_14-32-01-123456/
Log saved  : ccat_mapmaker/output/RCW92_2024-07-08_14-32-01-123456/run.log
============================================================
```

The `WNF-exclusion` line only appears if `wnf_exclude_sigma` is enabled and
finds an outlier.

## Configuring a run

All settings live in `config.toml`. The most commonly changed ones:

| Setting | Section | What it controls |
|---|---|---|
| `input_dirs` | `[data]` | Paths to data files (glob patterns OK) |
| `file_format` | `[data]` | `"g3"` or `"h5"` |
| `format` | `[data]` | `"simulation"` or `"lab"` |
| `ra0_deg`, `dec0_deg`, `target` | `[map]` | Map centre (omit all three to auto-detect) |
| `xlen_deg`, `ylen_deg` | `[map]` | Map width/height in degrees; omit either to size that axis from the data |
| `res_arcmin` | `[map]` | Pixel size in arcminutes |
| `n_iterations` | `[pipeline]` | Common-mode subtraction iterations (3-5 is typical) |
| `start_offset_s` | `[pipeline]` | Skip this many seconds at the start |
| `max_duration_s` | `[pipeline]` | Process at most this many seconds (comment out for full obs) |
| `sample_step` | `[pipeline]` | Retain every Nth time sample for all detectors (`1` by default) |
| `chunk_duration_s` | `[pipeline]` | Fixed chunk duration in seconds, or `"auto"` to size chunks from available memory |
| `chunk_memory_fraction` | `[pipeline]` | Fraction of available RAM targeted by auto sizing (default `0.5`) |
| `chunk_memory_gb` | `[pipeline]` | Optional upper bound for the auto chunk-memory target in GiB (`0` means no explicit cap) |
| `cosmic_rays`, `highpass`, `notch` | `[pipeline.*]` | Cleaning blocks to enable, see below |
| `auto_exclude_threshold` | `[pipeline]` | Exclude detectors with noise > N x median noise (`0` to disable) |
| `wnf_exclude_sigma` | `[pipeline]` | Exclude detectors by white-noise-floor outlier, on top of `auto_exclude_threshold` (`0` to disable) |
| `weight_cap_factor` | `[pipeline]` | Cap each detector's inverse-variance map weight at N x the median weight (`0` for uniform weighting) |
| `save_psd` | `[output]` | Save PSD diagnostics (`true` by default), independently of `save_png` |
| `output_dir` | `[output]` | Where to write outputs |

## Cleaning steps

`sample_step = 2` retains the first sample in the selected time window and
every second sample thereafter, continuously across chunks and files. The
stride applies before preparation, cleaning, mapmaking, and diagnostics; the
effective sample rate is divided by the step. This is unfiltered subsampling,
so signals above the new Nyquist frequency can alias, and cleaning, detector
weights, map coverage, and noise estimates can change. Sample-based cleaning
parameters count retained samples. Input data and pointing may still be read
at full resolution. If the first usable chunk contains fewer than four retained
samples, increase the chunk duration or reduce the step.

Set `chunk_duration_s = "auto"` to estimate the largest processing chunk from
available memory. Auto sizing applies to every pass, including preparation:
longer preparation chunks change PSD frequency resolution and may affect
per-chunk medians, detector noise estimates, and weights. Larger chunks also
move cleaning boundaries, which can affect gap filling and spike detection.
The memory estimate is conservative but not a hard process-memory limit;
input formats that materialize an entire source frame can exceed the target
for a large frame.

Gap filling is always invoked. The main map passes then invoke each cleaning
step whose configuration block is present, in the fixed order cosmic rays,
high-pass, notch. The `clean_steps` list does not control these passes.

```toml
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

Add or remove the corresponding blocks to enable or disable steps.
`cosmic_rays` removes sudden spikes by interpolation, `highpass` removes slow
atmospheric drift, and `notch` removes mains-line pickup and its harmonics.

PSD anomaly detection is not currently connected to the main pipeline.

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
2. **Initial maps**: one data scan produces both the naive map and the initial common-mode map, sharing input reads and cleaning.
3. **Iterative passes**: repeats common-mode subtraction using the previous map as a sky signal prediction, refining the atmosphere estimate each time.
4. **Save**: writes all maps, diagnostics, and metadata to a timestamped output directory.

## Output files

| File | Contents |
|---|---|
| `overview.png` | All maps in one grid, start here |
| `naive.npy` / `.png` | Map with no common-mode subtraction |
| `it_0.npy` / `.png` | After first common-mode pass |
| `it_N.npy` / `.png` | Map after N common-mode iterations |
| `hits.npy` / `.png` | Initial-pass sum of detector weights per pixel; raw sample count only with uniform weights |
| `noise.npy` / `.png` | Initial-pass sample-scatter estimate; not yet validated as an uncertainty for weighted or correlated samples |
| `null.npy` / `time_null.png` | Final-pass even/odd chunk-parity half-difference, if `compute_time_null = true` |
| `detsplit_null.npy` / `.png` | Final-pass balanced random detector-split half-difference, if `compute_detsplit_null = true` |
| `time_split_weight_0.npy`, `time_split_weight_1.npy` | Final-pass time-split per-pixel weight sums, if enabled |
| `detector_split_weight_0.npy`, `detector_split_weight_1.npy` | Final-pass detector-split per-pixel weight sums, if enabled |
| `ra_edges.npy`, `dec_edges.npy` | Pixel grid edges (for loading maps in numpy) |
| `diagnostics.png` | Peak signal, off-source RMS, convergence, and runtime per pass |
| `psd.png` | Final-pass matched-detector PSD before vs. after common-mode subtraction |
| `psd.npz` | Frequencies, detector names, segment counts and per-detector before/after PSDs |
| `weighting_psd.png` | Preparation PSD distribution and white-noise band |
| `weighting_psd.npz` | Preparation PSD frequency summary and per-detector white-noise floors |
| `config.toml` | Exact copy of the input configuration |
| `metadata.json` | Run status, provenance, settings, input manifest, processing and map diagnostics |
| `run.log` | Run log with timings and errors, including failed runs |
| `boresight_comparison.png` | Focal plane offset comparison (if `compare_boresight = true`) |
| `per_detector/centroids.json` | Per-detector peak RA/Dec centroids (if per-detector maps enabled) |

`metadata.json` records the Git commit, branch and dirty flag when available;
the dirty flag does not capture the uncommitted changes themselves. Missing Git
information is reported in the log and metadata. Package versions may be null
if distribution metadata is unavailable. The input manifest distinguishes
discovered files from files handed to the reader; an opened file need not
contribute samples to the selected time window.

The preparation counts cover all detectors and yielded chunks. Map-pass flag
counts cover retained detectors in chunks that were not entirely input-flagged;
`fully_flagged_chunks` counts the skipped chunks separately. Flag categories
can overlap. Chunk sizes are time samples per chunk. Convergence's off-source
RMS uses pixels below the final map's median, not a verified source-free region;
map quality statistics and null maps are diagnostics, not fidelity tests.
Non-finite diagnostic values are stored as JSON null.

Null maps are `(split_0 - split_1) / 2`, with `NaN` where either split lacks
coverage. Their weight maps are sums of detector weights, not necessarily
sample counts. Inspect both split weights: unequal coverage or weighting
changes the null's expected noise, and partial overlap is not an all-sky test.
The detector split is balanced in detector count, uses seed 0, and requires at
least two retained detectors; its assignments and overlap counts are recorded
in `metadata.json`. The time split depends on chunk boundaries, so auto chunk
sizing can change it between runs.

Both splits reuse the same common-mode estimate and previous full-sky map;
they are **not independently processed reconstructions**. Common artifacts can
cancel, while different coverage and filtering can leave sky residuals. A
passing null does not prove a map is unbiased, and the half-difference is not
automatically a calibrated noise estimate.

### PSD diagnostics

`save_psd = true` (the default) saves the PSD plots independently of
`save_png`; the `.npz` files additionally require `save_numpy = true`. The
comparison samples up to 16 evenly spaced retained detectors in the **final**
map pass. Both spectra use the same matched segments after baseline removal and
configured cleaning, immediately before and after common-mode subtraction.
Each detector's spectrum is the mean of its usable Welch segments; the plot
shows the median and 25th–75th percentile across those detectors. The segment
length is the smaller of 2048 and the first usable chunk length, with a Hann
window, 50% overlap within chunks and constant detrending. Segments containing
input or cleaning flags or non-finite samples are omitted from both spectra.
Frequency resolution, detector identities and segment counts are recorded in
`metadata.json`. Short chunks may contribute no segments; low frequencies
below the resolution cannot be assessed. Signal units are inherited from the
input, not independently calibrated. Chunks with invalid or different sample
rates are excluded from the comparison and reported in the log and metadata.

`weighting_psd.png` shows the separate preparation-pass periodogram used by
the current noise-floor and weighting calculation. **Its estimator is
unchanged:** only chunks exactly as long as the first usable chunk contribute,
and input flags are not masked. `preparation_processing` records eligible and
used time samples, chunk length and non-finite detector count; the log warns
about omitted chunks or flags. In auto chunk mode this can represent only part
of the observation. The two plots use different estimators, detector sets and
time selections and should not be directly compared as a filtering transfer
function.

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
.
├── README.md
├── environment.yml
├── tests/
└── ccat_mapmaker/
    ├── config.toml
    ├── run_mapmaker.py    entry point and pipeline orchestration
    ├── mapmaker/
    │   ├── coord.py       pointing coordinates and map-centre resolution
    │   ├── map.py         sky grid and detector-sample binning
    │   ├── output.py      maps, plots, and metadata
    │   ├── reader.py      .g3/.h5 input and chunk iteration
    │   └── tod.py         timestream cleaning, noise, and common-mode routines
    └── notebooks/
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'spt3g'`**
The `ccat` conda environment is not active. Run `conda activate ccat`.

**`FileNotFoundError` on startup**
Check that `input_dirs` in `config.toml` points to a directory that contains files matching `file_format`. The path is relative to the location of `config.toml`.

**Map looks empty or has very few hits**
Try omitting `ra0_deg` / `dec0_deg` / `target` so the mapmaker auto-detects the centre, or increase `xlen_deg` / `ylen_deg`.

**Run is slow**
Reduce `max_duration_s` to process a shorter time window, reduce `n_iterations`,
or set `chunk_duration_s = "auto"` and adjust the memory target.
