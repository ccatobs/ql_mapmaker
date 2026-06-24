# CCAT Quick-Look Mapmaker

Standalone command-line mapmaker for CCAT Prime Cam. Reads `.g3` detector data files, runs iterative common-mode subtraction, and saves sky maps to disk.

## Setup

### First time on a new machine

1. Install [Anaconda](https://www.anaconda.com/download) if not already present.

2. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```
   This installs Python 3.11, numpy, scipy, astropy, matplotlib, spt3g, h5py, and jupyter.

3. Activate it:
   ```bash
   conda activate ccat
   ```

### Every time you work on this project

Activate the environment before doing anything -- `spt3g` is not available outside it:

```bash
conda activate ccat
```

Your prompt should change from `(base)` to `(ccat)`. To deactivate when done:

```bash
conda deactivate
```

## Running the mapmaker

From the `CCAT/` directory:

```bash
conda activate ccat
python ccat_mapmaker/run_mapmaker.py --config ccat_mapmaker/config.toml
```

Output maps are saved to `ccat_mapmaker/output/` as `.npy` and `.png` files, one per pipeline stage.

## Configuring a run

Edit `config.toml`. The most commonly changed settings are:

| Setting | What it controls |
|---|---|
| `input_dirs` | Path(s) to the `.g3` data files |
| `ra0_deg`, `dec0_deg` | Map centre (omit to auto-detect from data) |
| `res_arcmin` | Pixel size |
| `n_iterations` | Number of common-mode subtraction iterations |
| `start_offset_s`, `max_duration_s` | Time window within the observation |

## Pipeline overview

Each run does the following:

1. **First pass** -- scans all chunks to compute per-detector baselines and map centre.
2. **Naive map** -- bins cleaned signal directly into a map with no common-mode subtraction.
3. **Initial common-mode pass** -- estimates the atmosphere as the mean across detectors and subtracts it, then rebins.
4. **Iterative passes** -- repeats common-mode subtraction using the previous map as a sky signal prediction, improving the atmosphere estimate each time.

## Code structure

```
run_mapmaker.py     entry point, drives the pipeline
config.toml         settings (edit this)
mapmaker/
    reader.py       reads .g3 files and rebuffers into fixed-duration chunks
    pointing.py     converts quaternion pointing to RA/Dec
    signal.py       I/Q to frequency shift conversion (real data only)
    cleaning.py     cosmic ray removal and high-pass filtering
    common_mode.py  common-mode noise subtraction
    binning.py      bins detector samples into sky pixels
    output.py       saves maps and diagnostics to disk
```

## Output files

| File | Contents |
|---|---|
| `naive.npy / .png` | Map with no common-mode subtraction |
| `it_0.npy / .png` | First common-mode pass |
| `it_N.npy / .png` | Map after N common-mode iterations |
| `hits.npy / .png` | Detector samples per pixel |
| `noise.npy / .png` | Noise estimate (1/√hits) |
| `ra_edges.npy`, `dec_edges.npy` | Pixel grid edges |
| `overview.png` | All maps in one grid |
| `diagnostics.png` | Peak signal, off-source RMS, convergence, and runtime per pass |
| `psd.png` | Timestream PSD before vs after common-mode subtraction |
| `metadata.json` | Run parameters and observation info |
