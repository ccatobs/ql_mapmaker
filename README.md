# CCAT/FYST Prime-Cam Quick-Look Mapmaker

A command-line mapmaker for CCAT Prime Cam detector data. It reads `.g3` or
`.h5` simulation and lab data, processes detector timestreams, and writes sky
maps and diagnostics.

## Get started

The recommended setup uses a lightweight Conda distribution such as
[Miniforge](https://github.com/conda-forge/miniforge) or
[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). From the
repository root:

```bash
conda env create -f environment.yml
conda activate ccat
```

Set the input data path and format in `ccat_mapmaker/config.toml`, then run:

```bash
python ccat_mapmaker/run_mapmaker.py --config ccat_mapmaker/config.toml
```

The application currently requires `spt3g` for every run, including HDF5
simulation data. To use the included Jupyter notebooks, install JupyterLab
into the active environment:

```bash
conda install -c conda-forge jupyterlab
```

## Documentation

See the [detailed guide](ccat_mapmaker/README.md) for configuration, tests,
cleaning options, outputs, diagnostics, and troubleshooting.

## Tests

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

Some integration tests use local sample data; see the detailed guide for
sample locations and environment-variable overrides.
