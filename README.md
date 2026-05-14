# CRYOtools

A Python package for processing and analysing DKIST Cryo-NIRSP spectropolarimetric data. CRYOtools provides tools for ingesting Level-1 data products, fitting high-resolution coronal emission line spectra, deriving physical parameters (Doppler velocities, non-thermal line widths), and producing publication-quality visualisations.

> **Note:** This package is still a work in progress. Expect bugs.

---

## Requirements

- Python ≥ 3.10
- The following Python packages (see [Dependencies](#dependencies) for details):
  `numpy`, `scipy`, `matplotlib`, `astropy`, `jax`, `numba`, `sunpy`, `dkist`, `hvpy`, `tqdm`

---

## Installation

### 1. Create and activate a virtual environment (recommended)


### 2. Install JAX

JAX installation depends on your hardware. Install the appropriate variant **before** installing CRYOtools:

**CPU only:**
```bash
pip install "jax[cpu]"
```

**NVIDIA GPU (CUDA 12):**
```bash
pip install "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other platforms and CUDA versions.

### 3. Install CRYOtools

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Richardjmorton/CRYOtools.git
cd CRYOtools
pip install -e .
```

The `-e` flag installs in editable mode, meaning changes to the source files are reflected immediately without reinstalling.

### 4. Install remaining dependencies

```bash
pip install numpy scipy matplotlib astropy numba sunpy dkist hvpy tqdm
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | Optimisation, signal processing, interpolation |
| `matplotlib` | Plotting and visualisation |
| `astropy` | FITS I/O, WCS, physical units and constants |
| `jax` | JAX-accelerated spectral fitting pipeline |
| `numba` | JIT-compiled convolution kernels (alternative fitting backend) |
| `sunpy` | Solar map handling and coordinate transforms |
| `dkist` | DKIST data product ingestion |
| `hvpy` | Helioviewer API for AIA/SDO context imagery |
| `tqdm` | Progress bars |

---

## Package structure

```
CRYOtools/
├── fit.py        # JAX-based spectral modelling pipeline
├── io.py         # Data ingestion: FITS/ASDF readers, atlas loaders, writers
├── util.py       # Shared utilities: WCS grids, cadence, physical conversions
├── plotting.py   # Visualisation: slit maps, parameter maps, model comparisons
├── fit_test.py   # Experimental NumPy/Numba fitting backend
└── models/       # Bundled reference solar and telluric spectra
```

---

## DKIST data access

Cryo-NIRSP Level-1 data products are accessed via the [DKIST Data Centre](https://data.dkist.nso.edu). You will need a DKIST account and to have downloaded or have access to the relevant dataset directories before using the ingestion routines in `CRYOtools.io`.

---

## Licence

MIT
