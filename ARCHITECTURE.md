# CRYOtools Architecture

## Repository layout
- `pyproject.toml` defines the package metadata for the distributable `CRYOtools` project.【F:pyproject.toml†L1-L18】
- `CRYOtools/fit.py` houses the primary JAX-based spectral modelling pipeline, including convolution helpers, solar/telluric atlas interpolation, slit-fitting routines, and result collation utilities.【F:CRYOtools/fit.py†L1-L435】
- `CRYOtools/io.py` contains dataset ingestion helpers for DKIST CryoNIRSP products, reference atlas loaders, FITS/ASDF readers, Stokes vector rotation, and writers for fitted spectra.【F:CRYOtools/io.py†L1-L401】
- `CRYOtools/util.py` provides shared astronomy-specific utilities for interpreting FITS headers, deriving coordinate grids, cadence estimation, and various physical conversions.【F:CRYOtools/util.py†L1-L377】
- `CRYOtools/plotting.py` offers end-user visualisation utilities for slit locations, context imagery, atlas inspection, fitted parameter maps, and model-vs-data comparisons.【F:CRYOtools/plotting.py†L1-L563】
- `CRYOtools/fit_test.py` is an experimental/legacy NumPy+Numba implementation of the spectral fitting workflow that mirrors the core pipeline with different performance trade-offs.【F:CRYOtools/fit_test.py†L1-L200】
- `CRYOtools/models/` bundles reference solar and telluric spectra that are accessed via `importlib.resources` by the I/O layer.【F:CRYOtools/io.py†L47-L88】

## Data ingestion layer (`CRYOtools.io`)
The I/O module abstracts access to DKIST CryoNIRSP Level-1 products and auxiliary atlases:

- `load_asdf` locates the observation ASDF file within a directory and logs the dataset shape to help diagnose large loads.【F:CRYOtools/io.py†L16-L44】
- `_read_solar_model` and `_read_telluric_model` expose packaged reference spectra via `importlib.resources`, ensuring deployments do not need external downloads.【F:CRYOtools/io.py†L47-L88】
- `find_L1_files`, `get_all_head`, and `_get_all_head` scan for matching FITS files and fetch their headers in parallel to minimise I/O overhead.【F:CRYOtools/io.py†L90-L166】
- `restore_slit_coords` reloads previously derived slit, time, and spectral coordinate arrays, enabling downstream processing without recomputation.【F:CRYOtools/io.py†L168-L198】
- `rotate_stokes` converts DKIST Stokes vectors into the solar radial frame using stored slit coordinates, keeping polarimetric analyses consistent.【F:CRYOtools/io.py†L200-L229】
- `_fits_mp`, `_unpack_and_run`, and `get_fits_data` stream FITS cubes in parallel, allocate observation-appropriate arrays, apply optional co-adding, photometric scaling, and Stokes rotation, returning both data and contributing filenames.【F:CRYOtools/io.py†L232-L372】
- `write_model_results` serialises optimiser outputs into a structured `spectrum_fits` directory, producing compressed `.npz` archives keyed by source FITS names.【F:CRYOtools/io.py†L375-L401】

## Shared utilities (`CRYOtools.util`)
Support functions centralise survey metadata handling and physical conversions:

- `return_obs_info` inspects header keywords to recover scan dimensions, wavelengths, and slit lengths while optionally echoing diagnostic text.【F:CRYOtools/util.py†L21-L50】
- `get_slit_coords` and `get_slit_coords_mp` reconstruct spatial and temporal sampling grids from FITS WCS information, caching them to disk for reuse.【F:CRYOtools/util.py†L53-L259】
- `get_spectral_coords` derives dispersion coordinates in nanometres and saves them alongside spatial grids.【F:CRYOtools/util.py†L101-L136】
- `get_slit_samp` summarises raster geometry by computing along-slit sampling and step width, forming the spatial scale for later plots.【F:CRYOtools/util.py†L137-L176】
- `calculate_cadence` estimates median temporal cadence, with optional quick-look plotting, enabling temporal interpretation of stare observations.【F:CRYOtools/util.py†L261-L289】
- `shift_to_v` and `calc_ntlw` convert spectral shifts and widths into physical Doppler velocities and non-thermal broadening measures using astropy units and constants.【F:CRYOtools/util.py†L290-L353】
- `ensure_directory` and `print_exposure` wrap common filesystem setup and header reporting patterns used across the package.【F:CRYOtools/util.py†L356-L377】

## Spectral modelling (`CRYOtools.fit`)
The modelling module implements the high-resolution spectral fitting workflow powered by JAX:

- Low-level kernels such as `gaussian_filter_1d`, `gaussian`, `fft_shift`, and `do_conv` perform convolution, analytic profile evaluation, and frequency-domain shifting of model spectra.【F:CRYOtools/fit.py†L30-L75】
- `get_lags` and `get_lags_lin` estimate velocity offsets between observed spectra and atlas templates (solar and telluric) via cross-correlation to initialise optimisation bounds.【F:CRYOtools/fit.py†L84-L129】
- `vac2air`, `get_solar_model`, and `get_telluric_model` convert wavelength scales and interpolate packaged atlases onto observation coordinates, providing the baseline continuum and absorption models.【F:CRYOtools/fit.py†L131-L155】
- The functional `fit_slit` routine iterates along a slit, periodically seeding bounds with differential evolution before refining with BFGS, storing optimiser results per spatial position.【F:CRYOtools/fit.py†L158-L201】
- The `fit_data` class wraps the same logic in an object-oriented interface that caches spectra, weights, and atlas models. Its `fit_slit` method reuses shared JAX-compiled inner functions, alternating global (`differential_evolution`) and local (`minimize`) solvers while updating velocity bounds dynamically.【F:CRYOtools/fit.py†L204-L378】
- `calculate_model` and `plot_model_vs_data` provide simple post-fit diagnostics by reconstructing best-fit spectra and comparing them to observations.【F:CRYOtools/fit.py†L381-L400】
- `_pull_fit_res` and `pull_fit_res` aggregate stored optimisation outputs across files into stacked parameter and merit arrays using multiprocessing, enabling rapid downstream analysis of large campaigns.【F:CRYOtools/fit.py†L404-L435】

## Visualisation utilities (`CRYOtools.plotting`)
Plotting helpers streamline quick-look inspection of data products and fit outputs:

- Functions such as `plot_slit_locs` and `aia_context_plot` visualise slit trajectories over time and embed them in helioviewer AIA context imagery, including optional rotation corrections.【F:CRYOtools/plotting.py†L24-L148】
- `which_line`, `plot_line_example`, and `quick_plot_emission` locate coronal lines, display representative spectra, and build emissivity maps from integrated intensities.【F:CRYOtools/plotting.py†L149-L270】
- `plot_solar_atlas` and `plot_telluric_atlas` overlay packaged reference spectra on the observation grid for validation.【F:CRYOtools/plotting.py†L273-L288】
- `plot_all_params_scan` and `plot_all_params_ss` lay out comprehensive parameter panels for raster and sit-and-stare datasets, respectively, handling colour scaling, extent calculation, and legend placement.【F:CRYOtools/plotting.py†L291-L445】
- `plot_coronal_params` focuses on amplitude, Doppler shift, and line-width maps with consistent colour bars and scaling controls.【F:CRYOtools/plotting.py†L447-L510】
- `plot_spectrogram_wf` juxtaposes observed spectrograms, fitted models, and residuals to highlight modelling performance and dual-beam artefacts.【F:CRYOtools/plotting.py†L514-L563】

## Alternative fitting prototype (`CRYOtools.fit_test`)
`fit_test.py` mirrors the main pipeline with a NumPy/Numba backend:

- It implements JIT-compiled convolution and Gaussian routines via Numba, along with shared-memory infrastructure for multiprocessing across slit positions.【F:CRYOtools/fit_test.py†L1-L200】
- The module reuses the same atlas loaders, cross-correlation lag estimation, parameter bounds, and optimisation strategy but trades JAX for explicit `SharedMemory` arrays and ProcessPool executors to experiment with different performance characteristics.【F:CRYOtools/fit_test.py†L20-L200】

# Testing
- Always run `python -m compileall CRYOtools`