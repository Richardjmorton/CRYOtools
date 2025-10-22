"""Utilities for crafting reusable test inputs from real CRYO observation data.

This module helps transform a collection of FITS files from real observations
into the compact fixtures referenced in the utils test plan.  Each helper
focuses on producing a single type of artifact so that tests can mix and match
only what they need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from astropy.io import fits
from astropy.io.fits import Header
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
import astropy.constants as const

# ---------------------------------------------------------------------------
# General FITS helpers
# ---------------------------------------------------------------------------


# Keywords that are essential to the utils helpers and therefore always copied
# when we trim a header down to the minimal fixture state.
REQUIRED_HEADER_PREFIXES: Tuple[str, ...] = (
    "NAXIS",
    "CTYPE",
    "CUNIT",
    "CRPIX",
    "CRVAL",
    "CDELT",
    "PC",
    "DATE",
    "TIME",
    "CN",
    "CAM_FPS",
)


def _load_primary_header(path: Union[str, Path]) -> Header:
    """Read the primary HDU header from ``path`` without data blocks."""
    with fits.open(path) as hdul:
        header = hdul[0].header.copy()
    return header


def extract_observation_headers(
    fits_paths: Sequence[Union[str, Path]],
    *,
    keep_full_header: bool = False,
    extra_keys: Sequence[str] | None = None,
) -> List[Header]:
    """Return lightweight FITS headers tailored for the utils tests.

    Parameters
    ----------
    fits_paths
        Sequence of paths pointing at real level-1 FITS files.
    keep_full_header
        If ``True`` the complete header is preserved.  Otherwise, only WCS and
        raster bookkeeping keywords are retained so the fixtures stay small.
    extra_keys
        Optional explicit keywords to keep even when ``keep_full_header`` is
        ``False``.
    """
    headers: List[Header] = []
    for path in fits_paths:
        header = _load_primary_header(path)
        if keep_full_header:
            headers.append(header)
            continue

        trimmed = Header()
        # Preserve mandatory keys first.
        for key in header.keys():
            if key == "" or key == "COMMENT" or key == "HISTORY":
                continue
            if key in (extra_keys or []):
                trimmed[key] = header[key]
                continue
            if key.startswith(REQUIRED_HEADER_PREFIXES):
                trimmed[key] = header[key]
        headers.append(trimmed)
    return headers


# ---------------------------------------------------------------------------
# Coordinate fixtures
# ---------------------------------------------------------------------------


def generate_slit_coordinate_inputs(
    headers: Sequence[Header],
    *,
    scan_subset: Optional[Sequence[int]] = None,
    meas_subset: Optional[Sequence[int]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ``hpxy_coords`` and ``time_coords`` arrays from real headers.

    Parameters
    ----------
    headers
        Sequence of already trimmed headers (see
        :func:`extract_observation_headers`).
    scan_subset
        Optional list of scan step indices (1-based as stored in the header)
        that should be included.  By default every scan in ``headers`` is used.
    meas_subset
        Optional list of measurement indices per scan (also 1-based) to keep.
    output_dir
        If provided the arrays are written as ``*.npy`` files to this
        directory so that parametrised tests can reuse them without re-running
        the WCS machinery.
    """

    if not headers:
        raise ValueError("At least one header is required to build slit inputs.")

    scan_indices = np.array([hdr["CNCURSCN"] for hdr in headers], dtype=int)
    meas_indices = np.array([hdr["CNCMEAS"] for hdr in headers], dtype=int)

    n_scan_steps = int(scan_indices.max())
    n_meas_per_step = int(meas_indices.max())
    slit_length = int(headers[0]["NAXIS2"])

    if scan_subset is not None:
        scan_mask = np.isin(scan_indices, list(scan_subset))
    else:
        scan_mask = np.ones_like(scan_indices, dtype=bool)

    if meas_subset is not None:
        meas_mask = np.isin(meas_indices, list(meas_subset))
    else:
        meas_mask = np.ones_like(meas_indices, dtype=bool)

    combined_mask = scan_mask & meas_mask
    selected_headers = [hdr for hdr, keep in zip(headers, combined_mask) if keep]

    hpxy_coords = np.zeros((2, n_scan_steps, n_meas_per_step, slit_length))
    time_coords = np.zeros((n_scan_steps, n_meas_per_step), dtype="datetime64[ms]")

    for hdr in selected_headers:
        wcs = WCS(hdr)
        # WCS expects pixel indices, hence the (0-based) range below.
        xy = wcs.array_index_to_world(0, np.arange(slit_length), 0)[1]
        x = xy.Tx.to(u.arcsec).value
        y = xy.Ty.to(u.arcsec).value
        obstime = xy[0].obstime.to_datetime()

        scan_idx = int(hdr["CNCURSCN"]) - 1
        meas_idx = int(hdr["CNCMEAS"]) - 1
        hpxy_coords[0, scan_idx, meas_idx, :] = x
        hpxy_coords[1, scan_idx, meas_idx, :] = y
        time_coords[scan_idx, meas_idx] = np.datetime64(obstime, "ms")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / "hpxy_coords.npy", hpxy_coords)
        np.save(output_path / "time_coords.npy", time_coords)

    return hpxy_coords, time_coords


def generate_spectral_coordinate_input(
    header: Header,
    *,
    pixel_indices: Optional[Sequence[int]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Derive the spectral coordinate array in nanometers for tests.

    Parameters
    ----------
    header
        Representative FITS header containing dispersion WCS keywords.
    pixel_indices
        Optional iterable of spectral pixel indices to extract.  When omitted a
        dense array covering the entire dispersion axis is returned.
    output_path
        If set, the resulting ``spec_coords`` array is persisted as ``.npy``.
    """

    nwv = int(header["NAXIS1"])
    pixels = np.arange(nwv) if pixel_indices is None else np.array(pixel_indices)

    wcs = WCS(header)
    coords = wcs.array_index_to_world(0, 0, pixels)[0]
    spec_coords = coords.to(u.nm).value

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, spec_coords)

    return spec_coords


# ---------------------------------------------------------------------------
# Derived numeric arrays for physics/unit tests
# ---------------------------------------------------------------------------


def build_sample_numeric_arrays(
    hpxy_coords: np.ndarray,
    spec_coords: np.ndarray,
    *,
    line_core_index: Optional[int] = None,
    thermal_temperature: Optional[u.Quantity] = 8e4 * u.K,
) -> Dict[str, np.ndarray]:
    """Generate compact numeric arrays tailored for unit tests.

    The returned dictionary provides:

    ``along_slit``
        Two-row array of helioprojective coordinates for the first raster step.
    ``wavelength_shifts``
        Dispersion offsets relative to ``line_core_index`` (defaults to the
        spectral midpoint).
    ``fwhm_samples``
        Synthetic full-width-half-maximum samples derived from the local
        gradient of ``spec_coords``.
    ``thermal_velocities``
        Thermal velocity estimates converted to km/s using the supplied plasma
        temperature (defaults to 80 000 K).
    """

    if hpxy_coords.ndim != 4:
        raise ValueError("`hpxy_coords` must have shape (2, n_scan, n_meas, n_slit)")

    along_slit = hpxy_coords[:, 0, 0, :]

    if spec_coords.size == 0:
        raise ValueError("`spec_coords` must contain at least one element")

    if line_core_index is None:
        line_core_index = int(spec_coords.size // 2)

    line_core_wavelength = spec_coords[line_core_index]
    wavelength_shifts = spec_coords - line_core_wavelength

    # Estimate a stable FWHM surrogate from the local slope to keep values
    # deterministic even when the incoming spectra are noisy.
    gradient = np.gradient(spec_coords)
    fwhm_samples = np.abs(gradient) * 2.355

    # Convert thermal velocity for hydrogen at the supplied temperature.  Tests
    # that care about units can wrap this array with ``u.km / u.s``.
    if thermal_temperature is None:
        thermal_temperature = 8e4 * u.K

    thermal_velocity = np.sqrt(2 * const.k_B * thermal_temperature / const.m_p)
    thermal_velocities = np.full_like(spec_coords, thermal_velocity.to(u.km / u.s).value)

    return {
        "along_slit": along_slit,
        "wavelength_shifts": wavelength_shifts,
        "fwhm_samples": fwhm_samples,
        "thermal_velocities": thermal_velocities,
    }


# ---------------------------------------------------------------------------
# Time axis helpers
# ---------------------------------------------------------------------------


def build_timestamp_grid(
    headers: Sequence[Header],
    *,
    key: str = "DATE-AVG",
    sort: bool = True,
) -> np.ndarray:
    """Construct a monotonic ``numpy.datetime64`` array from FITS headers."""

    times: List[np.datetime64] = []
    for hdr in headers:
        if key not in hdr:
            raise KeyError(f"Header missing required time keyword: {key}")
        obstime = Time(hdr[key], format="isot", scale="utc")
        times.append(np.datetime64(obstime.to_datetime(), "ms"))

    time_array = np.array(times, dtype="datetime64[ms]")
    if sort:
        time_array = np.sort(time_array)
    return time_array


__all__ = [
    "extract_observation_headers",
    "generate_slit_coordinate_inputs",
    "generate_spectral_coordinate_input",
    "build_sample_numeric_arrays",
    "build_timestamp_grid",
]
