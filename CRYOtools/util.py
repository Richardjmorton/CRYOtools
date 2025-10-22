import os

import numpy as np

from astropy.time import Time
from astropy.io.fits import Header

from astropy.wcs import WCS
import astropy.units as u
import astropy.constants as const

import warnings

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

from tqdm import tqdm


def return_obs_info(hdrs: Sequence[Header], verbose: bool = True) -> Tuple[int, int, int, int]:
    """Return key observation dimensions from a sequence of FITS headers.

    Parameters
    ----------
    hdrs
        Sequence of FITS headers describing each exposure.
    verbose
        If ``True`` print diagnostic scan information to stdout.

    Returns
    -------
    tuple[int, int, int, int]
        Number of scan steps, measurements per step, wavelength samples, and
        spatial pixels along the slit.
    """

    if verbose and hdrs:
        # Show information on 1st and 2nd direction scanning (step size and number)
        diag_prefixes = ("CNP1DSS", "CNP1DNSP", "CNP2DSS", "CNP2DNSP")
        first_header = hdrs[0]
        for prefix in diag_prefixes:
            matching_keys = [key for key in first_header.keys() if key.startswith(prefix)]
            for key in matching_keys:
                print(f"{key}: {first_header[key]}")


    n_scan_steps = np.max(np.array([hdr['CNCURSCN'] for hdr in hdrs]))
    n_meas_at_step = np.max(np.array([hdr['CNCMEAS'] for hdr in hdrs]))

    n_wv = hdrs[0]['NAXIS1']
    n_along_slit = hdrs[0]['NAXIS2']

    return n_scan_steps, n_meas_at_step, n_wv, n_along_slit


def get_slit_coords(
    hdrs: Sequence[Header],
    output_dir: Union[str, os.PathLike] = './outputs/',
    saver: Optional[Callable[[str, np.ndarray], None]] = np.save,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive slit spatial coordinates and acquisition times from FITS headers.

    Parameters
    ----------
    hdrs
        Sequence of FITS headers describing each exposure.
    output_dir
        Directory where derived coordinate arrays should be stored when ``saver``
        is provided.
    saver
        Callable invoked with ``(path, array)`` to persist outputs. Pass ``None``
        to disable writing, which is useful for tests.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``hpxy_coords`` with shape ``(2, n_scan_steps, n_meas_at_step, slit_len)``
        and ``time_coords`` with shape ``(n_scan_steps, n_meas_at_step)``.
    """
    n_scan_steps = np.max(np.array([hdr['CNCURSCN'] for hdr in hdrs]))
    n_meas_at_step = np.max(np.array([hdr['CNCMEAS'] for hdr in hdrs]))

    n_alongSlit = hdrs[0]['NAXIS2']
    hpxy_coords = np.zeros((2,n_scan_steps, n_meas_at_step,n_alongSlit))
    time_coords = np.zeros((n_scan_steps, n_meas_at_step),dtype ='datetime64[ms]')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") ## TO ELIMINATE datafix warnings
        for hd in hdrs:
            wcs = WCS(hd)
            xy = wcs.array_index_to_world(0,np.arange(n_alongSlit),0)[1]
            # Extract world coordinates (Tx, Ty) and observation time along the slit.
            x,y,obstime = xy.Tx.value,xy.Ty.value,xy[0].obstime
            CNCURSCN,CNCMEAS = hd['CNCURSCN'],hd['CNCMEAS']
            hpxy_coords[0,CNCURSCN-1,CNCMEAS-1,:] = x
            hpxy_coords[1,CNCURSCN-1,CNCMEAS-1,:] = y
            time_coords[CNCURSCN-1,CNCMEAS-1] = obstime.to_datetime()

    ## Save the dataset coordinates to the location of your choice (defaults to working directory)
    if saver is not None:
        output_dir_path = os.fspath(output_dir)
        ensure_directory(output_dir_path)
        saver(os.path.join(output_dir_path, 'hpxy_coords.npy'), hpxy_coords)
        saver(os.path.join(output_dir_path, 'time_coords.npy'), time_coords)

    print(f"Helioprojective XY Coordinates Shape: {hpxy_coords.shape}")
    print(f"Datetime Coordinates Shape: {time_coords.shape}")

    return hpxy_coords, time_coords

def get_spectral_coords(
    hdrs: Sequence[Header],
    output_dir: Union[str, os.PathLike] = './outputs/',
    saver: Optional[Callable[[str, np.ndarray], None]] = np.save,
) -> np.ndarray:
    """Compute the spectral dispersion axis in nanometers using WCS tools.

    Parameters
    ----------
    hdrs
        Sequence of FITS headers describing each exposure.
    output_dir
        Directory where derived spectral coordinates should be stored when
        ``saver`` is provided.
    saver
        Callable invoked with ``(path, array)`` to persist outputs. Pass ``None``
        to disable writing, which is useful for tests.

    Returns
    -------
    numpy.ndarray
        Array of spectral coordinates (in nm) corresponding to detector pixels.

    Notes
    -----
    The dispersion axis is not strictly linear (``CTYPE1`` is ``AWAV-GRA``), so the
    WCS transformation is used to produce physically meaningful coordinates.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # TO ELIMINATE datafix warnings
        wcs = WCS(hdrs[0])
        nwv = hdrs[0]['NAXIS1']
        spec_coords = wcs.array_index_to_world(0,0,np.arange(nwv))[0].to(u.nm).value

    # Save the dataset coordinates (defaults to working directory)
    if saver is not None:
        output_dir_path = os.fspath(output_dir)
        ensure_directory(output_dir_path)
        saver(os.path.join(output_dir_path, 'spec_coords.npy'), spec_coords)
    print(f"Spectral Coordinates Shape: {spec_coords.shape}")

    return spec_coords

def get_slit_samp(
    hpxy_coords: np.ndarray,
    n_scan_steps: int,
    verbose: bool = True
) -> Tuple[float, float]:
    """Determine the raster slit step size and sampling along the slit.

    Parameters
    ----------
    hpxy_coords
        Array of helioprojective coordinates generated by :func:`get_slit_coords`.
    n_scan_steps
        Number of scan steps in the raster sequence.
    verbose
        If ``True`` print the derived sampling values to stdout.

    Returns
    -------
    tuple[float, float]
        Sampling along the slit (arcsec per pixel) and raster step size (arcsec).

    Notes
    -----
    The computed sampling incorporates any WCS ``PC_ij`` matrix transformations,
    providing an accurate physical spacing.
    """

    if hpxy_coords.shape[-1] < 2:
        raise ValueError("`hpxy_coords` must span at least two pixels along the slit.")

    base_vec = hpxy_coords[:, 0, 0, 1] - hpxy_coords[:, 0, 0, 0]
    slit_samp = np.linalg.norm(base_vec)

    if n_scan_steps <= 1:
        step_width = 0.0
    else:
        step_vec = hpxy_coords[:, 1, 0, 0] - hpxy_coords[:, 0, 0, 0]
        step_width = np.linalg.norm(step_vec)
    
    if verbose:
        print(f'Raster step size in arcsec: {step_width}')
        print(f'Sampling along the slit (arcsec per pixel): {slit_samp}')

    return slit_samp, step_width


def _get_sc_mp(
    header_str: str,
    n_alongSlit: int,
    i: int
) -> Tuple[int, int, np.ndarray, np.ndarray, datetime]:
    """Reconstruct WCS from a header string and extract slit coordinates.

    The ``i`` index is included to preserve ordering from :func:`enumerate` but is
    otherwise unused.
    """
    header = Header.fromstring(header_str,sep="\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # TO ELIMINATE datafix warnings
        wcs = WCS(header)
        xy = wcs.array_index_to_world(0, np.arange(n_alongSlit), 0)[1]
        x, y, obstime = xy.Tx.value, xy.Ty.value, xy[0].obstime
    return header['CNCURSCN'], header['CNCMEAS'], x, y, obstime.to_datetime()


def _unpack_and_run(args: Tuple[str, int, int]) -> Tuple[int, int, np.ndarray, np.ndarray, datetime]:
    """Unpack tuple arguments and invoke :func:`_get_sc_mp`."""
    return _get_sc_mp(*args)


def get_slit_coords_mp(
    hdrs: Sequence[Header],
    cpu_max: int =4,
    output_dir: Union[str, os.PathLike] ='./outputs/',
    saver: Optional[Callable[[str, np.ndarray], None]] = np.save,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract slit coordinates from FITS headers using multiprocessing.

    Parameters
    ----------
    hdrs
        Sequence of FITS headers describing each exposure.
    cpu_max
        Upper bound on the number of worker processes to spawn.
    output_dir
        Directory where derived coordinate arrays should be stored when ``saver``
        is provided.
    saver
        Callable invoked with ``(path, array)`` to persist outputs. Pass ``None``
        to disable writing, which is useful for tests.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Multiprocessed equivalents of the ``hpxy_coords`` and ``time_coords``
        arrays described in :func:`get_slit_coords`.
    """

    n_scanSteps = np.max(np.array([hdr['CNCURSCN'] for hdr in hdrs]))
    n_measAtStep = np.max(np.array([hdr['CNCMEAS'] for hdr in hdrs]))
    n_alongSlit = hdrs[0]['NAXIS2']
    hpxy_coords = np.zeros((2,n_scanSteps,n_measAtStep,n_alongSlit))
    time_coords = np.zeros((n_scanSteps,n_measAtStep),dtype ='datetime64[ms]')

    # Serialize headers to strings (fully picklable) 
    # Prepare list of (header_str, n_alongSlit) tuples
    tasks = [(hdr.tostring(sep="\n",padding=False, endcard=True), n_alongSlit,i) for i,hdr in enumerate(hdrs)]

    ncpus = min(os.cpu_count() , cpu_max)
    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        results = list(tqdm(executor.map(_unpack_and_run, tasks),
                                     total=len(hdrs), desc="Processing"))

        for res in results:
            CNCURSCN,CNCMEAS,x,y,obstime = res
            hpxy_coords[0,CNCURSCN-1,CNCMEAS-1,:] = x
            hpxy_coords[1,CNCURSCN-1,CNCMEAS-1,:] = y
            time_coords[CNCURSCN-1,CNCMEAS-1] = obstime
    
    # Save the dataset coordinates (defaults to working directory)
    if saver is not None:
        output_dir_path = os.fspath(output_dir)
        ensure_directory(output_dir_path)
        saver(os.path.join(output_dir_path, 'hpxy_coords.npy'), hpxy_coords)
        saver(os.path.join(output_dir_path, 'time_coords.npy'), time_coords)

    print(f"Helioprojective XY Coordinates Shape: {hpxy_coords.shape}")
    print(f"Datetime Coordinates Shape: {time_coords.shape}")

    return hpxy_coords, time_coords

def _flatten_time_axis(tc: np.ndarray) -> np.ndarray:
    """Return a flattened ``datetime64`` array preserving acquisition order."""

    flattened = np.asarray(tc).squeeze()
    if flattened.ndim == 0:
        flattened = flattened[None]
    return flattened


def _timedelta_to_seconds(deltas: Iterable[np.timedelta64]) -> np.ndarray:
    """Convert an iterable of ``numpy.timedelta64`` objects to seconds."""

    deltas = np.asarray(list(deltas), dtype='timedelta64[ns]')
    return deltas.astype(np.float64) * 1e-9


def calculate_cadence(tc: np.ndarray, plot_cad: bool = False) -> float:
    """Estimate the median cadence of observations in seconds.

    Parameters
    ----------
    tc
        Array of acquisition timestamps as produced by :func:`get_slit_coords`.
    plot_cad
        If ``True`` plot and save the cadence series using :mod:`matplotlib`.

    Returns
    -------
    float
        Median cadence in seconds between successive exposures.
    """
    flattened = _flatten_time_axis(tc)
    if flattened.size < 2:
        raise ValueError("Cadence calculation requires at least two timestamps.")

    diffs = np.diff(flattened)
    cad_seconds = _timedelta_to_seconds(diffs)
    med_cad = float(np.median(cad_seconds))

    if plot_cad:
        import matplotlib.pyplot as plt  # Local import to avoid mandatory dependency

        fig, ax = plt.subplots()
        ax.plot(cad_seconds, '.', label='Cadence (s)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Seconds')
        ax.legend()
        fig.savefig('cadence_values.png')
        plt.close(fig)

    return med_cad

def shift_to_v(
    delta_lam: np.ndarray,
    lam_0: float = 1074.7,
    lam_cor: float = 0.7
) -> u.Quantity:
    """Convert wavelength shifts to Doppler velocities in km/s.

    Parameters
    ----------
    delta_lam
        Array of measured wavelength offsets.
    lam_0
        Rest wavelength (nm) of the observed spectral line.
    lam_cor
        Empirical correction applied to the measured wavelength shifts.
    """
    c_km_s = const.c.to(u.km / u.s)
    # correction to measured shifts (from fitting procedure)
    corrected = (delta_lam - lam_cor) / lam_0
    F = corrected - 1
    return ((F ** 2 - 1) / (F ** 2 + 1)) * c_km_s  # Doppler formula


def calc_ntlw(
    data: np.ndarray,
    res_pow: float,
    wave: float = 1074.7,
    v_th: Union[float, Sequence[float], np.ndarray] = 21.0
) -> u.Quantity:
    """Calculate non-thermal line widths for a spectral line.

    Parameters
    ----------
    data
        Observed full-width-half-maximum (FWHM) values in nanometers.
    res_pow
        Instrument resolving power.
    wave
        Central wavelength (nm) of the observed spectral line.
    v_th
        Thermal velocity (km/s). May be scalar or broadcastable array.

    Returns
    -------
    astropy.units.Quantity
        Non-thermal line width expressed in km/s.
    """

    if np.ndim(v_th) == 0:
        v_th = np.full_like(data, v_th, dtype=float)
    else:
        v_th = np.asarray(v_th, dtype=float)
        if v_th.shape != data.shape:
            raise ValueError(
                "`v_th` must be either a scalar or the same shape as `data` "
                f"(got {v_th.shape}, expected {data.shape})"
            )

    v_th = v_th * u.km / u.s  # km/s
    fac = 2 * np.sqrt(np.log(2))
    w_i = wave * u.nm / res_pow  # nm
    scale = wave * u.nm / const.c.to(u.nm / u.s)

    fwhm = np.sqrt(2) * fac * data * u.nm  # nm
    ntlw = np.sqrt((fwhm ** 2 - w_i ** 2) / (scale ** 2 * fac ** 2) - v_th.to(u.nm / u.s) ** 2)

    return ntlw.to(u.km / u.s)


def ensure_directory(directory_path: Union[str, os.PathLike]) -> None:
    """Create *directory_path* if it does not already exist."""

    path = os.fspath(directory_path)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def print_exposure(hdrs: Sequence[Header]) -> None:
    """Pretty-print key exposure-related FITS header values.

    Parameters
    ----------
    hdrs
        Sequence of FITS headers; only the first header is used for printing.
    """
    exposureKeys = ['XPOSURE','TEXPOSUR','CAM_FPS','CNNSCI','CNNNDR','CNMODNST','CNNMEAS']
    for key in exposureKeys:
        if key not in hdrs[0]:
            continue
        comment = hdrs[0].comments.get(key, '')
        print(f"{key.ljust(10)} {comment.ljust(25)} {hdrs[0][key]} ")

    if 'CAM_FPS' in hdrs[0]:
        cadence_ms = 1000.0 / hdrs[0]['CAM_FPS']
        print(f"Time between ramps (i.e. triggers): {cadence_ms} msec")

    if 'DATE-AVG' in hdrs[0] and 'DATE-AVG' in hdrs[-1]:
        delta = Time(hdrs[-1]['DATE-AVG']).to_datetime() - Time(hdrs[0]['DATE-AVG']).to_datetime()
        print(f"Times between first and last trigger: {delta}")

