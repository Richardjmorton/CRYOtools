import glob
import os
from importlib import resources
from typing import Any, List, Optional, Tuple

import dkist
import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

from CRYOtools.util import return_obs_info


def load_asdf(dataset_directory: str) -> Optional[Any]:
    """Load the ASDF dataset associated with ``dataset_directory``.

    Parameters
    ----------
    dataset_directory:
        Folder containing the ``*.asdf`` file. The directory should end with a
        path separator so that :func:`glob.glob` can match files directly.

    Returns
    -------
    Optional[Any]
        The dataset returned by :func:`dkist.load_dataset`, or ``None`` when no
        (or multiple) ASDF files are present in the directory.
    """

    asdf_filename = glob.glob(dataset_directory + "*.asdf")

    if len(asdf_filename) == 1:
        dset = dkist.load_dataset(asdf_filename[0])
        print(f"Dataset data shape: {dset.data.shape}")
        return dset

    if len(asdf_filename) == 0:
        print("No files found")
    else:
        print("Multiple files found.")

    return None


def _read_solar_model(file: Optional[str] = None) -> np.ndarray:
    """Return the bundled solar reference spectrum as a NumPy array.

    Parameters
    ----------
    file:
        Optional override for the bundled filename.

    Returns
    -------
    np.ndarray
        Solar reference spectrum sampled on the model wavelength grid.
    """

    if not file:
        file = "solar_merged_20200720_600_33300_100.out"

    with resources.path(__package__ + ".models", file) as path:
        # Skip the header rows provided by the source model file.
        return np.loadtxt(path, skiprows=3)


def _read_telluric_model(file: Optional[str] = None) -> np.ndarray:
    """Return the bundled telluric absorption model as a NumPy array.

    Parameters
    ----------
    file:
        Optional override for the bundled filename.

    Returns
    -------
    np.ndarray
        Telluric transmission spectrum sampled on the model wavelength grid.
    """

    if not file:
        file = "hitran_1micron_h20_trans.npy"

    with resources.path(__package__ + ".models", file) as path:
        return np.load(path)


def find_L1_files(dataset_directory: str) -> Optional[List[str]]:
    """Return a sorted list of L1 FITS files located in ``dataset_directory``.

    Parameters
    ----------
    dataset_directory:
        Path prefix pointing to the directory of FITS products. The directory
        should include a trailing separator when concatenating.

    Returns
    -------
    Optional[List[str]]
        List of matching filenames, or ``None`` when none are found.
    """

    L1_filenames = glob.glob(dataset_directory + "*L1.fits")
    L1_filenames.sort()

    if not L1_filenames:
        print("No L1 FITS files found in the directory.")
        return None

    return L1_filenames


def _get_all_head(file: str) -> Header:
    """Load the FITS header from extension 1 for a single file.

    Parameters
    ----------
    file:
        Path to the FITS file.

    Returns
    -------
    Header
        Header for the science extension used when loading data values.
    """

    return fits.getheader(file, ext=1)


def get_all_head(dataset_directory: str, cpu_max: int = 4) -> Optional[List[Header]]:
    """Load FITS headers for all L1 files in ``dataset_directory`` in parallel.

    Parameters
    ----------
    dataset_directory:
        Directory containing the DKIST L1 FITS products.
    cpu_max:
        Maximum number of worker threads to use when fetching headers.

    Returns
    -------
    Optional[List[Header]]
        List of headers corresponding to :func:`find_L1_files`, or ``None`` when
        no files are present.
    """

    L1_filenames = find_L1_files(dataset_directory)

    if not L1_filenames:
        return None

    ncpus = min(os.cpu_count(), cpu_max)

    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        hdrs = list(
            tqdm(
                executor.map(_get_all_head, L1_filenames),
                total=len(L1_filenames),
                desc="Getting headers",
            )
        )

    return hdrs


def restore_slit_coords(dataset_directory: str) -> List[np.ndarray]:
    """Load the slit coordinate arrays stored alongside the data.

    Parameters
    ----------
    dataset_directory:
        Directory that contains the ``*_coords.npy`` files.

    Returns
    -------
    List[np.ndarray]
        Coordinate arrays describing the slit location, time sampling and
        spectral sampling.
    """

    coord_list = ["hpxy_coords.npy", "time_coords.npy", "spec_coords.npy"]

    data: List[np.ndarray] = []
    for coord in coord_list:
        filepath = dataset_directory + coord

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            data.append(np.load(filepath))
        except Exception as exc:  # pragma: no cover - defensive exception context
            raise RuntimeError(f"Error loading file '{filepath}': {exc}") from exc

    return data


def rotate_stokes(data: np.ndarray, dataset_directory: str) -> np.ndarray:
    """Rotate Stokes vectors from the DKIST frame into the solar radial frame.

    Parameters
    ----------
    data:
        Stokes vector cube arranged as ``(stokes, scan, meas, slit, wavelength)``.
    dataset_directory:
        Directory that contains the slit coordinate files used to compute the
        rotation angle.

    Returns
    -------
    np.ndarray
        Rotated data cube with Stokes ``Q`` aligned with the solar radial
        direction.
    """

    hpxy_coords, _ = restore_slit_coords(dataset_directory)

    linpol_rotate_angle = np.arctan2(hpxy_coords[1, :, 0, :], hpxy_coords[0, :, 0, :])
    cos2_rot = np.cos(2.0 * linpol_rotate_angle)[:, None, :, None]
    sin2_rot = np.sin(2.0 * linpol_rotate_angle)[:, None, :, None]

    data_rotated = np.copy(data)
    # Apply the Mueller matrix rotation to the linear polarisation components.
    data_rotated[1] = cos2_rot * data[1] + sin2_rot * data[2]
    data_rotated[2] = -sin2_rot * data[1] + cos2_rot * data[2]

    return data_rotated


def _fits_mp(header_str: str, L1_filename: str, n_stokes: int) -> Tuple[int, int, List[np.ndarray]]:
    """Worker function used when reading data in parallel.

    Parameters
    ----------
    header_str:
        FITS header serialised to a string for pickling safety.
    L1_filename:
        Filename of the Stokes ``I`` FITS file.
    n_stokes:
        Number of Stokes parameters to load.

    Returns
    -------
    Tuple[int, int, List[np.ndarray]]
        The current scan index, measurement index and the loaded images.
    """

    header = Header.fromstring(header_str, sep="\n")

    images: List[np.ndarray] = []
    images.append(fits.getdata(L1_filename, ext=1).squeeze())

    if n_stokes > 1:
        images.append(fits.getdata(L1_filename.replace("_I_", "_Q_"), ext=1).squeeze())
        images.append(fits.getdata(L1_filename.replace("_I_", "_U_"), ext=1).squeeze())
        images.append(fits.getdata(L1_filename.replace("_I_", "_V_"), ext=1).squeeze())

    return header["CNCURSCN"], header["CNCMEAS"], images


def _unpack_and_run(args: Tuple[str, str, int]) -> Tuple[int, int, List[np.ndarray]]:
    """Bridge function to unpack arguments for :func:`_fits_mp`.

    Parameters
    ----------
    args:
        Tuple containing the arguments for :func:`_fits_mp`.

    Returns
    -------
    Tuple[int, int, List[np.ndarray]]
        Result of invoking :func:`_fits_mp` with the supplied arguments.
    """

    return _fits_mp(*args)


def get_fits_data(
    dataset_directory: str,
    n_stokes: int = 1,
    cpu_max: int = 4,
    file_start: Optional[int] = None,
    file_end: Optional[int] = None,
    convert_phot: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Read Cryo FITS files from ``dataset_directory`` using multiprocessing.

    Parameters
    ----------
    dataset_directory:
        Directory containing the DKIST L1 FITS products.
    n_stokes:
        Number of Stokes parameters to load (1 for intensity only, 4 for IQUV).
    cpu_max:
        Upper bound on the number of worker processes used to read data.
    file_start:
        Index of the first file to read. Values greater than zero skip early
        files in the scan sequence.
    file_end:
        Exclusive index of the last file to read, allowing a truncated load.
    convert_phot:
        When ``True`` scale the data to parts-per-million of the disk-center
        intensity.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        The loaded data cube and the list of FITS files that contributed to it.
    """

    hdrs = get_all_head(dataset_directory)
    files = find_L1_files(dataset_directory)

    if hdrs is None or files is None:
        raise FileNotFoundError("No FITS headers or files were located in the directory.")

    if file_end is not None:
        files = files[:file_end]
        hdrs = hdrs[:file_end]

    file_start = 0 if file_start is None else file_start
    if file_start:
        files = files[file_start:]
        hdrs = hdrs[file_start:]

    n_scan_steps, n_meas_at_step, n_wv, n_along_slit = return_obs_info(hdrs)

    # Pre-allocate the data cube. For scanning observations we keep a single
    # measurement slot because repeats will be co-added, whereas sit-and-stare
    # observations retain the measurement dimension.
    if n_scan_steps > 1:
        data = np.zeros((n_stokes, n_scan_steps, 1, n_along_slit, n_wv), dtype=float)
        coadd_index = np.zeros(n_scan_steps)
    else:
        data = np.zeros(
            (n_stokes, 1, n_meas_at_step - file_start, n_along_slit, n_wv),
            dtype=float,
        )
        coadd_index = np.zeros(n_meas_at_step)

    print("Defined data set shape to load: ", data.shape)

    # Serialize headers to strings so they are safely picklable across processes.
    tasks = [(hdr.tostring(sep="\n"), file, n_stokes) for hdr, file in zip(hdrs, files)]

    ncpus = min(os.cpu_count(), cpu_max)

    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        results = list(
            tqdm(executor.map(_unpack_and_run, tasks), total=len(files), desc="Reading in data")
        )

        for current_scan, current_meas, images in results:
            if n_scan_steps > 1:
                data[:, current_scan - 1, 0, :, :] += np.array(images)
                coadd_index[current_scan - 1] += 1
            else:
                data[:, 0, current_meas - 1 - file_start, :, :] = np.array(images)

    if n_scan_steps > 1:
        # Normalise the co-added images by the number of measurements.
        data = data / coadd_index[None, :, None, None, None]

    if convert_phot:
        data = data * 1e6

    if n_stokes > 1:
        data = rotate_stokes(data, dataset_directory)

    return data, files


def write_model_results(folder: str, file: str, res: np.ndarray) -> str:
    """Persist model results to ``folder/spectrum_fits``.

    Parameters
    ----------
    folder:
        Output directory base path.
    file:
        L1 FITS filename whose results are being saved.
    res:
        Model fit results to serialise via :func:`numpy.savez_compressed`.

    Returns
    -------
    str
        Path to the saved ``.npz`` archive.
    """

    output_directory = os.path.join(folder, "spectrum_fits")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = os.path.splitext(os.path.basename(file))[0]
    save_file = os.path.join(output_directory, f"{file_name}.npz")
    np.savez_compressed(save_file, res=res)

    return save_file
