import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import glob
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jacobian
from jax.scipy.signal import correlate

import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.signal import correlate as corr_sp
from scipy.signal import correlation_lags

from concurrent.futures import ProcessPoolExecutor

from scipy.optimize import OptimizeResult

from tqdm.auto import tqdm

from CRYOtools.io import _read_solar_model, _read_telluric_model


@jax.jit
def gaussian_filter_1d(input_array: jnp.ndarray, sigma: float = 2.0) -> jnp.ndarray:
    """Apply a 1-D Gaussian convolution to the supplied signal.

    Parameters
    ----------
    input_array
        One-dimensional signal to smooth. The function assumes evenly spaced
        samples.
    sigma
        Width of the Gaussian kernel expressed in pixels. Larger values
        increase the smoothing.

    Returns
    -------
    jax.numpy.ndarray
        Smoothed copy of ``input_array`` with edge handling performed by
        mirroring the signal.
    """

    # make the radius of the filter equal to truncate standard deviations
    radius = 30  # jnp.array(4 * sigma + 0.5, jnp.int32)
    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    signal_ext = jnp.array([input_array[::-1], input_array, input_array[::-1]])
    ln = len(input_array)
    smooth = correlate(signal_ext.reshape(3 * ln), phi_x[::-1], mode="same")

    return smooth[ln : 2 * ln]

@jax.jit
def gaussian(x: jnp.ndarray, amplitude: float, mean: float, sigma: float) -> jnp.ndarray:
    """Evaluate a Gaussian profile.

    Parameters
    ----------
    x
        Coordinates at which to evaluate the profile.
    amplitude
        Peak amplitude of the Gaussian.
    mean
        Central value of the distribution.
    sigma
        Standard deviation of the Gaussian in the same units as ``x``.

    Returns
    -------
    jax.numpy.ndarray
        Gaussian profile sampled at ``x``.
    """

    expo = (x - mean) / sigma
    return amplitude * jnp.exp(-(expo**2) / 2)


@jax.jit
def fft_shift(input_array: jnp.ndarray, shift: float) -> jnp.ndarray:
    """Apply a sub-pixel shift using the Fourier shift theorem.

    Parameters
    ----------
    input_array
        Signal to shift in the frequency domain.
    shift
        Requested translation in pixels. Positive values shift the signal
        toward higher indices.

    Returns
    -------
    jax.numpy.ndarray
        Shifted signal in the spatial domain.
    """

    N = input_array.shape[0]
    x = jnp.arange(-N / 2, N / 2)
    kx = -1j * 2 * np.pi * x / N
    finput = jnp.fft.fftshift(jnp.fft.fftn(input_array))
    shifted_finput = finput * jnp.exp(-(kx * shift))
    shifted_input = jnp.real(jnp.fft.ifftn(jnp.fft.ifftshift(shifted_finput)))
    return shifted_input
    


def do_conv(x: jnp.ndarray, y: jnp.ndarray, Rpow_log: float) -> jnp.ndarray:
    """Convolve ``y`` with a Gaussian kernel derived from the resolving power.

    Parameters
    ----------
    x
        Wavelength grid corresponding to ``y``.
    y
        Spectrum to be convolved.
    Rpow_log
        Natural logarithm of the assumed resolving power.

    Returns
    -------
    jax.numpy.ndarray
        Smoothed spectrum sampled on ``x``.
    """

    fwhm_wv = x.mean() / jnp.exp(Rpow_log)
    sigm_wv = fwhm_wv / (2.0 * jnp.sqrt(2.0 * jnp.log(2)))
    dwv = x[0] - x[1]
    kern_pix = sigm_wv / jnp.abs(dwv)
    y_conv = gaussian_filter_1d(y, sigma=kern_pix)
    return y_conv


def standard(x: jnp.ndarray) -> jnp.ndarray:
    """Return the z-score normalisation of ``x``.

    Parameters
    ----------
    x
        Input samples to normalise.

    Returns
    -------
    jax.numpy.ndarray
        Normalised values with zero mean and unit variance.
    """

    return (x - x.mean()) / x.std()


def parabola(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Return the extremum of the parabola passing through three points.

    Parameters
    ----------
    x, y
        Abscissa and ordinate samples of three adjacent points.

    Returns
    -------
    float
        Estimated sub-pixel shift corresponding to the parabola minimum.
    """

    return x[2] - (y[2] - y[1]) / (y[2] - 2.0 * y[1] + y[0]) - 0.5


def get_lags(
    data: jnp.ndarray,
    atlas: jnp.ndarray,
    spec_coords: jnp.ndarray,
    Solar: bool = True,
) -> float:
    """Estimate the pixel shift between ``data`` and a reference atlas.

    Parameters
    ----------
    data
        Observed spectrum sampled on ``spec_coords``.
    atlas
        Reference spectrum to correlate against ``data``.
    spec_coords
        Wavelength coordinates corresponding to both spectra.
    Solar
        When ``True`` correlate against the solar atlas window. Otherwise the
        telluric window is used.

    Returns
    -------
    float
        Sub-pixel lag (in pixels) that maximises the cross-correlation.
    """

    if Solar:
        index = np.where((spec_coords > 1074.85) & (spec_coords < 1075.05))[0]
        atlas_cp = deepcopy(atlas)
    else:
        index = np.where((spec_coords > 1074.27) & (spec_coords < 1074.4))[0]
        # telluric profiles narrow, so broaden by a reasonable amount
        atlas_cp = do_conv(spec_coords, atlas * np.median(data), 10.7)

    x1 = data[index]
    x2 = atlas_cp[index]
    corr = corr_sp(standard(x1), standard(x2), "same", method="fft")
    lags = correlation_lags(x1.size, x2.size, "same")

    # get three points around maxima
    max_loc = np.argmax(corr)
    locs = np.array([-1, 0, 1]) + max_loc
    # get sub-pixel estimate of shift
    shift = parabola(lags[locs], corr[locs])
    return -shift


def get_lags_lin(
    data: jnp.ndarray,
    atlas: jnp.ndarray,
    spec_coords: jnp.ndarray,
    Solar: bool = True,
) -> float:
    """Cross-correlate after removing a linear background trend.

    Parameters
    ----------
    data, atlas, spec_coords, Solar
        See :func:`get_lags`. A linear background is removed from ``data`` prior
        to correlation to reduce continuum bias.

    Returns
    -------
    float
        Sub-pixel lag (in pixels) after background correction.
    """

    if Solar:
        index = np.where((spec_coords > 1074.85) & (spec_coords < 1075.05))[0]
        atlas_cp = deepcopy(atlas)
    else:
        index = np.where((spec_coords > 1074.27) & (spec_coords < 1074.4))[0]
        # telluric profiles narrow, so broaden by a reasonable amount
        atlas_cp = do_conv(spec_coords, atlas * np.median(data), 10.7)

    # estimate and subtract linear function
    ind_1074 = np.argmin(spec_coords - 1074)
    y_est = data[ind_1074]
    c = y_est - 0.25 * 1074

    x1 = data[index] - (0.25 * spec_coords[index] + c)
    x2 = atlas_cp[index]
    corr = corr_sp(standard(x1), standard(x2), "same", method="fft")
    lags = correlation_lags(x1.size, x2.size, "same")

    # get three points around maxima
    max_loc = np.argmax(corr)
    locs = np.array([-1, 0, 1]) + max_loc
    # get sub-pixel estimate of shift
    shift = parabola(lags[locs], corr[locs])
    return -shift


def vac2air(wave_vac: np.ndarray) -> np.ndarray:
    """Convert wavelengths from vacuum (Å) to air (Å).

    Parameters
    ----------
    wave_vac
        Wavelength grid in vacuum Angstroms.

    Returns
    -------
    numpy.ndarray
        Converted wavelengths using the standard refractive index relation.
    """

    wave_air = np.copy(wave_vac)
    ww = wave_vac >= 2000
    sigma2 = (1e4 / wave_vac[ww]) ** 2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - sigma2) + 0.00015998 / (38.9 - sigma2)
    wave_air[ww] = wave_vac[ww] / n
    return wave_air


def get_solar_model(wave_len: np.ndarray) -> np.ndarray:
    """Sample the bundled solar atlas on the requested wavelength grid.

    Parameters
    ----------
    wave_len
        Target wavelength coordinates (in nm) for interpolation.

    Returns
    -------
    numpy.ndarray
        Solar reference spectrum resampled onto ``wave_len``.
    """

    dc = _read_solar_model()

    dcwv = vac2air(1e7 / dc[:, 0] * 10)
    dcsp = dc[:, 1]

    ftscor = np.interp(wave_len * 10, dcwv[::-1], dcsp[::-1])
    return ftscor


def get_telluric_model(wave_len: np.ndarray) -> np.ndarray:
    """Sample the bundled HITRAN telluric transmission model.

    Parameters
    ----------
    wave_len
        Target wavelength coordinates (in nm) for interpolation.

    Returns
    -------
    numpy.ndarray
        Telluric transmission values sampled on ``wave_len``.
    """

    # Load HITRAN MODEL TELLURIC SPECTRUM FOR WATER
    hdat = _read_telluric_model()
    hwv = np.flip(hdat[0])
    hitran = np.flip(hdat[1])
    ftsatm = np.interp(wave_len, hwv, hitran)
    return ftsatm
          

def fit_slit(
    img: jnp.ndarray,
    spec_coords: jnp.ndarray,
    do_diff: int = 0,
    wgts: Optional[jnp.ndarray] = None,
    use_tqdm: bool = False,
) -> List[OptimizeResult]:
    """Fit each spatial pixel in ``img`` with the forward model.

    Parameters
    ----------
    img
        Two-dimensional array with shape ``(slit_len, wavelength_len)``.
    spec_coords
        Wavelength coordinates corresponding to ``img``.
    do_diff
        Interval for re-running the global differential evolution search.
        ``0`` triggers the search on every pixel (default behaviour).
    wgts
        Optional weights applied to the residual when evaluating the loss
        function. When omitted unit weights are assumed.
    use_tqdm
        If ``True`` report progress with :mod:`tqdm`.

    Returns
    -------
    list[scipy.optimize.OptimizeResult]
        Result of :func:`scipy.optimize.minimize` for each spatial pixel along
        the slit.
    """

    res_slits: List[OptimizeResult] = []
    n_rows = img.shape[0]

    if do_diff == 0:
        do_diff = n_rows

    fts_cor = get_solar_model(spec_coords)
    fts_atm = get_telluric_model(spec_coords)
    log_fts_atm = jnp.log(fts_atm)
    bounds = create_bounds()

    if wgts is None:
        wgts = jnp.ones(n_rows, jnp.float64)

    iterator = tqdm(img, desc="Processing", disable=not use_tqdm)
    for i, y in enumerate(iterator):

        velS_est = get_lags(y, fts_cor, Solar=True)
        velT_est = get_lags(y, fts_atm, Solar=False)
        bounds[5] = (velS_est - 0.2, velS_est + 0.2)
        bounds[6] = (velT_est - 0.2, velT_est + 0.2)
        bounds[8] = (np.nanmedian(y) - 10, np.nanmedian(y) + 10)

        if i % do_diff == 0:
            res = differential_evolution(
                loss,
                bounds,
                args=(spec_coords, y, wgts),
                tol=1.0e-2,
                maxiter=800,
                popsize=1,
            )
        res = minimize(
            loss,
            res.x,
            args=(spec_coords, y, wgts),
            jac=jac_loss,
            method="BFGS",
        )
        res_slits.append(res)

    return res_slits


class fit_data:
    """Convenience wrapper that fits every row in a Cryo slit image."""

    def __init__(self, data: np.ndarray, spec_coords: np.ndarray, do_diff: int = 0) -> None:
        """Store the dataset to be fitted and configure optimisation cadence."""

        self.data = data
        self.spec_coords = spec_coords
        self.n_along_slit = data.shape[0]
        self.n_wv = data.shape[1]
        self.bounds: Optional[List[Tuple[float, float]]] = None

        if do_diff == 0:
            self.do_diff = self.n_along_slit
        else:
            self.do_diff = do_diff

    def create_bounds(self, bounds: Optional[List[Tuple[float, float]]] = None) -> None:
        """Define parameter bounds for the optimisation.

        Parameters
        ----------
        bounds
            Optional set of bounds. When omitted the canonical values described
            in the Cryo fitting notebooks are used.
        """

        if bounds is None:
            line_amp = (0, 50)
            del_lam = (
                1074.63 - 6 / 3e5 * 1074.63 - 1074.0,
                1074.63 + 6 / 3e5 * 1074.63 - 1074.0,
            )
            sigma = (0.055, 0.1)
            rpow_log = (np.log(30000), np.log(65000))
            opac = (0.5, 10)
            vel_solar = (-2, 2)
            vel_tell = (-3.2, -2.5)
            strayfrac = (0.0, 0.5)
            icont = (0, 0)
            icont_lin = (0, 1)

            bounds = [
                line_amp,
                del_lam,
                sigma,
                rpow_log,
                opac,
                vel_solar,
                vel_tell,
                strayfrac,
                icont,
                icont_lin,
            ]

        self.bounds = bounds

    def build_model(self, params: Sequence[float]) -> jnp.ndarray:
        """Evaluate the forward model using the provided parameter vector."""

        (
            amp,
            lam_0,
            sigma,
            Rpow_log,
            opac,
            velS,
            velT,
            strayfrac,
            icont,
            icont_lin,
        ) = params

        # ifit -- coronal line
        gfit = gaussian(self.spec_coords - 1074.0, amp, lam_0, sigma)

        ftsSmod = jnp.copy(self.fts_cor)
        # shift coronal data
        ftsSmod = fft_shift(ftsSmod, velS)

        # scale and shift telluric spectra
        ftsTmod = jnp.exp(opac * self.log_fts_atm)
        ftsTmod = fft_shift(ftsTmod, velT)

        ftsmod = ftsSmod * ftsTmod

        # add straylight
        ftsmod = (ftsmod + strayfrac) / (1.0 + strayfrac)
        # scale for total
        ftsmod = ftsmod * (icont + icont_lin * self.spec_coords)

        # note that telluric absorption is also applied to the coronal line
        ifit = ftsmod + gfit * ftsTmod

        # convolution for spectrograph line spread function
        # Gaussian convolution of the FTS atlas
        fwhm_wv = self.spec_coords.mean() / jnp.exp(Rpow_log)
        sigm_wv = fwhm_wv / (2.0 * jnp.sqrt(2.0 * jnp.log(2)))
        dwv = self.spec_coords[0] - self.spec_coords[1]
        kern_pix = sigm_wv / jnp.abs(dwv)
        ifit = gaussian_filter_1d(ifit, sigma=kern_pix)

        return ifit

    def loss(self, params: Sequence[float], y: jnp.ndarray) -> float:
        """Return the weighted mean squared error for a candidate parameter set."""

        y_hat = self.build_model(params)
        return float(jnp.mean((y_hat - y) ** 2 * self.wgts))

    def fit_slit(
        self,
        wgts: Optional[jnp.ndarray] = None,
        use_tqdm: bool = False,
        min_tol: float = 1e-4,
    ) -> List[OptimizeResult]:
        """Fit each spatial position in :attr:`data`.

        Parameters
        ----------
        wgts
            Optional per-wavelength weights. Unit weights are assumed when
            ``None``.
        use_tqdm
            When ``True`` display a progress bar while fitting the slit.
        min_tol
            Convergence tolerance passed to :func:`scipy.optimize.minimize`.

        Returns
        -------
        list[scipy.optimize.OptimizeResult]
            Result objects for each slit position.
        """

        res_slits: List[OptimizeResult] = []

        # some parameters are made explicit to work with JAX functions
        fts_cor = get_solar_model(self.spec_coords)
        self.fts_cor = fts_cor
        fts_atm = get_telluric_model(self.spec_coords)
        self.fts_atm = fts_atm
        log_fts_atm = jnp.log(fts_atm)
        self.log_fts_atm = log_fts_atm

        if self.bounds is None:
            self.create_bounds()

        if wgts is None:
            self.wgts = jnp.ones(self.n_wv, jnp.float64)
            wgts = self.wgts
        else:
            self.wgts = wgts

        @jax.jit
        def build_model_int(params: Sequence[float], x: jnp.ndarray) -> jnp.ndarray:
            (
                amp,
                lam_0,
                sigma,
                Rpow_log,
                opac,
                velS,
                velT,
                strayfrac,
                icont,
                icont_lin,
            ) = params

            # ifit -- coronal line
            gfit = gaussian(x - 1074.0, amp, lam_0, sigma)

            ftsSmod = jnp.copy(fts_cor)
            # shift coronal data
            ftsSmod = fft_shift(ftsSmod, velS)

            # scale and shift telluric spectra
            ftsTmod = jnp.exp(opac * log_fts_atm)
            ftsTmod = fft_shift(ftsTmod, velT)

            ftsmod = ftsSmod * ftsTmod

            # add straylight
            ftsmod = (ftsmod + strayfrac) / (1.0 + strayfrac)
            # scale for total
            ftsmod = ftsmod * (icont + icont_lin * x)

            # note that telluric absorption is also applied to the coronal line
            ifit = ftsmod + gfit * ftsTmod

            # convolution for spectrograph line spread function
            # Gaussian convolution of the FTS atlas
            fwhm_wv = x.mean() / jnp.exp(Rpow_log)
            sigm_wv = fwhm_wv / (2.0 * jnp.sqrt(2.0 * jnp.log(2)))
            dwv = x[0] - x[1]
            kern_pix = sigm_wv / jnp.abs(dwv)
            ifit = gaussian_filter_1d(ifit, sigma=kern_pix)

            return ifit

        @jax.jit
        def loss_int(params: Sequence[float], y: jnp.ndarray, x: jnp.ndarray, wgts: jnp.ndarray) -> float:
            y_hat = build_model_int(params, x)
            return jnp.mean((y_hat - y) ** 2 * wgts)

        jac_loss = jax.jit(jacobian(loss_int))

        x = self.spec_coords
        iterator = tqdm(range(self.n_along_slit), desc="Processing", disable=not use_tqdm)

        for i in iterator:
            y = self.data[i]

            if i % self.do_diff == 0:
                velS_est = get_lags_lin(y, fts_cor, x, Solar=True)
                velT_est = get_lags_lin(y, fts_atm, x, Solar=False)
                assert self.bounds is not None  # for type checkers
                self.bounds[5] = (velS_est - 0.5, velS_est + 0.5)
                self.bounds[6] = (velT_est - 0.5, velT_est + 0.5)
                # NOTE: Review whether the constant bound limits remain appropriate.
                self.bounds[8] = (-1000, -200)
                res = differential_evolution(
                    loss_int,
                    self.bounds,
                    args=(y, x, wgts),
                    tol=1.0e-2,
                    maxiter=800,
                    popsize=1,
                )
            res = minimize(
                loss_int,
                res.x,
                args=(y, x, wgts),
                jac=jac_loss,
                method="BFGS",
                tol=min_tol,
            )
            res_slits.append(res)

        self.res_slits = res_slits
        return self.res_slits

    def calculate_model(self) -> List[jnp.ndarray]:
        """Return the forward-model profiles for the fitted results.

        Notes
        -----
        When no optimisation has been performed a singleton list containing an
        empty array is returned for review.
        """

        if hasattr(self, "res_slits"):
            return [self.build_model(r.x) for r in self.res_slits]
        return [jnp.array([])]

    def plot_model_vs_data(self, vmin: float = -1.0, vmax: float = 1.0) -> None:
        """Plot the observed data, fitted model, and residuals."""

        self.model = self.calculate_model()

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(self.data)
        ax[0].set_title("Data")
        ax[1].imshow(self.model)
        ax[1].set_title("Model")
        pl = ax[2].imshow(self.data - self.model, vmin=vmin, vmax=vmax)
        # plt.colorbar(pl)
        ax[2].set_title("Residual \n clipped {0} to {1}".format((vmin, vmax)))
        plt.tight_layout()

def _pull_fit_res(file: str) -> List[List[Any]]:
    """Load a saved optimisation result from ``file``.

    Parameters
    ----------
    file
        Path to the ``.npz`` archive containing an array of optimisation
        results.

    Returns
    -------
    list[list[Any]]
        Optimiser parameter vectors and merit function values for each slit
        position.
    """

    dat = np.load(file, allow_pickle=True)

    return [[m.x, m.fun] for m in dat["res"]]


def pull_fit_res(dataset_directory: str, cpu_max: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate all stored fit results into contiguous arrays.

    Parameters
    ----------
    dataset_directory
        Root directory that contains the ``spectrum_fits`` outputs.
    cpu_max
        Maximum number of workers used when reading the ``.npz`` files.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The stacked fit parameters with shape ``(n_params, n_files, n_slits)``
        and the corresponding merit function values.
    """

    fit_directory = dataset_directory + "spectrum_fits/"

    file_list = glob.glob(fit_directory + "*npz")
    file_list.sort()

    ncpus = min(os.cpu_count(), cpu_max)

    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        results = list(
            tqdm(executor.map(_pull_fit_res, file_list), total=len(file_list), desc="Getting fit data")
        )

    fit_results: List[List[np.ndarray]] = []
    merit_results: List[List[float]] = []
    for res in results:
        fit_results.append([m[0] for m in res])
        merit_results.append([float(m[1]) for m in res])

    fit_results_arr = np.array(fit_results)
    merit_results_arr = np.array(merit_results)
    return np.transpose(fit_results_arr, [2, 0, 1]), merit_results_arr

