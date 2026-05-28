"""JAX-based spectral fitting pipeline for Cryo-NIRSP coronal lines.

Module-level side effects
-------------------------
Two import-time side effects are intentional but worth knowing about:

1. ``os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")``
   is applied so that XLA exposes multiple host devices for any
   future :func:`jax.pmap`-style usage. ``setdefault`` means a user-supplied
   ``XLA_FLAGS`` is **never** clobbered. The flag also only takes effect
   if it is set before JAX initialises, so import :mod:`CRYOtools.fit`
   before anything that imports JAX directly.
2. ``jax.config.update("jax_enable_x64", True)`` is required for the
   forward model to converge on narrow coronal lines; do not disable.

For explicit control, call :func:`configure` instead of relying on these
defaults.
"""

import os

# setdefault avoids clobbering a user-supplied XLA_FLAGS environment variable.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import functools
import glob
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.scipy.signal import correlate

import numpy as np

import optimistix as optx

import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.signal import correlate as corr_sp
from scipy.signal import correlation_lags

from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from CRYOtools.io import _read_solar_model, _read_telluric_model
from CRYOtools.config import (
    InstrumentConfig,
    LineConfig,
    CRYO_NIRSP,
    detect_line_config,
)


def configure(
    device_count: Optional[int] = None,
    enable_x64: bool = True,
) -> None:
    """Set JAX/XLA runtime options explicitly.

    Call this before importing JAX downstream if the import-time defaults
    are not what you want. ``XLA_FLAGS`` is only honoured by XLA if set
    before JAX initialises; if JAX has already been imported, changing
    ``device_count`` here will have no effect.

    Parameters
    ----------
    device_count
        Forces XLA to expose this many host devices via ``XLA_FLAGS``. If
        ``None``, leaves ``XLA_FLAGS`` unchanged.
    enable_x64
        Whether to enable 64-bit precision in JAX. Strongly recommended
        ``True`` for this package.
    """

    if device_count is not None:
        os.environ["XLA_FLAGS"] = (
            f"--xla_force_host_platform_device_count={int(device_count)}"
        )
    jax_config.update("jax_enable_x64", enable_x64)



@functools.partial(jax.jit, static_argnames=("radius",))
def gaussian_filter_1d(
    input_array: jnp.ndarray,
    sigma: float = 2.0,
    radius: int = 30,
) -> jnp.ndarray:
    """Apply a 1-D Gaussian convolution to the supplied signal.

    Parameters
    ----------
    input_array
        One-dimensional signal to smooth. The function assumes evenly spaced
        samples.
    sigma
        Width of the Gaussian kernel expressed in pixels. Larger values
        increase the smoothing.
    radius
        Half-width of the convolution kernel in pixels. Marked ``static`` so
        that the kernel array has a compile-time-known shape. Choose
        ``radius`` outside JIT via :func:`compute_lsf_radius` so it bounds
        the worst-case ``sigma`` produced under the configured
        ``Rpow_log`` bounds and ``spec_coords`` dispersion.

    Returns
    -------
    jax.numpy.ndarray
        Smoothed copy of ``input_array`` with edge handling performed by
        mirroring the signal.
    """

    sigma2 = sigma * sigma
    x = jnp.arange(-radius, radius + 1)
    phi_x = jnp.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    signal_ext = jnp.array([input_array[::-1], input_array, input_array[::-1]])
    ln = len(input_array)
    smooth = correlate(signal_ext.reshape(3 * ln), phi_x[::-1], mode="same")

    return smooth[ln : 2 * ln]


def compute_lsf_radius(
    spec_coords: np.ndarray,
    rpow_log_min: float,
    safety_factor: float = 4.5,
    minimum: int = 30,
) -> int:
    """Return a static kernel radius (pixels) for :func:`gaussian_filter_1d`.

    Computed outside JIT so that the radius is a Python ``int`` known at
    trace time. Sized to comfortably contain the worst-case (widest) LSF
    sigma produced by the configured lower bound on ``Rpow_log`` at the
    sampling implied by ``spec_coords``.

    Parameters
    ----------
    spec_coords
        Wavelength axis (nm). Only the dispersion ``|spec_coords[1] -
        spec_coords[0]|`` and the mean wavelength are used.
    rpow_log_min
        Lower bound of ``log(R)`` (natural log of the resolving power).
        Lower R \u2192 wider LSF \u2192 larger required radius.
    safety_factor
        Multiplier on the worst-case sigma. ``4.5`` captures > 99.999% of
        a Gaussian; the kernel weight outside is negligible.
    minimum
        Floor on the returned radius. Defaults to 30 so we never go below
        the previous hard-coded value.

    Returns
    -------
    int
        Radius in pixels suitable to pass as the ``radius=`` argument of
        :func:`gaussian_filter_1d`.
    """

    spec_coords = np.asarray(spec_coords)
    dwv = abs(float(spec_coords[1] - spec_coords[0]))
    wave_mean = float(spec_coords.mean())
    fwhm_wv = wave_mean / float(np.exp(rpow_log_min))
    sigma_wv = fwhm_wv / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pix = sigma_wv / dwv
    return max(minimum, int(np.ceil(safety_factor * sigma_pix)))


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
    # Use the positive dispersion magnitude so the variable name matches its value.
    dwv = jnp.abs(x[1] - x[0])
    kern_pix = sigm_wv / dwv
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
    line_config: LineConfig,
    instrument_config: InstrumentConfig = CRYO_NIRSP,
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
    line_config
        Line-specific configuration. The ``solar_window`` and
        ``telluric_window`` fields are used as the cross-correlation
        bandpass.
    instrument_config
        Instrument-specific configuration. The ``nominal_log_rpow`` field
        is used to pre-broaden the telluric atlas before correlation.
    Solar
        When ``True`` correlate against the solar atlas window. Otherwise the
        telluric window is used.

    Returns
    -------
    float
        Sub-pixel lag (in pixels) that maximises the cross-correlation.

    Raises
    ------
    ValueError
        If the requested cross-correlation window is unset on
        ``line_config``.
    """

    if Solar:
        window = line_config.solar_window
        if window is None:
            raise ValueError(
                f"line_config '{line_config.name}' has no solar_window; "
                "set it before requesting a Solar cross-correlation lag."
            )
        index = np.where((spec_coords > window[0]) & (spec_coords < window[1]))[0]
        atlas_cp = atlas
    else:
        window = line_config.telluric_window
        if window is None:
            raise ValueError(
                f"line_config '{line_config.name}' has no telluric_window; "
                "set it before requesting a telluric cross-correlation lag."
            )
        index = np.where((spec_coords > window[0]) & (spec_coords < window[1]))[0]
        # Telluric features are narrow; broaden the HITRAN model to the
        # instrument's nominal resolution before correlating.
        atlas_cp = do_conv(
            spec_coords,
            atlas * np.median(data),
            instrument_config.nominal_log_rpow,
        )

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
    line_config: LineConfig,
    instrument_config: InstrumentConfig = CRYO_NIRSP,
    Solar: bool = True,
) -> float:
    """Cross-correlate after removing a linear background trend.

    Parameters
    ----------
    data, atlas, spec_coords, line_config, instrument_config, Solar
        See :func:`get_lags`. A linear background is subtracted from
        ``data`` prior to correlation to reduce continuum bias. The
        background is anchored at ``line_config.x_ref`` with slope
        ``line_config.continuum_slope_estimate``.

    Returns
    -------
    float
        Sub-pixel lag (in pixels) after background correction.

    Raises
    ------
    ValueError
        If the requested cross-correlation window is unset on
        ``line_config``.
    """

    if Solar:
        window = line_config.solar_window
        if window is None:
            raise ValueError(
                f"line_config '{line_config.name}' has no solar_window; "
                "set it before requesting a Solar cross-correlation lag."
            )
        index = np.where((spec_coords > window[0]) & (spec_coords < window[1]))[0]
        atlas_cp = atlas
    else:
        window = line_config.telluric_window
        if window is None:
            raise ValueError(
                f"line_config '{line_config.name}' has no telluric_window; "
                "set it before requesting a telluric cross-correlation lag."
            )
        index = np.where((spec_coords > window[0]) & (spec_coords < window[1]))[0]
        # Telluric features are narrow; broaden the HITRAN model to the
        # instrument's nominal resolution before correlating.
        atlas_cp = do_conv(
            spec_coords,
            atlas * np.median(data),
            instrument_config.nominal_log_rpow,
        )

    # Estimate and subtract a linear background anchored at the line centre.
    x_ref = line_config.x_ref
    slope = line_config.continuum_slope_estimate
    ind_ref = np.argmin(np.abs(spec_coords - x_ref))
    y_est = data[ind_ref]
    # Background line passes through (x_ref, y_est) with the configured slope.
    c = y_est - slope * x_ref

    x1 = data[index] - (slope * spec_coords[index] + c)
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
          

@functools.partial(jax.jit, static_argnames=("radius",))
def _build_model(
    params: jnp.ndarray,
    x: jnp.ndarray,
    fts_cor: jnp.ndarray,
    log_fts_atm: jnp.ndarray,
    x_ref: float,
    radius: int = 30,
) -> jnp.ndarray:
    """Forward model for a Cryo-NIRSP coronal-line spectrum.

    This is the single canonical implementation of the forward model used
    by both the differential-evolution global search and the
    :mod:`optimistix` local refinement. It is defined at module level so
    that its JIT cache key is stable across calls.

    Parameters
    ----------
    params
        Length-10 parameter vector ordered as ``[amp, lam_0, sigma,
        Rpow_log, opac, velS, velT, strayfrac, c0, c1]``. ``lam_0`` is the
        line-centre offset from ``x_ref`` in nm; ``c0`` is the continuum
        level at ``x_ref`` (\u03BCB\u2299) and ``c1`` is the continuum slope
        (\u03BCB\u2299 / nm).
    x
        Wavelength grid (nm) for the slit pixel being fitted.
    fts_cor
        Solar reference spectrum sampled on ``x``.
    log_fts_atm
        ``log`` of the telluric transmission model sampled on ``x``.
    x_ref
        Air rest wavelength of the targeted coronal line in nm. Anchors
        both the gaussian profile and the linear continuum.
    radius
        Half-width of the LSF convolution kernel in pixels. Static
        argument; choose via :func:`compute_lsf_radius` so it bounds the
        worst-case sigma under the configured ``Rpow_log`` lower bound.

    Returns
    -------
    jax.numpy.ndarray
        Model spectrum sampled on ``x``.
    """

    (
        amp,
        lam_0,
        sigma,
        Rpow_log,
        opac,
        velS,
        velT,
        strayfrac,
        c0,
        c1,
    ) = params

    # Coronal emission line, centred at x_ref + lam_0.
    gfit = gaussian(x - x_ref, amp, lam_0, sigma)

    # Shift solar reference.
    ftsSmod = fft_shift(fts_cor, velS)

    # Scale and shift telluric transmission.
    ftsTmod = jnp.exp(opac * log_fts_atm)
    ftsTmod = fft_shift(ftsTmod, velT)

    ftsmod = ftsSmod * ftsTmod
    # Straylight contamination.
    ftsmod = (ftsmod + strayfrac) / (1.0 + strayfrac)
    # Linear continuum anchored at the line centre: c0 is the continuum
    # level at x_ref and c1 is the slope (per nm). Anchoring at x_ref
    # decouples the two parameters; the previous (icont + icont_lin * x)
    # parameterisation made them nearly degenerate because x \u2248 1074 nm.
    ftsmod = ftsmod * (c0 + c1 * (x - x_ref))

    # Telluric absorption is applied to the coronal line as well.
    ifit = ftsmod + gfit * ftsTmod

    # Spectrograph line-spread-function convolution.
    fwhm_wv = x.mean() / jnp.exp(Rpow_log)
    sigm_wv = fwhm_wv / (2.0 * jnp.sqrt(2.0 * jnp.log(2)))
    # Positive dispersion magnitude — variable name matches value.
    dwv = jnp.abs(x[1] - x[0])
    kern_pix = sigm_wv / dwv
    ifit = gaussian_filter_1d(ifit, sigma=kern_pix, radius=radius)

    return ifit


@functools.partial(jax.jit, static_argnames=("radius",))
def _loss(
    params: jnp.ndarray,
    args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
    radius: int = 30,
) -> jnp.ndarray:
    """Weighted mean-squared-error loss.

    The signature matches the :func:`optimistix.minimise` convention
    ``fn(y, args)`` (with the additional static ``radius`` argument) so
    the same function can be used for both the scipy
    :func:`differential_evolution` global search and the optimistix local
    refinement, via a thin wrapper produced by
    :func:`_get_loss_callable`.

    Parameters
    ----------
    params
        Length-10 parameter vector (see :func:`_build_model`).
    args
        Tuple ``(y, x, wgts, fts_cor, log_fts_atm, x_ref)``.
    radius
        Static LSF kernel radius (see :func:`_build_model`).
    """

    y, x, wgts, fts_cor, log_fts_atm, x_ref = args
    y_hat = _build_model(params, x, fts_cor, log_fts_atm, x_ref, radius=radius)
    return jnp.mean((y_hat - y) ** 2 * wgts)


@functools.lru_cache(maxsize=16)
def _get_loss_callable(radius: int) -> Callable[[jnp.ndarray, Tuple], jnp.ndarray]:
    """Return a ``(params, args) -> scalar`` callable with ``radius`` baked in.

    Cached by radius so repeated requests with the same value return the
    same function object. That stability matters because
    :func:`optimistix.minimise` traces its loss function; a fresh closure
    per call would force a re-trace.

    Internal: the returned callable is not itself ``@jax.jit``-decorated.
    It delegates to the module-level jitted :func:`_loss` which keys its
    JIT cache on the static ``radius`` value.
    """

    def loss_fn(params: jnp.ndarray, args: Tuple) -> jnp.ndarray:
        return _loss(params, args, radius=radius)

    return loss_fn


# Parameter-vector length used by sentinel results when a pixel fit fails.
_N_PARAMS = 10


def _make_sentinel_result(message: str) -> OptimizeResult:
    """Return an OptimizeResult that signals a failed pixel fit.

    Used by :meth:`fit_data.fit_slit` to record failures without
    interrupting the slit. Sentinels preserve the uniform shape that
    :func:`pull_fit_res` assumes when stacking results across files.
    """

    return OptimizeResult(
        x=np.full(_N_PARAMS, np.nan),
        fun=np.nan,
        success=False,
        message=message,
    )


class fit_data:
    """Convenience wrapper that fits every row in a Cryo slit image."""

    def __init__(
        self,
        data: np.ndarray,
        spec_coords: np.ndarray,
        do_diff: Optional[int] = None,
        line_config: Optional[LineConfig] = None,
        instrument_config: Optional[InstrumentConfig] = None,
    ) -> None:
        """Store the dataset to be fitted and configure optimisation cadence.

        Parameters
        ----------
        data
            Two-dimensional array with shape ``(n_along_slit, n_wavelength)``.
        spec_coords
            One-dimensional wavelength axis (air nm).
        do_diff
            Interval at which differential-evolution is rerun.

            * ``None`` (default) or ``0`` \u2014 run DE only once, at the
              first pixel of the slit, and warm-start every subsequent
              pixel from the previous fit. Fastest; risks getting stuck
              if the slit conditions vary strongly.
            * ``1`` \u2014 run DE at every pixel. Most robust against local
              minima; slowest.
            * any ``k > 1`` \u2014 run DE every ``k`` pixels and warm-start
              the rest.
        line_config
            Per-line configuration. When omitted the line is auto-detected
            from ``spec_coords`` via :func:`config.detect_line_config`.
        instrument_config
            Per-instrument configuration. Defaults to :data:`config.CRYO_NIRSP`.
        """

        self.data = data
        self.spec_coords = spec_coords
        self.n_along_slit = data.shape[0]
        self.n_wv = data.shape[1]
        self.bounds: Optional[List[Tuple[float, float]]] = None

        if line_config is None:
            line_config = detect_line_config(spec_coords)
        self.line_config = line_config

        if instrument_config is None:
            instrument_config = CRYO_NIRSP
        self.instrument_config = instrument_config

        # Canonical sentinel: None or 0 both mean "DE once per slit".
        if do_diff is None or do_diff == 0:
            self.do_diff = self.n_along_slit
        else:
            self.do_diff = int(do_diff)

        # Precompute reference spectra and LSF kernel radius so that
        # build_model / loss / calculate_model are usable before
        # fit_slit() is ever called. The atlas I/O cost is paid once
        # per fit_data instance; loading is ~100 ms.
        self.fts_cor = jnp.asarray(get_solar_model(spec_coords))
        self.fts_atm = jnp.asarray(get_telluric_model(spec_coords))
        self.log_fts_atm = jnp.log(self.fts_atm)
        # Default weights: unity. fit_slit overrides if the caller passes
        # explicit weights.
        self.wgts = jnp.ones(self.n_wv, jnp.float64)
        # Kernel radius bounded by the worst-case (lowest-R) LSF sigma
        # under the configured Rpow_log lower bound.
        self.radius = compute_lsf_radius(
            spec_coords,
            instrument_config.rpow_log_bounds[0],
        )

    def create_bounds(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        c0_range: Tuple[float, float] = (0.0, 200.0),
        c1_range: Tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        """Define parameter bounds for the optimisation.

        Parameters
        ----------
        bounds
            Explicit 10-tuple bounds list. When omitted the defaults from
            :meth:`config.LineConfig.make_bounds` are used.
        c0_range, c1_range
            Lower and upper bounds for the continuum-level (\u03BCB\u2299)
            and continuum-slope (\u03BCB\u2299/nm) parameters when ``bounds``
            is None. Defaults match the typical Fe XIII data range.
        """

        if bounds is None:
            bounds = self.line_config.make_bounds(
                self.instrument_config,
                c0_range=c0_range,
                c1_range=c1_range,
            )

        self.bounds = bounds

    def build_model(
        self,
        params: Union[Sequence[float], np.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Evaluate the forward model using the provided parameter vector.

        Thin delegate around the canonical module-level :func:`_build_model`.
        Usable immediately after construction \u2014 ``fts_cor``,
        ``log_fts_atm`` and the LSF kernel radius are populated in
        ``__init__``.
        """

        return _build_model(
            jnp.asarray(params),
            jnp.asarray(self.spec_coords),
            self.fts_cor,
            self.log_fts_atm,
            self.line_config.x_ref,
            radius=self.radius,
        )

    def loss(
        self,
        params: Union[Sequence[float], np.ndarray, jnp.ndarray],
        y: Union[Sequence[float], np.ndarray, jnp.ndarray],
    ) -> float:
        """Return the weighted mean squared error for a candidate parameter set.

        Thin delegate around the canonical module-level :func:`_loss`.
        Usable immediately after construction; ``self.wgts`` defaults to
        unit weights and is overridden by :meth:`fit_slit` if non-trivial
        weights are supplied.
        """

        args = (
            jnp.asarray(y),
            jnp.asarray(self.spec_coords),
            self.wgts,
            self.fts_cor,
            self.log_fts_atm,
            self.line_config.x_ref,
        )
        return float(_loss(jnp.asarray(params), args, radius=self.radius))

    def fit_slit(
        self,
        wgts: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        use_tqdm: bool = False,
        min_tol: float = 1e-4,
        max_steps: int = 512,
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
            Convergence tolerance for the :mod:`optimistix` BFGS solver.
            Used as both ``rtol`` and ``atol``.
        max_steps
            Maximum number of BFGS iterations per pixel.

        Returns
        -------
        list[scipy.optimize.OptimizeResult]
            One result per slit position, in order. Failed or non-finite
            pixels are reported as sentinel results with ``x`` filled with
            NaN, ``fun`` NaN, ``success=False``. The list is always the
            full length ``self.n_along_slit`` so downstream stacking by
            :func:`pull_fit_res` sees uniform shapes.

        Notes
        -----
        The forward model and loss live at module level (:func:`_build_model`,
        :func:`_loss`) so the JAX JIT cache hits on the second and subsequent
        calls with the same spectral grid shape and ``radius``. The local
        refinement uses :func:`optimistix.minimise`, which keeps the entire
        BFGS iteration inside JIT and removes the per-step host/device
        synchronisation that previously dominated runtime.
        """

        res_slits: List[OptimizeResult] = []

        if self.bounds is None:
            self.create_bounds()

        if wgts is not None:
            self.wgts = jnp.asarray(wgts)
        # else: self.wgts already populated in __init__

        # Local aliases for the hot loop.
        fts_cor = self.fts_cor
        fts_atm = self.fts_atm
        log_fts_atm = self.log_fts_atm
        wgts_arr = self.wgts
        x = jnp.asarray(self.spec_coords)
        x_ref = float(self.line_config.x_ref)
        radius = int(self.radius)
        solver = optx.BFGS(rtol=min_tol, atol=min_tol)

        # Cached optimistix-compatible wrapper with `radius` baked in. Stable
        # function identity across pixels and across fit_data instances with
        # the same radius means optimistix's internal trace cache hits.
        loss_callable = _get_loss_callable(radius)
        # Bound version of _loss for scipy.differential_evolution.
        de_loss = functools.partial(_loss, radius=radius)

        # `force_de_next` tracks whether the previous pixel produced a usable
        # warm-start. A failed / non-converged pixel poisons the chain, so we
        # force a fresh DE search on the next pixel.
        force_de_next = True
        last_res: Optional[OptimizeResult] = None

        iterator = tqdm(range(self.n_along_slit), desc="Processing", disable=not use_tqdm)

        for i in iterator:
            y_np = np.asarray(self.data[i])

            # Skip pixels with NaN / inf data: record sentinel and move on.
            if not np.all(np.isfinite(y_np)):
                res_slits.append(_make_sentinel_result(
                    f"non-finite data at pixel {i}"
                ))
                force_de_next = True
                last_res = None
                continue

            y = jnp.asarray(y_np)
            loss_args = (y, x, wgts_arr, fts_cor, log_fts_atm, x_ref)

            try:
                run_de = (i % self.do_diff == 0) or force_de_next

                if run_de:
                    velS_est = get_lags_lin(
                        y, fts_cor, x, self.line_config, self.instrument_config, Solar=True
                    )
                    velT_est = get_lags_lin(
                        y, fts_atm, x, self.line_config, self.instrument_config, Solar=False
                    )
                    assert self.bounds is not None  # for type checkers
                    self.bounds[5] = (velS_est - 0.5, velS_est + 0.5)
                    self.bounds[6] = (velT_est - 0.5, velT_est + 0.5)

                    de_res = differential_evolution(
                        de_loss,
                        self.bounds,
                        args=(loss_args,),
                        tol=1.0e-2,
                        maxiter=800,
                        popsize=15,  # DE population = popsize * len(bounds) = 150
                    )
                    params0 = jnp.asarray(de_res.x)
                else:
                    assert last_res is not None
                    params0 = jnp.asarray(last_res.x)

                sol = optx.minimise(
                    loss_callable,
                    solver,
                    params0,
                    args=loss_args,
                    max_steps=max_steps,
                    throw=False,
                )

                # Recompute the loss once so downstream consumers can still
                # read ``.fun`` as they did with scipy.optimize.minimize.
                value_np = np.asarray(sol.value)
                if not np.all(np.isfinite(value_np)):
                    raise FloatingPointError("optimistix returned non-finite parameters")

                fun_val = float(_loss(sol.value, loss_args, radius=radius))
                success = bool(sol.result == optx.RESULTS.successful)
                res = OptimizeResult(
                    x=value_np,
                    fun=fun_val,
                    success=success,
                    message=str(sol.result),
                )
                # Poisoned warm-start chain: if BFGS didn't converge, the
                # solution may still be near a minimum but is not trustworthy
                # as a starting point for the next pixel.
                force_de_next = not success

            except Exception as exc:  # noqa: BLE001 — broad catch is intentional
                # Don't let one bad pixel kill the whole slit.
                res = _make_sentinel_result(
                    f"{type(exc).__name__} at pixel {i}: {exc}"
                )
                force_de_next = True

            res_slits.append(res)
            last_res = res

        self.res_slits = res_slits
        return self.res_slits

    def calculate_model(self) -> np.ndarray:
        """Return the forward-model profiles as a 2-D array.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_along_slit, n_wavelength)``. Failed
            (sentinel) pixels in :attr:`res_slits` produce NaN rows.

        Raises
        ------
        RuntimeError
            If :meth:`fit_slit` has not been called yet.
        """

        if not hasattr(self, "res_slits"):
            raise RuntimeError(
                "calculate_model() requires fit_slit() to have been called first."
            )

        rows: List[np.ndarray] = []
        for r in self.res_slits:
            params = np.asarray(r.x)
            if not np.all(np.isfinite(params)):
                # Sentinel row: NaNs propagate cleanly through imshow / arithmetic.
                rows.append(np.full(self.n_wv, np.nan))
            else:
                rows.append(np.asarray(self.build_model(params)))
        return np.stack(rows)

    def plot_model_vs_data(self, vmin: float = -1.0, vmax: float = 1.0) -> None:
        """Plot the observed data, fitted model, and residuals."""

        self.model = self.calculate_model()  # 2-D numpy ndarray
        data_np = np.asarray(self.data)
        extent = [self.spec_coords[0], self.spec_coords[-1], 0, self.n_along_slit]
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(data_np, extent=extent, aspect=0.005, origin='lower')
        ax[0].set_title("Data")

        ax[1].imshow(self.model, extent=extent, aspect=0.005, origin='lower')
        ax[1].set_title("Model")

        residual = data_np - self.model  # both 2-D, same shape
        pl = ax[2].imshow(residual, vmin=vmin, vmax=vmax,
                          extent=extent, aspect=0.005, origin='lower')
        # plt.colorbar(pl)
        ax[2].set_title("Residual \n clipped {0} to {1}".format(vmin, vmax))

        for a in ax:
            a.set_xlabel('Wavelength (nm)')
        ax[0].set_ylabel('Pixel along slit')

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
        Maximum number of I/O worker threads.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The stacked fit parameters with shape ``(n_params, n_files, n_slits)``
        and the corresponding merit function values.

    Notes
    -----
    Uses :class:`ThreadPoolExecutor` rather than processes because the
    workload is dominated by ``np.load`` (which releases the GIL) and a
    process pool would otherwise pay the cold-import cost of JAX / scipy
    in every worker.
    """

    fit_directory = dataset_directory + "spectrum_fits/"

    file_list = sorted(glob.glob(fit_directory + "*npz"))

    ncpus = min(os.cpu_count(), cpu_max)

    with ThreadPoolExecutor(max_workers=ncpus) as executor:
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

