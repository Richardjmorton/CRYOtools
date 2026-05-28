"""Line- and instrument-specific configuration for CRYOtools fitting.

This module centralises every wavelength-dependent and instrument-dependent
number that was previously hardcoded throughout :mod:`fit.py` and
:mod:`util.py`. Two pre-built line configurations are provided
(:data:`FE_XIII_1074`, :data:`FE_XIII_1080`) along with one instrument
configuration (:data:`CRYO_NIRSP`). Use :func:`detect_line_config` to pick the
right line configuration automatically from a wavelength axis.

Notes
-----
Reference wavelengths are stored as air-frame values to match the convention
used by the existing pipeline (FITS WCS ``CTYPE1 = 'AWAV-GRA'`` and the
solar-atlas vacuum-to-air conversion in :func:`fit.get_solar_model`).

The :class:`LineConfig` for Fe XIII 1079.8 nm intentionally leaves its
cross-correlation windows unset; populate ``solar_window`` and
``telluric_window`` empirically before fitting that line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentConfig:
    """Instrument-specific parameters shared by all spectral lines.

    Parameters
    ----------
    name
        Human-readable instrument name (used in diagnostic messages only).
    rpow_log_bounds
        Lower and upper bounds on ``log(R)`` (natural log of the resolving
        power) used during fitting.
    nominal_log_rpow
        ``log(R)`` used to pre-broaden the HITRAN telluric atlas inside
        :func:`fit.get_lags` and :func:`fit.get_lags_lin` so the reference
        spectrum approximately matches the data resolution before
        cross-correlation. Empirically tuned; not part of the optimisation.
    """

    name: str
    rpow_log_bounds: Tuple[float, float]
    nominal_log_rpow: float


CRYO_NIRSP = InstrumentConfig(
    name="Cryo-NIRSP",
    rpow_log_bounds=(math.log(30000.0), math.log(65000.0)),
    nominal_log_rpow=10.7,  # log(~44355), nominal R for Cryo-NIRSP
)


# ---------------------------------------------------------------------------
# Line configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LineConfig:
    """Per-line constants for the Cryo-NIRSP coronal-line forward model.

    Parameters
    ----------
    name
        Human-readable line label.
    x_ref
        Reference rest wavelength of the line in nm (air-frame convention,
        matching DKIST L1 ``CTYPE1 = 'AWAV-GRA'`` and the solar atlas
        vacuum-to-air conversion).
    vel_search_half_width
        Half-width (km/s) of the symmetric velocity window used to bound
        the line centre during fitting. The bound on the ``lam_0`` parameter
        is computed via :meth:`del_lam_bounds`.
    solar_window
        ``(min, max)`` wavelength interval in nm enclosing a clean
        photospheric (scattered solar) feature used to estimate the solar
        velocity offset by cross-correlation against the bundled solar
        atlas. ``None`` disables solar cross-correlation for this line.
    telluric_window
        ``(min, max)`` wavelength interval in nm enclosing a clean H\u2082O
        telluric feature used to estimate the telluric velocity offset.
        ``None`` disables telluric cross-correlation for this line.
    continuum_slope_estimate
        Empirical slope (\u03BCB\u2299/nm) used as a fixed estimate when
        subtracting a linear background in :func:`fit.get_lags_lin`. Not
        part of the forward-model parameter vector.
    """

    name: str
    x_ref: float
    vel_search_half_width: float = 6.0  # km/s
    solar_window: Optional[Tuple[float, float]] = None
    telluric_window: Optional[Tuple[float, float]] = None
    continuum_slope_estimate: float = 0.25  # \u03BCB\u2299 / nm
    continuum_ref: float = 0.0   # wavelength (nm)

    # ----- derived quantities -------------------------------------------------

    def del_lam_bounds(self) -> Tuple[float, float]:
        """Symmetric bound on the line-centre offset ``lam_0`` in nm.

        The bound corresponds to a Doppler velocity window of
        ``\u00B1 vel_search_half_width`` around :attr:`x_ref`.
        """

        c_km_s = 299_792.458
        delta = self.vel_search_half_width / c_km_s * self.x_ref
        return (-delta, delta)

    def make_bounds(
        self,
        instrument: InstrumentConfig,
        c0_range: Tuple[float, float] = (0.0, 200.0),
        c1_range: Tuple[float, float] = (-10.0, 10.0),
    ) -> List[Tuple[float, float]]:
        """Return the 10-element parameter-bounds list for the forward model.

        Parameter ordering matches :func:`fit._build_model`::

            [amp, lam_0, sigma, Rpow_log, opac,
             velS, velT, strayfrac, c0, c1]

        ``c0`` is the continuum level at the line centre (\u03BCB\u2299);
        ``c1`` is the continuum slope (\u03BCB\u2299/nm). Both are referenced
        to ``x_ref``: the model evaluates the continuum as
        ``c0 + c1 * (spec_coords - x_ref)``.
        """

        return [
            (0.0, 50.0),                # 0: amp                 [\u03BCB\u2299]
            self.del_lam_bounds(),      # 1: lam_0 offset        [nm]
            (0.055, 0.1),               # 2: sigma               [nm]
            instrument.rpow_log_bounds, # 3: Rpow_log
            (0.5, 10.0),                # 4: opac (telluric scale)
            (-2.0, 2.0),                # 5: velS                [pixels]
            (-3.2, -2.5),               # 6: velT                [pixels]
            (0.0, 0.5),                 # 7: strayfrac
            c0_range,                   # 8: c0 (continuum at x_ref) [\u03BCB\u2299]
            c1_range,                   # 9: c1 (continuum slope)    [\u03BCB\u2299/nm]
        ]


# Pre-built line configurations.
# Empirically-verified solar / telluric features to aid model fitting

FE_XIII_1074 = LineConfig(
    name="Fe XIII 1074.7 nm",
    x_ref=1074.63,                          # air, matches the previous create_bounds value
    solar_window=(1074.85, 1075.05),        # photospheric scattered-light feature
    telluric_window=(1074.27, 1074.40),     # H2O telluric feature
    continuum_ref=1074.0.
)


FE_XIII_1080 = LineConfig(
    name="Fe XIII 1079.8 nm",
    x_ref=1079.79,                          # NIST/CHIANTI; air convention for spec_coords
    solar_window=(1078.2,1078.8),           # photospheric scattered-light feature
    telluric_window=(1079.9,1080.1),        # H2O telluric feature
    continuum_ref=1080.2,
)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


# Order matters: detect_line_config returns the first match. The Fe XIII pair
# are well separated (~5 nm) so this is unambiguous in practice.
_KNOWN_LINES: Tuple[LineConfig, ...] = (FE_XIII_1074, FE_XIII_1080)


def detect_line_config(spec_coords: np.ndarray) -> LineConfig:
    """Pick a :class:`LineConfig` whose ``x_ref`` falls inside ``spec_coords``.

    Parameters
    ----------
    spec_coords
        One-dimensional wavelength axis (air nm).

    Returns
    -------
    LineConfig
        The first preset whose :attr:`LineConfig.x_ref` is bracketed by
        ``spec_coords.min()`` and ``spec_coords.max()``.

    Raises
    ------
    ValueError
        If no preset matches; callers should then pass ``line_config``
        explicitly.
    """

    arr = np.asarray(spec_coords)
    if arr.size == 0:
        raise ValueError("`spec_coords` is empty; cannot auto-detect line.")

    spec_min = float(arr.min())
    spec_max = float(arr.max())

    for cfg in _KNOWN_LINES:
        if spec_min < cfg.x_ref < spec_max:
            return cfg

    raise ValueError(
        f"Could not auto-detect a known line in spec_coords spanning "
        f"[{spec_min:.3f}, {spec_max:.3f}] nm. Pass `line_config` explicitly."
    )
