"""Unit tests for :mod:`CRYOtools.util` covering the critical helper APIs."""

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("astropy.time")
pytest.importorskip("astropy.io.fits")

from astropy.io.fits import Header
from astropy.time import Time

from CRYOtools import util


def _make_header(
    scan: int,
    meas: int,
    *,
    naxis1: int = 3,
    naxis2: int = 4,
) -> Header:
    """Return a minimal FITS header with the bookkeeping keys used in tests."""

    header = Header()
    header["CNCURSCN"] = scan
    header["CNCMEAS"] = meas
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    return header


@pytest.fixture
def fake_wcs(monkeypatch):
    """Patch :class:`astropy.wcs.WCS` with a deterministic stub for tests."""

    class FakeWorld:
        def __init__(self, x_values: np.ndarray, y_values: np.ndarray, obstime: Time):
            self.Tx = SimpleNamespace(value=x_values)
            self.Ty = SimpleNamespace(value=y_values)
            self._obstime = obstime

        def __getitem__(self, item):  # pragma: no cover - trivial
            return SimpleNamespace(obstime=self._obstime)

    class FakeWCS:
        def __init__(self, header: Header):
            self._header = header

        def array_index_to_world(self, *args, **kwargs):
            slit_length = int(self._header["NAXIS2"])
            scan = int(self._header["CNCURSCN"])
            meas = int(self._header["CNCMEAS"])
            x_vals = np.full(slit_length, float(scan))
            y_vals = np.arange(slit_length, dtype=float) + float(meas)
            obstime = Time(datetime(2020, 1, scan, 0, 0, meas))
            return (None, FakeWorld(x_vals, y_vals, obstime))

    monkeypatch.setattr(util, "WCS", FakeWCS)
    return FakeWCS


def test_return_obs_info_reports_extents(capsys):
    """`return_obs_info` should compute raster extents and emit diagnostics."""

    header_a = _make_header(1, 2)
    header_a["CNP1DSS1"] = 42
    header_b = _make_header(3, 1)

    result = util.return_obs_info([header_a, header_b])

    captured = capsys.readouterr()
    assert "CNP1DSS1" in captured.out
    assert result == (3, 2, 3, 4)


def test_return_obs_info_silent_when_requested(capsys):
    """`return_obs_info` should avoid printing diagnostics when verbose is False."""

    header = _make_header(2, 4)
    result = util.return_obs_info([header], verbose=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert result == (2, 4, 3, 4)


def test_get_slit_coords_populates_arrays_without_saver(fake_wcs, monkeypatch):
    """`get_slit_coords` should fill coordinate arrays and skip saving when asked."""

    headers = [_make_header(1, 1), _make_header(1, 2)]

    # Ensure the saver shortcut bypasses filesystem interactions.
    def fail_if_called(path):  # pragma: no cover - defensive guard
        raise AssertionError("ensure_directory should not be invoked when saver=None")

    monkeypatch.setattr(util, "ensure_directory", fail_if_called)

    hpxy, times = util.get_slit_coords(headers, saver=None)

    assert np.all(hpxy[0, 0, 0] == 1.0)
    assert np.all(hpxy[1, 0, 1] == np.arange(4) + 2.0)
    assert times.shape == (1, 2)


def test_get_slit_coords_invokes_custom_saver(fake_wcs, tmp_path):
    """`get_slit_coords` should route persistence through the provided saver callable."""

    headers = [_make_header(1, 1)]
    saved = {}

    def recorder(path, array):
        saved[os.path.basename(path)] = array

    util.get_slit_coords(headers, output_dir=tmp_path, saver=recorder)

    assert set(saved) == {"hpxy_coords.npy", "time_coords.npy"}
    np.testing.assert_array_equal(saved["hpxy_coords.npy"][0, 0, 0], np.full(4, 1.0))


def test_get_spectral_coords_respects_custom_saver(fake_wcs, tmp_path):
    """`get_spectral_coords` should use the provided saver and return the spectrum."""

    header = _make_header(1, 1)
    saved = {}

    def recorder(path, array):
        saved[os.path.basename(path)] = array

    spectrum = util.get_spectral_coords([header], output_dir=tmp_path, saver=recorder)

    assert spectrum.shape == (3,)
    assert saved["spec_coords.npy"].shape == (3,)


def test_get_slit_samp_validates_slit_length():
    """`get_slit_samp` should reject inputs lacking at least two slit pixels."""

    hpxy = np.zeros((2, 1, 1, 1))
    with pytest.raises(ValueError):
        util.get_slit_samp(hpxy, n_scan_steps=1)


def test_get_slit_samp_returns_sampling_values():
    """`get_slit_samp` should compute along-slit sampling and raster step size."""

    hpxy = np.zeros((2, 2, 1, 3))
    # Along-slit vector from pixel 0 to 1 has length sqrt(5).
    hpxy[:, 0, 0, 0] = (0.0, 0.0)
    hpxy[:, 0, 0, 1] = (1.0, 2.0)
    # Step between scan 0 and 1 is sqrt(2).
    hpxy[:, 1, 0, 0] = (1.0, 1.0)

    slit_samp, step_width = util.get_slit_samp(hpxy, n_scan_steps=2, verbose=False)

    assert pytest.approx(slit_samp) == np.sqrt(5.0)
    assert pytest.approx(step_width) == np.sqrt(2.0)


def test_calculate_cadence_returns_median():
    """`calculate_cadence` should report the median gap between timestamps."""

    timestamps = np.array([
        np.datetime64("2020-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:03"),
        np.datetime64("2020-01-01T00:00:08"),
    ])

    cadence = util.calculate_cadence(timestamps)

    assert cadence == 5.0


def test_calculate_cadence_requires_multiple_samples():
    """`calculate_cadence` should raise when fewer than two timestamps are provided."""

    timestamps = np.array([np.datetime64("2020-01-01T00:00:00")])
    with pytest.raises(ValueError):
        util.calculate_cadence(timestamps)


def test_shift_to_v_returns_quantity_with_units():
    """`shift_to_v` should attach km/s units to the Doppler velocity output."""

    delta = np.array([0.7])  # exactly cancels the correction term
    velocity = util.shift_to_v(delta)

    assert velocity.unit.is_equivalent("km/s")
    assert velocity.value.shape == (1,)


def test_calc_ntlw_supports_broadcast_and_validates_shape():
    """`calc_ntlw` should broadcast scalar thermal velocities and enforce shape checks."""

    data = np.full((2,), 0.02)
    result = util.calc_ntlw(data, res_pow=50000, v_th=21.0)
    assert result.unit.is_equivalent("km/s")

    with pytest.raises(ValueError):
        util.calc_ntlw(data, res_pow=50000, v_th=np.array([21.0, 22.0, 23.0]))


def test_ensure_directory_accepts_pathlike(tmp_path):
    """`ensure_directory` should create directories when given :class:`pathlib.Path` inputs."""

    target = tmp_path / "nested" / "leaf"
    util.ensure_directory(target)

    assert target.exists()


def test_print_exposure_handles_missing_comments(capsys):
    """`print_exposure` should avoid crashing when FITS comments are absent."""

    header = _make_header(1, 1)
    header["CAM_FPS"] = 20.0
    util.print_exposure([header])

    captured = capsys.readouterr()
    assert "CAM_FPS" in captured.out
    assert "Time between ramps" in captured.out

