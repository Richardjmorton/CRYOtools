"""Parallel slit-level batch processing for CRYOtools fits.

Provides :func:`batch_fit_slits`, a process-pool driver that runs many
:meth:`CRYOtools.fit.fit_data.fit_slit` calls in parallel \u2014 one slit per
worker task. Process-based parallelism is used because the inner fit
loop is GIL-bound (see the timing diagnostics in the project notes); a
:class:`concurrent.futures.ThreadPoolExecutor` does not scale, but
:class:`~concurrent.futures.ProcessPoolExecutor` does.

Each worker subprocess imports JAX cold and pays a one-time JIT
compilation cost (~5\u201315 s) on its first slit. ``ProcessPoolExecutor``
keeps workers alive across tasks, so the compile cost amortises over
however many slits each worker ends up handling.

Example
-------
>>> # In your top-level script (note the __main__ guard \u2014 required on
>>> # macOS/Windows because of how multiprocessing spawns workers):
>>>
>>> if __name__ == "__main__":
...     from CRYOtools.batch import batch_fit_slits
...     saved, failed = batch_fit_slits(
...         data_arrays=[data[0, i, 0] for i in range(n_slits)],
...         files=files[::5],
...         spec_coords=spec_coords,
...         wgts=wgts,
...         output_path=path,
...         n_workers=4,
...     )
...     print(f"Saved {len(saved)} slits, {len(failed)} failed")
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tqdm.auto import tqdm

# Import LineConfig / InstrumentConfig lazily inside the worker so the
# parent process doesn't pull in JAX at module-import time. The types
# below are referenced only for annotations on the public function.
from CRYOtools.config import InstrumentConfig, LineConfig


# ---------------------------------------------------------------------------
# Worker — must be importable at module level so multiprocessing can pickle it
# ---------------------------------------------------------------------------


@dataclass
class _SlitTask:
    """Pickled payload sent to each worker process.

    Bundled into a dataclass so the worker signature is single-argument,
    which works cleanly with ``ProcessPoolExecutor.map``.
    """

    data: np.ndarray              # shape (n_along_slit, n_wavelength)
    file: str                     # output filename stem
    spec_coords: np.ndarray       # shared 1-D wavelength axis
    wgts: Optional[np.ndarray]    # per-wavelength weights, or None
    output_path: str              # passed to write_model_results
    do_diff: Optional[int]        # passed to fit_data(...)
    line_config: Optional[LineConfig]
    instrument_config: Optional[InstrumentConfig]


@dataclass
class _SlitOutcome:
    """Result returned from each worker.

    Exactly one of ``saved_path`` / ``error`` is non-None.
    """

    file: str
    saved_path: Optional[str] = None
    error: Optional[str] = None


def _slit_worker(task: _SlitTask) -> _SlitOutcome:
    """Worker entry point: fit one slit and persist the result.

    Imports of :mod:`CRYOtools.fit` and :mod:`CRYOtools.io` happen inside
    the function body so they are deferred to the worker process. That
    keeps the parent process JAX-free until it actually needs JAX
    (typically: never \u2014 the parent only orchestrates).

    Exceptions are caught and serialised into the returned
    :class:`_SlitOutcome` so the parent can decide whether to log-and-skip
    or re-raise. Letting an exception escape here would crash the worker.
    """

    try:
        # Deferred imports: JAX cold-load happens here, in the worker.
        from CRYOtools.fit import fit_data
        from CRYOtools.io import write_model_results

        # Test hook: when the env var is set, monkey-patch the atlas
        # loaders with synthetic stubs. Harmless when unset.
        if os.environ.get("CRYOTOOLS_TEST_STUBS"):
            try:
                from CRYOtools import _test_stubs  # type: ignore
                _test_stubs.apply()
            except ImportError:
                pass

        slit = fit_data(
            task.data,
            task.spec_coords,
            do_diff=task.do_diff,
            line_config=task.line_config,
            instrument_config=task.instrument_config,
        )
        results = slit.fit_slit(
            use_tqdm=False,
            wgts=task.wgts,
        )
        saved_path = write_model_results(task.output_path, task.file, results)
        return _SlitOutcome(file=task.file, saved_path=saved_path)
    except Exception:  # noqa: BLE001 \u2014 broad catch is intentional
        # Capture the full traceback into a string so the parent can log
        # it without needing access to the worker's exception object.
        return _SlitOutcome(
            file=task.file,
            error=traceback.format_exc(),
        )


def _worker_initialiser() -> None:
    """Run once per worker process before any task is dispatched.

    Caps XLA to a single host device so multiple workers do not all
    spawn the maximum thread pool. The flag must be set *before* JAX is
    imported \u2014 because the actual JAX import is deferred to the first
    call of :func:`_slit_worker`, doing this in the initialiser is
    sufficient.
    """

    # Workers each get a single XLA device. Multiple workers running in
    # parallel still share the physical CPU; XLA's intra-op thread pool
    # within each worker will overlap with the others. In practice this
    # produces good throughput without the heavy oversubscription seen
    # when each worker forces the default 4-device flag from fit.py.
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    # Stay on CPU regardless of any GPU visible to the worker.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def batch_fit_slits(
    data_arrays: Sequence[np.ndarray],
    files: Sequence[str],
    spec_coords: np.ndarray,
    wgts: Optional[np.ndarray],
    output_path: str,
    do_diff: Optional[int] = None,
    n_workers: int = 4,
    line_config: Optional[LineConfig] = None,
    instrument_config: Optional[InstrumentConfig] = None,
    use_tqdm: bool = True,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Fit many slits in parallel via a process pool.

    Each slit becomes one task; tasks are dispatched to ``n_workers``
    subprocesses that share an :class:`OS.environ` configuration
    capping XLA to a single host device per worker. Workers stay alive
    across tasks, so the JAX JIT-compile cost is paid once per worker
    and amortises over however many slits that worker ends up handling.

    Parameters
    ----------
    data_arrays
        Iterable of per-slit 2-D arrays, each of shape
        ``(n_along_slit, n_wavelength)``. One slit per array.
    files
        Output filename stems aligned 1\u20131 with ``data_arrays``. Each
        slit's results are written via
        :func:`CRYOtools.io.write_model_results` to
        ``<output_path>/spectrum_fits/<file>.npz``.
    spec_coords
        Shared 1-D wavelength axis (air nm).
    wgts
        Shared per-wavelength weights, or ``None`` for unit weights.
    output_path
        Directory passed to :func:`CRYOtools.io.write_model_results`.
        The function creates ``<output_path>/spectrum_fits/`` as needed.
    do_diff
        Forwarded to :class:`CRYOtools.fit.fit_data`. ``None`` keeps the
        package default (one DE call per slit). A small positive integer
        reruns DE more often \u2014 useful when slit conditions vary
        strongly enough that a single DE seed at pixel 0 isn't good
        enough.
    n_workers
        Number of subprocesses. Sensible values are 2\u20136 on a typical
        8-physical-core machine. Higher than physical-core-count almost
        never helps and may hurt.
    line_config, instrument_config
        Forwarded to :class:`CRYOtools.fit.fit_data`. ``None`` means
        auto-detect / use defaults.
    use_tqdm
        Display a progress bar (one tick per completed slit).

    Returns
    -------
    saved_paths
        List of ``.npz`` paths actually written. May be shorter than
        ``files`` if any slits failed.
    failures
        List of ``(file, traceback_string)`` for slits that raised an
        exception. Empty on a clean run.

    Notes
    -----
    The caller must gate the invocation behind
    ``if __name__ == "__main__":`` on macOS and Windows; this is a
    requirement of :mod:`multiprocessing` rather than of this function.
    Linux users can ignore it but adding it doesn't hurt.

    Raises
    ------
    ValueError
        If ``data_arrays`` and ``files`` have different lengths.
    """

    data_arrays = list(data_arrays)
    files = list(files)
    if len(data_arrays) != len(files):
        raise ValueError(
            f"data_arrays and files length mismatch: "
            f"{len(data_arrays)} vs {len(files)}"
        )

    # Materialise wgts and spec_coords into NumPy to keep the pickled
    # payload free of JAX device arrays (which can sometimes pickle
    # inefficiently or require the worker to materialise on a device
    # before unpickling). Lightweight; no measurable cost.
    spec_coords_np = np.asarray(spec_coords)
    wgts_np = None if wgts is None else np.asarray(wgts)

    tasks: List[_SlitTask] = [
        _SlitTask(
            data=np.asarray(d),
            file=f,
            spec_coords=spec_coords_np,
            wgts=wgts_np,
            output_path=output_path,
            do_diff=do_diff,
            line_config=line_config,
            instrument_config=instrument_config,
        )
        for d, f in zip(data_arrays, files)
    ]

    saved_paths: List[str] = []
    failures: List[Tuple[str, str]] = []

    # mp_context='spawn' is the cross-platform-safe choice: workers are
    # fresh interpreters that go through _worker_initialiser before
    # touching JAX. On Linux the default 'fork' would also work but
    # fork+JAX is occasionally fragile.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_worker_initialiser,
    ) as executor:
        # Submit all and use as_completed so the progress bar advances
        # as each slit finishes rather than in submission order.
        future_to_file = {
            executor.submit(_slit_worker, task): task.file for task in tasks
        }
        iterator = as_completed(future_to_file)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(tasks), desc="Fitting slits")

        for fut in iterator:
            outcome = fut.result()
            if outcome.error is None:
                saved_paths.append(outcome.saved_path)  # type: ignore[arg-type]
            else:
                failures.append((outcome.file, outcome.error))

    if failures:
        # Print a compact summary; full tracebacks remain available in
        # the returned ``failures`` list for programmatic inspection.
        print(
            f"\nbatch_fit_slits: {len(failures)} slit(s) failed:",
            *[f"  {f}: {err.splitlines()[-1] if err else ''}"
              for f, err in failures],
            sep="\n",
        )

    return saved_paths, failures
