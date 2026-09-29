"""Warm worker: one subprocess runs a batch of fits for one method.

Imports and a small warm-up fit (numba JIT, torch initialization) happen once per batch and
are recorded as overhead; runtime_seconds covers fit_transform only. Every fit writes its own
atomic result, so a crash or a parent-enforced timeout loses at most the fit in progress.
"""

import os
import random
import resource
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .registry import get_method
from .schemas import MethodRunResult
from .utils import read_json, versions, write_json

WARMUP_ROWS = 64


def _reset_peak():
    """Reset the kernel's peak-RSS mark so VmHWM describes the next fit only."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _peak_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _fit(method, X, seed, params):
    random.seed(seed)
    np.random.seed(seed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        wall, cpu = time.perf_counter(), time.process_time()
        Y, diagnostics = method.fit_transform(X, seed, params)
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
        diagnostics.update(method.after_fit())
    return Y, diagnostics, wall, cpu, [str(w.message) for w in captured]


def _save_embedding(directory, Y):
    tmp = directory / "embedding.tmp.npy"
    np.save(tmp, Y, allow_pickle=False)
    os.replace(tmp, directory / "embedding.npy")


def run_job(method, job, X, package_versions, overhead):
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "embedding.npy").unlink(missing_ok=True)
    result = MethodRunResult(
        job["method_id"], job["parameters"], job["seed"], "failed", 0.0, package_versions=package_versions
    )
    exact_peak = _reset_peak()
    try:
        params, adjustments = method.validate_input(X, job["parameters"])
        result.parameters = params
        result.warnings.extend(adjustments)
        try:
            Y, diagnostics, wall, cpu, caught = _fit(method, X, job["seed"], params)
        except Exception as first:  # the only predefined numerical retry
            if job["method_id"] != "lle":
                raise
            failed = {"parameters": params, "error_type": type(first).__name__, "error_message": str(first)}
            params = dict(params, reg=0.01, eigen_solver="dense")
            result.parameters, result.retry_count = params, 1
            result.warnings.append("LLE retry: reg=0.01 and dense eigensolver after numerical failure")
            Y, diagnostics, wall, cpu, caught = _fit(method, X, job["seed"], params)
            diagnostics["attempt_history"] = [failed]
        result.warnings.extend(caught)
        if Y.shape != (len(X), 2) or not np.isfinite(Y).all():
            raise ValueError("Invalid embedding shape or non-finite coordinates")
        _save_embedding(directory, np.asarray(Y, dtype=np.float64))
        result.embedding = "embedding.npy"
        result.diagnostics = diagnostics
        result.status = "success"
        result.runtime_seconds = wall
        result.timing = {"fit_wall_seconds": wall, "fit_cpu_seconds": cpu}
    except Exception as exc:  # noqa: BLE001 -- intentional isolation/audit boundary
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        result.diagnostics["traceback"] = traceback.format_exc()
    finally:
        result.peak_memory_mb = _peak_mb()
        result.timing.update(overhead, peak_memory_scope="this fit" if exact_peak else "process lifetime")
        write_json(directory / "result.json", dict(asdict(result), input_sha256=job["input_sha256"]))


def main():
    batch = read_json(sys.argv[1])
    progress = Path(sys.argv[2])
    start = time.perf_counter()
    method = get_method(batch["method_id"])
    package_versions = versions()
    import_seconds = time.perf_counter() - start
    inputs = {}

    def load(path):
        if path not in inputs:
            inputs[path] = np.load(path, allow_pickle=False)
        return inputs[path]

    warm, warmup_error = time.perf_counter(), None
    try:
        X0 = load(batch["jobs"][0]["input_path"])[:WARMUP_ROWS]
        params, _ = method.validate_input(X0, batch["jobs"][0]["parameters"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            method.fit_transform(X0, 0, params)
            method.after_fit()
    except Exception as exc:  # noqa: BLE001 -- warm-up is best effort; real fits report their own errors
        warmup_error = f"{type(exc).__name__}: {exc}"
    overhead = {
        "batch_import_seconds": import_seconds,
        "batch_warmup_seconds": time.perf_counter() - warm,
        "batch_warmup_error": warmup_error,
        "batch_size": len(batch["jobs"]),
    }
    for index, job in enumerate(batch["jobs"]):
        write_json(progress, {"index": index, "pid": os.getpid()})
        run_job(method, job, load(job["input_path"]), package_versions, overhead)
    write_json(progress, {"index": len(batch["jobs"]), "pid": os.getpid(), "done": True})


if __name__ == "__main__":
    main()
