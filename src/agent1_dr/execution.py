"""Batched method fits with result caching, per-fit timeouts and isolated fallbacks."""

import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .registry import get_method
from .schemas import MethodRunResult
from .utils import checksum, fit_code_hash, read_json, versions, write_json

POLL_SECONDS = 0.5
# Imports plus the small warm-up fit must finish before the first real fit starts.
WARMUP_ALLOWANCE_SECONDS = 900


def _env():
    return dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONHASHSEED="0",
        PYTHONNOUSERSITE="1",
        LD_LIBRARY_PATH=str(Path(sys.prefix) / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )


def _finish(job, result):
    """Record checksums so that later stages can verify the cached fit."""
    d = job["directory"]
    write_json(d / "result.json", result)
    write_json(
        d / "run_manifest.json",
        {
            "result_sha256": checksum(d / "result.json"),
            "embedding_sha256": checksum(d / "embedding.npy") if result["status"] == "success" else None,
        },
    )
    print(f"{result['method_id']} seed={result['seed']} {result['status']} {result['runtime_seconds']:.2f}s "
          f"-> {d}", flush=True)
    return result


def _valid(job, result):
    if result["input_sha256"] != job["input_sha256"] or result["status"] not in {"success", "failed", "timeout"}:
        return False
    if result["status"] != "success":
        return True
    y = np.load(job["directory"] / "embedding.npy")
    return y.shape == (job["n_rows"], 2) and bool(np.isfinite(y).all())


def _cached(job):
    d = job["directory"]
    try:
        if read_json(d / "request.json") != job["request"]:
            return None
        result = read_json(d / "result.json")
        saved = read_json(d / "run_manifest.json")
        if saved["result_sha256"] != checksum(d / "result.json"):
            return None
        if result["status"] == "success" and saved["embedding_sha256"] != checksum(d / "embedding.npy"):
            return None
        return result if _valid(job, result) else None
    except (OSError, ValueError, KeyError):
        return None


def _collect(job):
    """A fit that the worker completed in this batch, or None."""
    try:
        if read_json(job["directory"] / "request.json") != job["request"]:
            return None
        result = read_json(job["directory"] / "result.json")
        return _finish(job, result) if _valid(job, result) else None
    except (OSError, ValueError, KeyError):
        return None


def _terminal(job, status, error_type, message, seconds):
    result = asdict(
        MethodRunResult(
            job["request"]["method_id"],
            job["params"],
            job["seed"],
            status,
            seconds,
            error_type=error_type,
            error_message=message,
            package_versions=versions(),
        )
    )
    result["input_sha256"] = job["input_sha256"]
    return _finish(job, result)


def _run_batch(method_id, batch, timeout, log_dir):
    for job in batch:
        job["directory"].mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "run_manifest.json", "embedding.npy"):
            (job["directory"] / name).unlink(missing_ok=True)
        write_json(job["directory"] / "request.json", job["request"])
    stamp = f"{method_id}_{os.getpid()}_{time.time_ns()}"
    batch_path, progress = log_dir / f".batch_{stamp}.json", log_dir / f".progress_{stamp}.json"
    write_json(
        batch_path,
        {
            "method_id": method_id,
            "jobs": [
                {
                    "method_id": method_id,
                    "input_path": str(j["input_path"]),
                    "seed": j["seed"],
                    "parameters": j["params"],
                    "directory": str(j["directory"]),
                    "input_sha256": j["input_sha256"],
                }
                for j in batch
            ],
        },
    )
    timed_out = None
    with open(log_dir / "worker.log", "a") as log:
        log.write(f"--- batch {stamp}: {len(batch)} fits\n")
        log.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent1_dr.worker", str(batch_path), str(progress)],
            env=_env(),
            stdout=log,
            stderr=log,
        )
        current, since = -1, time.monotonic()
        while proc.poll() is None:
            time.sleep(POLL_SECONDS)
            try:
                index = read_json(progress)["index"]
            except (OSError, ValueError, KeyError):
                index = -1
            now = time.monotonic()
            if index != current:
                current, since = index, now
            if now - since > (WARMUP_ALLOWANCE_SECONDS if current < 0 else timeout):
                proc.kill()
                proc.wait()
                timed_out = now - since
    try:
        current = read_json(progress)["index"]
    except (OSError, ValueError, KeyError):
        pass
    batch_path.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    return proc.returncode, max(current, 0), timed_out


def run_fits(method_id, jobs, timeout, force=False, log_dir=None):
    """Fit one method on several (input, seed, parameter) jobs; results are returned in job order.

    Each job: {"input_path", "seed", "params", "directory"}. Valid cached results are reused
    unless force is set. The worker is warm for the whole batch; a fit that exceeds the
    per-fit timeout, or crashes the worker, receives a terminal status and the remaining fits
    continue in a fresh worker.
    """
    code = fit_code_hash()
    shapes, digests = {}, {}
    for job in jobs:
        path = str(job["input_path"])
        if path not in digests:
            digests[path] = checksum(path)
            shapes[path] = np.load(path, mmap_mode="r").shape[0]
        job["directory"] = Path(job["directory"])
        job["input_sha256"], job["n_rows"] = digests[path], shapes[path]
        job["request"] = {
            "method_id": method_id,
            "seed": job["seed"],
            "parameters": job["params"],
            "input_sha256": job["input_sha256"],
            "code_hash": code,
        }
    results = [None] * len(jobs)
    pending = []
    for i, job in enumerate(jobs):
        results[i] = None if force else _cached(job)
        if results[i] is None:
            pending.append(i)
    log_dir = Path(log_dir or jobs[0]["directory"].parent)
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending:
        returncode, index, timed_out = _run_batch(method_id, [jobs[i] for i in pending], timeout, log_dir)
        remaining = []
        for position, i in enumerate(pending):
            results[i] = _collect(jobs[i])
            if results[i] is None:
                remaining.append((position, i))
        if remaining and (timed_out is not None or returncode != 0):
            # The fit in progress when the worker stopped is terminal; the rest get a fresh worker.
            culprit = next((i for position, i in remaining if position == index), remaining[0][1])
            if timed_out is not None:
                results[culprit] = _terminal(jobs[culprit], "timeout", "TimeoutExpired",
                                             f"Fit exceeded {timeout:.0f}s", timed_out)
            else:
                results[culprit] = _terminal(jobs[culprit], "failed", "WorkerCrash",
                                             f"Worker exited {returncode}; see {log_dir / 'worker.log'}", 0.0)
            remaining = [(p, i) for p, i in remaining if i != culprit]
        elif remaining:
            for _, i in remaining:
                results[i] = _terminal(jobs[i], "failed", "MissingResult", "Worker finished without a result", 0.0)
            remaining = []
        pending = [i for _, i in remaining]
    return results


def parameters(cfg, method_id, X):
    return dict(get_method(method_id).default_params(X), **cfg.methods.get(method_id, {}))
