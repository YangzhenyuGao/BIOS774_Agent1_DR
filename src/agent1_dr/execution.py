import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .registry import get_method
from .schemas import MethodRunResult
from .utils import checksum, code_hash, read_json, versions, write_json


def run_method(method_id, input_path, seed, params, directory, timeout, force=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    request = {
        "method_id": method_id,
        "seed": seed,
        "parameters": params,
        "input_sha256": checksum(input_path),
        "code_hash": code_hash(),
    }
    try:
        if not force and read_json(directory / "request.json") == request:
            result = read_json(directory / "result.json")
            saved = read_json(directory / "run_manifest.json")
            if saved["result_sha256"] != checksum(directory / "result.json"):
                raise ValueError("Cached result checksum mismatch")
            if result["input_sha256"] != request["input_sha256"]:
                raise ValueError("Cached input checksum mismatch")
            if result["status"] in {"failed", "timeout"}:
                return result
            if saved["embedding_sha256"] != checksum(directory / "embedding.npy"):
                raise ValueError("Cached embedding checksum mismatch")
            y = np.load(directory / "embedding.npy")
            if np.isfinite(y).all() and y.shape == (len(np.load(input_path)), 2):
                return result
    except (OSError, ValueError, KeyError):
        pass
    write_json(directory / "request.json", request)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONHASHSEED=str(seed),
        PYTHONNOUSERSITE="1",
        LD_LIBRARY_PATH=str(Path(sys.prefix) / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )
    history = []
    # Only LLE has a predefined numerical retry. Other failures remain visible, without speculative changes.
    for retry in range(2 if method_id == "lle" else 1):
        current = dict(request)
        if retry:
            current["parameters"] = dict(params, reg=0.01, eigen_solver="dense")
        write_json(directory / "attempt_request.json", current)
        t = time.monotonic()
        with open(directory / f"attempt_{retry}.log", "w") as log:
            try:
                p = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "agent1_dr.worker",
                        str(input_path),
                        str(directory / "attempt_request.json"),
                        str(directory),
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                    timeout=timeout,
                    check=False,
                )
                if p.returncode != 0:
                    raise RuntimeError(f"Worker exited {p.returncode}; see attempt log")
                result = read_json(directory / "result.json")
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                status = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed"
                result = asdict(
                    MethodRunResult(
                        method_id,
                        current["parameters"],
                        seed,
                        status,
                        time.monotonic() - t,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        package_versions=versions(),
                    )
                )
        history.append(dict(result))
        if result["status"] == "success" or result["status"] == "timeout":
            break
    result["retry_count"] = len(history) - 1
    if len(history) > 1:
        result["runtime_seconds"] = sum(r["runtime_seconds"] for r in history)
        result["warnings"].append("LLE retry: reg=0.01 and dense eigensolver after numerical failure")
        result["diagnostics"]["attempt_history"] = history
    result["input_sha256"] = request["input_sha256"]
    write_json(directory / "result.json", result)
    write_json(
        directory / "run_manifest.json",
        {
            "result_sha256": checksum(directory / "result.json"),
            "embedding_sha256": checksum(directory / "embedding.npy")
            if result["status"] == "success"
            else None,
        },
    )
    print(f"{method_id} seed={seed} {result['status']} {result['runtime_seconds']:.2f}s", flush=True)
    return result


def parameters(cfg, method_id, X):
    return dict(get_method(method_id).default_params(X), **cfg.methods.get(method_id, {}))
