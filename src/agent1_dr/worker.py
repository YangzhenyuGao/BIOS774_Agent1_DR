"""One subprocess per fit, with atomic result files and full failure evidence."""

import random
import resource
import sys
import time
import traceback
import warnings
from dataclasses import asdict

import numpy as np

from .registry import get_method
from .schemas import MethodRunResult
from .utils import read_json, versions, write_json


def main():
    input_path, request_path, output = sys.argv[1:]
    req = read_json(request_path)
    random.seed(req["seed"])
    np.random.seed(req["seed"])
    t = time.monotonic()
    result = MethodRunResult(
        req["method_id"], req["parameters"], req["seed"], "failed", 0, package_versions=versions()
    )
    captured = []
    try:
        X = np.load(input_path)
        method = get_method(req["method_id"])
        params, adjustments = method.validate_input(X, req["parameters"])
        result.parameters = params
        result.warnings.extend(adjustments)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            Y, diagnostics = method.fit_transform(X, req["seed"], params)
        if Y.shape != (len(X), 2) or not np.isfinite(Y).all():
            raise ValueError("Invalid embedding shape or non-finite coordinates")
        np.save(output + "/embedding.npy", Y)
        result.embedding = "embedding.npy"
        result.diagnostics = diagnostics
        result.status = "success"
    except Exception as exc:  # noqa: BLE001 -- intentional isolation/audit boundary
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        result.diagnostics["traceback"] = traceback.format_exc()
    finally:
        result.warnings.extend(str(w.message) for w in captured)
        result.runtime_seconds = time.monotonic() - t
        result.peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        write_json(output + "/result.json", asdict(result))


if __name__ == "__main__":
    main()
