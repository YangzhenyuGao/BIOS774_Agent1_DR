import hashlib
import json
import os
import socket
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .config import ROOT
from .schemas import StageManifest

PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "anndata",
    "scanpy",
    "umap-learn",
    "pydiffmap",
    "medmnist",
    "torch",
    "gpytorch",
    "matplotlib",
    "weasyprint",
    "pypdf",
    "openai",
]


def versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "NOT INSTALLED"
    return result


def json_default(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, default=json_default, indent=2, allow_nan=False))
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def fit_code_hash():
    """Hash of the code that can change a fitted embedding (method wrappers and worker)."""
    h = hashlib.sha256()
    package = ROOT / "src/agent1_dr"
    files = sorted((package / "methods").glob("*.py")) + [
        package / name for name in ("worker.py", "registry.py", "schemas.py")
    ]
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def manifest(stage, cfg, paths):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return StageManifest(
        stage,
        cfg.hash,
        code_hash(),
        datetime.now(UTC).isoformat(),
        {str(Path(p).relative_to(cfg.output)): checksum(p) for p in paths},
        versions(),
        commit.stdout.strip() if commit.returncode == 0 else None,
        socket.gethostname(),
        os.environ.get("SLURM_JOB_ID"),
    )


def complete(stage, cfg, directory):
    directory = Path(directory)
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.name not in {"stage_manifest.json", "pilot_complete.flag"}
        and not p.name.endswith(".tmp")
        and not p.name.startswith(".")
    ]
    write_json(directory / "stage_manifest.json", manifest(stage, cfg, files))


def valid_stage(cfg, directory):
    try:
        m = read_json(Path(directory) / "stage_manifest.json")
        return (
            m["config_hash"] == cfg.hash
            and m["code_hash"] == code_hash()
            and all(
                (cfg.output / p).is_file() and checksum(cfg.output / p) == sha
                for p, sha in m["files"].items()
            )
        )
    except (OSError, ValueError, KeyError):
        return False
