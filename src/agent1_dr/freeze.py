"""Freeze validated dataset evidence: checksum every artifact and make it read-only."""

import os
import stat
import subprocess
from datetime import UTC, datetime

from .config import ROOT
from .utils import checksum, code_hash, read_json, write_json
from .validation import dataset_pdf, validate

FROZEN = ROOT / "outputs" / "FROZEN_MANIFEST.json"


def _read_only(path):
    mode = os.stat(path).st_mode
    os.chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def freeze(configs):
    for cfg in configs:
        result = validate(cfg)
        if result["status"] != "PASSED":
            raise ValueError(f"{cfg.dataset.name} must validate before freezing")
    files = {}
    for cfg in configs:
        for p in sorted(cfg.output.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(ROOT))] = checksum(p)
        files[str(dataset_pdf(cfg).relative_to(ROOT))] = checksum(dataset_pdf(cfg))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    record = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "git_commit": commit.stdout.strip() or None,
        "code_hash": code_hash(),
        "config_hashes": {cfg.dataset.name: cfg.hash for cfg in configs},
        "n_files": len(files),
        "files": files,
    }
    write_json(FROZEN, record)
    for cfg in configs:
        for p in [*cfg.output.rglob("*"), cfg.output]:
            if p.is_file() or p.is_dir():
                _read_only(p)
        _read_only(dataset_pdf(cfg))
    _read_only(FROZEN)
    return record


def verify_frozen():
    record = read_json(FROZEN)
    bad = [p for p, sha in record["files"].items() if not (ROOT / p).is_file() or checksum(ROOT / p) != sha]
    if bad:
        raise ValueError(f"Frozen evidence changed: {bad[:5]}{' ...' if len(bad) > 5 else ''}")
    return record
