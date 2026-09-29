import numpy as np
from scipy import sparse

from agent1_dr.config import Config
from agent1_dr.inspector import inspect_data
from agent1_dr.planner import plan


def profile(X, names=None):
    from dataclasses import asdict

    return asdict(inspect_data("x", X, None, {}, names))


def test_count_matrix_branch_from_profile_not_name():
    rng = np.random.default_rng(0)
    X = sparse.random(50, 1500, density=0.05, random_state=0, data_rvs=lambda n: rng.integers(1, 20, n)).tocsr()
    names = np.array([f"MT-{i}" if i < 5 else f"G{i}" for i in range(1500)])
    p = plan(profile(X, names), Config())
    assert p["branch"] == "count_matrix"
    qc = p["steps"][0]
    assert qc["action"] == "observation_qc" and "max_mito_fraction" in qc["parameters"]
    assert "sparsity 0.95" in qc["reason"]


def test_bounded_intensity_and_continuous_branches():
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 256, size=(40, 300), dtype=np.uint8)
    assert plan(profile(pixels), Config())["branch"] == "bounded_intensity"
    assert plan(profile(rng.normal(size=(40, 6))), Config())["branch"] == "continuous"
    counts_but_narrow = rng.integers(0, 3, size=(40, 20)).astype(float)
    assert plan(profile(counts_but_narrow), Config())["branch"] != "count_matrix"
