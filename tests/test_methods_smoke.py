import math

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_swiss_roll

from agent1_dr.execution import run_fits
from agent1_dr.registry import METHOD_IDS, get_method


@pytest.mark.parametrize("mid", METHOD_IDS)
@pytest.mark.parametrize("dataset", ["blobs", "swiss"])
def test_real_method(mid, dataset, tmp_path):
    X = (
        make_blobs(n_samples=45, n_features=5, random_state=4)[0]
        if dataset == "blobs"
        else make_swiss_roll(n_samples=45, random_state=4)[0]
    )
    X = (X - X.mean(0)) / X.std(0)
    method = get_method(mid)
    params = method.default_params(X)
    if mid == "gplvm":
        params.update(epochs=20, inducing_points=8, batch_size=44)
    np.save(tmp_path / "input.npy", X)
    # Two identical jobs in one warm batch must reproduce each other.
    jobs = [{"input_path": tmp_path / "input.npy", "seed": 774, "params": params, "directory": tmp_path / str(i)}
            for i in (0, 1)]
    results = run_fits(mid, jobs, 180)
    embeddings = []
    for job, r in zip(jobs, results):
        assert r["status"] == "success", r
        assert r["method_id"] == mid and r["parameters"] and r["package_versions"]
        assert r["timing"]["fit_wall_seconds"] == r["runtime_seconds"] >= 0
        y = np.load(job["directory"] / "embedding.npy")
        assert y.shape == (45, 2) and np.isfinite(y).all()
        embeddings.append(y)
        if mid == "gplvm":
            diag = r["diagnostics"]
            assert diag["optimization_steps"] == 20 * math.ceil(45 / 44)
            assert diag["latent_displacement"] > 1e-5 and diag["final_loss"] < diag["initial_loss"]
    # Eigenvector sign flips do not alter distances; test actual geometry reproducibility.
    from sklearn.metrics import pairwise_distances

    np.testing.assert_allclose(
        pairwise_distances(embeddings[0]), pairwise_distances(embeddings[1]), rtol=1e-3, atol=1e-3
    )


def test_failure_and_timeout_isolation(tmp_path):
    rng = np.random.default_rng(1)
    np.save(tmp_path / "small.npy", rng.normal(size=(10, 4)))
    jobs = [
        {"input_path": tmp_path / "small.npy", "seed": 1, "params": {"nonsense": 1}, "directory": tmp_path / "bad"},
        {"input_path": tmp_path / "small.npy", "seed": 1, "params": {"svd_solver": "full"}, "directory": tmp_path / "ok"},
    ]
    bad, ok = run_fits("pca", jobs, 60)
    assert bad["status"] == "failed" and bad["error_type"] and ok["status"] == "success"
    # A fit that outlives the per-fit timeout is killed; later fits in the batch still run.
    np.save(tmp_path / "large.npy", rng.normal(size=(3000, 10)))
    slow = {"perplexity": 30, "max_iter": 5000, "init": "pca", "n_jobs": 1, "learning_rate": "auto"}
    jobs = [
        {"input_path": tmp_path / "large.npy", "seed": 1, "params": slow, "directory": tmp_path / "slow"},
        {"input_path": tmp_path / "small.npy", "seed": 1, "params": {"perplexity": 3, "max_iter": 250,
                                                                     "init": "pca", "n_jobs": 1},
         "directory": tmp_path / "after"},
    ]
    timed_out, after = run_fits("tsne", jobs, 2)
    assert timed_out["status"] == "timeout" and after["status"] == "success"


def test_corrupted_run_is_recomputed_and_valid_cache_reused(tmp_path):
    X = np.random.default_rng(2).normal(size=(20, 4))
    np.save(tmp_path / "input.npy", X)
    job = {"input_path": tmp_path / "input.npy", "seed": 4, "params": {"svd_solver": "full"},
           "directory": tmp_path / "run"}
    run_fits("pca", [dict(job)], 60)
    expected = np.load(tmp_path / "run/embedding.npy")
    stamp = (tmp_path / "run/result.json").stat().st_mtime_ns
    run_fits("pca", [dict(job)], 60)
    assert (tmp_path / "run/result.json").stat().st_mtime_ns == stamp
    np.save(tmp_path / "run/embedding.npy", np.zeros((20, 2)))
    run_fits("pca", [dict(job)], 60)
    np.testing.assert_allclose(np.load(tmp_path / "run/embedding.npy"), expected)
