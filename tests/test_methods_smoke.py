import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_swiss_roll

from agent1_dr.execution import run_method
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
    results = []
    for repeat in [0, 1]:
        directory = tmp_path / str(repeat)
        r = run_method(mid, tmp_path / "input.npy", 774, params, directory, 180)
        assert r["status"] == "success", r
        assert r["method_id"] == mid and r["parameters"] and r["package_versions"]
        y = np.load(directory / "embedding.npy")
        assert y.shape == (45, 2) and np.isfinite(y).all()
        results.append(y)
        if mid == "gplvm":
            diag = r["diagnostics"]
            assert diag["optimization_steps"] == 20 and diag["latent_displacement"] > 1e-5
            assert diag["final_loss"] < diag["initial_loss"]
    # Eigenvector sign flips do not alter distances; test actual geometry reproducibility.
    from sklearn.metrics import pairwise_distances

    np.testing.assert_allclose(
        pairwise_distances(results[0]), pairwise_distances(results[1]), rtol=1e-3, atol=1e-3
    )


def test_failure_and_timeout_isolation(tmp_path):
    np.save(tmp_path / "input.npy", np.ones((10, 4)))
    r = run_method("pca", tmp_path / "input.npy", 1, {"nonsense": 1}, tmp_path / "bad", 60)
    assert r["status"] == "failed" and r["error_type"]
    np.save(tmp_path / "input.npy", np.random.default_rng(1).normal(size=(10, 4)))
    r = run_method("pca", tmp_path / "input.npy", 1, {"svd_solver": "full"}, tmp_path / "ok", 60)
    assert r["status"] == "success"
    r = run_method("pca", tmp_path / "input.npy", 1, {"svd_solver": "full"}, tmp_path / "timeout", 0.0001)
    assert r["status"] == "timeout"


def test_corrupted_run_is_recomputed(tmp_path):
    X = np.random.default_rng(2).normal(size=(20, 4))
    np.save(tmp_path / "input.npy", X)
    directory = tmp_path / "run"
    run_method("pca", tmp_path / "input.npy", 4, {"svd_solver": "full"}, directory, 60)
    expected = np.load(directory / "embedding.npy")
    np.save(directory / "embedding.npy", np.zeros((20, 2)))
    run_method("pca", tmp_path / "input.npy", 4, {"svd_solver": "full"}, directory, 60)
    np.testing.assert_allclose(np.load(directory / "embedding.npy"), expected)
