import anndata
import numpy as np
import pandas as pd
import pytest

from agent1_dr.config import Config
from agent1_dr.data import load_data
from agent1_dr.data.pbmc3k import align_labels
from agent1_dr.sampling import cohort_indices


def test_alignment():
    a = anndata.AnnData(np.ones((2, 3)), obs=pd.DataFrame({"louvain": ["B", "T"]}, index=["b", "a"]))
    assert align_labels(["a", "b", "c"], a).tolist() == ["T", "B", "unannotated"]
    with pytest.raises(ValueError):
        align_labels(["a", "z"], a)


def test_cohort_metadata_roundtrip_without_pickle(tmp_path):
    from dataclasses import asdict

    from agent1_dr import pilot
    from agent1_dr.inspector import inspect_data

    cfg = Config()
    data = load_data(cfg)
    labels = pd.Series(data[1], dtype=object).to_numpy()
    ids = pd.Series(data[2], dtype=object).to_numpy()
    objects = (data[0], labels, ids, *data[3:])
    profile = asdict(inspect_data("synthetic", data[0], labels, {}, None))
    _, expected_labels, expected_ids = pilot.prepare_cohort(cfg, tmp_path, 30, objects, profile)
    for name, expected in [("labels", expected_labels), ("ids", expected_ids)]:
        saved = np.load(tmp_path / f"{name}.npy", allow_pickle=False)
        assert saved.dtype.kind == "U"
        np.testing.assert_array_equal(saved, expected)
    np.testing.assert_array_equal(labels.astype(str), data[1])


def test_deterministic_loader_and_sampling():
    a, b = load_data(Config()), load_data(Config())
    np.testing.assert_array_equal(a[0], b[0])
    labels = np.repeat(["a", "b", "c"], [70, 20, 10])
    idx = cohort_indices(100, 30, 774, labels)
    assert len(idx) == 30 and len(set(idx)) == 30
    assert dict(zip(*np.unique(labels[idx], return_counts=True))) == {"a": 21, "b": 6, "c": 3}
    np.testing.assert_array_equal(idx, cohort_indices(100, 30, 774, labels))


def test_preprocessing_fit_does_not_use_labels():
    from agent1_dr.preprocessing import prepare

    cfg = Config()
    data = load_data(cfg)
    changed = (data[0], np.random.default_rng(2).permutation(data[1]), *data[2:])
    a = prepare(cfg, data, len(data[2]))
    b = prepare(cfg, changed, len(data[2]))
    np.testing.assert_allclose(a[0], b[0])
    assert a[3]["labels_used_for_fit"] is False


def test_new_numeric_dataset_without_labels(tmp_path):
    from agent1_dr.config import Dataset
    from agent1_dr.preprocessing import prepare

    path = tmp_path / "data.npz"
    np.savez(path, X=np.random.default_rng(1).normal(size=(40, 5)))
    cfg = Config(dataset=Dataset(name="numeric", path=str(path)))
    data = load_data(cfg)
    assert data[1] is None
    X, labels, ids, plan, _ = prepare(cfg, data, 30)
    assert X.shape == (30, 5) and len(ids) == len(labels) == 30
    assert not plan["actions"][0]["parameters"]["stratified"]
