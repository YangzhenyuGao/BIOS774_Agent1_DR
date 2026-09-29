import pytest

from agent1_dr.config import ROOT, Config, Pilot, Selection, load_config


def test_invalid_config():
    with pytest.raises(ValueError):
        Config(pilot=Pilot(max_samples=10))
    with pytest.raises(ValueError):
        Config(selection=Selection(max_final_methods=1))
    with pytest.raises(ValueError):
        Config(selection=Selection(use_labels_for_selection=True))


def test_config_hash():
    a = load_config(ROOT / "config/pbmc3k.yaml")
    b = load_config(ROOT / "config/pathmnist.yaml")
    assert a.hash != b.hash
    assert a.hash == load_config(ROOT / "config/pbmc3k.yaml").hash
