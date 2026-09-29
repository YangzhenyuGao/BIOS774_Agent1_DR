from agent1_dr.registry import METHOD_IDS, get_method


def test_registry():
    assert len(METHOD_IDS) == 10
    assert set(METHOD_IDS) == {
        "pca",
        "kernel_pca",
        "mds",
        "isomap",
        "lle",
        "laplacian",
        "diffusion_maps",
        "gplvm",
        "tsne",
        "umap",
    }
    for mid in METHOD_IDS:
        assert get_method(mid).method_id == mid
