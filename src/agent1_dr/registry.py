from importlib import import_module

METHOD_IDS = (
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
)
STOCHASTIC = {"mds", "gplvm", "tsne", "umap"}


def get_method(method_id):
    if method_id not in METHOD_IDS:
        raise ValueError(f"Unknown method {method_id}")
    return import_module(f"agent1_dr.methods.{method_id}").Method()
