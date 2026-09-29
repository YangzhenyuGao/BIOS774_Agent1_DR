"""Small CPU import, GP and one-page PDF environment checks."""
import importlib
import json
from pathlib import Path
import torch
import gpytorch
from weasyprint import HTML
from pypdf import PdfReader
from agent1_dr.utils import versions

modules = ['numpy', 'pandas', 'scipy', 'sklearn', 'anndata', 'scanpy', 'umap', 'pydiffmap',
           'medmnist', 'torch', 'gpytorch', 'matplotlib', 'seaborn', 'yaml', 'joblib', 'tqdm',
           'psutil', 'jinja2', 'weasyprint', 'pypdf', 'openai', 'pytest']
for name in modules:
    importlib.import_module(name)
    print(f'IMPORT PASS {name}', flush=True)
x = torch.tensor([[0.], [1.]])
k = gpytorch.kernels.RBFKernel()(x).to_dense()
assert torch.isfinite(k).all()
HTML(string='<h1>Agent environment PDF smoke test</h1>').write_pdf('logs/environment_test.pdf')
assert len(PdfReader('logs/environment_test.pdf').pages) == 1
Path('logs/package_versions.json').write_text(json.dumps(versions(), indent=2))
print('CPU GP and PDF checks PASSED')
