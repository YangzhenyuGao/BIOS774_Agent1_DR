"""Run twice on compute node: verify official cached loaders and tiny preprocessing."""
from pathlib import Path
from agent1_dr.config import ROOT, load_config
from agent1_dr.data import load_data
from agent1_dr.preprocessing import prepare
from agent1_dr.utils import write_json
import numpy as np

records = []
for name in ['pbmc3k', 'pathmnist']:
    cfg = load_config(ROOT/f'config/{name}.yaml')
    first = load_data(cfg)
    files = list((ROOT/f'data/raw/{name}').glob('*'))
    before = {str(p): p.stat().st_mtime_ns for p in files if p.is_file()}
    second = load_data(cfg)
    after = {str(p): p.stat().st_mtime_ns for p in files if p.is_file()}
    assert before == after, 'Second load rewrote cached data'
    assert first[3]['files'] == second[3]['files']
    np.testing.assert_array_equal(first[2], second[2])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[0].shape == second[0].shape
    X, labels, ids, plan, cohort = prepare(cfg, first, 60)
    assert X.shape[0] == len(ids) == len(labels) == 60 and np.isfinite(X).all()
    records.append(dict(dataset=name, raw_shape=list(first[0].shape), tiny_shape=list(X.shape),
                        cache_unchanged=True, provenance=first[3], preprocessing_plan=plan))
    print(f'{name}: real cached-loader and tiny preprocessing PASSED', flush=True)
write_json(Path('logs/data_checks.json'), records)
