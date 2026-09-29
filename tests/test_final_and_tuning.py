from agent1_dr.config import Config
from agent1_dr.final import project, scaling_model
from agent1_dr.tuning import choose


def probe(method, exponent, seconds_at_1000, memory_growth=0.0):
    return [
        {"method": method, "n": n, "status": "success", "seconds": seconds_at_1000 * (n / 1000) ** exponent,
         "peak_memory_mb": 300 + memory_growth * (n / 1000) ** 2}
        for n in (500, 1000, 2000, 4000)
    ]


def test_scaling_model_recovers_exponents():
    model = scaling_model(probe("isomap", 2.0, 1.0, memory_growth=50) + probe("pca", 1.0, 0.01))
    assert abs(model["isomap"]["time_exponent"] - 2.0) < 1e-9
    # Increments above the smallest probe over-estimate a + c*n^2 growth: conservative by design.
    assert 2.0 <= model["isomap"]["memory_exponent"] < 2.5
    assert model["pca"]["memory_exponent"] == 1.0  # no measurable growth: conservative default


def test_projection_budget_binds_for_quadratic_method():
    cfg = Config()
    model = scaling_model(probe("isomap", 2.0, 1.0) + probe("pca", 1.0, 0.01))
    # 2x safety: 2 * (n/1000)^2 s must stay within the 40-minute final budget -> n <= ~34,600
    assert project(model, 20000, cfg)["feasible"]
    assert not project(model, 40000, cfg)["feasible"]
    cheap = scaling_model(probe("pca", 1.0, 0.01))
    assert project(cheap, 20000, cfg)["feasible"]


def rows(qualities, severe=()):
    return [
        {"grid_index": i, "quality": q, "severe_warning": i in severe, "severe_reasons": ["x"] if i in severe else [],
         "status": "success", "runtime_seconds": 1.0}
        for i, q in enumerate(qualities)
    ]


def test_tuning_keeps_default_within_margin():
    cfg = Config()
    winner, reason = choose(rows([0.600, 0.603]), {}, cfg)
    assert winner["grid_index"] == 0 and "default kept" in reason
    winner, _ = choose(rows([0.600, 0.650]), {}, cfg)
    assert winner["grid_index"] == 1
    winner, reason = choose(rows([0.700, 0.650], severe=(0,)), {}, cfg)
    assert winner["grid_index"] == 1 and "ineligible" in reason
