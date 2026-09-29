"""Strict nested configuration; relative paths are rooted at the project."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Project:
    seed: int = 774
    output_root: str = "outputs"


@dataclass
class Dataset:
    name: str = "synthetic"
    split: str = "train"
    max_final_samples: int = 3000
    synthetic_samples: int = 90
    path: str = ""
    label_column: str = ""
    id_column: str = ""


@dataclass
class Pilot:
    max_samples: int = 1000
    n_repeats_stochastic: int = 3
    timeout_minutes_per_run: float = 30
    n_neighbors_eval: int = 15
    n_neighbors_secondary: int = 50
    n_subsamples: int = 5
    subsample_fraction: float = 0.8


@dataclass
class Selection:
    retain_pca_baseline: bool = True
    max_final_methods: int = 4
    quality_relative_to_best: float = 0.90
    use_labels_for_selection: bool = False
    # Pre-registered v2 quality weights (docs/METHOD_NOTES.md); components lie in [0, 1].
    weight_trustworthiness: float = 0.35
    weight_recall: float = 0.35
    weight_global: float = 0.20
    weight_stability: float = 0.10
    # Practical-equivalence floor for quality differences and runtime materiality.
    equivalence_floor: float = 0.005
    runtime_material_ratio: float = 2.0
    runtime_material_seconds: float = 1.0
    # Severe diagnostics: collapsed layout, near-disconnected diffusion graph.
    collapse_absolute: float = 0.05
    collapse_relative: float = 0.25
    spectral_gap_min: float = 1e-6


@dataclass
class Preprocessing:
    min_genes: int = 200
    min_cells: int = 3
    max_genes: int = 2500
    max_mito_fraction: float = 0.05
    n_top_genes: int = 2000
    pca_components: int = 50
    clip: float = 10
    # Profile rules: a count matrix is sparse, non-negative, integer-valued and wide.
    count_min_sparsity: float = 0.5
    count_min_features: int = 1000


@dataclass
class Final:
    probe_sizes: list = field(default_factory=lambda: [500, 1000, 2000, 4000])
    size_ladder: list = field(
        default_factory=lambda: [1000, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 15000, 20000]
    )
    time_budget_minutes: float = 40
    memory_budget_gb: float = 24
    safety_factor: float = 2.0


@dataclass
class Reporting:
    mode: str = "deterministic"
    fail_if_llm_requested_but_unavailable: bool = True
    model: str = "gpt-4.1-mini"
    # Deliverable name under reports/ (for example generated_report_1); empty keeps it in outputs/.
    report_name: str = ""


@dataclass
class Config:
    project: Project = field(default_factory=Project)
    dataset: Dataset = field(default_factory=Dataset)
    pilot: Pilot = field(default_factory=Pilot)
    selection: Selection = field(default_factory=Selection)
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    reporting: Reporting = field(default_factory=Reporting)
    final: Final = field(default_factory=Final)
    methods: dict = field(default_factory=dict)

    def __post_init__(self):
        # Dataclass annotations alone do not validate runtime YAML types.
        for section in (
            self.project,
            self.dataset,
            self.pilot,
            self.selection,
            self.preprocessing,
            self.reporting,
            self.final,
        ):
            for name, definition in section.__dataclass_fields__.items():
                value = getattr(section, name)
                expected = definition.type
                if expected is float:
                    valid = type(value) in (int, float)
                else:
                    valid = type(value) is expected
                if not valid:
                    raise ValueError(f"{type(section).__name__}.{name} has invalid type")
        from .registry import METHOD_IDS

        if not isinstance(self.methods, dict) or any(k not in METHOD_IDS for k in self.methods):
            raise ValueError("methods must map supported method IDs to parameters")
        if self.dataset.name not in {"pbmc3k", "pathmnist", "synthetic", "numeric"}:
            raise ValueError("dataset.name must be pbmc3k, pathmnist, synthetic or numeric")
        if self.dataset.name == "numeric" and not self.dataset.path:
            raise ValueError("numeric datasets require dataset.path")
        if self.dataset.split not in {"train", "val", "test"}:
            raise ValueError("invalid dataset split")
        if self.pilot.max_samples < 6 or self.dataset.max_final_samples < 6:
            raise ValueError("cohorts must contain at least 6 observations")
        if self.pilot.n_repeats_stochastic < 2 or self.pilot.timeout_minutes_per_run <= 0:
            raise ValueError("need >=2 stochastic repeats and a positive timeout")
        if not 1 <= self.pilot.n_neighbors_eval < self.pilot.max_samples / 2:
            raise ValueError("require 1 <= k < pilot.max_samples/2")
        if not 2 <= self.selection.max_final_methods <= 10:
            raise ValueError("max_final_methods must be 2..10")
        if not 0 < self.selection.quality_relative_to_best <= 1:
            raise ValueError("quality threshold must be (0,1]")
        if self.selection.use_labels_for_selection:
            raise ValueError("Label-based selection is intentionally unsupported in this baseline")
        if self.reporting.mode not in {"deterministic", "openai"}:
            raise ValueError("reporting.mode must be deterministic or openai")
        if self.preprocessing.pca_components < 2 or self.preprocessing.n_top_genes < 2:
            raise ValueError("at least two features/components required")
        sel = self.selection
        w = [sel.weight_trustworthiness, sel.weight_recall, sel.weight_global, sel.weight_stability]
        if min(w) < 0 or abs(sum(w) - 1) > 1e-9:
            raise ValueError("quality weights must be non-negative and sum to 1")
        if sel.equivalence_floor < 0 or sel.runtime_material_ratio < 1 or sel.runtime_material_seconds < 0:
            raise ValueError("invalid equivalence or runtime-materiality setting")
        if not 0 < sel.collapse_absolute < 1 or not 0 < sel.collapse_relative <= 1:
            raise ValueError("collapse thresholds must lie in (0, 1)")
        if self.pilot.n_subsamples < 2 or not 0.5 < self.pilot.subsample_fraction < 1:
            raise ValueError("need >=2 subsamples with fraction in (0.5, 1)")
        if self.pilot.n_neighbors_secondary < self.pilot.n_neighbors_eval:
            raise ValueError("secondary neighborhood must be at least the primary k")
        f = self.final
        if not f.probe_sizes or min(f.probe_sizes) < 6 or f.safety_factor < 1 or f.time_budget_minutes <= 0:
            raise ValueError("invalid final-cohort planning settings")
        if self.preprocessing.min_genes < 0 or self.preprocessing.min_cells < 1:
            raise ValueError("invalid QC limits")
        if not 0 <= self.preprocessing.max_mito_fraction <= 1:
            raise ValueError("mitochondrial fraction must be [0,1]")

    @property
    def output(self):
        return ROOT / self.project.output_root / self.dataset.name

    @property
    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def load_config(path):
    data = yaml.safe_load((ROOT / "config/base.yaml").read_text()) or {}
    override = yaml.safe_load(Path(path).read_text()) or {}
    for key, value in override.items():
        if isinstance(value, dict):
            data.setdefault(key, {}).update(value)
        else:
            data[key] = value
    types = {
        "project": Project,
        "dataset": Dataset,
        "pilot": Pilot,
        "selection": Selection,
        "preprocessing": Preprocessing,
        "reporting": Reporting,
        "final": Final,
    }
    return Config(**{k: types[k](**v) if k in types else v for k, v in data.items()})
