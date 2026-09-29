"""Serializable evidence contracts."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetProfile:
    name: str
    n_samples: int
    n_features: int
    statistics: dict
    provenance: dict


@dataclass
class PreprocessingPlan:
    actions: list[dict]
    labels_used_for_fit: bool = False


@dataclass
class MethodRunResult:
    method_id: str
    parameters: dict
    seed: int
    status: str
    runtime_seconds: float
    peak_memory_mb: float | None = None
    warnings: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    package_versions: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    embedding: Any = None


@dataclass
class EvaluationResult:
    trustworthiness: float
    neighbor_recall: float
    silhouette: float | None
    silhouette_note: str
    k: int
    distance_metric: str = "euclidean"
    metric_sample_size: int = 0


@dataclass
class SelectionDecision:
    method: str
    prior_suitability: str
    pilot_evidence: dict
    decision: str
    reason: str


@dataclass
class StageManifest:
    stage: str
    config_hash: str
    code_hash: str
    created_at: str
    files: dict
    package_versions: dict
    git_commit: str | None
    hostname: str
    job_id: str | None


@dataclass
class ReportEvidenceBundle:
    dataset: str
    files: dict
    reporting_mode: str
    validated: bool
