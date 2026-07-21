"""Shared versioned contract for the spatial evaluation harness."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


SUBMISSION_SCHEMA_VERSION = "ms-vista-spatial-submission/v2"
RUN_MANIFEST_SCHEMA_VERSION = "ms-vista-spatial-run/v2"
BENCHMARK_MANIFEST_SCHEMA_VERSION = "ms-vista-spatial-benchmark/v2"
REPORT_SCHEMA_VERSION = "ms-vista-spatial-report/v2"
HARNESS_VERSION = "1.1.0"
ABLATION_SAMPLES_PER_DATASET = 100
JUDGE_REVISION = "Qwen3-30B-A3B-Instruct-2507"

MODES = ("main", "noimage", "noimgpp")
PROMPT_MODES = ("noncot", "cot")
MODE_LABELS = {
    "main": "main",
    "noimage": "no_image",
    "noimgpp": "no_image_plus",
}
REQUIRED_CONDITIONS = tuple(
    f"{MODE_LABELS[mode]}_{prompt_mode}"
    for mode in MODES
    for prompt_mode in PROMPT_MODES
)
ABLATION_MODES = {"noimage", "noimgpp"}

DATASETS = (
    "BLINK",
    "CV-Bench-2D",
    "CV-Bench-3D",
    "MMVP",
    "RealWorldQA",
    "VStarBench",
    "MMSIBench_wo_circular",
    "3DSRBench",
    "VSR_MCQ",
    "SpatialBench",
    "MindCube",
    "OmniSpatial",
    "SAT-Real",
)

DATASET_DISPLAY_NAMES = {
    "BLINK": "BLINK",
    "CV-Bench-2D": "CV-Bench (2D)",
    "CV-Bench-3D": "CV-Bench (3D)",
    "MMVP": "MMVP",
    "RealWorldQA": "RealWorldQA",
    "VStarBench": "V*Bench",
    "MMSIBench_wo_circular": "MMSI-Bench",
    "3DSRBench": "3DSRBench",
    "VSR_MCQ": "VSR",
    "SpatialBench": "SpatialBench",
    "MindCube": "MindCube",
    "OmniSpatial": "OmniSpatial",
    "SAT-Real": "SAT (Real)",
}

DATASET_TYPES = {
    "BLINK": "2D",
    "CV-Bench-2D": "2D",
    "MMVP": "2D",
    "RealWorldQA": "2D",
    "VStarBench": "2D",
    "VSR_MCQ": "2D",
    "SpatialBench": "2D",
    "CV-Bench-3D": "3D",
    "MMSIBench_wo_circular": "3D",
    "3DSRBench": "3D",
    "MindCube": "3D",
    "OmniSpatial": "3D",
    "SAT-Real": "dynamic",
}


def condition_for(mode: str, prompt_mode: str) -> str:
    if mode not in MODE_LABELS:
        raise ValueError(f"Unsupported spatial mode: {mode}")
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported spatial prompt mode: {prompt_mode}")
    return f"{MODE_LABELS[mode]}_{prompt_mode}"


def _safe_id_part(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Spatial question identifiers cannot be empty")
    return re.sub(r"\s+", "_", text)


def question_id(dataset: str, source_index: Any, rotation: int | None = None) -> str:
    base = f"{_safe_id_part(dataset)}:{_safe_id_part(source_index)}"
    return f"{base}:r{rotation}" if rotation is not None else base


def evaluation_group_id(dataset: str, source_index: Any) -> str:
    return f"{_safe_id_part(dataset)}:{_safe_id_part(source_index)}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def load_ablation_manifest(path: str | Path) -> Dict[str, set[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Ablation manifest must be a JSON object keyed by dataset")
    unknown = sorted(set(raw) - set(DATASETS))
    missing = sorted(set(DATASETS) - set(raw))
    if unknown or missing:
        raise ValueError(
            "Ablation manifest dataset mismatch: "
            f"unknown={unknown or 'none'}, missing={missing or 'none'}"
        )
    selected: Dict[str, set[str]] = {}
    for dataset in DATASETS:
        values = raw.get(dataset)
        if not isinstance(values, list) or len(values) != ABLATION_SAMPLES_PER_DATASET:
            raise ValueError(
                f"Ablation manifest entry for {dataset} must contain exactly "
                f"{ABLATION_SAMPLES_PER_DATASET} indices"
            )
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError(f"Ablation manifest entry for {dataset} contains an empty index")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Ablation manifest entry for {dataset} contains duplicate indices")
        selected[dataset] = set(normalized)
    return selected


def parse_csv_values(value: str, allowed: Iterable[str], label: str) -> list[str]:
    parsed = [item.strip() for item in str(value or "").split(",") if item.strip()]
    allowed_set = set(allowed)
    unknown = [item for item in parsed if item not in allowed_set]
    if unknown:
        raise ValueError(f"Unsupported {label}: {', '.join(unknown)}")
    if not parsed:
        raise ValueError(f"At least one {label} is required")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"Duplicate {label} values are not allowed")
    return parsed
