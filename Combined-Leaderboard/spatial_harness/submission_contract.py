"""Versioned public contract for the paper-aligned Track-3 harness."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from spatial_harness.run_track3_vllm import (
    DATASETS,
    HARNESS_CONTRACT,
    MODES,
    PROMPT_MODES,
)


BENCHMARK_MANIFEST_SCHEMA_VERSION = "ms-vista-spatial-benchmark/v3"
SUBMISSION_SCHEMA_VERSION = "ms-vista-spatial-submission/v3"
RUN_MANIFEST_SCHEMA_VERSION = "ms-vista-spatial-run/v3"
REPORT_SCHEMA_VERSION = "ms-vista-spatial-report/v3"
HARNESS_VERSION = "2.0.0"
INFERENCE_FAILURE_SCHEMA_VERSION = "ms-vista-track3-inference-failure/v1"
INFERENCE_FAILURE_POLICY = "context-limit-terminal-failure-v1"
INFERENCE_FAILURE_DISPOSITION = "scored_incorrect_without_judge"
INFERENCE_FAILURE_JUDGE_METHOD = "inference_failure"
INFERENCE_FAILURE_MCQ_ANSWER = "0"
INFERENCE_FAILURE_VQA_ANSWER = "INFERENCE_FAILED"

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
DATASET_TYPES = {
    "BLINK": "2D",
    "CV-Bench-2D": "2D",
    "CV-Bench-3D": "3D",
    "MMVP": "2D",
    "RealWorldQA": "2D",
    "VStarBench": "2D",
    "MMSIBench_wo_circular": "3D",
    "3DSRBench": "3D",
    "VSR_MCQ": "2D",
    "SpatialBench": "2D",
    "MindCube": "3D",
    "OmniSpatial": "3D",
    "SAT-Real": "dynamic",
}
EVALUATION_POLICY = {
    "answer_type": {
        "main": "mcq_and_vqa",
        "noimage": "mcq_and_vqa",
        "noimgpp": "text_mcq_only",
    },
    "noimage_image": "same_size_gray",
    "noimgpp_image": "same_size_gray",
    "noimgpp_answer": "Cannot determine from the image",
    "circular_modes": ["main"],
    "circular_datasets": ["SAT-Real", "SpatialBench"],
    "scoring": "all_rotations_correct_within_evaluation_group",
}
JUDGE_DECODING = {
    "strategy": "greedy",
    "temperature": 0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
    "max_tokens": 4,
}


def condition_for(mode: str, prompt_mode: str) -> str:
    try:
        mode_label = MODE_LABELS[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported Track-3 mode: {mode}") from exc
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported Track-3 prompt mode: {prompt_mode}")
    return f"{mode_label}_{prompt_mode}"


def _safe_id_part(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Track-3 identifiers cannot be empty")
    return re.sub(r"\s+", "_", text)


def public_question_id(dataset: str, index: Any) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported Track-3 dataset: {dataset}")
    return f"{_safe_id_part(dataset)}:{_safe_id_part(index)}"


def public_evaluation_group(dataset: str, group: Any) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported Track-3 dataset: {dataset}")
    return f"{_safe_id_part(dataset)}:{_safe_id_part(group)}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def validate_contract_identity(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != BENCHMARK_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Track-3 benchmark manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("harness_contract") != HARNESS_CONTRACT:
        raise ValueError("Track-3 benchmark manifest uses a different evaluation protocol")
    if manifest.get("harness_version") != HARNESS_VERSION:
        raise ValueError("Track-3 benchmark manifest uses a different harness version")
    if manifest.get("datasets") != list(DATASETS):
        raise ValueError("Track-3 benchmark manifest does not contain the canonical datasets")
    if manifest.get("required_conditions") != list(REQUIRED_CONDITIONS):
        raise ValueError("Track-3 benchmark manifest does not contain all six conditions")
    if manifest.get("evaluation_policy") != EVALUATION_POLICY:
        raise ValueError("Track-3 benchmark manifest evaluation policy is not canonical")
