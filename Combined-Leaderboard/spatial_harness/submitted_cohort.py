"""Frozen identifiers for an imported, self-reported evaluation cohort.

This is not an official benchmark/protocol attestation. It binds comparisons
to the exact submitted sample catalogue without inventing data/prompt hashes.
"""

import re

SCHEMA_VERSION = "ms-vista-submitted-spatial-cohort/v1"
POLICY = {
    "scoring": "evaluation_group_all_rows_correct",
    "server_ground_truth_evaluation": False,
    "comparability": "same_cohort_only",
}


def validate_manifest(manifest, datasets, conditions):
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("scope_kind") != "submitted_sample_catalog"
        or manifest.get("official_protocol_attested") is not False
        or manifest.get("provenance_status") != "submitter_declared_unattested"
        or manifest.get("evaluation_policy") != POLICY
        or manifest.get("datasets") != list(datasets)
        or manifest.get("dataset_count") != len(datasets)
        or manifest.get("required_conditions") != list(conditions)
        or manifest.get("primary_condition") != "main_noncot"
    ):
        raise ValueError("Invalid self-reported cohort policy or dimensions")
    for field in ("benchmark_version", "harness_version", "label"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Submitted cohort is missing {field}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_archive_sha256", ""))):
        raise ValueError("Submitted cohort has no valid source archive digest")
    for field in ("condition_counts", "condition_group_counts"):
        _counts(manifest.get(field), conditions)
    for field in ("dataset_condition_counts", "dataset_condition_group_counts"):
        values = manifest.get(field)
        if not isinstance(values, dict) or set(values) != set(datasets):
            raise ValueError("Submitted cohort dataset counts are incomplete")
        for counts in values.values():
            _counts(counts, conditions)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"questions", "submission_template"}:
        raise ValueError("Submitted cohort public catalogue artifacts are incomplete")
    for key, filename in (("questions", "questions.jsonl"), ("submission_template", "submission_template.jsonl")):
        artifact = artifacts[key]
        if (
            not isinstance(artifact, dict)
            or artifact.get("filename") != filename
            or type(artifact.get("rows")) is not int or artifact["rows"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", "")))
        ):
            raise ValueError("Invalid submitted cohort catalogue artifact")
    return manifest


def _counts(values, conditions):
    if (
        not isinstance(values, dict) or set(values) != set(conditions)
        or any(type(value) is not int or value <= 0 for value in values.values())
    ):
        raise ValueError("Submitted cohort condition counts must be positive integers")
