#!/usr/bin/env python3
"""Repackage reviewed external v2 runs into a separate self-reported cohort.

Never runs archive code, inference, or a judge. Source verdicts supply claimed
credit only; private contact/answer-key fields are excluded from public files.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from config import EVAL_CONDITIONS, SPATIAL_DATASET_KEYS
from review_track3_v2 import COMBOS, META_KEYS, check_archive, claimed_credit, load_rows, review_model
from spatial_harness.artifact_package import (
    ANSWERS_SCHEMA_VERSION, RAW_OUTPUTS_SCHEMA_VERSION, ARCHIVE_NAME,
    PACKAGE_SCHEMA_VERSION, VERIFICATION_LEVEL, SCORE_SOURCE, SCORE_UNIT,
    aggregate_claimed_scores, canonical_json_bytes, sha256_bytes, sha256_file,
    write_artifact_package, write_gzip_jsonl,
)
from spatial_harness.submitted_cohort import SCHEMA_VERSION, POLICY
from spatial_submission import _load_public_spatial_contract

VERSION = "spatial-cot-eval-v2-submitted-2026-08-12"
MODEL_DETAILS = {
    "gpt-5_2025-08-07": ("GPT-5 (2025-08-07)", "OpenAI", "closed"),
    "gpt-4o_2024-11-20": ("GPT-4o (2024-11-20)", "OpenAI", "closed"),
    "google/gemma-4-31B-it": ("Gemma-4-31B-it", "Google", "open_weights"),
}


def condition(combo):
    return combo.replace("noimgpp", "no_image_plus").replace("noimg", "no_image")


def sample_catalog(archive, base):
    catalog = {}
    for combo in COMBOS:
        for (dataset, index), row in load_rows(archive, base + f"predictions/pred_{combo}.jsonl", combo, verdict=False).items():
            question_id = f"{dataset}:{index}"
            metadata = {"question_id": question_id, "dataset_key": dataset,
                        "evaluation_group": f"{dataset}:{row['group']}", "answer_type": row["answer_type"]}
            previous = catalog.setdefault(question_id, {**metadata, "conditions": []})
            if any(previous[key] != value for key, value in metadata.items()):
                raise ValueError("Question metadata changes across conditions")
            previous["conditions"].append(condition(combo))
    return [catalog[key] for key in sorted(catalog)]


def write_contract(destination, catalog, source_sha256):
    destination.mkdir()
    questions = b"".join(canonical_json_bytes(row) for row in catalog)
    template = b"".join(canonical_json_bytes({"question_id": row["question_id"], "condition": cond, "answer": ""})
                        for cond in EVAL_CONDITIONS for row in catalog if cond in row["conditions"])
    counts = {dataset: Counter() for dataset in SPATIAL_DATASET_KEYS}
    groups = {dataset: defaultdict(set) for dataset in SPATIAL_DATASET_KEYS}
    for row in catalog:
        for cond in row["conditions"]:
            counts[row["dataset_key"]][cond] += 1
            groups[row["dataset_key"]][cond].add(row["evaluation_group"])
    dataset_counts = {dataset: dict(counts[dataset]) for dataset in SPATIAL_DATASET_KEYS}
    dataset_groups = {dataset: {cond: len(groups[dataset][cond]) for cond in EVAL_CONDITIONS} for dataset in SPATIAL_DATASET_KEYS}
    manifest = {
        "schema_version": SCHEMA_VERSION, "benchmark_version": VERSION,
        "label": "Submitted v2 cohort (2026-08-12)", "scope_kind": "submitted_sample_catalog",
        "official_protocol_attested": False, "provenance_status": "submitter_declared_unattested",
        "source_archive_sha256": source_sha256, "harness_version": "spatial-cot-eval-v2",
        "evaluation_policy": POLICY, "datasets": SPATIAL_DATASET_KEYS,
        "dataset_count": len(SPATIAL_DATASET_KEYS), "required_conditions": EVAL_CONDITIONS,
        "primary_condition": "main_noncot", "unique_question_ids": len(catalog),
        "dataset_condition_counts": dataset_counts, "dataset_condition_group_counts": dataset_groups,
        "condition_counts": {cond: sum(values[cond] for values in dataset_counts.values()) for cond in EVAL_CONDITIONS},
        "condition_group_counts": {cond: sum(values[cond] for values in dataset_groups.values()) for cond in EVAL_CONDITIONS},
        "artifacts": {
            "questions": {"filename": "questions.jsonl", "rows": len(catalog), "sha256": sha256_bytes(questions)},
            "submission_template": {"filename": "submission_template.jsonl", "rows": sum(sum(c.values()) for c in counts.values()), "sha256": sha256_bytes(template)},
        },
    }
    (destination / "questions.jsonl").write_bytes(questions)
    (destination / "submission_template.jsonl").write_bytes(template)
    manifest_bytes = canonical_json_bytes(manifest)
    (destination / "manifest.json").write_bytes(manifest_bytes)
    _load_public_spatial_contract(manifest_bytes, template, questions, allow_submitted_cohort=True)
    return manifest_bytes


def converted_rows(archive, base):
    for combo in COMBOS:
        predictions = load_rows(archive, base + f"predictions/pred_{combo}.jsonl", combo, verdict=False)
        verdicts = load_rows(archive, base + f"verdicts/judged_{combo}.jsonl", combo, verdict=True)
        if predictions.keys() != verdicts.keys():
            raise ValueError("Prediction and verdict keys differ")
        for key in sorted(predictions):
            pred, verdict = predictions[key], verdicts[key]
            if any(pred[field] != verdict[field] for field in META_KEYS):
                raise ValueError("Paired metadata differs")
            missing = not (pred["output"] or "").strip()
            credit = int(claimed_credit(verdict))
            if missing and credit:
                raise ValueError("Missing output cannot receive positive claimed credit")
            identity = {"dataset": pred["dataset"], "question_id": f"{pred['dataset']}:{pred['index']}",
                        "evaluation_group": f"{pred['dataset']}:{pred['group']}", "condition": condition(combo)}
            answer = {
                **identity, "schema_version": ANSWERS_SCHEMA_VERSION, "answer_type": pred["answer_type"],
                "final_answer": "MISSING_SOURCE_OUTPUT" if missing else (verdict["judged"] if pred["answer_type"] == "mcq" else pred["output"]),
                "claimed_credit": credit,
            }
            raw = {**identity, "schema_version": RAW_OUTPUTS_SCHEMA_VERSION,
                   "raw_output": pred["output"], "output_status": "missing_source_output" if missing else "present",
                   "submitted_judge_output": verdict["judged"],
                   "final_answer_origin": "missing_output_marker" if missing else ("submitted_judge_label" if pred["answer_type"] == "mcq" else "source_model_output")}
            yield answer, raw


def prepare(source: Path, output: Path):
    if output.exists():
        raise ValueError("Output already exists; use a new private directory")
    source_digest = sha256_file(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"source_archive_sha256": source_digest, "benchmark_version": VERSION, "models": []}
    with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        check_archive(archive)
        stage = Path(temporary) / "bundle"
        stage.mkdir(mode=0o700)
        bases = sorted(name.removesuffix("metadata.json") for name in archive.namelist() if name.endswith("__track3/metadata.json"))
        if len(bases) != len(MODEL_DETAILS):
            raise ValueError("Expected exactly the three reviewed model exports")
        catalog = None
        model_names = set()
        for base in bases:
            review = review_model(archive, base, {})
            if not review["submitted_arithmetic_consistent"] or any(item["empty_outputs_with_claimed_credit"] for item in review["conditions"].values()):
                raise ValueError("Submitted arithmetic or missing-output credit is inconsistent")
            current = sample_catalog(archive, base)
            if catalog is not None and current != catalog:
                raise ValueError("Models do not have identical sample catalogues")
            catalog = current
        manifest_bytes = write_contract(stage / "contract", catalog, source_digest)
        for base in bases:
            metadata = json.loads(archive.read(base + "metadata.json"))
            model_id = metadata.get("model_id")
            if model_id not in MODEL_DETAILS or model_id in model_names or metadata.get("kit_version") != "spatial-cot-eval-v2" or metadata.get("date") != "2026-08-12":
                raise ValueError("Unknown/duplicate model or source harness")
            model_names.add(model_id)
            display, organization, access = MODEL_DETAILS[model_id]
            target = stage / base.rstrip("/").rsplit("/", 1)[-1]
            target.mkdir()
            answers, raw_rows = [], []
            for answer, raw in converted_rows(archive, base):
                answers.append(answer)
                raw_rows.append(raw)
            missing = sum(row["output_status"] == "missing_source_output" for row in raw_rows)
            configuration = {key: metadata.get(key) for key in ("model_id", "date", "decoding", "judge_model", "kit_version")}
            manifest = {
                "schema_version": PACKAGE_SCHEMA_VERSION, "verification_level": VERIFICATION_LEVEL, "score_source": SCORE_SOURCE,
                "model": {"name": model_id, "display_name": display, "revision": None, "access": access, "organization": organization},
                "benchmark": {"version": VERSION, "manifest_sha256": sha256_bytes(manifest_bytes)},
                "evaluation": {"harness_contract": "spatial-cot-eval-v2", "harness_version": "submitter-declared-v2",
                               "configuration_sha256": sha256_bytes(canonical_json_bytes(configuration)), "submitted_configuration": configuration,
                               "judge": {"name": metadata.get("judge_model"), "revision": None},
                               "provenance_status": "submitter_declared_unattested", "source_archive_sha256": source_digest},
                "evidence": {"answer_rows": len(answers), "raw_output_rows": len(raw_rows), "missing_output_rows": missing,
                             "raw_output_scope": "source_outputs_with_missing_records" if missing else "complete_model_response",
                             "missing_output_policy": "Source null/empty output retained; MISSING_SOURCE_OUTPUT is an administrative marker, not a model response; original zero credit retained; failure cause unknown."},
                "scoring": {"source": SCORE_SOURCE, "unit": SCORE_UNIT},
            }
            write_gzip_jsonl(target / "answers.jsonl.gz", answers)
            write_gzip_jsonl(target / "raw_outputs.jsonl.gz", raw_rows)
            package = write_artifact_package(target / ARCHIVE_NAME, manifest,
                                             aggregate_claimed_scores(answers, SPATIAL_DATASET_KEYS, EVAL_CONDITIONS),
                                             target / "answers.jsonl.gz", target / "raw_outputs.jsonl.gz")
            summary["models"].append({"model": display, "source_model": model_id, "package": str(package.relative_to(stage)),
                                       "rows": len(answers), "missing_output_rows": missing, "package_sha256": sha256_file(package)})
            del answers, raw_rows
        (stage / "review.json").write_bytes(canonical_json_bytes(summary))
        os.replace(stage, output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output), indent=2))
