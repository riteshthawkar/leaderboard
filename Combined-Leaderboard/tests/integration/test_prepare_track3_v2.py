import io
import json
import zipfile

import pytest

from config import EVAL_CONDITIONS, SPATIAL_DATASET_KEYS
from import_spatial_results import build_import_plan
from prepare_track3_v2 import COMBOS, converted_rows, sample_catalog, write_contract
from spatial_harness.artifact_package import (
    ARCHIVE_NAME, PACKAGE_SCHEMA_VERSION, SCORE_SOURCE, SCORE_UNIT, VERIFICATION_LEVEL,
    aggregate_claimed_scores, sha256_bytes, write_artifact_package, write_gzip_jsonl,
)


def _source(*, missing_credit=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for combo in COMBOS:
            mode, pmode = combo.split("_", 1)
            predictions, verdicts = [], []
            for index, dataset in enumerate(SPATIAL_DATASET_KEYS):
                row = {"dataset": dataset, "index": "q1", "group": "g1", "answer_type": "mcq" if index != 1 else "vqa", "mode": mode, "pmode": pmode}
                predictions.append({**row, "output": None if index == 0 else "actual model output", "options": {"A": "option"}})
                verdicts.append({**row, "gt": "A", "cannot_label": "A", "judged": "B" if index == 0 and not missing_credit else "A"})
            archive.writestr(f"predictions/pred_{combo}.jsonl", "".join(json.dumps(row) + "\n" for row in predictions))
            archive.writestr(f"verdicts/judged_{combo}.jsonl", "".join(json.dumps(row) + "\n" for row in verdicts))
    return zipfile.ZipFile(io.BytesIO(stream.getvalue()))


def test_missing_output_is_preserved_and_vqa_verdict_is_not_a_model_answer():
    with _source() as archive:
        converted = list(converted_rows(archive, ""))
    answer, raw = next(pair for pair in converted if pair[1]["raw_output"] is None)
    assert answer["claimed_credit"] == 0 and answer["final_answer"] == "MISSING_SOURCE_OUTPUT"
    assert raw["raw_output"] is None and raw["output_status"] == "missing_source_output"
    assert all("gt" not in record and "cannot_label" not in record for pair in converted for record in pair)
    vqa = next(answer for answer, _ in converted if answer["answer_type"] == "vqa")
    assert vqa["final_answer"] == "actual model output"
    with _source(missing_credit=True) as archive, pytest.raises(ValueError, match="positive claimed credit"):
        list(converted_rows(archive, ""))


def test_cohort_package_validates_and_import_preserves_closed_model_metadata(tmp_path):
    with _source() as archive:
        catalog = sample_catalog(archive, "")
        converted = list(converted_rows(archive, ""))
    manifest_bytes = write_contract(tmp_path / "contract", catalog, "f" * 64)
    answers, raw = zip(*converted)
    answer_path, raw_path = tmp_path / "answers.jsonl.gz", tmp_path / "raw.jsonl.gz"
    write_gzip_jsonl(answer_path, answers)
    write_gzip_jsonl(raw_path, raw)
    package = write_artifact_package(tmp_path / ARCHIVE_NAME, {
        "schema_version": PACKAGE_SCHEMA_VERSION, "verification_level": VERIFICATION_LEVEL, "score_source": SCORE_SOURCE,
        "model": {"name": "source/snapshot", "display_name": "Closed model", "organization": "Example", "access": "closed"},
        "benchmark": {"version": json.loads(manifest_bytes)["benchmark_version"], "manifest_sha256": sha256_bytes(manifest_bytes)},
        "evaluation": {"harness_contract": "submitted-v2", "harness_version": "v2", "configuration_sha256": "e" * 64, "judge": {"name": "Submitted judge", "revision": None}, "provenance_status": "submitter_declared_unattested"},
        "evidence": {"answer_rows": 78, "raw_output_rows": 78, "raw_output_scope": "source_outputs_with_missing_records", "missing_output_rows": 6},
        "scoring": {"source": SCORE_SOURCE, "unit": SCORE_UNIT},
    }, aggregate_claimed_scores(answers, SPATIAL_DATASET_KEYS, EVAL_CONDITIONS), answer_path, raw_path)
    with pytest.raises(ValueError, match="not production ready"):
        build_import_plan([package], contract_dir=tmp_path / "contract")
    item = build_import_plan([package], contract_dir=tmp_path / "contract", allow_submitted_cohort=True)[0]
    assert item.display_name == "Closed model" and item.score.model_meta["access"] == "closed"
    assert "compute_dtype" not in item.score.model_meta and "weight_loading" not in item.score.model_meta
    assert item.run_metadata["missing_output_rows"] == 6
    assert item.score.metadata["spatial_run"]["provenance_status"] == "submitter_declared_unattested"
