import hashlib
import json
from pathlib import Path

import pytest

from evaluation.assemble_v4_ranking_bundle import (
    AssemblyError,
    normalize_source_config_names,
    source_identity,
)
from evaluation.extract_canonical_answers import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    METHOD,
    extractor_contract_sha256,
)
from evaluation.package_v4_ranking_bundle import (
    EXTRACTOR_MAX_TOKENS,
    PackagingError,
    validate_extraction_identity,
    validate_track_evidence,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_source_identity_requires_one_reasoning_profile(tmp_path):
    source = tmp_path / "model"
    for track, profile in (
        ("do_you_see_me", "thinking"),
        ("minds_eye", "nonthinking"),
    ):
        _write_json(
            source / f"{track}.run_config.json",
            {
                "model_id": "org/model",
                "model_revision": "a" * 40,
                "reasoning_profile": profile,
            },
        )

    with pytest.raises(AssemblyError, match="inconsistent"):
        source_identity(source)


def test_normalize_source_config_names_updates_artifact_map(tmp_path):
    model_dir = tmp_path / "model"
    tracks = {}
    for track in ("do_you_see_me", "minds_eye"):
        source_name = f"{track}.v11_run_config.json"
        payload = b'{"pipeline_revision":"source"}\n'
        path = model_dir / source_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        tracks[track] = {
            "source_run_config": source_name,
            "artifacts": {
                source_name: {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            },
        }
    manifest = {"tracks": tracks}

    normalize_source_config_names(model_dir, manifest)

    for track in tracks:
        target_name = f"{track}.source_run_config.json"
        assert (model_dir / target_name).is_file()
        assert manifest["tracks"][track]["source_run_config"] == target_name
        assert set(manifest["tracks"][track]["artifacts"]) == {target_name}


def test_v4_identity_requires_gold_blind_pinned_contract():
    contract = extractor_contract_sha256(
        DEFAULT_EXTRACTOR_MODEL,
        EXTRACTOR_MAX_TOKENS,
        DEFAULT_EXTRACTOR_REVISION,
    )
    identity = (
        METHOD,
        DEFAULT_EXTRACTOR_MODEL,
        DEFAULT_EXTRACTOR_REVISION,
        contract,
        (
            "question",
            "answer_type",
            "task",
            "response_metadata",
            "candidate_response",
        ),
        False,
        False,
        False,
    )
    validate_extraction_identity(identity)

    with pytest.raises(PackagingError, match="gold-blind"):
        validate_extraction_identity((*identity[:7], True))


def test_track_evidence_rejects_source_response_hash_mismatch(
    tmp_path,
    monkeypatch,
):
    from evaluation import package_v4_ranking_bundle as package

    monkeypatch.setitem(package.EXPECTED_ROWS, "do_you_see_me", 1)
    root = tmp_path / "bundle"
    model_dir = root / "model"
    question_id = "question-1"
    _write_jsonl(
        model_dir / "do_you_see_me.diagnostics.jsonl",
        [
            {
                "question_id": question_id,
                "output": "answer 2",
                "answer_extraction_method": METHOD,
                "extractor_source_output_sha256": "0" * 64,
                "extracted_answer": "2",
            }
        ],
    )
    _write_jsonl(
        model_dir / "do_you_see_me.evidence_extraction.jsonl",
        [
            {
                "question_id": question_id,
                "status": "committed",
                "answer": "2",
                "response_sha256": "0" * 64,
            }
        ],
    )
    manifest = {
        "strict_answer_count": 1,
        "invalid_commitment_count": 0,
        "unresolved_answer_count": 0,
        "evidence_extraction": {"status_counts": {"committed": 1}},
    }

    with pytest.raises(PackagingError, match="source-response hash mismatch"):
        validate_track_evidence(root, "model", "do_you_see_me", manifest)
