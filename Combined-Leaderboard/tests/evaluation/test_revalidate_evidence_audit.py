import hashlib
import json
from pathlib import Path

from evaluation.extract_canonical_answers import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    EVIDENCE_VALIDATION_METHOD,
    FAIL_CLOSED_FALLBACK_METHOD,
    METHOD,
    revalidate_extractor_row,
)
from evaluation.revalidate_evidence_audit import revalidate_audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_revalidation_preserves_qwen_answer_and_repairs_only_evidence_status(
    tmp_path, monkeypatch
):
    response = (
        r"The correct option for the fifth image at \( t = 1.0 \) is **C**."
    )
    response_sha256 = hashlib.sha256(response.encode("utf-8")).hexdigest()
    candidate = {
        "model_slug": "provider-model",
        "source_relative_dir": "provider-model",
        "track": "minds_eye",
        "question_id": "q1",
        "answer_type": "mcq_letter",
        "task": "spatial_prediction",
        "category": "prior_model_extractor",
        "question": "Choose one option.",
        "response": response,
        "response_finish_reason": "stop",
        "response_completion_tokens": 20,
        "response_sha256": response_sha256,
        "current_submission_answer": "__UNRESOLVED__",
    }
    extractor_output = json.dumps(
        {
            "verdict": "COMMITTED",
            "answer": "C",
            "evidence": (
                "The correct option for the fifth image at $ t = 1.0 $ is **C**."
            ),
        },
        separators=(",", ":"),
    )
    row = {
        "model_slug": candidate["model_slug"],
        "source_relative_dir": candidate["source_relative_dir"],
        "track": candidate["track"],
        "question_id": candidate["question_id"],
        "answer_type": candidate["answer_type"],
        "task": candidate["task"],
        "category": candidate["category"],
        "response_finish_reason": "stop",
        "response_sha256": response_sha256,
        "method": METHOD,
        "extractor_contract_sha256": "c" * 64,
        "ground_truth_loaded": False,
        "ground_truth_supplied_to_extractor": False,
        "extractor_model": DEFAULT_EXTRACTOR_MODEL,
        "extractor_revision": DEFAULT_EXTRACTOR_REVISION,
        "extractor_output": extractor_output,
        "extractor_verdict": "COMMITTED",
        "answer": "",
        "evidence": json.loads(extractor_output)["evidence"],
        "status": "unsupported_by_evidence",
        "finish_reason": "stop",
        "completion_tokens": 15,
    }
    input_audit = tmp_path / "audit.jsonl"
    output_audit = tmp_path / "audit.revalidated-v2.jsonl"
    _write_jsonl(input_audit, [row])
    source_hash_before = hashlib.sha256(input_audit.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "evaluation.revalidate_evidence_audit.load_candidates",
        lambda *args, **kwargs: [candidate],
    )

    result = revalidate_audit(
        tmp_path,
        tmp_path,
        input_audit,
        output_audit,
        excluded_variants=set(),
    )

    revalidated = json.loads(output_audit.read_text(encoding="utf-8"))
    assert result["transitions"] == {"unsupported_by_evidence->committed": 1}
    assert result["source_audit_sha256"] == source_hash_before
    assert hashlib.sha256(input_audit.read_bytes()).hexdigest() == source_hash_before
    assert revalidated["answer"] == "C"
    assert revalidated["evidence"] == row["evidence"]
    assert revalidated["extractor_output"] == extractor_output
    assert revalidated["status"] == "committed"
    assert revalidated["evidence_validation_method"] == EVIDENCE_VALIDATION_METHOD
    assert revalidated["source_audit_sha256"] == source_hash_before
    assert "gold_answer" not in revalidated


def test_revalidation_preserves_fail_closed_terminal_fallback():
    candidate = {
        "answer_type": "mcq_letter",
        "task": "analogies",
        "response": "The response ends without a final selection.",
        "current_submission_answer": "__UNRESOLVED__",
    }
    row = {
        "extractor_output": "{",
        "extractor_verdict": "UNRESOLVED",
        "answer": "",
        "evidence": "",
        "status": "unresolved",
        "terminal_fallback_method": FAIL_CLOSED_FALLBACK_METHOD,
        "terminal_fallback_from_status": "invalid_extractor_output",
    }

    result = revalidate_extractor_row(
        candidate,
        row,
        source_audit_sha256="a" * 64,
    )

    assert result["status"] == "unresolved"
    assert result["answer"] == ""
    assert result["extractor_output"] == "{"
    assert result["terminal_fallback_method"] == FAIL_CLOSED_FALLBACK_METHOD
    assert result["evidence_validation_method"] == EVIDENCE_VALIDATION_METHOD
