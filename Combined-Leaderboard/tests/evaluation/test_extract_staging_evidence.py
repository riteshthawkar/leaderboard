import asyncio
import hashlib
import json
from argparse import Namespace
from types import SimpleNamespace

from evaluation.do_you_see_me.config import TRACK
from evaluation.extract_canonical_answers import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    METHOD,
    candidate_key,
    extractor_contract_sha256,
    load_audit_checkpoint,
    run,
    seed_embedded_evidence,
)
from evaluation.extract_staging_evidence import _run


def test_staging_uses_v4_once_and_writes_reusable_evidence(tmp_path, monkeypatch):
    questions = tmp_path / "questions.jsonl"
    source = tmp_path / "do_you_see_me.inference.diagnostics.jsonl"
    evidence = tmp_path / "do_you_see_me.evidence_extraction.jsonl"
    diagnostics = tmp_path / "do_you_see_me.diagnostics.jsonl"
    submission = tmp_path / "do_you_see_me_submission.jsonl"
    response = "Reasoning. Final answer is C."
    question = {
        "question_id": "q1",
        "question": "Which option is correct?",
        "answer_type": "mcq_letter",
        "image": "unused.png",
    }
    questions.write_text(json.dumps(question) + "\n", encoding="utf-8")
    source.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "source_subset": "",
                "answer_type": "mcq_letter",
                "output": response,
                "finish_reason": "stop",
                "completion_tokens": 8,
                "inference_method": "visual-inference-output-sha256-v1",
                "inference_output_sha256": hashlib.sha256(
                    response.encode()
                ).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    requests = []

    class Completions:
        async def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "verdict": "COMMITTED",
                                    "answer": "C",
                                    "evidence": "Final answer is C.",
                                }
                            )
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=18),
            )

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        async def close(self):
            return None

    async def ready(*args, **kwargs):
        return None

    monkeypatch.setattr("evaluation.extract_staging_evidence.AsyncOpenAI", Client)
    monkeypatch.setattr(
        "evaluation.extract_staging_evidence.wait_for_extractor_clients", ready
    )
    args = Namespace(
        model_slug="test-model",
        track="do_you_see_me",
        source=source,
        evidence=evidence,
        diagnostics=diagnostics,
        out=submission,
        questions=questions,
        endpoint=["http://127.0.0.1:8000/v1"],
        api_key="EMPTY",
        model=DEFAULT_EXTRACTOR_MODEL,
        revision=DEFAULT_EXTRACTOR_REVISION,
        max_tokens=256,
        concurrency=1,
        timeout=60,
        endpoint_start_timeout=60,
        retries=2,
        report_every=1,
        track_config=TRACK,
    )

    report = asyncio.run(_run(args))

    assert len(requests) == 1
    assert requests[0]["response_format"]["json_schema"]["name"] == (
        "commitment_extraction"
    )
    evidence_row = json.loads(evidence.read_text(encoding="utf-8"))
    assert evidence_row["method"] == METHOD
    assert evidence_row["status"] == "committed"
    assert evidence_row["evidence"] == "Final answer is C."
    diagnostic = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert diagnostic["answer_extraction_method"] == METHOD
    assert diagnostic["extractor_evidence"] == "Final answer is C."
    assert json.loads(submission.read_text(encoding="utf-8"))["answer"] == "C"
    assert report["extractor_contract_sha256"] == extractor_contract_sha256(
        DEFAULT_EXTRACTOR_MODEL, 256, DEFAULT_EXTRACTOR_REVISION
    )


def test_canonical_audit_seeds_public_v4_evidence_without_model_call(
    tmp_path, monkeypatch
):
    canonical = tmp_path / "canonical"
    model_dir = canonical / "test-model"
    model_dir.mkdir(parents=True)
    output = tmp_path / "audit.jsonl"
    response = "Final answer is C."
    candidate = {
        "model_slug": "test-model",
        "source_relative_dir": "test-model",
        "track": "do_you_see_me",
        "question_id": "q1",
        "answer_type": "mcq_letter",
        "task": "",
        "category": "deterministic_explicit",
        "question": "Which option is correct?",
        "response": response,
        "response_finish_reason": "stop",
        "response_completion_tokens": 5,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "current_submission_answer": "UNRESOLVED",
    }
    contract = extractor_contract_sha256(
        DEFAULT_EXTRACTOR_MODEL, 256, DEFAULT_EXTRACTOR_REVISION
    )
    row = {
        **{
            key: candidate[key]
            for key in (
                "model_slug",
                "source_relative_dir",
                "track",
                "question_id",
                "answer_type",
                "task",
                "category",
                "response_finish_reason",
                "response_sha256",
            )
        },
        "method": METHOD,
        "extractor_contract_sha256": contract,
        "ground_truth_loaded": False,
        "ground_truth_supplied_to_extractor": False,
        "extractor_model": DEFAULT_EXTRACTOR_MODEL,
        "extractor_revision": DEFAULT_EXTRACTOR_REVISION,
        "extractor_verdict": "COMMITTED",
        "answer": "C",
        "evidence": "Final answer is C.",
        "status": "committed",
        "extractor_output": json.dumps(
            {
                "verdict": "COMMITTED",
                "answer": "C",
                "evidence": "Final answer is C.",
            }
        ),
        "finish_reason": "stop",
        "completion_tokens": 18,
    }
    (model_dir / "do_you_see_me.evidence_extraction.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    by_key = {candidate_key(candidate): candidate}

    seeded = seed_embedded_evidence(
        canonical,
        output,
        by_key,
        contract,
        DEFAULT_EXTRACTOR_MODEL,
        DEFAULT_EXTRACTOR_REVISION,
    )
    existing, retries = load_audit_checkpoint(output, by_key, contract)

    assert seeded == 1
    assert len(existing) == 1
    assert retries == {}

    args = SimpleNamespace(
        project_root=tmp_path,
        canonical_root=canonical,
        policy="all",
        exclude_variants=[],
        model=DEFAULT_EXTRACTOR_MODEL,
        revision=DEFAULT_EXTRACTOR_REVISION,
        max_tokens=256,
        output=output,
        endpoints=[],
    )
    monkeypatch.setattr(
        "evaluation.extract_canonical_answers.load_candidates",
        lambda *_args, **_kwargs: [candidate],
    )
    monkeypatch.setattr(
        "evaluation.extract_canonical_answers.AsyncOpenAI",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("complete embedded evidence contacted an endpoint")
        ),
    )
    asyncio.run(run(args))
