import asyncio
import hashlib
import json
from argparse import Namespace
from types import SimpleNamespace

from evaluation.answer_extraction_contract import METHOD
from evaluation.do_you_see_me.config import TRACK
from evaluation.extract_staging_answers import _run


def test_model_only_extraction_writes_ready_submission_without_answer_recovery(
    tmp_path, monkeypatch
):
    questions = tmp_path / "questions.jsonl"
    source = tmp_path / "do_you_see_me.inference.diagnostics.jsonl"
    diagnostics = tmp_path / "do_you_see_me.diagnostics.jsonl"
    submission = tmp_path / "do_you_see_me_submission.jsonl"
    question_rows = [
        {
            "question_id": "q1",
            "question": "Which option is correct?",
            "answer_type": "mcq_letter",
            "image": "unused.png",
        },
        {
            "question_id": "q2",
            "question": "Which option is correct?",
            "answer_type": "mcq_letter",
            "image": "unused.png",
        },
    ]
    questions.write_text(
        "".join(json.dumps(row) + "\n" for row in question_rows), encoding="utf-8"
    )
    outputs = ["Reasoning. Final answer is C.", "A or B could both be correct."]
    source.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": question["question_id"],
                    "answer_type": "mcq_letter",
                    "output": output,
                    "finish_reason": "stop",
                    "inference_method": "visual-inference-output-sha256-v1",
                    "inference_output_sha256": hashlib.sha256(
                        output.encode()
                    ).hexdigest(),
                }
            )
            + "\n"
            for question, output in zip(question_rows, outputs, strict=True)
        ),
        encoding="utf-8",
    )
    requests = []

    class Completions:
        async def create(self, **request):
            requests.append(request)
            payload = json.loads(request["messages"][1]["content"])
            answer = "C" if payload["candidate_response"].startswith("Reasoning") else ""
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps({"answer": answer})),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=4),
            )

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        async def close(self):
            return None

    monkeypatch.setattr("evaluation.extract_staging_answers.AsyncOpenAI", Client)
    args = Namespace(
        track="do_you_see_me",
        source=source,
        diagnostics=diagnostics,
        out=submission,
        questions=questions,
        endpoint=["http://127.0.0.1:8000/v1"],
        api_key="EMPTY",
        model="Qwen/Qwen3-8B",
        revision="revision-a",
        max_tokens=64,
        concurrency=2,
        timeout=60,
        retries=2,
        checkpoint_every=1,
        report_every=1,
        track_config=TRACK,
    )

    report = asyncio.run(_run(args))

    assert len(requests) == 2
    for request in requests:
        payload = json.loads(request["messages"][1]["content"])
        assert set(payload) == {"candidate_response", "expected_answer_format"}
        assert "question" not in payload
        assert request["response_format"]["json_schema"]["schema"]["properties"][
            "answer"
        ]["enum"] == ["", "A", "B", "C", "D", "E", "F"]
    rows = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert [row["answer_extraction_method"] for row in rows] == [METHOD, METHOD]
    assert [row["extractor_answer"] for row in rows] == ["C", ""]
    assert [row["extractor_status"] for row in rows] == ["extracted", "empty"]
    submission_rows = [json.loads(line) for line in submission.read_text().splitlines()]
    assert [row["answer"] for row in submission_rows] == ["C", "UNRESOLVED"]
    assert report["extracted_count"] == 1
    assert report["empty_count"] == 1
