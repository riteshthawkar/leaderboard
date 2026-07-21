import argparse
import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest


from scoring.task_scorer import TaskScorer  # noqa: E402

from evaluation.common.visual_pipeline import (  # noqa: E402
    EvaluationPipelineError,
    MANDATORY_ANSWER_EXTRACTION_METHOD_ID,
    MANDATORY_EXTRACTOR_MODEL_ID,
    MANDATORY_EXTRACTOR_MODEL_REVISION,
    MANDATORY_EXTRACTOR_PROMPT_SHA256,
    VisualTrackConfig,
    export_submission,
    final_answer,
    load_questions,
    read_diagnostics,
    write_diagnostics,
)
from evaluation import prepare_visual_data  # noqa: E402
from evaluation.common.vllm_runner import (  # noqa: E402
    _invalid_raw_records,
    _json_object,
    _resume_records,
    _run,
)
from evaluation.do_you_see_me.config import TRACK as DYSM_TRACK  # noqa: E402
from evaluation.minds_eye.config import TRACK as MINDS_EYE_TRACK  # noqa: E402


def test_standardized_visual_submission_names():
    assert DYSM_TRACK.default_output_path().name == "do_you_see_me_submission.jsonl"
    assert MINDS_EYE_TRACK.default_output_path().name == "minds_eye_submission.jsonl"


def _stub_output(question: dict) -> str:
    if question.get("task") == "form_constancy":
        return "<answer>Yes</answer>"
    return {
        "integer": "<answer>7</answer>",
        "mcq_index_1_4": "<answer>2</answer>",
        "mcq_letter": "<answer>C</answer>",
        "text": "<answer>sample</answer>",
    }[question["answer_type"]]


def _extracted(question: dict) -> dict:
    output = _stub_output(question)
    answer = final_answer(
        output,
        str(question.get("answer_type") or "text"),
        str(question.get("task") or ""),
    )
    return {
        "output": output,
        "extracted_answer": answer,
        "answer_extraction_method": MANDATORY_ANSWER_EXTRACTION_METHOD_ID,
        "extractor_status": "resolved",
        "extractor_model": MANDATORY_EXTRACTOR_MODEL_ID,
        "extractor_model_revision": MANDATORY_EXTRACTOR_MODEL_REVISION,
        "extractor_quantization": "unquantized",
        "extractor_runtime": "vllm 0.25.1",
        "extractor_prompt_sha256": MANDATORY_EXTRACTOR_PROMPT_SHA256,
        "extractor_ground_truth_access": False,
        "extractor_image_access": False,
        "extractor_source_diagnostics": "diagnostics.jsonl",
        "extractor_source_output_sha256": hashlib.sha256(
            output.encode("utf-8")
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    ("track", "task_id"),
    ((DYSM_TRACK, "do_you_see_me"), (MINDS_EYE_TRACK, "minds_eye")),
)
def test_visual_pipeline_exports_full_backend_compatible_bundle(tmp_path, track, task_id):
    questions = load_questions(track.questions_path, track)
    records = [
        {**question, **_extracted(question)}
        for question in questions
    ]
    output_path = tmp_path / f"{task_id}.jsonl"

    report = export_submission(records, questions, output_path)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert report["row_count"] == len(questions)
    assert len(rows) == len(questions)
    assert list(rows[0]) == ["question_id", "condition", "answer"]
    assert {row["condition"] for row in rows} == {"standard"}

    scorer = TaskScorer(task_id)
    scorer._questions = {}
    scorer._gt = {
        row["question_id"]: {"answer": row["answer"], "group": "format-check"}
        for row in rows
    }
    score = scorer.score(output_path, "format-check")
    assert score.total_samples == len(rows)
    assert score.accuracy == 1.0


def test_visual_pipeline_refuses_partial_or_failed_output(tmp_path):
    questions = [
        {"question_id": "q1", "answer_type": "integer"},
        {"question_id": "q2", "answer_type": "mcq_letter"},
    ]

    with pytest.raises(EvaluationPipelineError, match="coverage is incomplete"):
        export_submission(
            [{"question_id": "q1", "answer_type": "integer", "output": "1"}],
            questions,
            tmp_path / "partial.jsonl",
        )

    with pytest.raises(EvaluationPipelineError, match="require attention"):
        export_submission(
            [
                {"question_id": "q1", "answer_type": "integer", "output": "1"},
                {
                    "question_id": "q2",
                    "answer_type": "mcq_letter",
                    "output": None,
                    "error": "timeout",
                },
            ],
            questions,
            tmp_path / "failed.jsonl",
        )
    assert not (tmp_path / "partial.jsonl").exists()
    assert not (tmp_path / "failed.jsonl").exists()


def test_final_answer_extraction_matches_supported_answer_types():
    assert final_answer("reasoning </think> Final answer: -12", "integer") == "-12"
    assert final_answer("<answer>1,234</answer>", "integer") == "1234"
    assert final_answer("<answer>3</answer>", "mcq_index_1_4") == "3"
    assert final_answer("The selected option is d.", "mcq_letter") == "D"
    assert final_answer("Final response: UNFME", "text") == "UNFME"


def test_diagnostics_round_trip_preserves_errors(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    records = [
        {
            "question_id": "q1",
            "source_subset": "dysm_2d_v1",
            "answer_type": "integer",
            "output": None,
            "error": "request timed out",
        }
    ]

    write_diagnostics(path, records)

    assert read_diagnostics([path]) == records


def test_vllm_resume_keeps_all_complete_raw_records(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    questions = [
        {
            "question_id": "q1",
            "source_subset": "dysm_2d_v1",
            "answer_type": "integer",
        },
        {
            "question_id": "q2",
            "source_subset": "dysm_2d_v1",
            "answer_type": "mcq_letter",
        },
        {
            "question_id": "q3",
            "source_subset": "dysm_2d_v1",
            "answer_type": "integer",
        },
    ]
    write_diagnostics(
        path,
        [
            {**questions[0], "output": "<answer>7</answer>"},
            {**questions[1], "output": None, "error": "timeout"},
            {**questions[2], "output": "not a number"},
        ],
    )

    resumed = _resume_records(path, questions)

    assert list(resumed) == ["q1", "q3"]


def test_vllm_resume_rejects_diagnostics_from_another_question_bundle(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    write_diagnostics(
        path,
        [
            {
                "question_id": "other",
                "source_subset": "dysm_2d_v1",
                "answer_type": "integer",
                "output": "1",
            }
        ],
    )

    with pytest.raises(EvaluationPipelineError, match="unknown question IDs"):
        _resume_records(
            path,
            [
                {
                    "question_id": "q1",
                    "source_subset": "dysm_2d_v1",
                    "answer_type": "integer",
                }
            ],
        )


def test_vllm_strict_partial_validation_reports_missing_and_invalid_outputs():
    questions = [
        {"question_id": "q1", "answer_type": "integer"},
        {"question_id": "q2", "answer_type": "mcq_letter"},
    ]

    assert _invalid_raw_records(
        [{"question_id": "q1", "answer_type": "integer", "output": "none"}],
        questions,
    ) == [
        "q2 (missing)",
    ]


def test_vllm_chat_template_kwargs_require_a_json_object():
    assert _json_object('{"enable_thinking":false}') == {"enable_thinking": False}

    with pytest.raises(argparse.ArgumentTypeError, match="JSON object"):
        _json_object("[]")


def test_vllm_runner_builds_request_checkpoints_and_closes_client(tmp_path, monkeypatch):
    package_dir = tmp_path / "track"
    prompt_dir = package_dir / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "noncot.txt").write_text("Return only the answer.\n", encoding="utf-8")
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "Choose an option.",
                "answer_type": "mcq_letter",
                "source_subset": "subset_v1",
                "image_url": "data:image/png;base64,AA==",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    track = VisualTrackConfig(
        task_id="test_track",
        label="Test track",
        source_subsets=("subset_v1",),
        questions_path=questions_path,
        package_dir=package_dir,
    )
    requests = []
    closed_clients = []

    class FakeCompletions:
        async def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="<answer>A</answer>"))]
            )

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            closed_clients.append(self)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    diagnostics_path = tmp_path / "diagnostics.jsonl"
    args = SimpleNamespace(
        api_key="EMPTY",
        chat_template_kwargs={"enable_thinking": False},
        checkpoint_every=1,
        concurrency=1,
        endpoints="http://127.0.0.1:8011/v1",
        extra_body={},
        image_root=None,
        limit=0,
            max_retries=0,
            max_tokens=16,
            max_final_answer_tokens=200,
            model="fake/model",
            prompt_mode="noncot",
            questions=questions_path,
            request_timeout=30,
        repetition_penalty=1.0,
        resume=False,
        seed=0,
        temperature=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
            frequency_penalty=0.0,
            top_p=0.95,
            stop=None,
            include_stop_str_in_output=False,
            extract_all_only=False,
            inference_only=True,
        )

    questions, records = asyncio.run(_run(args, track, diagnostics_path))

    assert len(questions) == len(records) == 1
    assert records[0]["output"] == "<answer>A</answer>"
    assert requests[0]["model"] == "fake/model"
    assert requests[0]["temperature"] == 1.0
    assert requests[0]["top_p"] == 0.95
    assert requests[0]["presence_penalty"] == 0.0
    assert requests[0]["frequency_penalty"] == 0.0
    assert requests[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    assert diagnostics_path.is_file()
    assert len(closed_clients) == 1


def test_visual_dataset_verification_requires_every_question_image(tmp_path, monkeypatch):
    package_dir = tmp_path / "track"
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "Choose one.",
                "answer_type": "mcq_letter",
                "source_subset": "subset_v1",
                "image": "images/example.png",
                "image_url": "https://example.test/example.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    track = VisualTrackConfig(
        task_id="test_track",
        label="Test track",
        source_subsets=("subset_v1",),
        questions_path=questions_path,
        package_dir=package_dir,
    )
    monkeypatch.setattr(prepare_visual_data, "TRACKS", (track,))
    image_path = tmp_path / "dataset" / "subset_v1" / "images" / "example.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-placeholder")

    report = prepare_visual_data.verify_visual_data(tmp_path / "dataset")

    assert report["total_questions"] == 1
    assert report["tracks"] == {"test_track": 1}
    image_path.unlink()
    with pytest.raises(EvaluationPipelineError, match=r"1 image\(s\) are missing"):
        prepare_visual_data.verify_visual_data(tmp_path / "dataset")
