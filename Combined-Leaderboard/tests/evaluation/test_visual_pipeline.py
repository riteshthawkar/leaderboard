import asyncio
import base64
import hashlib
import json
from types import SimpleNamespace

from evaluation.common.vllm_runner import (
    _answer_is_supported_by_output,
    _extract_one,
    _extractor_answer,
    _infer_one,
    _resume_records,
    main,
)
from evaluation.common.visual_pipeline import (
    VisualTrackConfig,
    export_submission,
    final_answer,
    image_for_openai,
    record_answer,
    write_diagnostics,
)


def test_final_answer_rejects_accidental_choice_letters_and_unclosed_reasoning():
    assert final_answer("reasoning without a final choice", "mcq_letter") == ""
    assert final_answer("<think>There are 7 candidates", "integer") == ""
    assert final_answer("<|begin_of_box|>D", "mcq_letter") == ""


def test_final_answer_accepts_glm_native_box_delimiters():
    assert (
        final_answer(
            "Reasoning that mentions A.\n<|begin_of_box|>D<|end_of_box|>",
            "mcq_letter",
        )
        == "D"
    )


def test_final_answer_accepts_innermost_complete_glm_native_box():
    assert (
        final_answer(
            "<|begin_of_box|><answer><|begin_of_box|>\n"
            "<|begin_of_box|>E<|end_of_box|>",
            "mcq_letter",
        )
        == "E"
    )
    assert final_answer("<|begin_of_box|><answer>G</answer>", "mcq_letter") == ""


def test_final_answer_accepts_innermost_complete_answer_block():
    assert (
        final_answer(
            "The format is <answer>LETTER, so I choose "
            "<answer>B</answer>",
            "mcq_letter",
        )
        == "B"
    )


def test_final_answer_accepts_unambiguous_odd_figure_commitments():
    assert (
        final_answer(
            "Most figures share the concept. Figure D does not adhere to it.",
            "mcq_letter",
        )
        == "D"
    )
    assert (
        final_answer(
            "The figure that does not adhere to this concept is B.",
            "mcq_letter",
        )
        == "B"
    )
    assert (
        final_answer(
            "Figure C does not adhere, but figure D does not adhere either.",
            "mcq_letter",
        )
        == ""
    )


def test_extractor_answer_accepts_glm_control_token_wrappers():
    assert (
        _extractor_answer(
            "<think>The supported final choice is D.</think>\n"
            "<|begin_of_box|><answer>D</answer>",
            "mcq_letter",
        )
        == "D"
    )


def test_native_box_answer_is_supported_by_source_output():
    assert _answer_is_supported_by_output(
        "Reasoning.\n<|begin_of_box|>D<|end_of_box|>", "D", "mcq_letter"
    )


def test_final_answer_normalizes_only_unambiguous_integer_words():
    assert final_answer("There are two spheres in total.", "integer") == "2"
    assert final_answer("There are zero blue shapes in the scene.", "integer") == "0"
    assert final_answer("There may be two or three spheres.", "integer") == ""


def test_record_answer_preserves_raw_output_and_prefers_extracted_answer():
    record = {
        "output": "<think>unfinished reasoning",
        "extracted_answer": "6",
    }

    assert record_answer(record, "integer") == "6"
    assert record["output"] == "<think>unfinished reasoning"


def test_export_submission_can_opt_in_to_exact_raw_output_fallback(tmp_path):
    output_path = tmp_path / "submission.jsonl"
    unfinished = "  <think>Still comparing options A and B\n"

    result = export_submission(
        [
            {"question_id": "q1", "output": "<answer>C</answer>"},
            {"question_id": "q2", "output": unfinished},
        ],
        [
            {"question_id": "q1", "answer_type": "mcq_letter"},
            {"question_id": "q2", "answer_type": "mcq_letter"},
        ],
        output_path,
        raw_output_fallback=True,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {"question_id": "q1", "condition": "standard", "answer": "C"},
        {"question_id": "q2", "condition": "standard", "answer": unfinished},
    ]
    assert result["raw_output_fallback_question_ids"] == ["q2"]


def test_runner_finalizes_existing_faulty_output_without_inference(tmp_path):
    questions = tmp_path / "questions.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    output = tmp_path / "submission.jsonl"
    faulty_output = "<|begin_of_box|>Option 5<|end_of_box|>"
    questions.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "Choose an option.",
                "answer_type": "mcq_index_1_4",
                "image": "unused.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics.write_text(
        json.dumps({"question_id": "q1", "output": faulty_output}) + "\n",
        encoding="utf-8",
    )
    track = VisualTrackConfig(
        task_id="test",
        label="Test",
        source_subsets=(),
        questions_path=questions,
        package_dir=tmp_path,
    )

    result = main(
        track,
        [
            "--model",
            "unused/model",
            "--questions",
            str(questions),
            "--diagnostics",
            str(diagnostics),
            "--out",
            str(output),
            "--finalize-existing-diagnostics",
            "--raw-output-fallback",
        ],
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["answer"] == faulty_output


def test_resume_accepts_a_provenance_preserving_extracted_answer(tmp_path):
    diagnostics = tmp_path / "diagnostics.jsonl"
    raw_output = "<think>The count may be 4, but after checking it is 6"
    write_diagnostics(
        diagnostics,
        [
            {
                "question_id": "q1",
                "answer_type": "integer",
                "output": raw_output,
                "answer_extraction_method": "same-served-model-text-only-v1",
                "extractor_model": "test/model",
                "extractor_output": "<answer>6</answer>",
                "extracted_answer": "6",
            }
        ],
    )

    resumed = _resume_records(
        diagnostics,
        [{"question_id": "q1", "answer_type": "integer"}],
    )

    assert resumed["q1"]["output"] == raw_output
    assert resumed["q1"]["extracted_answer"] == "6"


def test_same_model_extractor_is_text_only_and_preserves_raw_output():
    requests = []
    tokenization_requests = []

    class Completions:
        async def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="<answer>6</answer>"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=5),
            )

    class TokenizeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"count": 1}

    class TokenizeClient:
        async def post(self, path, *, json):
            tokenization_requests.append((path, json))
            return TokenizeResponse()

    raw_output = "<think>The count may be 4, but after checking it is 6"
    result = asyncio.run(
        _extract_one(
            SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
            asyncio.Semaphore(1),
            {
                "question_id": "q1",
                "answer_type": "integer",
                "output": raw_output,
            },
            {
                "question_id": "q1",
                "question": "How many objects are visible?",
                "answer_type": "integer",
            },
            model="test/model",
            max_tokens=200,
            seed=17,
            max_final_answer_tokens=10,
            tokenize_client=TokenizeClient(),
        )
    )

    assert result["output"] == raw_output
    assert result["extracted_answer"] == "6"
    assert result["answer_extraction_method"] == "same-served-model-text-only-v1"
    assert result["extractor_model"] == "test/model"
    assert result["extractor_output"] == "<answer>6</answer>"
    assert result["extractor_completion_tokens"] == 5
    request = requests[0]
    assert request["temperature"] == 0.0
    assert request["top_p"] == 1.0
    assert request["seed"] == 17
    assert request["max_tokens"] == 200
    assert all("image" not in json.dumps(message) for message in request["messages"])
    payload = json.loads(request["messages"][1]["content"])
    assert payload == {
        "question": "How many objects are visible?",
        "answer_type": "integer",
        "candidate_response": raw_output,
    }
    assert tokenization_requests == [
        (
            "/tokenize",
            {
                "model": "test/model",
                "prompt": "6",
                "add_special_tokens": False,
            },
        )
    ]


def test_same_model_extractor_rejects_unsupported_or_ambiguous_answers():
    outputs = iter(
        [
            "<answer>9</answer>",
            "<answer>UNRESOLVED</answer>",
            "I think the answer is <answer>6</answer>",
        ]
    )

    class Completions:
        async def create(self, **request):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=next(outputs)),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=5),
            )

    async def extract():
        return await _extract_one(
            SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
            asyncio.Semaphore(1),
            {
                "question_id": "q1",
                "answer_type": "integer",
                "output": "The candidates are 4 and 6.",
            },
            {
                "question_id": "q1",
                "question": "How many objects are visible?",
                "answer_type": "integer",
            },
            model="test/model",
            max_tokens=200,
            seed=0,
            max_final_answer_tokens=None,
        )

    unsupported, unresolved, malformed = [
        asyncio.run(extract()) for _ in range(3)
    ]

    assert unsupported["extractor_error"] == (
        "The extracted answer is not stated in the candidate response."
    )
    assert unresolved["extractor_error"] == "The extractor returned UNRESOLVED."
    assert malformed["extractor_error"] == (
        "The extractor did not return a parseable answer block."
    )
    assert all(
        "extracted_answer" not in result
        for result in (unsupported, unresolved, malformed)
    )


def test_mcq_extractor_support_requires_an_explicit_option_or_answer():
    assert _answer_is_supported_by_output(
        "Option C appears to follow the pattern.", "C", "mcq_letter"
    )
    assert _answer_is_supported_by_output(
        "The final answer is A.", "A", "mcq_letter"
    )
    assert not _answer_is_supported_by_output(
        "A complex analysis continues without choosing.", "A", "mcq_letter"
    )
    assert not _answer_is_supported_by_output(
        "At t=1.0 the shape rotates.", "1", "mcq_index_1_4"
    )


def test_openai_image_payload_preserves_original_bytes(tmp_path):
    original = b"\x89PNG\r\n\x1a\noriginal-benchmark-image-bytes"
    image = tmp_path / "sample.png"
    image.write_bytes(original)

    payload = image_for_openai({"question_id": "q1", "image": str(image)}, None)

    header, encoded = payload.split(",", 1)
    assert header == "data:image/png;base64"
    assert base64.b64decode(encoded) == original


def test_diagnostics_retain_finish_reason_and_completion_tokens(tmp_path):
    output = tmp_path / "diagnostics.jsonl"
    write_diagnostics(
        output,
        [
            {
                "question_id": "q1",
                "source_subset": "subset",
                "answer_type": "mcq_letter",
                "output": "<answer>A</answer>",
                "finish_reason": "stop",
                "completion_tokens": 8,
                "final_answer_tokens": 1,
                "answer_extraction_method": "same-model-judge-v1",
                "extractor_model": "test/model",
                "extractor_output": "<answer>A</answer>",
                "extractor_finish_reason": "stop",
                "extractor_completion_tokens": 4,
                "extractor_source_diagnostics": "attempt-1.diagnostics.jsonl",
                "extractor_source_output_sha256": "abc123",
                "extractor_attempts": [
                    {
                        "answer_extraction_method": "same-model-judge-v1",
                        "extractor_output": "<answer>UNRESOLVED</answer>",
                    }
                ],
                "extracted_answer": "A",
            }
        ],
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["finish_reason"] == "stop"
    assert row["completion_tokens"] == 8
    assert row["final_answer_tokens"] == 1
    assert row["answer_extraction_method"] == "same-model-judge-v1"
    assert row["extractor_model"] == "test/model"
    assert row["extractor_output"] == "<answer>A</answer>"
    assert row["extractor_completion_tokens"] == 4
    assert row["extractor_source_diagnostics"] == "attempt-1.diagnostics.jsonl"
    assert row["extractor_source_output_sha256"] == "abc123"
    assert row["extractor_attempts"][0]["extractor_output"] == (
        "<answer>UNRESOLVED</answer>"
    )
    assert row["extracted_answer"] == "A"


def test_archived_extractor_source_preserves_current_output_and_judge_history():
    requests = []

    class Completions:
        async def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="<answer>I, L, E</answer>"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=9),
            )

    current_output = "The candidates might be I, T, H or I, L, O."
    archived_output = "The letters from left to right are I, L, E."
    result = asyncio.run(
        _extract_one(
            SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
            asyncio.Semaphore(1),
            {
                "question_id": "q1",
                "answer_type": "text",
                "output": current_output,
                "answer_extraction_method": "same-served-model-text-only-v1",
                "extractor_model": "test/model",
                "extractor_output": "<answer>UNRESOLVED</answer>",
                "extractor_error": "The extractor returned UNRESOLVED.",
            },
            {
                "question_id": "q1",
                "question": "Which letters are visible?",
                "answer_type": "text",
            },
            model="test/model",
            max_tokens=200,
            seed=0,
            max_final_answer_tokens=None,
            candidate_output=archived_output,
            source_diagnostics="attempt-1.diagnostics.jsonl",
        )
    )

    assert result["output"] == current_output
    assert result["extracted_answer"] == "I, L, E"
    assert result["extractor_source_diagnostics"] == (
        "attempt-1.diagnostics.jsonl"
    )
    assert result["extractor_source_output_sha256"] == hashlib.sha256(
        archived_output.encode("utf-8")
    ).hexdigest()
    assert result["extractor_attempts"] == [
        {
            "answer_extraction_method": "same-served-model-text-only-v1",
            "extractor_model": "test/model",
            "extractor_output": "<answer>UNRESOLVED</answer>",
            "extractor_error": "The extractor returned UNRESOLVED.",
        }
    ]
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["candidate_response"] == archived_output


def test_vllm_request_honors_capped_and_uncapped_max_tokens(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nbenchmark-image")
    requests = []
    tokenization_requests = []

    class Completions:
        async def create(self, **request):
            requests.append(request)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="1"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=1),
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )

    class TokenizeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"count": 1}

    class TokenizeClient:
        async def post(self, path, *, json):
            tokenization_requests.append((path, json))
            return TokenizeResponse()

    tokenize_client = TokenizeClient()
    item = {
        "question_id": "q1",
        "question": "How many objects are visible?",
        "image": str(image),
    }

    async def infer(max_tokens, *, stop=None, include_stop_str_in_output=False):
        return await _infer_one(
            client,
            asyncio.Semaphore(1),
            item,
            image_root=None,
            system_prompt="Return the answer.",
            model="test/model",
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=0,
            extra_body={},
            stop=stop or [],
            include_stop_str_in_output=include_stop_str_in_output,
            max_final_answer_tokens=200,
            tokenize_client=tokenize_client,
        )

    asyncio.run(
        infer(
            None,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
    )
    asyncio.run(infer(200))
    asyncio.run(infer(8192))

    assert "max_tokens" not in requests[0]
    assert requests[0]["stop"] == ["</answer>"]
    assert requests[0]["extra_body"]["include_stop_str_in_output"] is True
    assert requests[1]["max_tokens"] == 200
    assert "stop" not in requests[1]
    assert requests[2]["max_tokens"] == 8192
    assert tokenization_requests == [
        (
            "/tokenize",
            {
                "model": "test/model",
                "prompt": "1",
                "add_special_tokens": False,
            },
        ),
        (
            "/tokenize",
            {
                "model": "test/model",
                "prompt": "1",
                "add_special_tokens": False,
            },
        ),
    ]


def test_uncapped_response_rejects_final_answer_over_token_limit(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nbenchmark-image")

    class Completions:
        async def create(self, **request):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="<think>reasoning</think><answer>result</answer>"
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=500),
            )

    class TokenizeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"count": 201}

    class TokenizeClient:
        async def post(self, path, *, json):
            return TokenizeResponse()

    result = asyncio.run(
        _infer_one(
            SimpleNamespace(chat=SimpleNamespace(completions=Completions())),
            asyncio.Semaphore(1),
            {
                "question_id": "q1",
                "question": "What is shown?",
                "answer_type": "text",
                "image": str(image),
            },
            image_root=None,
            system_prompt="Return the answer.",
            model="test/model",
            max_tokens=None,
            temperature=0.1,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=0,
            extra_body={},
            stop=["</answer>"],
            include_stop_str_in_output=True,
            max_final_answer_tokens=200,
            tokenize_client=TokenizeClient(),
        )
    )

    assert result["output"].endswith("</answer>")
    assert result["final_answer_tokens"] == 201
    assert result["error"] == (
        "Extracted final answer uses 201 tokens; the limit is 200."
    )
