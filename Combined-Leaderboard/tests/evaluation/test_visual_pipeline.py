import asyncio
import base64
import hashlib
import json
import sys
from types import SimpleNamespace

from evaluation.common.vllm_runner import (
    _answer_is_supported_by_output,
    _extract_one,
    _extractor_answer,
    _infer_one,
    _resume_records,
    _run,
    main,
)
from evaluation.common.visual_pipeline import (
    INVALID_FORMAT_ANSWER,
    INVALID_FORMAT_REASON,
    MANDATORY_ANSWER_EXTRACTION_METHOD_ID,
    MANDATORY_EXTRACTOR_MODEL_ID,
    MANDATORY_EXTRACTOR_MODEL_REVISION,
    MANDATORY_EXTRACTOR_PROMPT_SHA256,
    VisualTrackConfig,
    export_submission,
    final_answer,
    image_for_openai,
    record_answer,
    write_diagnostics,
)


def _extracted_record(
    question_id: str,
    output: str,
    answer: str | None = None,
    *,
    status: str = "resolved",
) -> dict:
    record = {
        "question_id": question_id,
        "output": output,
        "answer_extraction_method": MANDATORY_ANSWER_EXTRACTION_METHOD_ID,
        "extractor_status": status,
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
        "extractor_output": (
            f"<answer>{answer}</answer>"
            if answer is not None
            else "<answer>UNRESOLVED</answer>"
        ),
    }
    if answer is not None:
        record["extracted_answer"] = answer
    else:
        record["extractor_error"] = "The extractor returned UNRESOLVED."
    return record


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


def test_final_answer_prefers_the_last_latex_boxed_commitment():
    assert (
        final_answer(
            "The intermediate count is 2, but the final answer is \\boxed{3}.",
            "integer",
        )
        == "3"
    )
    assert final_answer("Therefore, \\fbox{C}.", "mcq_letter") == "C"


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


def test_extractor_answer_normalizes_explicit_uppercase_letter_sequences():
    assert (
        _extractor_answer(
            "<answer>B A C</answer>", "text", "letter_disambiguation"
        )
        == "BAC"
    )
    assert (
        _extractor_answer(
            "<answer>G, H, T</answer>", "text", "letter_disambiguation"
        )
        == "GHT"
    )
    assert (
        _extractor_answer(
            "<answer>b a c</answer>", "text", "letter_disambiguation"
        )
        == ""
    )


def test_native_box_answer_is_supported_by_source_output():
    assert _answer_is_supported_by_output(
        "Reasoning.\n<|begin_of_box|>D<|end_of_box|>", "D", "mcq_letter"
    )


def test_final_answer_normalizes_only_unambiguous_integer_words():
    assert final_answer("There are two spheres in total.", "integer") == "2"
    assert final_answer("There are zero blue shapes in the scene.", "integer") == "0"
    assert final_answer("There may be two or three spheres.", "integer") == ""


def test_final_answer_does_not_treat_question_coordinates_as_the_count():
    assert (
        final_answer(
            "There is one white square to the right of the black square at "
            "position (row 2, column 2) in the same row.",
            "integer",
        )
        == "1"
    )
    assert final_answer("The referenced cell is at row 2, column 2.", "integer") == ""


def test_final_answer_enforces_task_specific_text_contracts():
    assert final_answer("Final answer: yes", "text", "form_constancy") == "Yes"
    assert final_answer("I believe the answer is yes.", "text", "form_constancy") == ""
    assert final_answer("<answer>abc</answer>", "text", "letter_disambiguation") == "ABC"
    assert final_answer("The visible letters are ABC.", "text", "letter_disambiguation") == ""
    assert final_answer("ABCDEFGHIJ", "text", "letter_disambiguation") == ""


def test_record_answer_requires_pinned_independent_extractor_provenance():
    raw_output = "<think>unfinished reasoning"
    record = _extracted_record("q1", raw_output, "6")

    assert record_answer(record, "integer") == "6"
    assert record["output"] == raw_output
    record.pop("extractor_model_revision")
    assert record_answer(record, "integer") == ""


def test_export_submission_marks_unparseable_output_incorrect(tmp_path):
    output_path = tmp_path / "submission.jsonl"
    unfinished = "<think>" + ("Still comparing options A and B. " * 1_500)
    assert len(unfinished) > 30_000
    records = [
        _extracted_record("q1", "<answer>C</answer>", "C"),
        _extracted_record("q2", unfinished, status="unresolved"),
    ]

    result = export_submission(
        records,
        [
            {"question_id": "q1", "answer_type": "mcq_letter"},
            {"question_id": "q2", "answer_type": "mcq_letter"},
        ],
        output_path,
        mark_unparseable_incorrect=True,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {"question_id": "q1", "condition": "standard", "answer": "C"},
        {
            "question_id": "q2",
            "condition": "standard",
            "answer": INVALID_FORMAT_ANSWER,
        },
    ]
    assert result["invalid_format_count"] == 1
    assert result["invalid_format_question_ids"] == ["q2"]
    assert records[1]["submission_status"] == "invalid_format"
    assert records[1]["format_failure_reason"] == INVALID_FORMAT_REASON
    assert records[1]["raw_output_characters"] == len(unfinished)
    assert records[1]["raw_output_bytes"] == len(unfinished.encode("utf-8"))
    assert records[1]["raw_output_sha256"] == hashlib.sha256(
        unfinished.encode("utf-8")
    ).hexdigest()
    assert result["invalid_format_question_ids"] == ["q2"]


def test_export_submission_marks_malformed_text_contract_incorrect(tmp_path):
    output_path = tmp_path / "submission.jsonl"
    raw_output = "The visible letters appear to be A and B."
    records = [
        _extracted_record("q1", raw_output, status="unsupported")
    ]

    result = export_submission(
        records,
        [
            {
                "question_id": "q1",
                "answer_type": "text",
                "task": "letter_disambiguation",
            }
        ],
        output_path,
        mark_unparseable_incorrect=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["answer"] == INVALID_FORMAT_ANSWER
    assert result["invalid_format_question_ids"] == ["q1"]
    assert records[0]["task"] == "letter_disambiguation"


def test_runner_refuses_to_finalize_output_without_independent_extraction(tmp_path):
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
            "--mark-unparseable-incorrect",
        ],
    )

    assert result == 2
    assert not output.exists()


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


def test_independent_extractor_is_text_only_and_preserves_raw_output():
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
            model=MANDATORY_EXTRACTOR_MODEL_ID,
            max_tokens=200,
            seed=17,
            max_final_answer_tokens=10,
            tokenize_client=TokenizeClient(),
            extractor_model_label=MANDATORY_EXTRACTOR_MODEL_ID,
            extractor_model_revision=MANDATORY_EXTRACTOR_MODEL_REVISION,
            extractor_runtime="vllm 0.25.1",
            source_diagnostics="diagnostics.jsonl",
            extractor_chat_template_kwargs={"enable_thinking": False},
        )
    )

    assert result["output"] == raw_output
    assert result["extracted_answer"] == "6"
    assert result["answer_extraction_method"] == MANDATORY_ANSWER_EXTRACTION_METHOD_ID
    assert result["extractor_status"] == "resolved"
    assert result["extractor_model"] == MANDATORY_EXTRACTOR_MODEL_ID
    assert result["extractor_model_revision"] == MANDATORY_EXTRACTOR_MODEL_REVISION
    assert result["extractor_ground_truth_access"] is False
    assert result["extractor_image_access"] is False
    assert result["extractor_output"] == "<answer>6</answer>"
    assert result["extractor_completion_tokens"] == 5
    request = requests[0]
    assert request["temperature"] == 0.0
    assert request["top_p"] == 1.0
    assert request["seed"] == 17
    assert request["max_tokens"] == 200
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert all(
        "image_url" not in json.dumps(message) for message in request["messages"]
    )
    payload = json.loads(request["messages"][1]["content"])
    assert payload == {
        "question": "How many objects are visible?",
        "answer_type": "integer",
        "task": "",
        "candidate_response": raw_output,
    }
    assert tokenization_requests == [
        (
                "/tokenize",
                {
                    "model": MANDATORY_EXTRACTOR_MODEL_ID,
                    "prompt": "6",
                    "add_special_tokens": False,
                },
        )
    ]


def test_independent_extractor_rejects_unsupported_or_ambiguous_answers():
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
    assert unsupported["extractor_status"] == "unsupported"
    assert unresolved["extractor_error"] == "The extractor returned UNRESOLVED."
    assert unresolved["extractor_status"] == "unresolved"
    assert malformed["extractor_error"] == (
        "The extractor did not return a parseable answer block."
    )
    assert malformed["extractor_status"] == "invalid_response"
    assert all(
        "extracted_answer" not in result
        for result in (unsupported, unresolved, malformed)
    )


def test_mcq_extractor_support_requires_an_explicit_option_or_answer():
    assert not _answer_is_supported_by_output(
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
    assert _answer_is_supported_by_output(
        "After comparing them, the correct option is C.", "C", "mcq_letter"
    )
    assert _answer_is_supported_by_output(
        "Reasoning ends here.\n<answer>2", "2", "mcq_index_1_4"
    )
    assert _answer_is_supported_by_output(
        "Option 2", "2", "mcq_index_1_4"
    )
    assert _answer_is_supported_by_output(
        "The shape appears in several places.\n\nOption 2",
        "2",
        "mcq_index_1_4",
    )
    assert not _answer_is_supported_by_output(
        "Option A differs. Option B differs. Option C differs. Option D differs.",
        "D",
        "mcq_letter",
    )
    assert not _answer_is_supported_by_output(
        "I think the correct answer is D. However, I might be wrong. "
        "Alternatively, option C could be correct, but I am not sure.",
        "D",
        "mcq_letter",
    )
    assert not _answer_is_supported_by_output(
        "I think the correct answer is D. However, I might be wrong. "
        "Alternatively, option C could be correct, but I am not sure.",
        "C",
        "mcq_letter",
    )
    assert _answer_is_supported_by_output(
        "I initially considered option C and was not sure. "
        "After checking again, the final answer is D.",
        "D",
        "mcq_letter",
    )


def test_integer_extractor_support_requires_a_concluding_commitment():
    assert _answer_is_supported_by_output(
        "After counting, the number of black octagons is 4.", "4", "integer"
    )
    assert _answer_is_supported_by_output(
        "There are no red shapes in the image.", "0", "integer"
    )
    assert _answer_is_supported_by_output(
        "The items sum to 3 + 2 = **5**.", "5", "integer"
    )
    assert not _answer_is_supported_by_output(
        "We inspect row 5, column 2. Row 10 contains a triangle.",
        "10",
        "integer",
    )
    assert not _answer_is_supported_by_output(
        "Row 9: black\nRow 10: white\nRow 11: black (response truncated)",
        "10",
        "integer",
    )
    assert not _answer_is_supported_by_output(
        "Therefore, the answer is 0. But I am not sure. "
        "Maybe the answer is 1, and the response ends here.",
        "0",
        "integer",
    )
    assert not _answer_is_supported_by_output(
        "Therefore, the answer is 0. But I am not sure. "
        "Maybe the answer is 1, and the response ends here.",
        "1",
        "integer",
    )
    assert _answer_is_supported_by_output(
        "I first thought the answer was 4 and I was not sure. "
        "After recounting, the answer is 5.",
        "5",
        "integer",
    )
    assert _answer_is_supported_by_output(
        "There is one cone. There are no other cones. "
        "Therefore, the total number of cones is 1.",
        "1",
        "integer",
    )
    assert not _answer_is_supported_by_output(
        "I think there are six octagons. Wait, I made a mistake. "
        "Let me count again: 1, 2, 3, 4, 5, 6, and the response ends.",
        "6",
        "integer",
    )


def test_form_constancy_extractor_support_accepts_native_answer_box():
    assert _answer_is_supported_by_output(
        "<|begin_of_box|>Yes<|end_of_box|>",
        "Yes",
        "text",
        "form_constancy",
    )


def test_letter_extractor_support_rejects_lowercase_natural_language_fragments():
    assert _answer_is_supported_by_output(
        "I V G", "IVG", "text", "letter_disambiguation"
    )
    assert _answer_is_supported_by_output(
        "The final letters are b, e, a.", "BEA", "text", "letter_disambiguation"
    )
    assert not _answer_is_supported_by_output(
        "be a", "BE", "text", "letter_disambiguation"
    )
    assert not _answer_is_supported_by_output(
        'The shape resembles a "U" or possibly some abstract letter.',
        "U",
        "text",
        "letter_disambiguation",
    )
    assert _answer_is_supported_by_output(
        'It could be U or V. The final letters are: U.',
        "U",
        "text",
        "letter_disambiguation",
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


def test_visual_inference_honors_capped_and_uncapped_max_tokens_without_parsing(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nbenchmark-image")
    requests = []

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
    assert all("final_answer_tokens" not in request for request in requests)


def test_visual_inference_does_not_deterministically_accept_or_reject_answer_text(tmp_path):
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
        )
    )

    assert result["output"].endswith("</answer>")
    assert "final_answer_tokens" not in result
    assert "error" not in result


def test_extract_all_routes_even_well_formed_raw_answers_through_extractor(
    tmp_path, monkeypatch
):
    image = tmp_path / "sample.png"
    image.write_bytes(b"unused-by-text-only-extractor")
    questions = tmp_path / "questions.jsonl"
    question_rows = [
        {
            "question_id": "q1",
            "question": "How many objects are visible?",
            "answer_type": "integer",
            "image": str(image),
        },
        {
            "question_id": "q2",
            "question": "Which option is correct?",
            "answer_type": "mcq_letter",
            "image": str(image),
        },
    ]
    questions.write_text(
        "".join(json.dumps(row) + "\n" for row in question_rows),
        encoding="utf-8",
    )
    diagnostics = tmp_path / "diagnostics.jsonl"
    write_diagnostics(
        diagnostics,
        [
            {
                "question_id": "q1",
                "answer_type": "integer",
                "output": "<answer>6</answer>",
            },
            {
                "question_id": "q2",
                "answer_type": "mcq_letter",
                "output": "Final answer: C",
            },
        ],
    )
    requests = []

    class Completions:
        async def create(self, **request):
            requests.append(request)
            payload = json.loads(request["messages"][1]["content"])
            answer = "6" if payload["candidate_response"].startswith("<answer>") else "C"
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=f"<answer>{answer}</answer>"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=5),
            )

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        async def close(self):
            return None

    class TokenizeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"count": 1}

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def post(self, _path, *, json):
            assert json["model"] == MANDATORY_EXTRACTOR_MODEL_ID
            return TokenizeResponse()

        async def aclose(self):
            return None

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
    )
    monkeypatch.setitem(
        sys.modules, "httpx", SimpleNamespace(AsyncClient=FakeAsyncClient)
    )
    track = VisualTrackConfig(
        task_id="test",
        label="Test",
        source_subsets=(),
        questions_path=questions,
        package_dir=tmp_path,
    )
    args = SimpleNamespace(
        questions=questions,
        limit=0,
        endpoints="http://localhost:8000/v1",
        api_key="EMPTY",
        request_timeout=10.0,
        max_retries=0,
        extract_all_only=True,
        max_final_answer_tokens=200,
        resume=True,
        extractor_attempts=2,
        concurrency=2,
        model=MANDATORY_EXTRACTOR_MODEL_ID,
        extractor_max_tokens=200,
        seed=0,
        extractor_model_id=MANDATORY_EXTRACTOR_MODEL_ID,
        extractor_model_revision=MANDATORY_EXTRACTOR_MODEL_REVISION,
        extractor_quantization="unquantized",
        extractor_runtime="vllm 0.25.1",
        checkpoint_every=1,
    )

    _, records = asyncio.run(_run(args, track, diagnostics))

    assert len(requests) == 2
    assert {record["extracted_answer"] for record in records} == {"6", "C"}
    for request in requests:
        assert request["model"] == MANDATORY_EXTRACTOR_MODEL_ID
        assert request["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": False
        }
        assert "image_url" not in json.dumps(request["messages"])
        payload = json.loads(request["messages"][1]["content"])
        assert "ground_truth" not in payload
        assert set(payload) == {
            "question",
            "answer_type",
            "task",
            "candidate_response",
        }
