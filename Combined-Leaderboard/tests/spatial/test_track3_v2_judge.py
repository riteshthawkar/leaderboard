import asyncio

from spatial_harness.judge_track3 import (
    JUDGE_VQA,
    aggregate,
    correct,
    explicit_abstention_letter,
    is_terminal_inference_failure,
    judge_one,
    parse_bit,
    parse_letter,
)
from spatial_harness.submission_contract import (
    INFERENCE_FAILURE_DISPOSITION,
    INFERENCE_FAILURE_JUDGE_METHOD,
    INFERENCE_FAILURE_POLICY,
    INFERENCE_FAILURE_SCHEMA_VERSION,
)


def test_vqa_judge_contract_and_correctness():
    assert "STRICTLY OUTPUT EXACTLY ONE CHARACTER" in JUDGE_VQA
    assert parse_bit("1\n") == "1"
    assert parse_bit("explanation") is None
    assert correct({"answer_type": "vqa", "judged": "1", "mode": "main"}) is True
    assert correct({"answer_type": "vqa", "judged": "0", "mode": "main"}) is False


def test_mcq_and_no_image_plus_correctness():
    assert parse_letter("C") == "C"
    assert parse_letter("C because") is None
    assert correct({"answer_type": "mcq", "judged": "B", "mode": "main", "gt": "B"})
    assert correct(
        {"answer_type": "mcq", "judged": "B", "mode": "noimage", "gt": "B"}
    )
    assert correct(
        {
            "answer_type": "mcq",
            "judged": "E",
            "mode": "noimgpp",
            "gt": "A",
            "cannot_label": "E",
        }
    )
    assert (
        explicit_abstention_letter(
            {
                "mode": "noimgpp",
                "cannot_label": "E",
                "output": "<answer>Cannot determine from the image</answer>",
            }
        )
        == "E"
    )


def test_group_aggregation_requires_every_rotation_to_be_correct():
    common = {
        "dataset": "SpatialBench",
        "mode": "main",
        "pmode": "noncot",
        "answer_type": "mcq",
    }
    items = [
        {**common, "index": "q1_r0", "group": "q1", "gt": "A", "judged": "A"},
        {**common, "index": "q1_r1", "group": "q1", "gt": "B", "judged": "A"},
        {**common, "index": "q2_r0", "group": "q2", "gt": "A", "judged": "A"},
        {**common, "index": "q2_r1", "group": "q2", "gt": "B", "judged": "B"},
    ]
    result = aggregate(items)["datasets"]["SpatialBench"]
    assert result["main_noncot"] == 0.5
    assert result["main_noncot_n"] == 2


def test_aggregation_drops_group_with_unresolved_member():
    result = aggregate(
        [
            {
                "dataset": "BLINK",
                "mode": "main",
                "pmode": "cot",
                "index": "q1_r0",
                "group": "q1",
                "answer_type": "mcq",
                "gt": "A",
                "judged": None,
            }
        ]
    )["datasets"]["BLINK"]
    assert result["main_cot"] is None
    assert result["main_cot_n"] == 0
    assert result["main_cot_unresolved"] == 1


def test_aggregation_reports_both_ablation_conditions_explicitly():
    common = {
        "dataset": "BLINK",
        "pmode": "noncot",
        "index": "q1",
        "group": "q1",
        "answer_type": "mcq",
        "gt": "A",
        "judged": "A",
    }
    result = aggregate(
        [
            {**common, "mode": "noimage"},
            {
                **common,
                "mode": "noimgpp",
                "cannot_label": "C",
                "judged": "C",
            },
        ]
    )["datasets"]["BLINK"]
    assert result["noimage_noncot"] == 1.0
    assert result["noimgpp_noncot"] == 1.0


def test_judge_records_attempt_provenance():
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            choice = type(
                "Choice",
                (),
                {"message": type("Message", (), {"content": "B"})()},
            )()
            return type("Response", (), {"choices": [choice]})()

    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": Completions()})()},
    )()
    item = {
        "answer_type": "mcq",
        "mode": "main",
        "options": {"A": "left", "B": "right"},
        "output": "B",
    }
    result = asyncio.run(
        judge_one(
            client,
            asyncio.Semaphore(1),
            item,
            "judge",
            2,
        )
    )
    assert result["judged"] == "B"
    assert result["judge_method"] == "paper_llm_judge"
    assert result["judge_attempts"] == 1
    assert captured["max_tokens"] == 4
    assert captured["extra_body"] == {
        "structured_outputs": {"choice": ["0", "A", "B"]}
    }


def test_explicit_abstention_records_zero_judge_attempts():
    item = {
        "answer_type": "mcq",
        "mode": "noimgpp",
        "options": {"A": "left", "B": "Cannot determine from the image"},
        "cannot_label": "B",
        "output": "<answer>Cannot determine from the image</answer>",
    }
    result = asyncio.run(
        judge_one(
            object(),
            asyncio.Semaphore(1),
            item,
            "judge",
            2,
        )
    )
    assert result["judged"] == "B"
    assert result["judge_method"] == "explicit_abstention"
    assert result["judge_attempts"] == 0


def test_terminal_context_failure_is_scored_incorrect_without_judge():
    item = {
        "answer_type": "mcq",
        "mode": "main",
        "options": {"A": "left", "B": "right"},
        "gt": "A",
        "output": "",
        "error": "BadRequestError: Input length (33432) exceeds model's maximum context length (32768).",
        "finish_reason": "inference_error",
        "terminal_failure": {
            "schema_version": INFERENCE_FAILURE_SCHEMA_VERSION,
            "policy": INFERENCE_FAILURE_POLICY,
            "category": "input_context_exceeded",
            "disposition": INFERENCE_FAILURE_DISPOSITION,
            "input_length": 33432,
            "max_model_len": 32768,
        },
    }

    assert is_terminal_inference_failure(item)
    result = asyncio.run(
        judge_one(
            object(),
            asyncio.Semaphore(1),
            item,
            "judge",
            2,
        )
    )

    assert result["judged"] == "0"
    assert result["judge_method"] == INFERENCE_FAILURE_JUDGE_METHOD
    assert result["judge_attempts"] == 0
    assert correct(result) is False
