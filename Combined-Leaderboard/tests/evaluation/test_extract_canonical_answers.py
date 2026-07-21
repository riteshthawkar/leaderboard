import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from evaluation.extract_canonical_answers import (
    candidate_category,
    classify_extractor_output,
    commitment_verdict,
    contract_exact,
    evidence_supports,
    EXTRACTOR_RESPONSE_FORMAT,
    extractor_contract_sha256,
    extractor_payload,
    finalize_audit_checkpoint,
    GroundTruthError,
    load_gold_answers,
    load_audit_checkpoint,
    parse_extractor_output,
    run,
    terminal_source_classification,
    finalize_persistent_extractor_failure,
    valid_answer,
    wait_for_extractor_clients,
)


def test_checkpoint_retries_only_blocking_rows_and_preserves_attempts(tmp_path):
    contract = "c" * 64
    response = "The final answer is B."
    candidates = {
        ("model", "minds_eye", "q1"): {
            "model_slug": "model",
            "track": "minds_eye",
            "question_id": "q1",
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        },
        ("model", "minds_eye", "q2"): {
            "model_slug": "model",
            "track": "minds_eye",
            "question_id": "q2",
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        },
    }
    common = {
        "method": "qwen3-8b-gold-blind-evidence-extractor-v4",
        "extractor_contract_sha256": contract,
        "ground_truth_loaded": False,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
    }
    completed = {
        **common,
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "status": "committed",
        "answer": "B",
    }
    blocking = {
        **common,
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q2",
        "status": "invalid_extractor_output",
        "extractor_output": "not json",
        "extractor_attempts": [{"status": "request_error"}],
    }
    checkpoint = tmp_path / "audit.jsonl"
    checkpoint.write_text(
        json.dumps(completed) + "\n" + json.dumps(blocking) + "\n",
        encoding="utf-8",
    )

    existing, retry_history = load_audit_checkpoint(
        checkpoint,
        candidates,
        contract,
    )

    assert set(existing) == {("model", "minds_eye", "q1")}
    assert retry_history[("model", "minds_eye", "q2")] == [
        {"status": "request_error"},
        {
            "status": "invalid_extractor_output",
            "extractor_output": "not json",
        },
    ]
    retained = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert retained == [completed]


def test_checkpoint_finalizer_atomically_terminalizes_persistent_failure(tmp_path):
    contract = "c" * 64
    response = "Reasoning. **Final Answer**\n\n\\boxed{4}"
    response_hash = hashlib.sha256(response.encode()).hexdigest()
    candidate = {
        "model_slug": "model",
        "track": "do_you_see_me",
        "question_id": "q1",
        "answer_type": "mcq_index_1_4",
        "task": "visual_closure",
        "response": response,
        "response_sha256": response_hash,
    }
    row = {
        "method": "qwen3-8b-gold-blind-evidence-extractor-v4",
        "extractor_contract_sha256": contract,
        "ground_truth_loaded": False,
        "ground_truth_supplied_to_extractor": False,
        "response_sha256": response_hash,
        "model_slug": "model",
        "track": "do_you_see_me",
        "question_id": "q1",
        "status": "invalid_extractor_output",
        "extractor_output": "{",
        "extractor_attempts": [{"status": "invalid_extractor_output"}],
    }
    checkpoint = tmp_path / "audit.jsonl"
    checkpoint.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = finalize_audit_checkpoint(
        checkpoint,
        {("model", "do_you_see_me", "q1"): candidate},
        contract,
    )

    assert result["rows"] == 1
    assert result["terminalized"] == 1
    finalized = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert finalized["status"] == "committed"
    assert finalized["answer"] == "4"
    assert len(finalized["extractor_attempts"]) == 2


def test_wait_for_extractor_clients_retries_until_identity_matches():
    class Models:
        def __init__(self):
            self.calls = 0

        async def list(self):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("starting")
            if self.calls == 2:
                return SimpleNamespace(data=[SimpleNamespace(id="wrong/model")])
            return SimpleNamespace(data=[SimpleNamespace(id="Qwen/Qwen3-8B")])

    models = Models()
    client = SimpleNamespace(models=models)
    asyncio.run(
        wait_for_extractor_clients(
            [client],
            ["http://127.0.0.1:8035/v1"],
            "Qwen/Qwen3-8B",
            1,
            poll_interval=0,
        )
    )
    assert models.calls == 3


def test_wait_for_extractor_clients_rejects_wrong_identity():
    class Models:
        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="wrong/model")])

    with pytest.raises(RuntimeError, match="wrong/model"):
        asyncio.run(
            wait_for_extractor_clients(
                [SimpleNamespace(models=Models())],
                ["http://127.0.0.1:8035/v1"],
                "Qwen/Qwen3-8B",
                0.001,
                poll_interval=0,
            )
        )


def test_complete_checkpoint_exits_without_contacting_endpoints(tmp_path, monkeypatch):
    contract = extractor_contract_sha256("Qwen/Qwen3-8B", 128)
    response = "The final answer is B."
    response_hash = hashlib.sha256(response.encode()).hexdigest()
    candidate = {
        "model_slug": "model",
        "track": "minds_eye",
        "question_id": "q1",
        "response_sha256": response_hash,
    }
    checkpoint = tmp_path / "audit.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                **candidate,
                "method": "qwen3-8b-gold-blind-evidence-extractor-v4",
                "extractor_contract_sha256": contract,
                "ground_truth_loaded": False,
                "status": "committed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evaluation.extract_canonical_answers.load_candidates",
        lambda *_args, **_kwargs: [candidate],
    )
    monkeypatch.setattr(
        "evaluation.extract_canonical_answers.AsyncOpenAI",
        lambda **_kwargs: pytest.fail("complete checkpoint contacted an endpoint"),
    )
    args = SimpleNamespace(
        project_root=tmp_path,
        canonical_root=tmp_path,
        policy="all",
        exclude_variants=[],
        model="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        max_tokens=128,
        output=checkpoint,
        endpoints=["http://127.0.0.1:8035/v1"],
    )

    asyncio.run(run(args))


def test_contract_exact_distinguishes_literal_track_formats():
    assert contract_exact("do_you_see_me", "integer", "3")
    assert not contract_exact("do_you_see_me", "integer", "There are 3.")
    assert contract_exact(
        "minds_eye", "mcq_letter", "<think>Rotate it.</think><answer>B</answer>"
    )
    assert not contract_exact("minds_eye", "mcq_letter", "The answer is B")


def test_candidate_category_separates_deterministic_and_high_risk_cases():
    assert (
        candidate_category(
            "minds_eye", "mcq_letter", {"output": "The final answer is C"}
        )
        == "deterministic_explicit"
    )
    assert (
        candidate_category(
            "do_you_see_me", "integer", {"output": "I count 2, then 3"}
        )
        == "heuristic_local_parse"
    )
    assert (
        candidate_category(
            "minds_eye",
            "mcq_letter",
            {"output": "A or B", "extracted_answer": "B"},
        )
        == "prior_model_extractor"
    )


def test_extractor_json_and_answer_domains_are_strict():
    assert parse_extractor_output(
        '{"verdict":"COMMITTED","answer":"B","evidence":"Answer B"}'
    ) == (
        "COMMITTED",
        "B",
        "Answer B",
    )
    assert parse_extractor_output('{"answer":"B","evidence":"Answer B"}') is None
    assert parse_extractor_output("not json") is None
    assert valid_answer("B", "mcq_letter")
    assert not valid_answer("BC", "mcq_letter")
    assert not valid_answer("5", "mcq_index_1_4")


def test_evidence_supports_explicit_and_semantic_zero_answers():
    response = "After checking, the final answer is C."
    assert evidence_supports(response, "the final answer is C", "C", "mcq_letter")
    assert not evidence_supports(response, "the final answer is B", "B", "mcq_letter")
    zero_response = "There are no red circles in the image."
    assert evidence_supports(zero_response, zero_response, "0", "integer")
    parenthesized = "Figure (e) does not have mirrored letters."
    assert evidence_supports(parenthesized, parenthesized, "E", "mcq_letter")
    truncated_tag = "Therefore, the correct option is:\n<answer>A"
    assert evidence_supports(truncated_tag, truncated_tag, "A", "mcq_letter")


def test_evidence_must_be_quoted_from_the_response():
    assert not evidence_supports(
        "The response discusses A and B.", "The final answer is B", "B", "mcq_letter"
    )
    assert not evidence_supports(
        "1. Inspect the objects. There are 3 objects.", "1.", "1", "integer"
    )
    assert evidence_supports("Reasoning complete.\n1.", "1.", "1", "integer")


def test_boxed_answer_tokens_are_explicit_commitments():
    native = "Reasoning complete. <|begin_of_box|>D<|end_of_box|>"
    assert evidence_supports(native, "D", "D", "mcq_letter")
    assert evidence_supports(
        native,
        "<|begin_of_box|>D<|end_of_box|>",
        "D",
        "mcq_letter",
    )
    word_integer = "<|begin_of_box|>Two<|end_of_box|>"
    assert evidence_supports(word_integer, "Two", "2", "integer")
    truncated = "Reasoning complete.\n<answer>\nB\n```"
    assert evidence_supports(truncated, "B", "B", "mcq_letter")
    latex = "Therefore, \\boxed{C}"
    assert evidence_supports(latex, "\\boxed{C}", "C", "mcq_letter")


def test_gold_mentions_and_rejected_options_are_not_commitments():
    response = "Option B could be right, while option C is also plausible."
    assert not evidence_supports(
        response, "Option B could be right", "B", "mcq_letter"
    )
    rejected = "Option B is not correct; the final answer is C."
    assert not evidence_supports(rejected, "Option B is not correct", "B", "mcq_letter")
    assert evidence_supports(rejected, "the final answer is C", "C", "mcq_letter")
    mismatch = "Option B does not match the expected transformation."
    assert not evidence_supports(mismatch, mismatch, "B", "mcq_letter")
    ambiguous_count = "There may be 2 or 3 circles."
    assert not evidence_supports(ambiguous_count, ambiguous_count, "3", "integer")
    ambiguous_options = "The answer is A or B."
    assert not evidence_supports(
        ambiguous_options, ambiguous_options, "B", "mcq_letter"
    )


def test_common_explicit_mcq_commitments_are_supported():
    direct = "The correct option is (d)."
    assert evidence_supports(direct, direct, "D", "mcq_letter")
    described = (
        "Option (a) shows the shape rotated further, which matches the expected "
        "transformation."
    )
    assert evidence_supports(described, described, "A", "mcq_letter")
    only_match = "Among the options, only D matches this transformation."
    assert evidence_supports(
        f"Reasoning. {only_match}\n</done>", only_match, "D", "mcq_letter"
    )
    odd_one_out = (
        "Therefore, the figure that does not adhere to the common visual concept "
        "is Figure C."
    )
    assert evidence_supports(
        f"Reasoning. {odd_one_out}\n</done>",
        odd_one_out,
        "C",
        "mcq_letter",
    )


def test_direct_integer_and_text_commitments_are_supported():
    count = "That's 5 octagons."
    assert evidence_supports(count, count, "5", "integer")
    no_letters = "There are no letters visible in the image."
    assert evidence_supports(no_letters, no_letters, "no letters", "text")
    letters = 'The letters visible from left to right are "T".'
    assert evidence_supports(letters, letters, "T", "text")
    unicode_letters = 'The letters seen from left to right are "光学".'
    assert evidence_supports(
        unicode_letters, unicode_letters, "光学", "text"
    )
    answer_should = "So, the answer should be 3."
    assert evidence_supports(
        f"Reasoning. {answer_should}\n</done>", answer_should, "3", "integer"
    )
    letter_sequence = "So, the letters from left to right are N, E, X, T."
    assert evidence_supports(
        f"Reasoning. {letter_sequence}\n</done>",
        letter_sequence,
        "N, E, X, T",
        "text",
    )
    no_recognizable = "there don't appear to be any recognizable letters"
    assert evidence_supports(
        f"Reasoning. {no_recognizable}\n</done>",
        no_recognizable,
        "no letters",
        "text",
    )


def test_bounded_visual_conclusions_are_supported():
    recursive = (
        "Therefore, the figure that does not adhere to the recursive structure "
        "concept shared by the majority is Figure D."
    )
    assert evidence_supports(
        f"Reasoning. {recursive}\n</done>", recursive, "D", "mcq_letter"
    )
    rotation = "The correct rotational transformation of the original shape is Option C"
    assert evidence_supports(
        f"Reasoning. {rotation}\n</done>", rotation, "C", "mcq_letter"
    )
    folding = (
        "Based on the analysis, the figure that can be constructed by folding "
        "the Original Shape is option A."
    )
    assert evidence_supports(
        f"Reasoning. {folding}\n</done>", folding, "A", "mcq_letter"
    )
    malformed = "<answer>Canswer>"
    assert evidence_supports(
        f"Reasoning. {malformed}\n</done>", malformed, "C", "mcq_letter"
    )
    markdown = "**Answer:** C"
    assert evidence_supports(
        f"Reasoning. {markdown}\n</done>", markdown, "C", "mcq_letter"
    )
    bold_figure = (
        "Therefore, the figure that does not adhere to this concept is "
        "**Figure D**"
    )
    assert evidence_supports(
        f"Reasoning. {bold_figure}\n</done>",
        bold_figure,
        "D",
        "mcq_letter",
    )
    reversed_odd = (
        "Based on this analysis, Figure B is the one that does not adhere to "
        "the underlying visual concept."
    )
    assert evidence_supports(
        f"Reasoning. {reversed_odd}\n</done>",
        reversed_odd,
        "B",
        "mcq_letter",
    )
    transformation_or_pattern = (
        "Therefore, the figure that does not adhere to the common transformation "
        "or pattern shared by the majority is Figure F."
    )
    assert evidence_supports(
        f"Reasoning. {transformation_or_pattern}\n</done>",
        transformation_or_pattern,
        "F",
        "mcq_letter",
    )


def test_classifier_compares_blind_commitment_to_gold_after_extraction():
    candidate = {
        "answer_type": "mcq_letter",
        "response": "The final answer is B.",
        "gold_answer": "B",
        "current_submission_answer": "A",
    }
    result = classify_extractor_output(
        candidate,
        '{"verdict":"COMMITTED","answer":"B",'
        '"evidence":"The final answer is B"}',
    )
    assert result["status"] == "gold_committed"
    assert result["verdict"] == "GOLD_COMMITTED"
    assert result["extractor_verdict"] == "COMMITTED"


def test_raw_extractor_classification_contains_no_correctness_verdict():
    candidate = {
        "answer_type": "mcq_letter",
        "response": "The final answer is B.",
        "current_submission_answer": "A",
    }
    result = classify_extractor_output(
        candidate,
        '{"verdict":"COMMITTED","answer":"B",'
        '"evidence":"The final answer is B"}',
    )
    assert result["status"] == "committed"
    assert result["extractor_verdict"] == "COMMITTED"
    assert "verdict" not in result


def test_classifier_canonicalizes_task_specific_letter_sequences():
    candidate = {
        "answer_type": "text",
        "task": "letter_disambiguation",
        "response": "The letters visible are E T O N.",
        "current_submission_answer": "E T O N",
    }
    result = classify_extractor_output(
        candidate,
        '{"verdict":"COMMITTED","answer":"E T O N",'
        '"evidence":"The letters visible are E T O N"}',
    )
    assert result["status"] == "committed"
    assert result["answer"] == "ETON"
    assert result["proposed_answer"] == "E T O N"


def test_classifier_preserves_explicit_out_of_domain_commitment():
    candidate = {
        "answer_type": "mcq_index_1_4",
        "task": "visual_closure",
        "response": "The final answer is 7.",
        "current_submission_answer": "7",
    }
    result = classify_extractor_output(
        candidate,
        '{"verdict":"COMMITTED","answer":"7",'
        '"evidence":"The final answer is 7"}',
    )
    assert result["status"] == "invalid_format_committed"
    assert result["answer"] == "__INVALID_FORMAT__"
    assert result["proposed_answer"] == "7"


def test_classifier_downgrades_unclosed_truncated_commitments():
    candidate = {
        "answer_type": "mcq_letter",
        "task": "mental_rotation",
        "response": "The final answer is B. However, checking the next relation",
        "response_finish_reason": "length",
        "current_submission_answer": "B",
    }
    result = classify_extractor_output(
        candidate,
        '{"verdict":"COMMITTED","answer":"B",'
        '"evidence":"The final answer is B"}',
    )
    assert result["status"] == "unresolved_truncated_response"
    assert result["answer"] == ""


def test_classifier_accepts_closed_answer_from_length_limited_response():
    candidate = {
        "answer_type": "mcq_letter",
        "task": "mental_rotation",
        "response": "Reasoning. <answer>B</answer> trailing text was cut",
        "response_finish_reason": "length",
        "current_submission_answer": "B",
    }
    result = classify_extractor_output(
        candidate,
        '{"verdict":"COMMITTED","answer":"B",'
        '"evidence":"<answer>B</answer>"}',
    )
    assert result["status"] == "committed"
    assert result["answer"] == "B"


@pytest.mark.parametrize(
    ("candidate", "status", "answer"),
    [
        (
            {
                "answer_type": "mcq_letter",
                "task": "paper_folding",
                "response": "None of the options match. Therefore, there is no correct answer among the given options.",
            },
            "invalid_format_committed",
            "__INVALID_FORMAT__",
        ),
        (
            {
                "answer_type": "text",
                "task": "letter_disambiguation",
                "response": "There are no letters visible in the image.",
            },
            "invalid_format_committed",
            "__INVALID_FORMAT__",
        ),
        (
            {
                "answer_type": "text",
                "task": "letter_disambiguation",
                "response": 'On the left, the letters "H-R-E" are visible. In the middle, the letters "PHILIPS" are shown.',
            },
            "invalid_format_committed",
            "__INVALID_FORMAT__",
        ),
        (
            {
                "answer_type": "mcq_index_1_4",
                "task": "visual_closure",
                "response": "Reasoning. **Final Answer**\n\n\\boxed{4}",
            },
            "committed",
            "4",
        ),
    ],
)
def test_terminal_source_classification_is_literal_and_domain_aware(
    candidate, status, answer
):
    result = terminal_source_classification(candidate)
    assert result is not None
    assert result["status"] == status
    assert result["answer"] == answer
    assert result["evidence"] in candidate["response"]


def test_terminal_source_classification_refuses_ordinary_reasoning_mentions():
    assert terminal_source_classification(
        {
            "answer_type": "mcq_letter",
            "task": "mental_rotation",
            "response": "Option A seems plausible, but I still need to inspect B and C.",
        }
    ) is None
    assert terminal_source_classification(
        {
            "answer_type": "mcq_index_1_4",
            "task": "visual_closure",
            "response": "Maybe \\boxed{4}, but I still need to inspect option 3.",
        }
    ) is None


def test_terminal_source_classification_rejects_long_direct_letter_stream():
    response = "BBOLELLEL" + "SE" * 1000
    result = terminal_source_classification(
        {
            "answer_type": "text",
            "task": "letter_disambiguation",
            "response": response,
            "response_finish_reason": "length",
        }
    )

    assert result is not None
    assert result["status"] == "invalid_format_committed"
    assert result["answer"] == "__INVALID_FORMAT__"
    assert result["proposed_answer"] == response
    assert result["evidence"] == response


def test_terminal_fallback_requires_prior_retry_history():
    candidate = {
        "answer_type": "mcq_index_1_4",
        "task": "visual_closure",
        "response": "\\boxed{4}",
    }
    row = {"status": "invalid_extractor_output"}
    assert finalize_persistent_extractor_failure(candidate, row) is None
    result = finalize_persistent_extractor_failure(
        candidate,
        {**row, "extractor_attempts": [{"status": "invalid_extractor_output"}]},
    )
    assert result is not None
    assert result["status"] == "committed"
    assert result["terminal_fallback_method"] == (
        "deterministic-terminal-response-classifier-v1"
    )


def test_blind_extractor_prompt_contains_no_gold_contract():
    from evaluation.extract_canonical_answers import SYSTEM_PROMPT

    normalized = " ".join(SYSTEM_PROMPT.casefold().split())
    assert "gold answer" not in normalized
    assert "reference answer" in normalized
    assert "not given" in normalized


def test_extractor_payload_and_schema_are_strictly_gold_blind():
    payload = extractor_payload(
        {
            "question": "Question text",
            "answer_type": "mcq_letter",
            "response": "The final answer is B.",
            "gold_answer": "B",
        }
    )
    assert payload == {
        "question": "Question text",
        "answer_type": "mcq_letter",
        "candidate_response": "The final answer is B.",
    }
    schema = EXTRACTOR_RESPONSE_FORMAT["json_schema"]["schema"]
    assert schema["properties"]["verdict"]["enum"] == [
        "COMMITTED",
        "UNRESOLVED",
    ]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["answer"]["maxLength"] == 200
    assert schema["properties"]["evidence"]["maxLength"] == 800
    assert extractor_contract_sha256("model", 128) != extractor_contract_sha256(
        "model", 256
    )


def test_commitment_verdict_is_deterministic():
    assert commitment_verdict("b", "B", "mcq_letter") == "GOLD_COMMITTED"
    assert commitment_verdict("C", "B", "mcq_letter") == "OTHER_COMMITTED"


def test_gold_loader_requires_exact_canonical_coverage(tmp_path):
    for track, question_id, answer_type in (
        ("do_you_see_me", "t1_example", "integer"),
        ("minds_eye", "t2_example", "mcq_letter"),
    ):
        task_dir = tmp_path / "tasks" / track
        task_dir.mkdir(parents=True)
        (task_dir / "questions.jsonl").write_text(
            json.dumps(
                {
                    "question_id": question_id,
                    "answer_type": answer_type,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"t1_example": "3"}), encoding="utf-8")
    with pytest.raises(GroundTruthError, match="1 missing"):
        load_gold_answers(tmp_path, [incomplete])

    complete = tmp_path / "complete.json"
    complete.write_text(
        json.dumps(
            {
                "do_you_see_me": {"t1_example": {"answer": "3"}},
                "minds_eye": {"t2_example": "B"},
            }
        ),
        encoding="utf-8",
    )
    answers, digest = load_gold_answers(tmp_path, [complete])
    assert answers == {"t1_example": "3", "t2_example": "B"}
    assert len(digest) == 64