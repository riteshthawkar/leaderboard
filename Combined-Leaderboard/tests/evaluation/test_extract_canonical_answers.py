import json

import pytest

from evaluation.extract_canonical_answers import (
    candidate_category,
    classify_extractor_output,
    commitment_verdict,
    contract_exact,
    evidence_supports,
    GroundTruthError,
    load_gold_answers,
    parse_extractor_output,
    valid_answer,
)


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
        '{"verdict":"GOLD_COMMITTED","answer":"B","evidence":"Answer B"}'
    ) == (
        "GOLD_COMMITTED",
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


def test_classifier_recomputes_verdict_from_gold():
    candidate = {
        "answer_type": "mcq_letter",
        "response": "The final answer is B.",
        "gold_answer": "B",
        "current_submission_answer": "A",
    }
    result = classify_extractor_output(
        candidate,
        '{"verdict":"OTHER_COMMITTED","answer":"B",'
        '"evidence":"The final answer is B"}',
    )
    assert result["status"] == "gold_committed"
    assert result["verdict"] == "GOLD_COMMITTED"
    assert result["extractor_reported_verdict"] == "OTHER_COMMITTED"


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