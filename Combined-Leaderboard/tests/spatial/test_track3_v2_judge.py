from spatial_harness.judge_track3 import (
    JUDGE_VQA,
    aggregate,
    correct,
    parse_bit,
    parse_letter,
)


def test_vqa_judge_contract_and_correctness():
    assert "STRICTLY OUTPUT EXACTLY ONE CHARACTER" in JUDGE_VQA
    assert parse_bit("1\n") == "1"
    assert parse_bit("explanation") is None
    assert correct({"answer_type": "vqa", "judged": "1", "mode": "main"}) is True
    assert correct({"answer_type": "vqa", "judged": "0", "mode": "main"}) is False


def test_mcq_and_no_image_plus_correctness():
    assert parse_letter("C") == "C"
    assert correct({"answer_type": "mcq", "judged": "B", "mode": "main", "gt": "B"})
    assert correct(
        {
            "answer_type": "mcq",
            "judged": "E",
            "mode": "noimgpp",
            "gt": "A",
            "cannot_label": "E",
        }
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