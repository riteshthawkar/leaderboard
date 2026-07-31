import json
from pathlib import Path

import pandas as pd

from analysis.research.analyze_track3_results import analyze
from spatial_harness.run_track3_vllm import DATASETS


def _write_model(path: Path, stronger: bool) -> None:
    rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        for group_index in range(6):
            base = (dataset_index + group_index) % 3
            for mode in ("main", "noimage", "noimgpp"):
                for prompt_mode in ("noncot", "cot"):
                    answer_type = (
                        "vqa"
                        if mode != "noimgpp" and group_index == 5
                        else "mcq"
                    )
                    direct_correct = base != 0
                    if stronger:
                        direct_correct = base != 2
                    condition_correct = direct_correct
                    if mode == "noimage":
                        condition_correct = group_index % 2 == 0
                    elif mode == "noimgpp":
                        condition_correct = group_index % 3 != 0
                    if prompt_mode == "cot":
                        condition_correct = condition_correct and group_index != 1
                    rows.append(
                        {
                            "dataset": dataset,
                            "index": f"g{group_index}",
                            "group": f"g{group_index}",
                            "answer_type": answer_type,
                            "mode": mode,
                            "pmode": prompt_mode,
                            "gt": "A",
                            "cannot_label": "C" if mode == "noimgpp" else None,
                            "judged": (
                                "1"
                                if answer_type == "vqa" and condition_correct
                                else "0"
                                if answer_type == "vqa"
                                else "C"
                                if mode == "noimgpp" and condition_correct
                                else "A"
                                if condition_correct
                                else "B"
                            ),
                            "judge_method": "paper_llm_judge",
                            "judge_attempts": 1,
                            "finish_reason": "stop",
                        }
                    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_track3_analysis_runs_the_frozen_eleven_test_family(tmp_path):
    first = tmp_path / "internvl.jsonl"
    second = tmp_path / "qwen.jsonl"
    _write_model(first, stronger=False)
    _write_model(second, stronger=True)
    output = tmp_path / "analysis"

    summary = analyze(
        [("internvl", first), ("qwen", second)],
        output,
        bootstrap_reps=100,
        permutation_reps=100,
        seed=17,
    )

    assert summary["status"] == "complete"
    assert summary["confirmatory_test_count"] == 11
    assert len(summary["quality"]) == 2
    endpoints = pd.read_csv(output / "confirmatory_endpoints.csv")
    assert len(endpoints) == 11
    assert set(endpoints["scope"]) == {"internvl", "qwen", "cross_model"}
    assert (endpoints["dataset_count"] == len(DATASETS)).all()
    group_results = pd.read_csv(output / "group_results.csv")
    assert set(group_results["condition"]) == {
        "main_noncot",
        "main_cot",
        "noimage_noncot",
        "noimage_cot",
        "noimgpp_noncot",
        "noimgpp_cot",
    }
    assert (output / "REPORT.md").is_file()
