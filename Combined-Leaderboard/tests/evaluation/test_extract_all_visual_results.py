import json
from argparse import Namespace
from pathlib import Path

import pytest

from evaluation.extract_all_visual_results import (
    BatchExtractionError,
    contract_record,
    discover_jobs,
    prepare_output,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def create_track(
    root: Path,
    relative_dir: str,
    track: str,
    ids: list[str],
    *,
    model_id: str,
    revision: str,
) -> None:
    directory = root / relative_dir
    write_jsonl(
        directory / f"{track}.diagnostics.jsonl",
        [
            {
                "question_id": question_id,
                "answer_type": "mcq_letter",
                "output": "Final answer is A.",
            }
            for question_id in ids
        ],
    )
    write_jsonl(
        directory / f"{track}_submission.jsonl",
        [
            {
                "question_id": question_id,
                "condition": "standard",
                "answer": "A",
            }
            for question_id in ids
        ],
    )
    (directory / ".run_config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "model_revision": revision,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ),
        encoding="utf-8",
    )
    (directory / "run_manifest.json").write_text("{}\n", encoding="utf-8")


def test_batch_preparation_discovers_variants_and_locks_source_hashes(tmp_path):
    project_root = tmp_path / "project"
    source_root = project_root / "evaluation" / "results" / "final"
    output_root = project_root / "evaluation" / "results" / "final-extracted-v11"
    ids_by_track = {
        "do_you_see_me": ["d1", "d2"],
        "minds_eye": ["m1", "m2"],
    }
    for track, identifiers in ids_by_track.items():
        write_jsonl(
            project_root / "tasks" / track / "questions.jsonl",
            [
                {
                    "question_id": question_id,
                    "question": "Choose one.",
                    "answer_type": "mcq_letter",
                }
                for question_id in identifiers
            ],
        )
        create_track(
            source_root,
            "model-a",
            track,
            identifiers,
            model_id="org/model-a",
            revision="a" * 40,
        )
        create_track(
            source_root,
            f"variant-b/{track}-run",
            track,
            identifiers,
            model_id="org/model-b",
            revision="b" * 40,
        )
    (source_root / "index.json").write_text("{}\n", encoding="utf-8")
    (source_root / "model-a" / "final_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )

    args = Namespace(
        model="Qwen/Qwen3-8B",
        revision="c" * 40,
        seed=0,
        max_tokens=200,
    )
    jobs = discover_jobs(source_root, project_root)
    contract = contract_record(args)
    manifest = prepare_output(source_root, output_root, jobs, contract)

    assert len(jobs) == 4
    assert {job.variant_id for job in jobs} == {"model-a", "variant-b"}
    assert manifest["variant_count"] == 2
    assert manifest["track_job_count"] == 4
    assert manifest["total_response_count"] == 8
    assert (output_root / "source_index.json").is_file()
    assert (output_root / "model-a" / "source_final_manifest.json").is_file()
    assert (output_root / "model-a" / "source_run_manifest.json").is_file()
    assert not (output_root / "index.json").exists()

    resumed = prepare_output(source_root, output_root, jobs, contract)
    assert resumed["source_inventory_sha256"] == manifest["source_inventory_sha256"]

    diagnostics = source_root / "model-a" / "minds_eye.diagnostics.jsonl"
    rows = diagnostics.read_text(encoding="utf-8")
    diagnostics.write_text(rows.replace("Final answer is A", "Final answer is B"), encoding="utf-8")
    changed_jobs = discover_jobs(source_root, project_root)
    with pytest.raises(BatchExtractionError, match="unsafe resume"):
        prepare_output(source_root, output_root, changed_jobs, contract)
