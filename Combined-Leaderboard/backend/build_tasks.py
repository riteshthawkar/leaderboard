"""
Builds the three task bundles for the leaderboard.

For each task it writes public questions/templates under ``tasks/<task_id>/``
and the private answer key under the ignored ``Ground_truths/`` tree:
  * questions.json            - public; the samples handed to users (no answers)
  * submission_template.jsonl - public empty final-answer template
  * ground_truth.json         - private sample/question answer map

Tasks:
  do_you_see_me  - drawn from the Do-You-See-Me benchmark (perception)
  minds_eye      - drawn from the Mind's-Eye benchmark (cognition)
  spatial        - the 13 public spatial benchmarks (Task 3). We do NOT
                   redistribute those datasets; instead we write a manifest +
                   a small illustrative SAMPLE set so the pipeline is testable.
                   Real ground truth is produced by evaluation/spatial_reasoning/build_server_bundle.py.

Run:  python backend/build_tasks.py
"""

import json
import random

from config import (
    TASKS,
    TASK_TAXONOMY,
    SPATIAL_DATASETS,
    SPATIAL_MANIFEST_FILE,
    NO_IMAGE_PLUS_OPTION,
    GOLDEN_SET_SEED,
    EVAL_CONDITIONS,
    SPATIAL_BENCHMARK_SCHEMA_VERSION,
    SPATIAL_DATASET_KEYS,
)
from build_golden_set import _rng, _load_do_you_see_me, _load_minds_eye


def _write_task_bundle(task_id, questions, ground_truth, conditions):
    paths = TASKS[task_id]["paths"]
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["ground_truth"].parent.mkdir(parents=True, exist_ok=True)

    with open(paths["questions"], "w", encoding="utf-8") as f:
        json.dump({
            "task_id": task_id,
            "label": TASKS[task_id]["label"],
            "section": TASKS[task_id]["section"],
            "layer": TASKS[task_id]["layer"],
            "version": "1.0",
            "total_samples": len(questions),
            "conditions": conditions,
            "samples": questions,
        }, f, indent=2)

    with open(paths["ground_truth"], "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    with open(paths["template_jsonl"], "w", encoding="utf-8") as f:
        for condition in conditions:
            for q in questions:
                id_field = "question_id" if q.get("question_id") else "sample_id"
                f.write(json.dumps({
                    id_field: q.get(id_field) or q.get("sample_id"),
                    "condition": condition,
                    "answer": "",
                }) + "\n")

    print(f"  [ok] {task_id}: {len(questions)} samples -> {paths['dir']}")


def build_visual_cognition_task(task_id, benchmark, loader):
    rng = _rng()
    questions, ground_truth = [], {}
    for key, (capability, layer, dimension, fmt) in TASK_TAXONOMY.items():
        bm, task = key.split("/", 1)
        if bm != benchmark:
            continue
        from config import GOLDEN_SET_SIZE_PER_TASK
        recs = loader(task, GOLDEN_SET_SIZE_PER_TASK, rng)
        for r in recs:
            sample_id = f"{benchmark}:{task}:{r['index']:04d}"
            questions.append({
                "sample_id": sample_id,
                "group": capability,
                "task": task,
                "capability": capability,
                "dimension": dimension,
                "format": fmt,
                "image_path": r["image_path"],
                "question": r["question"],
                "options": r["options"],
            })
            ground_truth[sample_id] = {
                "answer": r["answer"],
                "group": capability,
                "task": task,
                "capability": capability,
            }
    _write_task_bundle(task_id, questions, ground_truth, ["standard"])


def build_spatial_task(sample_per_dataset=3):
    """Write the spatial manifest + a small illustrative SAMPLE bundle.

    The real Task-3 ground truth is generated offline by evaluation/spatial_reasoning from
    each dataset's official source; this sample only exercises the pipeline.
    """
    # Manifest (the public spec users run the harness against).
    SPATIAL_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPATIAL_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": SPATIAL_BENCHMARK_SCHEMA_VERSION,
            "task_id": "spatial",
            "benchmark_version": "demo-only",
            "demo": True,
            "note": "This checked-in bundle only exercises local code paths. It cannot be used for leaderboard submissions.",
            "datasets": SPATIAL_DATASET_KEYS,
            "required_conditions": EVAL_CONDITIONS,
            "primary_condition": "main_noncot",
            "no_image_plus_option": NO_IMAGE_PLUS_OPTION,
            "dataset_count": len(SPATIAL_DATASETS),
            "condition_counts": {
                condition: len(SPATIAL_DATASETS) * sample_per_dataset
                for condition in EVAL_CONDITIONS
            },
            "approx_total_samples": sum(d.get("approx_n", 0) for d in SPATIAL_DATASETS),
            "dataset_catalog": SPATIAL_DATASETS,
        }, f, indent=2)
    print(f"  [ok] spatial manifest -> {SPATIAL_MANIFEST_FILE}")

    rng = random.Random(GOLDEN_SET_SEED)
    letters = ["A", "B", "C", "D"]
    questions, ground_truth = [], {}
    for dataset_index, ds in enumerate(SPATIAL_DATASETS):
        for i in range(sample_per_dataset):
            sid = f"{ds['id']}:{i:03d}"
            ans = rng.choice(letters)
            questions.append({
                "sample_id": sid,
                "group": ds["id"],
                "dataset": ds["name"],
                "type": ds["type"],
                "tags": ds["tags"],
                "question": "[SAMPLE] Multiple-choice spatial question; see harness "
                            "output for real items.",
                "options": {l: f"option {l}" for l in letters},
                "sample": True,
            })
            condition_answers = {
                condition: ("E" if condition.startswith("no_image_plus_") else ans)
                for condition in EVAL_CONDITIONS
            }
            ground_truth[sid] = {
                "answer": ans,
                "condition_answers": condition_answers,
                "conditions": EVAL_CONDITIONS,
                "group": ds["name"],
                "dataset": ds["name"],
                "dataset_key": SPATIAL_DATASET_KEYS[dataset_index],
                "evaluation_group": sid,
                "tags": ds["tags"],
            }
    _write_task_bundle("spatial", questions, ground_truth, EVAL_CONDITIONS)


def build_all():
    print("Building task bundles...")
    build_visual_cognition_task("do_you_see_me", "do_you_see_me", _load_do_you_see_me)
    build_visual_cognition_task("minds_eye", "minds_eye", _load_minds_eye)
    build_spatial_task()
    print("Done.")


if __name__ == "__main__":
    build_all()
