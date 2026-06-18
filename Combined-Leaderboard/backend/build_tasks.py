"""
Builds the three task bundles for the leaderboard.

For each task it writes, under backend/tasks/<task_id>/:
  * questions.json            - public; the samples handed to users (no answers)
  * ground_truth.json         - private; sample_id -> {answer, group, ...}
  * submission_template.json  - empty template (predictions per condition)
  * submission_template.csv   - flat CSV template

Tasks:
  do_you_see_me  - drawn from the Do-You-See-Me benchmark (perception)
  minds_eye      - drawn from the Mind's-Eye benchmark (imagery)
  spatial        - the 13 public spatial benchmarks (Task 3). We do NOT
                   redistribute those datasets; instead we write a manifest +
                   a small illustrative SAMPLE set so the pipeline is testable.
                   Real ground truth is produced by running spatial_harness.

Run:  python backend/build_tasks.py
"""

import csv
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
)
from build_golden_set import _rng, _load_do_you_see_me, _load_minds_eye


def _write_task_bundle(task_id, questions, ground_truth, conditions):
    paths = TASKS[task_id]["paths"]
    paths["dir"].mkdir(parents=True, exist_ok=True)

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

    template = {
        "schema_version": "1.0",
        "task_id": task_id,
        "model_meta": {
            "name": "YOUR_MODEL_NAME", "family": "", "params_b": None,
            "type": "MLM", "is_reasoning": False,
        },
        "run": {"seed": 0, "decoding": "greedy", "temperature": 0},
        "predictions": {c: {q["sample_id"]: "" for q in questions} for c in conditions},
    }
    with open(paths["template_json"], "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)

    with open(paths["template_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "group", "condition", "question", "prediction"])
        for q in questions:
            writer.writerow([q["sample_id"], q.get("group", ""), "standard",
                             q.get("question", ""), ""])

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

    The real Task-3 ground truth is generated offline by spatial_harness from
    each dataset's official source; this sample only exercises the pipeline.
    """
    # Manifest (the public spec users run the harness against).
    SPATIAL_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPATIAL_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "task_id": "spatial",
            "version": "1.0",
            "note": "We do not redistribute these datasets. The harness downloads "
                    "each from its official source on your machine. Verify every "
                    "hf_id/config (entries with verify=true) against VLMEvalKit.",
            "conditions": EVAL_CONDITIONS,
            "no_image_plus_option": NO_IMAGE_PLUS_OPTION,
            "dataset_count": len(SPATIAL_DATASETS),
            "approx_total_samples": sum(d.get("approx_n", 0) for d in SPATIAL_DATASETS),
            "datasets": SPATIAL_DATASETS,
        }, f, indent=2)
    print(f"  [ok] spatial manifest -> {SPATIAL_MANIFEST_FILE}")

    rng = random.Random(GOLDEN_SET_SEED)
    letters = ["A", "B", "C", "D"]
    questions, ground_truth = [], {}
    for ds in SPATIAL_DATASETS:
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
            ground_truth[sid] = {"answer": ans, "group": ds["id"],
                                 "dataset": ds["name"], "tags": ds["tags"]}
    _write_task_bundle("spatial", questions, ground_truth, EVAL_CONDITIONS)


def build_all():
    print("Building task bundles...")
    build_visual_cognition_task("do_you_see_me", "do_you_see_me", _load_do_you_see_me)
    build_visual_cognition_task("minds_eye", "minds_eye", _load_minds_eye)
    build_spatial_task()
    print("Done.")


if __name__ == "__main__":
    build_all()
