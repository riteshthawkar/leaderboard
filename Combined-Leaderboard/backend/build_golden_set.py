"""
Golden Set builder for the unified Visual Cognition leaderboard.

Draws a fixed, deterministic sample from every locally available task across
Do-You-See-Me (perception) and Mind's-Eye (visual cognition) and writes:

  * golden_set_questions.json      - public; the questions handed to users
  * golden_set_ground_truth.json   - private; sample_id -> answer + metadata
  * submission_template.jsonl      - empty final-answer template users fill in

Run:  python backend/build_golden_set.py
"""

import json
import random
from pathlib import Path

import pandas as pd

from config import (
    DO_YOU_SEE_ME_2D_ROOT,
    MINDS_EYE_DATA_ROOT,
    MINDS_EYE_ANSWER_FIELDS,
    TASK_TAXONOMY,
    GOLDEN_SET_DIR,
    GOLDEN_SET_QUESTIONS_FILE,
    GOLDEN_SET_GROUND_TRUTH_FILE,
    GOLDEN_SET_TEMPLATE_JSONL,
    GOLDEN_SET_SIZE_PER_TASK,
    GOLDEN_SET_SEED,
    PROJECT_ROOT,
)

# Default prompts for Do-You-See-Me tasks whose CSV lacks a 'question' column.
DYSM_DEFAULT_PROMPTS = {
    "letter_disambiguation": "Identify the letter or character shown in the image.",
    "visual_closure": "Identify the complete object or shape suggested by the partial image.",
    "visual_figure_ground": "Identify the figure that is embedded in the background of the image.",
    "visual_form_constancy": "Identify the target shape regardless of its size, position or orientation.",
    "geometric_dataset": "Identify the geometric shape shown in the image.",
    "color_and_shape_disambiguation": "Identify the shape and colour described for the image.",
    "visual_spatial": "Answer the spatial-reasoning question about the image.",
}


def _rng():
    return random.Random(GOLDEN_SET_SEED)


def _take(items, n, rng):
    """Deterministically take up to n items from a list."""
    items = list(items)
    if len(items) <= n:
        return items
    return rng.sample(items, n)


def _load_do_you_see_me(task, n, rng):
    """Yield golden records for one Do-You-See-Me task."""
    csv_path = DO_YOU_SEE_ME_2D_ROOT / task / "dataset_info.csv"
    if not csv_path.exists():
        print(f"  [skip] {task}: missing {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "answer" not in df.columns:
        print(f"  [skip] {task}: CSV lacks filename/answer columns")
        return []

    has_question = "question" in df.columns
    default_q = DYSM_DEFAULT_PROMPTS.get(task, "Answer the question for the given image.")

    rows = [r for _, r in df.iterrows() if not pd.isna(r.get("answer"))]
    rows = _take(rows, n, rng)

    records = []
    for i, row in enumerate(rows):
        filename = str(row["filename"]).strip()
        question = str(row["question"]).strip() if has_question and not pd.isna(row.get("question")) else default_q
        answer = str(row["answer"]).strip()
        records.append({
            "task": task,
            "benchmark": "do_you_see_me",
            "image": filename,
            "image_path": f"Do-You-See-Me/2D_DoYouSeeMe/dataset/{task}/{filename}",
            "question": question,
            "options": None,
            "answer": answer,
            "index": i,
        })
    return records


def _load_minds_eye(task, n, rng):
    """Yield golden records for one Mind's-Eye task."""
    json_path = MINDS_EYE_DATA_ROOT / task / "annotations.json"
    if not json_path.exists():
        print(f"  [skip] {task}: missing {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    answer_field = MINDS_EYE_ANSWER_FIELDS.get(task, "answer")
    keys = _take(sorted(data.keys()), n, rng)

    records = []
    for i, key in enumerate(keys):
        item = data[key]
        raw_answer = item.get(answer_field, "")
        answer = str(raw_answer).strip()
        if answer.startswith("(") and answer.endswith(")"):
            answer = answer[1:-1].strip()
        if not answer:
            continue

        q = item.get("question", "")
        if isinstance(q, list):
            q = q[0] if q else ""
        question = str(q).strip() or "Select the correct option (A, B, C or D) shown in the image."

        records.append({
            "task": task,
            "benchmark": "minds_eye",
            "image": key,
            "image_path": f"Mind-s-Eye/data/{task}/{key}",
            "question": question,
            "options": None,  # MCQ options are rendered inside the image
            "answer": answer,
            "index": i,
        })
    return records


def build_golden_set():
    rng = _rng()
    questions = []
    ground_truth = {}

    print(f"Building golden set ({GOLDEN_SET_SIZE_PER_TASK} samples/task, seed={GOLDEN_SET_SEED})")
    for key, (capability, layer, dimension, fmt) in TASK_TAXONOMY.items():
        benchmark, task = key.split("/", 1)
        if benchmark == "do_you_see_me":
            recs = _load_do_you_see_me(task, GOLDEN_SET_SIZE_PER_TASK, rng)
        elif benchmark == "minds_eye":
            recs = _load_minds_eye(task, GOLDEN_SET_SIZE_PER_TASK, rng)
        else:
            recs = []

        for r in recs:
            sample_id = f"{benchmark}:{task}:{r['index']:04d}"
            questions.append({
                "sample_id": sample_id,
                "benchmark": benchmark,
                "task": task,
                "capability": capability,
                "layer": layer,
                "dimension": dimension,
                "format": fmt,
                "image": r["image"],
                "image_path": r["image_path"],
                "question": r["question"],
                "options": r["options"],
            })
            ground_truth[sample_id] = {
                "answer": r["answer"],
                "benchmark": benchmark,
                "task": task,
                "capability": capability,
                "layer": layer,
                "dimension": dimension,
                "format": fmt,
            }
        print(f"  [ok]   {key}: {len(recs)} samples")

    GOLDEN_SET_DIR.mkdir(parents=True, exist_ok=True)

    with open(GOLDEN_SET_QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0",
            "total_samples": len(questions),
            "tasks": len(TASK_TAXONOMY),
            "samples": questions,
        }, f, indent=2)

    with open(GOLDEN_SET_GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    with open(GOLDEN_SET_TEMPLATE_JSONL, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({
                "sample_id": q["sample_id"],
                "condition": "standard",
                "answer": "",
            }) + "\n")

    print(f"\nWrote {len(questions)} samples across {len(TASK_TAXONOMY)} tasks to {GOLDEN_SET_DIR}")
    return questions, ground_truth


if __name__ == "__main__":
    build_golden_set()
