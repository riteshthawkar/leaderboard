from pathlib import Path

from evaluation.common import VisualTrackConfig


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

TRACK = VisualTrackConfig(
    task_id="minds_eye",
    label="Mind's Eye",
    source_subsets=("minds_eye_fresh_v1",),
    questions_path=PROJECT_ROOT / "tasks" / "minds_eye" / "questions.jsonl",
    package_dir=PACKAGE_DIR,
)
