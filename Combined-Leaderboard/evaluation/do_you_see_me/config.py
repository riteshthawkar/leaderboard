from pathlib import Path

from evaluation.common import VisualTrackConfig


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

TRACK = VisualTrackConfig(
    task_id="do_you_see_me",
    label="Do You See Me",
    source_subsets=("dysm_2d_v1", "dysm_3d_v1"),
    questions_path=PROJECT_ROOT / "tasks" / "do_you_see_me" / "questions.jsonl",
    package_dir=PACKAGE_DIR,
)
