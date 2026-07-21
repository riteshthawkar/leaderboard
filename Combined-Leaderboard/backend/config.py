"""
Configuration and paths for the leaderboard system.
"""

import math
import os
import re
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Base directories
COMBINED_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = COMBINED_ROOT.parent

# Load environment variables from the Combined-Leaderboard/.env file (if present)
# before any os.getenv() lookups below. This is where the judge API key lives.
# The .env lives in the Combined-Leaderboard package dir (this file's grandparent),
# which is NOT PROJECT_ROOT (the workspace root, one level further up).
ENV_FILE = COMBINED_ROOT / ".env"

_DOTENV_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _duplicate_dotenv_keys(path: Path) -> list[str]:
    """Return assignment names repeated in a dotenv file without exposing values."""
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Could not read environment file {path}: {exc}") from exc

    seen = set()
    duplicates = set()
    for line in lines:
        match = _DOTENV_ASSIGNMENT.match(line.strip())
        if not match:
            continue
        key = match.group(1)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def _assert_unique_dotenv_assignments(path: Path) -> None:
    duplicates = _duplicate_dotenv_keys(path)
    if duplicates:
        raise RuntimeError(
            f"Environment file {path} contains duplicate assignments for: "
            + ", ".join(duplicates)
            + ". Keep exactly one value for each variable."
        )


_assert_unique_dotenv_assignments(ENV_FILE)
load_dotenv(ENV_FILE)
if os.name != "nt" and ENV_FILE.exists():
    try:
        os.chmod(ENV_FILE, 0o600)
    except OSError as exc:
        warnings.warn(
            f"Could not restrict .env permissions: {exc}",
            RuntimeWarning,
            stacklevel=1,
        )


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, "").strip() or str(default)).expanduser().resolve()


def _optional_path_from_env(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def _normalize_database_url(url: str) -> str:
    # Several deployment platforms still expose postgres:// URLs. SQLAlchemy
    # expects postgresql://.
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive whole number.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive whole number.")
    return value


def _nonnegative_finite_float_from_env(name: str, default: float | str) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative finite number.") from exc
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"{name} must be a non-negative finite number.")
    return value


BACKEND_DIR = COMBINED_ROOT / "backend"

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local").strip().lower() or "local"
if DEPLOYMENT_MODE not in {"local", "public"}:
    raise RuntimeError("DEPLOYMENT_MODE must be either 'local' or 'public'.")
PUBLIC_DEPLOYMENT = DEPLOYMENT_MODE == "public"

# Runtime state defaults to the historical workspace-level location so existing
# local data keeps working. In production, mount a persistent volume and set
# LEADERBOARD_DATA_DIR, or set each path explicitly.
DATA_DIR = _path_from_env("LEADERBOARD_DATA_DIR", PROJECT_ROOT)
RESULTS_DIR = _path_from_env("RESULTS_DIR", DATA_DIR / "results")
AUTO_BACKUP_DIR = _path_from_env("AUTO_BACKUP_DIR", DATA_DIR / "backups")
AUTO_BACKUP_MIRROR_DIR = _optional_path_from_env("AUTO_BACKUP_MIRROR_DIR")

_production_environment = (
    PUBLIC_DEPLOYMENT
    or os.getenv("FLASK_ENV", "").strip().lower() == "production"
)
AUTO_BACKUP_ENABLED = _bool_from_env("AUTO_BACKUP_ENABLED", _production_environment)
AUTO_BACKUP_RUN_ON_START = _bool_from_env("AUTO_BACKUP_RUN_ON_START", True)
AUTO_BACKUP_INTERVAL_HOURS = _positive_int_from_env("AUTO_BACKUP_INTERVAL_HOURS", 48)
AUTO_BACKUP_RETENTION_COUNT = _positive_int_from_env("AUTO_BACKUP_RETENTION_COUNT", 15)
AUTO_BACKUP_POLL_SECONDS = _positive_int_from_env("AUTO_BACKUP_POLL_SECONDS", 300)
REQUIRE_OFFSITE_BACKUP = PUBLIC_DEPLOYMENT or _bool_from_env("REQUIRE_OFFSITE_BACKUP", False)
WEB_CONCURRENCY = _positive_int_from_env("WEB_CONCURRENCY", 1)
SQLITE_BUSY_TIMEOUT_MS = _positive_int_from_env("SQLITE_BUSY_TIMEOUT_MS", 5000)
MIN_FREE_DISK_BYTES = _positive_int_from_env("MIN_FREE_DISK_BYTES", 1024 * 1024 * 1024)
MIN_FREE_DISK_PERCENT = _positive_int_from_env("MIN_FREE_DISK_PERCENT", 5)
if MIN_FREE_DISK_PERCENT > 100:
    raise RuntimeError("MIN_FREE_DISK_PERCENT must be between 1 and 100.")

DEFAULT_DATABASE_URL = f"sqlite:///{DATA_DIR / 'leaderboard.db'}"
DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL", "").strip() or DEFAULT_DATABASE_URL)
AUTH_DATABASE_URL = _normalize_database_url(os.getenv("AUTH_DATABASE_URL", "").strip() or DATABASE_URL)
SUBMISSION_DATABASE_URL = _normalize_database_url(
    os.getenv("SUBMISSION_DATABASE_URL", "").strip() or DATABASE_URL
)


def _validate_sqlite_concurrency(
    public_deployment: bool,
    database_urls: tuple[str, ...],
    worker_count: int,
) -> None:
    if (
        public_deployment
        and any(url.lower().startswith("sqlite:") for url in database_urls)
        and worker_count != 1
    ):
        raise RuntimeError(
            "Public SQLite deployment requires WEB_CONCURRENCY=1. "
            "Use Gunicorn threads for limited concurrency."
        )


def _validate_backup_configuration(
    public_deployment: bool,
    enabled: bool,
    backup_dir: Path,
    mirror_dir: Path | None,
    require_mirror: bool,
) -> None:
    if public_deployment and not enabled:
        raise RuntimeError("Public deployment requires AUTO_BACKUP_ENABLED=true.")
    if require_mirror and mirror_dir is None:
        raise RuntimeError(
            "Public deployment requires AUTO_BACKUP_MIRROR_DIR on a second mounted filesystem."
        )
    if mirror_dir is None:
        return
    if mirror_dir == backup_dir:
        raise RuntimeError("AUTO_BACKUP_MIRROR_DIR must differ from AUTO_BACKUP_DIR.")
    if require_mirror and backup_dir.exists() and mirror_dir.exists():
        try:
            same_filesystem = os.stat(backup_dir).st_dev == os.stat(mirror_dir).st_dev
        except OSError as exc:
            raise RuntimeError(f"Could not inspect backup filesystems: {exc}") from exc
        if same_filesystem:
            raise RuntimeError(
                "AUTO_BACKUP_MIRROR_DIR must be mounted on a different filesystem."
            )


_validate_sqlite_concurrency(
    PUBLIC_DEPLOYMENT,
    (DATABASE_URL, AUTH_DATABASE_URL, SUBMISSION_DATABASE_URL),
    WEB_CONCURRENCY,
)
_validate_backup_configuration(
    PUBLIC_DEPLOYMENT,
    AUTO_BACKUP_ENABLED,
    AUTO_BACKUP_DIR,
    AUTO_BACKUP_MIRROR_DIR,
    REQUIRE_OFFSITE_BACKUP,
)
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# Ground truth data directories
# The benchmark datasets live alongside the Combined-Leaderboard folder, i.e.
# directly under PROJECT_ROOT (the Leaderboard-Project workspace root).
DO_YOU_SEE_ME_ROOT = Path(os.getenv("DO_YOU_SEE_ME_ROOT", PROJECT_ROOT / "Do-You-See-Me"))
MINDS_EYE_ROOT = Path(os.getenv("MINDS_EYE_ROOT", PROJECT_ROOT / "Mind-s-Eye"))

# Do-You-See-Me dataset paths
DO_YOU_SEE_ME_2D_ROOT = DO_YOU_SEE_ME_ROOT / "2D_DoYouSeeMe"
DO_YOU_SEE_ME_3D_ROOT = DO_YOU_SEE_ME_ROOT / "3D_DoYouSeeMe"

# Mind's-Eye dataset paths
MINDS_EYE_DATA_ROOT = MINDS_EYE_ROOT / "data"

# Private visual-task answer keys. These are never served to users. Spatial
# results use the versioned public-evidence contract instead of server scoring.
GROUND_TRUTHS_DIR = _path_from_env("GROUND_TRUTHS_DIR", COMBINED_ROOT / "Ground_truths")
GROUND_TRUTHS_SOURCE = os.getenv("GROUND_TRUTHS_SOURCE", "auto").strip().lower() or "auto"
if GROUND_TRUTHS_SOURCE not in {"auto", "local", "hf"}:
    raise RuntimeError("GROUND_TRUTHS_SOURCE must be auto, local, or hf.")
GROUND_TRUTHS_HF_REPO = os.getenv("GROUND_TRUTHS_HF_REPO", "").strip()
GROUND_TRUTHS_HF_REPO_TYPE = os.getenv("GROUND_TRUTHS_HF_REPO_TYPE", "dataset").strip().lower()
if GROUND_TRUTHS_HF_REPO_TYPE not in {"dataset", "datasets", "model", "models"}:
    raise RuntimeError(
        "GROUND_TRUTHS_HF_REPO_TYPE must be dataset or model."
    )
GROUND_TRUTHS_HF_REVISION = os.getenv("GROUND_TRUTHS_HF_REVISION", "main").strip() or "main"
GROUND_TRUTHS_HF_CACHE_DIR = _path_from_env(
    "GROUND_TRUTHS_HF_CACHE_DIR", DATA_DIR / "ground_truths_cache"
)
GROUND_TRUTHS_HF_FORCE_REFRESH = os.getenv(
    "GROUND_TRUTHS_HF_FORCE_REFRESH", "false"
).strip().lower() in {"1", "true", "yes", "on"}
HF_TOKEN = (
    os.getenv("HF_TOKEN", "").strip()
    or os.getenv("HUGGINGFACE_TOKEN", "").strip()
    or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    or os.getenv("HF_API_TOKEN", "").strip()
)
DYSM_2D_GROUND_TRUTH = GROUND_TRUTHS_DIR / "dysm_2d_v1" / "ground_truth.jsonl"
DYSM_3D_GROUND_TRUTH = GROUND_TRUTHS_DIR / "dysm_3d_v1" / "ground_truth.jsonl"
MINDS_EYE_FRESH_GROUND_TRUTH = GROUND_TRUTHS_DIR / "minds_eye_fresh_v1" / "ground_truth.jsonl"

# Supported tasks per benchmark
# These match the actual 2D_DoYouSeeMe dataset folder names.
DO_YOU_SEE_ME_TASKS = [
    "visual_spatial",
    "letter_disambiguation",
    "visual_form_constancy",
    "visual_figure_ground",
    "visual_closure",
    "geometric_dataset",
    "color_and_shape_disambiguation",
]

MINDS_EYE_TASKS = [
    "slippage",
    "abstract",
    "mental_rotation",
    "mental_composition",
    "paper_folding",
    "dynamic_isomorph",
    "symmetric_isomorph",
    "hierarchial_isomorph",
]

# Task answer field mappings for Mind's-Eye
MINDS_EYE_ANSWER_FIELDS = {
    "slippage": "violation",
    "abstract": "answer",
    "mental_rotation": "answer",
    "mental_composition": "answer",
    "paper_folding": "correct_option",
    "dynamic_isomorph": "fifth_label",
    "symmetric_isomorph": "asymmetric_label",
    "hierarchial_isomorph": "answer",
}

# Supported submission file extensions. Submissions are newline-delimited JSON:
# one row per sample/condition with at least {"question_id": "...", "answer": "..."}.
# Legacy local bundles may use sample_id; the scorer accepts both.
SUPPORTED_PREDICTION_FORMATS = [".jsonl"]

# Ollama configuration for option extraction
OLLAMA_MODEL = os.getenv("MINDS_EYE_OPTION_EXTRACTOR_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("MINDS_EYE_OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Unified "Visual Cognition" battery
# ---------------------------------------------------------------------------
# Golden set: a fixed, combined sample drawn from every locally available task.
# Users download the questions, run their model, and submit a single file with
# predictions for every golden-set sample.
GOLDEN_SET_DIR = Path(os.getenv("GOLDEN_SET_DIR", COMBINED_ROOT / "golden_set"))
GOLDEN_SET_QUESTIONS_FILE = GOLDEN_SET_DIR / "golden_set_questions.json"
GOLDEN_SET_GROUND_TRUTH_FILE = GOLDEN_SET_DIR / "golden_set_ground_truth.json"
GOLDEN_SET_TEMPLATE_JSONL = GOLDEN_SET_DIR / "submission_template.jsonl"

# How many samples to draw per task when building the golden set.
GOLDEN_SET_SIZE_PER_TASK = int(os.getenv("GOLDEN_SET_SIZE_PER_TASK", "25"))
GOLDEN_SET_SEED = int(os.getenv("GOLDEN_SET_SEED", "42"))

# Cognitive taxonomy: maps each "<benchmark>/<task>" to a cognitive facet
# (capability), a cognitive layer, the spatial dimension and the answer format.
# Layers:
#   perception  (L1) -> Do-You-See-Me   : "can the model SEE?"
#   cognition   (L2) -> Mind's-Eye      : "can the model reason / transform?"
#   reasoning   (L3) -> Spatial+robust  : "can it REASON without hallucinating?"
# (key, (capability, layer, dimension, format))
TASK_TAXONOMY = {
    # ---- Layer 1: Perception (Do-You-See-Me, free-form VQA) ----
    "do_you_see_me/visual_spatial": ("spatial_relation", "perception", "2D", "vqa"),
    "do_you_see_me/letter_disambiguation": ("form_discrimination", "perception", "2D", "vqa"),
    "do_you_see_me/visual_form_constancy": ("form_constancy", "perception", "2D", "vqa"),
    "do_you_see_me/visual_figure_ground": ("figure_ground", "perception", "2D", "vqa"),
    "do_you_see_me/visual_closure": ("visual_closure", "perception", "2D", "vqa"),
    "do_you_see_me/geometric_dataset": ("shape_discrimination", "perception", "2D", "vqa"),
    "do_you_see_me/color_and_shape_disambiguation": ("feature_binding", "perception", "2D", "vqa"),
    # ---- Layer 2: Visual Cognition (Mind's-Eye, MCQ) ----
    "minds_eye/mental_rotation": ("mental_rotation", "cognition", "3D", "mcq"),
    "minds_eye/paper_folding": ("spatial_visualization", "cognition", "3D", "mcq"),
    "minds_eye/mental_composition": ("mental_composition", "cognition", "3D", "mcq"),
    "minds_eye/dynamic_isomorph": ("dynamic_reasoning", "cognition", "dynamic", "mcq"),
    "minds_eye/symmetric_isomorph": ("symmetry_analysis", "cognition", "2D", "mcq"),
    "minds_eye/hierarchial_isomorph": ("hierarchical_reasoning", "cognition", "2D", "mcq"),
    "minds_eye/slippage": ("conceptual_slippage", "cognition", "2D", "mcq"),
}

# Paper-aligned aggregation for Mind's Eye. The public task names are
# normalized to capabilities by the scorer before this mapping is applied.
MINDS_EYE_ART_BY_CAPABILITY = {
    "analogical_reasoning": "abstraction",
    "visual_relation_abstraction": "abstraction",
    "hierarchical_reasoning": "abstraction",
    "conceptual_slippage": "relation",
    "dynamic_reasoning": "relation",
    "symmetry_analysis": "relation",
    "mental_composition": "transformation",
    "mental_rotation": "transformation",
    "spatial_visualization": "transformation",
}

# Human-readable layer labels and the benchmark each layer is sourced from.
LAYER_LABELS = {
    "perception": "Perception (Do You See Me)",
    "cognition": "Cognition (Mind's Eye)",
    "reasoning": "Spatial Reasoning & Robustness",
}

# Visual Cognition Index (VCI): weighted mean across the cognitive layers that
# have golden-set data. Within a layer, capabilities are sample-weighted.
# Weights are renormalised over the layers actually present in a submission.
VCI_LAYER_WEIGHTS = {
    "perception": _nonnegative_finite_float_from_env("VCI_W_PERCEPTION", 0.5),
    "cognition": _nonnegative_finite_float_from_env(
        "VCI_W_COGNITION",
        os.getenv("VCI_W_IMAGERY", "0.5"),
    ),
    "reasoning": _nonnegative_finite_float_from_env("VCI_W_REASONING", 0.0),
}
if VCI_LAYER_WEIGHTS["perception"] + VCI_LAYER_WEIGHTS["cognition"] <= 0:
    raise RuntimeError(
        "VCI_W_PERCEPTION and VCI_W_COGNITION cannot both be zero."
    )

# Spatial evaluation conditions. Every official submission must contain all six
# so comparisons use identical model, data, prompt and ablation revisions.
EVAL_CONDITIONS = [
    "main_noncot",
    "main_cot",
    "no_image_noncot",
    "no_image_cot",
    "no_image_plus_noncot",
    "no_image_plus_cot",
]
SPATIAL_DATASET_KEYS = [
    "BLINK",
    "CV-Bench-2D",
    "CV-Bench-3D",
    "MMVP",
    "RealWorldQA",
    "VStarBench",
    "MMSIBench_wo_circular",
    "3DSRBench",
    "VSR_MCQ",
    "SpatialBench",
    "MindCube",
    "OmniSpatial",
    "SAT-Real",
]
SPATIAL_BENCHMARK_SCHEMA_VERSION = "ms-vista-spatial-benchmark/v2"
SPATIAL_RUN_SCHEMA_VERSION = "ms-vista-spatial-run/v2"
SPATIAL_SUBMISSION_SCHEMA_VERSION = "ms-vista-spatial-submission/v2"
SPATIAL_REPORT_SCHEMA_VERSION = "ms-vista-spatial-report/v2"

# Production readiness guard for the Spatial track. Keep this disabled until
# the official 13-dataset bundle is mounted; the submission endpoint remains
# closed for the checked-in demo bundle regardless of this readiness setting.
REQUIRE_OFFICIAL_SPATIAL = _bool_from_env(
    "REQUIRE_OFFICIAL_SPATIAL",
    False,
)
OFFICIAL_SPATIAL_MIN_SAMPLES = _positive_int_from_env(
    "OFFICIAL_SPATIAL_MIN_SAMPLES",
    1000,
)

# ---------------------------------------------------------------------------
# Three-task architecture
# ---------------------------------------------------------------------------
# The leaderboard is organised into three independently-submitted tasks grouped
# into two UI sections:
#
#   Section "Visual Cognition"
#     task 1: do_you_see_me  (perception)  -> Do-You-See-Me benchmark
#     task 2: minds_eye      (cognition)   -> Mind's-Eye benchmark
#     -> the two task accuracies combine into the Visual Cognition Index (VCI)
#
#   Section "Spatial Reasoning Analysis"
#     task 3: spatial        (reasoning)   -> 13 public spatial benchmarks,
#            evaluated via a downloadable harness; carries the robustness
#            diagnostics (CoT-delta, shortcut, hallucination resistance).
#
# Each task ships a public sample set (no answers) + a private ground-truth file
# + submission templates. Submissions are keyed by model name, so a model can
# submit the three tasks at different times and accrue a combined profile.

TASKS_DIR = Path(os.getenv("TASKS_DIR", COMBINED_ROOT / "tasks"))


def _task_paths(task_id: str) -> dict:
    base = TASKS_DIR / task_id
    return {
        "dir": base,
        "questions": base / "questions.json",
        "questions_jsonl": base / "questions.jsonl",
        # Generated answer keys must never share the public tasks tree. The
        # production scorer normally uses the versioned sources above; this
        # path supports local bundle generation and the spatial demo only.
        "ground_truth": GROUND_TRUTHS_DIR / f"{task_id}_generated_v1" / "ground_truth.json",
        "template_jsonl": base / "submission_template.jsonl",
    }


def _spatial_task_paths() -> dict:
    return _task_paths("spatial")


# task_id -> descriptor
TASKS = {
    "do_you_see_me": {
        "task_id": "do_you_see_me",
        "order": 1,
        "label": "Do You See Me",
        "section": "visual_cognition",
        "layer": "perception",
        "group_by": "capability",
        "supports_diagnostics": False,
        "score_method": "dimension_balanced_task_macro",
        "score_description": (
            "Mean task accuracy within 2D and 3D, followed by an equal mean "
            "across the two dimensions."
        ),
        "paper_total_samples": 2612,
        "description": "Low-level visual perception: can the model SEE what is in the image?",
        "paths": _task_paths("do_you_see_me"),
        "ground_truth_sources": [DYSM_2D_GROUND_TRUTH, DYSM_3D_GROUND_TRUTH],
    },
    "minds_eye": {
        "task_id": "minds_eye",
        "order": 2,
        "label": "Mind's Eye",
        "section": "visual_cognition",
        "layer": "cognition",
        "group_by": "capability",
        "supports_diagnostics": False,
        "score_method": "unweighted_task_macro",
        "score_description": "Unweighted mean of the eight task accuracies.",
        "paper_total_samples": 800,
        "description": "Visual cognition: can the model transform and compose shapes in the mind's eye?",
        "paths": _task_paths("minds_eye"),
        "ground_truth_sources": [MINDS_EYE_FRESH_GROUND_TRUTH],
    },
    "spatial": {
        "task_id": "spatial",
        "order": 3,
        "label": "Spatial Reasoning Analysis",
        "section": "spatial",
        "layer": "reasoning",
        "group_by": "dataset",
        "supports_diagnostics": True,
        "score_method": "unweighted_dataset_macro",
        "score_description": "Unweighted mean of main non-CoT accuracy across the 13 datasets.",
        "required_conditions": EVAL_CONDITIONS,
        "primary_condition": "main_noncot",
        "description": "Spatial reasoning robustness across 13 public benchmarks, "
                       "with CoT / no-image / no-image++ diagnostics.",
        "paths": _spatial_task_paths(),
    },
}

SECTIONS = {
    "visual_cognition": {
        "id": "visual_cognition",
        "label": "Visual Cognition",
        "tasks": ["do_you_see_me", "minds_eye"],
        "primary_metric": "vci",
    },
    "spatial": {
        "id": "spatial",
        "label": "Spatial Reasoning Analysis",
        "tasks": ["spatial"],
        "primary_metric": "macro_accuracy",
    },
}

# ---------------------------------------------------------------------------
# Grading configuration
# ---------------------------------------------------------------------------
# Public submissions must contain final answers, not raw chain-of-thought or
# unprocessed model transcripts. Visual answers are checked against each
# question's declared format before deterministic exact matching. A malformed
# answer remains auditable in the submission but receives no credit.
JUDGE_MODEL = None

GRADING = {
    "do_you_see_me": {
        "method": "jsonl_exact",
        "paper": "Do You See Me (arXiv:2506.02022)",
        "judge_model": JUDGE_MODEL,
        "answer_types": ["mcq", "numeric"],
        "random_baseline": None,  # free-response perception tasks
    },
    "minds_eye": {
        "method": "jsonl_exact",
        "paper": "Mind's Eye (arXiv:2604.16054)",
        "judge_model": JUDGE_MODEL,
        "answer_types": ["mcq"],
        # Paper reports a Random-Choice baseline (4- or 6-option MCQ). Computed
        # per submission from the option counts; this is the default fallback.
        "random_baseline": 0.25,
    },
    "spatial": {
        "method": "judged_jsonl_exact",
        "paper": "CoT Degrades Visual Spatial Reasoning (arXiv:2604.16060)",
        "judge_model": os.getenv(
            "SPATIAL_JUDGE_REVISION",
            "Qwen3-30B-A3B-Instruct-2507",
        ).strip(),
        "answer_types": ["mcq"],
        "decoding": {"strategy": "greedy", "temperature": 0.0, "metric": "pass@1"},
        "random_baseline": None,
    },
}

# Per-model leaderboard store for the three-task system (keyed by model name).
LEADERBOARD_STORE_FILE = _path_from_env("LEADERBOARD_STORE_FILE", RESULTS_DIR / "leaderboard_store.json")

# Spatial Task-3 harness assets.
SPATIAL_HARNESS_DIR = COMBINED_ROOT / "evaluation" / "spatial_reasoning"
SPATIAL_MANIFEST_FILE = TASKS["spatial"]["paths"]["dir"] / "manifest.json"

# The 13 public spatial benchmarks combined for Task 3 (paper arXiv:2604.16060,
# Appendix Table 5). We do NOT redistribute these datasets (some, e.g.
# RealWorldQA, are CC BY-ND which forbids redistributing a combined copy).
# The harness downloads each dataset from its official source on the user's
# machine. Verify every `hf_id`/config against VLMEvalKit before running:
# entries flagged "verify": true are best-effort and must be confirmed.
SPATIAL_DATASETS = [
    {"id": "blink", "name": "BLINK", "type": "2D", "approx_n": 1900,
     "tags": ["depth", "relation", "count", "localization"],
     "hf_id": "BLINK-Benchmark/BLINK", "license": "Apache-2.0", "verify": False},
    {"id": "cvbench2d", "name": "CV-Bench (2D)", "type": "2D", "approx_n": 1400,
     "tags": ["relation", "count", "localization", "size"],
     "hf_id": "nyu-visionx/CV-Bench", "license": "Apache-2.0", "verify": False},
    {"id": "mmvp", "name": "MMVP", "type": "2D", "approx_n": 300,
     "tags": ["relation", "localization"],
     "hf_id": "MMVP/MMVP", "license": "MIT", "verify": True},
    {"id": "realworldqa", "name": "RealWorldQA", "type": "2D", "approx_n": 765,
     "tags": ["relation", "localization"],
     "hf_id": "xai-org/RealworldQA", "license": "CC BY-ND 4.0", "verify": False},
    {"id": "spatialbench", "name": "SpatialBench", "type": "2D", "approx_n": 174,
     "tags": ["relation", "localization", "size"],
     "hf_id": "RussRobin/SpatialBench", "license": "see source", "verify": True},
    {"id": "vsr", "name": "VSR", "type": "2D", "approx_n": 1200,
     "tags": ["relation", "orientation", "egocentric"],
     "hf_id": "cambridgeltl/vsr_zeroshot", "license": "Apache-2.0", "verify": True},
    {"id": "vstar", "name": "V*Bench", "type": "2D", "approx_n": 191,
     "tags": ["attribute", "relation"],
     "hf_id": "craigwu/vstar_bench", "license": "MIT", "verify": False},
    {"id": "3dsrbench", "name": "3DSRBench", "type": "3D", "approx_n": 5200,
     "tags": ["3d", "localization", "orientation"],
     "hf_id": "ccvl/3DSRBench", "license": "see source", "verify": False},
    {"id": "cvbench3d", "name": "CV-Bench (3D)", "type": "3D", "approx_n": 1200,
     "tags": ["depth", "3d", "relation"],
     "hf_id": "nyu-visionx/CV-Bench", "license": "Apache-2.0", "verify": False},
    {"id": "mindcube", "name": "MindCube", "type": "3D", "approx_n": 1000,
     "tags": ["multiview", "relation", "egocentric"],
     "hf_id": "MLL-Lab/MindCube", "license": "see source", "verify": True},
    {"id": "mmsibench", "name": "MMSI-Bench", "type": "3D", "approx_n": 1000,
     "tags": ["multiimage", "spatial"],
     "hf_id": "RunsenXu/MMSI-Bench", "license": "see source", "verify": False},
    {"id": "omnispatial", "name": "OmniSpatial", "type": "3D", "approx_n": 1500,
     "tags": ["spatial", "reasoning"],
     "hf_id": "qizekun/OmniSpatial", "license": "see source", "verify": True},
    {"id": "satreal", "name": "SAT (Real)", "type": "dynamic", "approx_n": 1000,
     "tags": ["dynamic", "egocentric", "action"],
     "hf_id": "array/SAT", "license": "see source", "verify": True},
]

# The extra answer option the harness appends for the no-image++ probe.
NO_IMAGE_PLUS_OPTION = "Cannot determine from the image"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for _t in TASKS.values():
    _t["paths"]["dir"].mkdir(parents=True, exist_ok=True)
