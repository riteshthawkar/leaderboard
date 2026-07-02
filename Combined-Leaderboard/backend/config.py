"""
Configuration and paths for the leaderboard system.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from the Combined-Leaderboard/.env file (if present)
# before any os.getenv() lookups below. This is where the judge API key lives.
# The .env lives in the Combined-Leaderboard package dir (this file's grandparent),
# which is NOT PROJECT_ROOT (the workspace root, one level further up).
load_dotenv(Path(__file__).parent.parent / ".env")
BACKEND_DIR = PROJECT_ROOT / "backend"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"

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

# Supported file extensions
SUPPORTED_PREDICTION_FORMATS = [".csv", ".json"]

# Ollama configuration for option extraction
OLLAMA_MODEL = os.getenv("MINDS_EYE_OPTION_EXTRACTOR_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("MINDS_EYE_OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Unified "Visual Cognition" battery
# ---------------------------------------------------------------------------
# Combined-Leaderboard package root (this file is backend/config.py).
COMBINED_ROOT = Path(__file__).parent.parent

# Golden set: a fixed, combined sample drawn from every locally available task.
# Users download the questions, run their model, and submit a single file with
# predictions for every golden-set sample.
GOLDEN_SET_DIR = Path(os.getenv("GOLDEN_SET_DIR", COMBINED_ROOT / "golden_set"))
GOLDEN_SET_QUESTIONS_FILE = GOLDEN_SET_DIR / "golden_set_questions.json"
GOLDEN_SET_GROUND_TRUTH_FILE = GOLDEN_SET_DIR / "golden_set_ground_truth.json"
GOLDEN_SET_TEMPLATE_CSV = GOLDEN_SET_DIR / "submission_template.csv"
GOLDEN_SET_TEMPLATE_JSON = GOLDEN_SET_DIR / "submission_template.json"

# How many samples to draw per task when building the golden set.
GOLDEN_SET_SIZE_PER_TASK = int(os.getenv("GOLDEN_SET_SIZE_PER_TASK", "25"))
GOLDEN_SET_SEED = int(os.getenv("GOLDEN_SET_SEED", "42"))

# Cognitive taxonomy: maps each "<benchmark>/<task>" to a cognitive facet
# (capability), a cognitive layer, the spatial dimension and the answer format.
# Layers:
#   perception  (L1) -> Do-You-See-Me   : "can the model SEE?"
#   imagery     (L2) -> Mind's-Eye      : "can the model IMAGINE / transform?"
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
    # ---- Layer 2: Mental Imagery (Mind's-Eye, MCQ) ----
    "minds_eye/mental_rotation": ("mental_rotation", "imagery", "3D", "mcq"),
    "minds_eye/paper_folding": ("spatial_visualization", "imagery", "3D", "mcq"),
    "minds_eye/mental_composition": ("mental_composition", "imagery", "3D", "mcq"),
    "minds_eye/dynamic_isomorph": ("dynamic_reasoning", "imagery", "dynamic", "mcq"),
    "minds_eye/symmetric_isomorph": ("symmetry_analysis", "imagery", "2D", "mcq"),
    "minds_eye/hierarchial_isomorph": ("hierarchical_reasoning", "imagery", "2D", "mcq"),
    "minds_eye/slippage": ("conceptual_slippage", "imagery", "2D", "mcq"),
}

# Human-readable layer labels and the benchmark each layer is sourced from.
LAYER_LABELS = {
    "perception": "Perception (Do-You-See-Me)",
    "imagery": "Mental Imagery (Mind's-Eye)",
    "reasoning": "Spatial Reasoning & Robustness",
}

# Visual Cognition Index (VCI): weighted mean across the cognitive layers that
# have golden-set data. Within a layer, capabilities are sample-weighted.
# Weights are renormalised over the layers actually present in a submission.
VCI_LAYER_WEIGHTS = {
    "perception": float(os.getenv("VCI_W_PERCEPTION", "0.5")),
    "imagery": float(os.getenv("VCI_W_IMAGERY", "0.5")),
    "reasoning": float(os.getenv("VCI_W_REASONING", "0.0")),
}

# Evaluation conditions (the paper's experimental design). Only "standard" is
# required; the rest unlock the robustness diagnostics when submitted.
EVAL_CONDITIONS = ["standard", "cot", "no_image", "no_image_plus"]

# ---------------------------------------------------------------------------
# Three-task architecture
# ---------------------------------------------------------------------------
# The leaderboard is organised into three independently-submitted tasks grouped
# into two UI sections:
#
#   Section "Visual Cognition"
#     task 1: do_you_see_me  (perception)  -> Do-You-See-Me benchmark
#     task 2: minds_eye      (imagery)     -> Mind's-Eye benchmark
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
        "ground_truth": base / "ground_truth.json",
        "template_json": base / "submission_template.json",
        "template_csv": base / "submission_template.csv",
    }


# task_id -> descriptor
TASKS = {
    "do_you_see_me": {
        "task_id": "do_you_see_me",
        "order": 1,
        "label": "Do-You-See-Me",
        "section": "visual_cognition",
        "layer": "perception",
        "group_by": "capability",
        "supports_diagnostics": False,
        "description": "Low-level visual perception: can the model SEE what is in the image?",
        "paths": _task_paths("do_you_see_me"),
    },
    "minds_eye": {
        "task_id": "minds_eye",
        "order": 2,
        "label": "Mind's-Eye",
        "section": "visual_cognition",
        "layer": "imagery",
        "group_by": "capability",
        "supports_diagnostics": False,
        "description": "Mental imagery: can the model IMAGINE and transform shapes in the mind's eye?",
        "paths": _task_paths("minds_eye"),
    },
    "spatial": {
        "task_id": "spatial",
        "order": 3,
        "label": "Spatial Reasoning Analysis",
        "section": "spatial",
        "layer": "reasoning",
        "group_by": "dataset",
        "supports_diagnostics": True,
        "description": "Spatial reasoning robustness across 13 public benchmarks, "
                       "with CoT / no-image / no-image++ diagnostics.",
        "paths": _task_paths("spatial"),
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
        "primary_metric": "accuracy",
    },
}

# ---------------------------------------------------------------------------
# Grading configuration (single shared LLM judge: GPT-4o)
# ---------------------------------------------------------------------------
# All three tasks are graded by the SAME judge model - GPT-4o - served through
# the OpenAI chat-completions API, so scores are uniform and directly
# comparable across tasks. Only the grading *method* differs by task, because
# the task formats differ:
#   * do_you_see_me - GPT-4o answer *extractor*, then exact match (MCQ label or
#     numeric value).                       (Do You See Me, arXiv:2506.02022)
#   * minds_eye     - GPT-4o answer *extractor*, then exact MCQ-label match.
#                                            (Mind's Eye, arXiv:2604.16054)
#   * spatial       - GPT-4o LLM-*as-judge* decides correct/incorrect.
#                                            (CoT Degrades..., arXiv:2604.16060)
#
# The only setting read from the environment (.env) is the API key:
#   OPENAI_API_KEY  = <your OpenAI key>
#   GRADING_TIMEOUT = per-call timeout in seconds (default: 60)
# When no key is configured (or a call fails) the grader falls back to
# deterministic normalised string / single-letter / numeric matching, so the
# leaderboard still works fully offline. The method actually used is recorded
# per task.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")

GRADING = {
    "do_you_see_me": {
        "method": "extract",
        "paper": "Do You See Me (arXiv:2506.02022)",
        "judge_model": JUDGE_MODEL,
        "answer_types": ["mcq", "numeric"],
        "random_baseline": None,  # free-response perception tasks
    },
    "minds_eye": {
        "method": "extract",
        "paper": "Mind's Eye (arXiv:2604.16054)",
        "judge_model": JUDGE_MODEL,
        "answer_types": ["mcq"],
        # Paper reports a Random-Choice baseline (4- or 6-option MCQ). Computed
        # per submission from the option counts; this is the default fallback.
        "random_baseline": 0.25,
    },
    "spatial": {
        "method": "judge",
        "paper": "CoT Degrades Visual Spatial Reasoning (arXiv:2604.16060)",
        "judge_model": JUDGE_MODEL,
        "answer_types": ["mcq"],
        "decoding": {"strategy": "greedy", "temperature": 0.0, "metric": "pass@1"},
        "random_baseline": None,
    },
}

# Per-model leaderboard store for the three-task system (keyed by model name).
LEADERBOARD_STORE_FILE = RESULTS_DIR / "leaderboard_store.json"

# Spatial Task-3 harness assets.
SPATIAL_HARNESS_DIR = COMBINED_ROOT / "spatial_harness"
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
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GOLDEN_SET_DIR.mkdir(parents=True, exist_ok=True)
for _t in TASKS.values():
    _t["paths"]["dir"].mkdir(parents=True, exist_ok=True)
