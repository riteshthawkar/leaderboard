"""
Generic scorer for a single task submission (do_you_see_me / minds_eye / spatial).

Scores a JSONL submission against the task's private ground truth and produces
a :class:`TaskScore` with overall + per-group accuracy and, for tasks that
support it (Task 3 spatial), the robustness diagnostics.

Public submissions must already contain final answers. The server matches
``question_id``/``sample_id`` + final answer deterministically and stores only
aggregate results, not the submitted file or raw predictions.
"""

import json
import math
import re
import statistics
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from visual_answer_contract import (
    canonical_answers_equal,
    is_canonical_visual_answer,
)

from config import (
    TASKS,
    MINDS_EYE_ART_BY_CAPABILITY,
    GRADING,
    GROUND_TRUTHS_DIR,
    GROUND_TRUTHS_SOURCE,
    GROUND_TRUTHS_HF_REPO,
    GROUND_TRUTHS_HF_REPO_TYPE,
    GROUND_TRUTHS_HF_REVISION,
    GROUND_TRUTHS_HF_CACHE_DIR,
    GROUND_TRUTHS_HF_FORCE_REFRESH,
    HF_TOKEN,
)
from constants import MAX_SUBMISSION_LINE_CHARS
from models.tasks import GroupResult, Diagnostics, TaskScore

try:
    from sqlite_runtime import harden_private_directory, harden_private_file
except ImportError:  # pragma: no cover - package import fallback
    from ..sqlite_runtime import harden_private_directory, harden_private_file

_VISUAL_CONDITIONS = ("standard",)
_SPATIAL_CONDITIONS = (
    "main_noncot",
    "main_cot",
    "no_image_noncot",
    "no_image_cot",
    "no_image_plus_noncot",
    "no_image_plus_cot",
)
_SPATIAL_LEGACY_CONDITION_ALIASES = {
    "standard": "main_noncot",
    "cot": "main_cot",
    "no_image": "no_image_noncot",
    "no_image_plus": "no_image_plus_noncot",
}
_ANSWER_FIELDS = ("answer", "prediction", "response", "output")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_OPTION_RE = re.compile(r"\b([A-Ja-j])\b")
_ID_FIELDS = ("sample_id", "question_id", "id")


class SubmissionValidationError(ValueError):
    """A user-correctable JSONL contract or coverage error."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = {key: value for key, value in details.items() if value is not None}

    def to_dict(self) -> dict:
        return {"code": self.code, **self.details}

_DYSM_ID_RE = re.compile(
    r"^t1_(?P<dimension>2d|3d)_(?P<task>.+)_(?P<difficulty>easy|medium|hard)_(?P<index>\d+)$"
)
_MINDS_EYE_ID_RE = re.compile(r"^t2_(?P<task>.+)_(?P<index>\d+)$")

_DYSM_CAPABILITY_BY_TASK = {
    "shape_discrimination": "shape_discrimination",
    "geometric_dataset": "shape_discrimination",
    "letter_disambiguation": "form_discrimination",
    "visual_form_constancy": "form_constancy",
    "form_constancy": "form_constancy",
    "visual_figure_ground": "figure_ground",
    "visual_closure": "visual_closure",
    "visual_spatial": "spatial_relation",
    "joint_shape_color": "feature_binding",
    "color_and_shape_disambiguation": "feature_binding",
    "shape_color_disambiguation": "feature_binding",
}
_MINDS_EYE_CAPABILITY_BY_TASK = {
    "analogies": "analogical_reasoning",
    "dynamic_isomorphism": "dynamic_reasoning",
    "dynamic_isomorph": "dynamic_reasoning",
    "hierarchical_isomorphism": "hierarchical_reasoning",
    "hierarchial_isomorph": "hierarchical_reasoning",
    "mental_composition": "mental_composition",
    "mental_rotation": "mental_rotation",
    "mrt": "mental_rotation",
    "paper_folding": "spatial_visualization",
    "slippage": "conceptual_slippage",
    "symmetric_structures": "symmetry_analysis",
    "symmetric_isomorph": "symmetry_analysis",
}
_MINDS_EYE_DIMENSION_BY_TASK = {
    "dynamic_isomorphism": "dynamic",
    "dynamic_isomorph": "dynamic",
    "mental_composition": "3D",
    "mental_rotation": "3D",
    "mrt": "3D",
    "paper_folding": "3D",
}
_MINDS_EYE_OPTION_COUNT_BY_TASK = {
    "analogies": 6,
    "slippage": 6,
}
_DYSM_ANSWER_TYPE_BY_TASK = {
    "form_constancy": "text",
    "joint_shape_color": "integer",
    "letter_disambiguation": "text",
    "shape_color_disambiguation": "integer",
    "shape_discrimination": "integer",
    "visual_closure": "mcq_index_1_4",
    "visual_figure_ground": "mcq_index_1_4",
    "visual_form_constancy": "mcq_index_1_4",
    "visual_spatial": "integer",
}
_VALID_GT_SOURCES = {"auto", "local", "hf"}
_HF_DOWNLOAD_TIMEOUT_SECONDS = 60

# Phrases that count as a correct "cannot determine" response in no-image++.
_CANNOT_DETERMINE = {
    "cannot determine", "cannot determine from the image", "cant determine",
    "can not determine", "indeterminate", "unknown", "not enough information",
    "insufficient information", "none of the above", "cannot be determined",
    "cannot tell", "no answer", "unanswerable",
}


def _normalize(value) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^\s*(answer|option|choice|final answer)\s*(is|=)?\s*[:\-]?\s*", "", s)
    s = s.strip().strip("().,:;\"'")
    s = re.sub(r"\s+", " ", s)
    return s


def _as_decimal(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        return None


def _extract_numeric(value) -> Optional[Decimal]:
    normalized = _normalize(value)
    if not normalized:
        return None
    if _NUMBER_RE.fullmatch(normalized):
        return _as_decimal(normalized)
    matches = _NUMBER_RE.findall(normalized)
    if len(matches) == 1:
        return _as_decimal(matches[0])
    return None


def _extract_option(value) -> Optional[str]:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return None
    normalized = _normalize(raw)
    if re.fullmatch(r"[a-j]", normalized):
        return normalized.upper()

    patterns = [
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|=)?\s*[:\-]?\s*\(?([A-Ja-j])\)?",
        r"\(([A-Ja-j])\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    tokens = _OPTION_RE.findall(raw)
    return tokens[-1].upper() if tokens else None


def _is_cannot_determine(pred) -> bool:
    return _normalize(pred) in _CANNOT_DETERMINE


def _match(pred, gold) -> bool:
    pred_norm, gold_norm = _normalize(pred), _normalize(gold)
    if not pred_norm:
        return False
    if pred_norm == gold_norm:
        return True

    gold_number = _extract_numeric(gold)
    if gold_number is not None:
        pred_number = _extract_numeric(pred)
        return pred_number is not None and pred_number == gold_number

    if re.fullmatch(r"[a-j]", gold_norm):
        return _extract_option(pred) == gold_norm.upper()

    return False


def _strict_visual_match(pred, gold, answer_type: str, task: str) -> tuple[bool, bool]:
    """Return ``(has_valid_format, is_correct)`` for a final visual answer."""
    valid_format = is_canonical_visual_answer(
        pred,
        answer_type=answer_type,
        task=task,
    )
    return valid_format, valid_format and canonical_answers_equal(
        pred,
        gold,
        answer_type=answer_type,
        task=task,
    )


def _row_id(row: dict) -> str:
    return str(next((row.get(field) for field in _ID_FIELDS if row.get(field)), "")).strip()


class TaskScorer:
    """Scores submissions for one task against its private ground truth."""

    def __init__(self, task_id: str):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        harden_private_directory(GROUND_TRUTHS_DIR)
        self.task_id = task_id
        self.task = TASKS[task_id]
        gt_sources = self.task.get("ground_truth_sources") or [self.task["paths"]["ground_truth"]]
        self.ground_truth_files = [Path(path) for path in gt_sources]
        self.ground_truth_file = self.ground_truth_files[0]
        self.questions_file = Path(self.task["paths"]["questions"])
        self.group_by = self.task.get("group_by", "group")
        self.supports_diagnostics = bool(self.task.get("supports_diagnostics"))
        self.allowed_conditions = tuple(
            self.task.get("required_conditions") or _VISUAL_CONDITIONS
        )
        self.primary_condition = str(
            self.task.get("primary_condition") or "standard"
        )
        self._gt: Optional[Dict[str, dict]] = None
        self._questions: Optional[Dict[str, dict]] = None
        # Submission-time grading is deterministic; GRADING is retained for
        # user-facing provenance and random-baseline metadata.
        self.grading_cfg = GRADING.get(task_id, {"method": "jsonl_exact", "answer_types": ["mcq"]})
        self._method_counts: ContextVar[Optional[Dict[str, int]]] = ContextVar(
            f"task_scorer_method_counts_{task_id}",
            default=None,
        )

    @property
    def ground_truth(self) -> Dict[str, dict]:
        if self._gt is None:
            gt: Dict[str, dict] = {}
            for gt_file in self.ground_truth_files:
                resolved_gt_file = self._resolve_ground_truth_file(gt_file)
                if not resolved_gt_file.exists():
                    raise FileNotFoundError(
                        f"Ground truth for task '{self.task_id}' not found: "
                        f"{gt_file}. Configure GROUND_TRUTHS_DIR or GROUND_TRUTHS_HF_REPO."
                    )
                harden_private_directory(resolved_gt_file.parent)
                harden_private_file(resolved_gt_file)
                loaded = self._load_ground_truth_file(resolved_gt_file)
                duplicates = set(gt).intersection(loaded)
                if duplicates:
                    raise ValueError(
                        f"Ground truth for task '{self.task_id}' has duplicate ids: "
                        f"{self._format_sample_ids(duplicates)}"
                    )
                gt.update(loaded)
            self._gt = gt
        return self._gt

    def resolved_ground_truth_files(self) -> List[Path]:
        """Return local answer-key files, fetching configured private HF files if needed."""
        return [self._resolve_ground_truth_file(path) for path in self.ground_truth_files]

    def _resolve_ground_truth_file(self, path: Path) -> Path:
        source = GROUND_TRUTHS_SOURCE if GROUND_TRUTHS_SOURCE in _VALID_GT_SOURCES else "auto"
        if source != "hf" and path.exists():
            return path
        if source in {"auto", "hf"}:
            downloaded = self._ensure_hf_ground_truth(path)
            if downloaded is not None:
                return downloaded
        return path

    def _ground_truth_relative_path(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(GROUND_TRUTHS_DIR.resolve())
        except ValueError:
            return Path(path.parent.name) / path.name

    def _hf_resolve_url(self, relative_path: Path) -> str:
        repo = GROUND_TRUTHS_HF_REPO.strip().strip("/")
        revision = quote(GROUND_TRUTHS_HF_REVISION, safe="")
        rel = quote(relative_path.as_posix(), safe="/")
        if repo.startswith(("http://", "https://")):
            return f"{repo}/resolve/{revision}/{rel}"
        if repo.startswith(("datasets/", "models/", "spaces/")):
            return f"https://huggingface.co/{repo}/resolve/{revision}/{rel}"
        if GROUND_TRUTHS_HF_REPO_TYPE in {"model", "models"}:
            return f"https://huggingface.co/{repo}/resolve/{revision}/{rel}"
        return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{rel}"

    def _ensure_hf_ground_truth(self, path: Path) -> Optional[Path]:
        if not GROUND_TRUTHS_HF_REPO:
            return None
        if not HF_TOKEN:
            raise FileNotFoundError(
                f"Ground truth for task '{self.task_id}' not found locally and "
                "HF_TOKEN/HUGGINGFACE_TOKEN is not configured."
            )

        relative_path = self._ground_truth_relative_path(path)
        cache_path = GROUND_TRUTHS_HF_CACHE_DIR / relative_path
        if cache_path.exists() and not GROUND_TRUTHS_HF_FORCE_REFRESH:
            harden_private_directory(cache_path.parent)
            harden_private_file(cache_path)
            return cache_path

        harden_private_directory(cache_path.parent)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        url = self._hf_resolve_url(relative_path)
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        try:
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=_HF_DOWNLOAD_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            harden_private_file(tmp_path)
            tmp_path.replace(cache_path)
            harden_private_file(cache_path)
        except requests.HTTPError as exc:
            status = getattr(response, "status_code", None)
            if status in {401, 403}:
                raise PermissionError(
                    "Unable to download private ground truth from Hugging Face: "
                    "check HF token permissions."
                ) from exc
            raise FileNotFoundError(
                f"Unable to download ground truth from Hugging Face path "
                f"'{relative_path.as_posix()}' (HTTP {status})."
            ) from exc
        except requests.RequestException as exc:
            raise FileNotFoundError(
                f"Unable to download ground truth from Hugging Face path "
                f"'{relative_path.as_posix()}': {exc}"
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        return cache_path

    def _load_ground_truth_file(self, path: Path) -> Dict[str, dict]:
        if path.suffix.lower() == ".jsonl":
            return self._load_jsonl_ground_truth(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Ground truth file must contain an object: {path}")
        parsed = {}
        for sample_id, value in data.items():
            meta = dict(value) if isinstance(value, dict) else {"answer": value}
            if "answer" not in meta:
                raise ValueError(f"Ground truth row for '{sample_id}' is missing answer")
            meta.setdefault("sample_id", str(sample_id))
            meta.update(self._derive_ground_truth_metadata(str(sample_id), path))
            parsed[str(sample_id)] = meta
        return parsed

    def _load_jsonl_ground_truth(self, path: Path) -> Dict[str, dict]:
        parsed = {}
        with open(path, "r", encoding="utf-8-sig") as f:
            for lineno, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid ground truth JSONL at {path}:{lineno}: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Invalid ground truth JSONL at {path}:{lineno}: row must be an object")
                sample_id = _row_id(row)
                if not sample_id:
                    raise ValueError(
                        f"Invalid ground truth JSONL at {path}:{lineno}: "
                        "missing sample_id/question_id"
                    )
                if "answer" not in row:
                    raise ValueError(f"Invalid ground truth JSONL at {path}:{lineno}: missing answer")
                if sample_id in parsed:
                    raise ValueError(f"Duplicate ground truth id '{sample_id}' in {path}")
                meta = dict(row)
                meta["sample_id"] = sample_id
                if "question_id" in row:
                    meta["question_id"] = sample_id
                meta.update(self._derive_ground_truth_metadata(sample_id, path))
                parsed[sample_id] = meta
        if not parsed:
            raise ValueError(f"Ground truth JSONL is empty: {path}")
        return parsed

    def _derive_ground_truth_metadata(self, sample_id: str, path: Path) -> Dict[str, str]:
        source_subset = path.parent.name
        dysm = _DYSM_ID_RE.match(sample_id)
        if dysm:
            task = dysm.group("task")
            dimension = dysm.group("dimension").upper()
            capability = _DYSM_CAPABILITY_BY_TASK.get(task, task)
            return {
                "benchmark": "do_you_see_me",
                "source_subset": source_subset,
                "task": task,
                "group": capability,
                "capability": capability,
                "dimension": dimension,
                "difficulty": dysm.group("difficulty"),
                "index": dysm.group("index"),
                "layer": "perception",
            }
        minds_eye = _MINDS_EYE_ID_RE.match(sample_id)
        if minds_eye:
            task = minds_eye.group("task")
            capability = _MINDS_EYE_CAPABILITY_BY_TASK.get(task, task)
            return {
                "benchmark": "minds_eye",
                "source_subset": source_subset,
                "task": task,
                "group": capability,
                "capability": capability,
                "dimension": _MINDS_EYE_DIMENSION_BY_TASK.get(task, "2D"),
                "index": minds_eye.group("index"),
                "layer": "cognition",
            }
        return {"source_subset": source_subset}

    @property
    def questions(self) -> Dict[str, dict]:
        """sample_id/question_id -> {question, options, format}; empty if no questions file."""
        if self._questions is None:
            self._questions = {}
            questions_jsonl = self.task["paths"].get("questions_jsonl")
            if questions_jsonl and questions_jsonl.exists():
                try:
                    with open(questions_jsonl, "r", encoding="utf-8-sig") as f:
                        samples = [json.loads(line) for line in f if line.strip()]
                    self._index_question_samples(samples)
                except Exception:
                    self._questions = {}
            elif self.questions_file.exists():
                try:
                    with open(self.questions_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    samples = data.get("samples", []) if isinstance(data, dict) else data
                    self._index_question_samples(samples)
                except Exception:
                    self._questions = {}
        return self._questions

    def _index_question_samples(self, samples) -> None:
        for s in samples or []:
            if not isinstance(s, dict):
                continue
            sid = _row_id(s)
            if sid:
                self._questions[sid] = {
                    "question": s.get("question", ""),
                    "options": s.get("options"),
                    "format": s.get("format") or s.get("answer_type") or "mcq",
                }

    # ------------------------------------------------------------- grading
    def _grade(self, pred, gold, sid) -> bool:
        """Grade one submitted final answer against the private ground truth."""
        method_counts = self._method_counts.get()
        if self.task_id in {"do_you_see_me", "minds_eye"}:
            info = self.ground_truth.get(sid, {})
            task = str(info.get("task") or "").strip()
            question = self.questions.get(sid, {})
            answer_type = str(question.get("format") or "").strip()
            if not answer_type:
                answer_type = (
                    "mcq_letter"
                    if self.task_id == "minds_eye"
                    else _DYSM_ANSWER_TYPE_BY_TASK.get(task, "text")
                )
            valid_format, correct = _strict_visual_match(
                pred,
                gold,
                answer_type,
                task,
            )
            if method_counts is not None:
                method = "jsonl_exact" if valid_format else "invalid_format"
                method_counts[method] = method_counts.get(method, 0) + 1
            return correct

        if method_counts is not None:
            method_counts["jsonl_exact"] = method_counts.get("jsonl_exact", 0) + 1
        return _match(pred, gold)

    def _random_baseline(self) -> Optional[float]:
        cfg = self.grading_cfg.get("random_baseline")
        if cfg is None:
            return None
        vals = []
        for q in self.questions.values():
            opts = q.get("options")
            if isinstance(opts, (list, tuple, dict)) and opts:
                vals.append(1.0 / len(opts))
        if vals:
            return sum(vals) / len(vals)
        if self.task_id == "minds_eye":
            # The released question rows keep options inside the image, so the
            # public JSONL does not contain an options array. Table 2 defines
            # VRA and VCS as six-option tasks and the remaining tasks as four.
            task_names = {
                str(info.get("task") or "").strip()
                for info in self.ground_truth.values()
                if info.get("task")
            }
            if task_names:
                task_baselines = [
                    1.0 / _MINDS_EYE_OPTION_COUNT_BY_TASK.get(task, 4)
                    for task in task_names
                ]
                return sum(task_baselines) / len(task_baselines)
        return float(cfg)

    def random_baseline(self) -> Optional[float]:
        """Return the benchmark-level chance baseline for public metadata."""
        return self._random_baseline()

    def _is_no_image_plus_correct(self, sid: str, pred, condition: str) -> bool:
        if _is_cannot_determine(pred):
            return True
        return self._grade(pred, self._gold_for_condition(sid, condition), sid)

    def _normalize_condition(self, value) -> str:
        condition = str(value or self.primary_condition).strip().lower()
        if self.task_id == "spatial":
            condition = _SPATIAL_LEGACY_CONDITION_ALIASES.get(condition, condition)
        return condition

    def _expected_ids(self, condition: str) -> set[str]:
        expected = set()
        for sample_id, meta in self.ground_truth.items():
            conditions = meta.get("conditions")
            if not conditions or condition in conditions:
                expected.add(sample_id)
        return expected

    def _gold_for_condition(self, sample_id: str, condition: str):
        meta = self.ground_truth[sample_id]
        condition_answers = meta.get("condition_answers")
        if isinstance(condition_answers, dict) and condition in condition_answers:
            return condition_answers[condition]
        return meta["answer"]

    # ------------------------------------------------------------------ parse
    def parse_submission(self, file_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict]:
        """Return (predictions_by_condition, model_meta) from a JSONL file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Submission file not found: {file_path}")
        if file_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Unsupported file format: {file_path.suffix.lower()}. Use .jsonl")
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return self.parse_submission_text(f.read())

    def parse_submission_text(self, text: str) -> Tuple[Dict[str, Dict[str, str]], Dict]:
        predictions, parsed_meta, _records = self.parse_submission_text_with_records(text)
        return predictions, parsed_meta

    def parse_submission_text_with_records(
        self, text: str
    ) -> Tuple[Dict[str, Dict[str, str]], Dict, List[Dict[str, str]]]:
        """Parse newline-delimited JSON rows.

        Expected row shape:
            {"question_id": "...", "answer": "...", "condition": "standard"}

        ``sample_id`` and ``id`` are accepted as identifier aliases.
        ``condition`` defaults to the task's primary condition. ``prediction``, ``response``,
        and ``output`` are accepted as answer aliases for harness compatibility.
        """
        predictions: Dict[str, Dict[str, str]] = {}
        records: List[Dict[str, str]] = []
        seen = set()
        parsed_rows = 0
        empty_answers = []
        maximum_rows = sum(
            len(self._expected_ids(condition))
            for condition in self.allowed_conditions
        )
        # Keep a small diagnostic allowance so ordinary wrong-version files
        # still receive precise unknown/missing-ID errors at coverage time.
        parse_row_limit = maximum_rows + 100

        for lineno, raw_line in enumerate(str(text or "").splitlines(), start=1):
            if len(raw_line) > MAX_SUBMISSION_LINE_CHARS:
                raise SubmissionValidationError(
                    "submission_line_too_long",
                    f"Line {lineno} is longer than the {MAX_SUBMISSION_LINE_CHARS:,}-character JSONL limit. Keep only the final answer and remove reasoning traces or embedded data.",
                    line_number=lineno,
                    max_characters=MAX_SUBMISSION_LINE_CHARS,
                )
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SubmissionValidationError(
                    "invalid_jsonl_syntax",
                    f"Line {lineno} is not valid JSON ({exc.msg}). Use one complete JSON object per line; do not wrap the file in a JSON array.",
                    line_number=lineno,
                    parser_message=exc.msg,
                ) from exc
            if not isinstance(row, dict):
                raise SubmissionValidationError(
                    "jsonl_row_not_object",
                    f"Line {lineno} must be a JSON object, but it contains {type(row).__name__}. Each line should look like {{\"question_id\": \"...\", \"answer\": \"...\"}}.",
                    line_number=lineno,
                    received_type=type(row).__name__,
                )

            id_values = [
                (field, str(row.get(field)).strip())
                for field in _ID_FIELDS
                if row.get(field) is not None and str(row.get(field)).strip()
            ]
            if not id_values:
                raise SubmissionValidationError(
                    "missing_question_id",
                    f"Line {lineno} is missing a question identifier. Copy the 'question_id' value from the provided template into every row.",
                    line_number=lineno,
                    accepted_fields=list(_ID_FIELDS),
                )
            distinct_ids = {value for _field, value in id_values}
            if len(distinct_ids) > 1:
                received = ", ".join(f"{field}={value!r}" for field, value in id_values)
                raise SubmissionValidationError(
                    "conflicting_question_ids",
                    f"Line {lineno} contains conflicting question identifiers ({received}). Keep one identifier field, preferably 'question_id'.",
                    line_number=lineno,
                    id_fields={field: value for field, value in id_values},
                )
            sample_id = id_values[0][1]

            answer_fields = [field for field in _ANSWER_FIELDS if field in row]
            if not answer_fields:
                raise SubmissionValidationError(
                    "missing_answer_field",
                    f"Line {lineno} for question_id '{sample_id}' has no answer field. Add 'answer' with the model's final response.",
                    line_number=lineno,
                    question_id=sample_id,
                    accepted_fields=list(_ANSWER_FIELDS),
                )
            if len(answer_fields) > 1:
                raise SubmissionValidationError(
                    "multiple_answer_fields",
                    f"Line {lineno} for question_id '{sample_id}' contains multiple answer fields ({', '.join(answer_fields)}). Keep only one, preferably 'answer'.",
                    line_number=lineno,
                    question_id=sample_id,
                    answer_fields=answer_fields,
                )
            answer_field = answer_fields[0]

            raw_condition = str(row.get("condition") or self.primary_condition).strip().lower()
            condition = self._normalize_condition(raw_condition)
            if condition not in self.allowed_conditions:
                if self.task_id != "spatial" and (
                    raw_condition in _SPATIAL_LEGACY_CONDITION_ALIASES
                    or raw_condition in _SPATIAL_CONDITIONS
                ):
                    raise SubmissionValidationError(
                        "condition_not_supported_for_task",
                        f"Line {lineno} uses condition '{raw_condition}', but {self.task['label']} accepts only the 'standard' condition. Remove the condition or set it to 'standard'.",
                        line_number=lineno,
                        question_id=sample_id,
                        condition=raw_condition,
                        task_id=self.task_id,
                    )
                raise SubmissionValidationError(
                    "invalid_submission_condition",
                    f"Line {lineno} uses unsupported condition '{condition}'. Allowed conditions are: {', '.join(self.allowed_conditions)}.",
                    line_number=lineno,
                    question_id=sample_id,
                    condition=condition,
                    allowed_conditions=list(self.allowed_conditions),
                )

            key = (condition, sample_id)
            if key in seen:
                raise SubmissionValidationError(
                    "duplicate_sample_output",
                    f"Line {lineno} repeats question_id '{sample_id}' for condition '{condition}'. Every question_id must appear exactly once per condition.",
                    line_number=lineno,
                    question_id=sample_id,
                    condition=condition,
                )
            seen.add(key)
            answer_value = row.get(answer_field)
            if isinstance(answer_value, (dict, list)):
                raise SubmissionValidationError(
                    "invalid_answer_type",
                    f"Line {lineno} for question_id '{sample_id}' has an object or array in '{answer_field}'. The final answer must be a string, number, or boolean.",
                    line_number=lineno,
                    question_id=sample_id,
                    answer_field=answer_field,
                    received_type=type(answer_value).__name__,
                )
            if isinstance(answer_value, float) and not math.isfinite(answer_value):
                raise SubmissionValidationError(
                    "invalid_answer_value",
                    f"Line {lineno} for question_id '{sample_id}' contains a non-finite numeric answer. Use a finite number or a text answer.",
                    line_number=lineno,
                    question_id=sample_id,
                    answer_field=answer_field,
                )
            answer = "" if answer_value is None else str(answer_value).strip()
            if not answer:
                empty_answers.append({"line_number": lineno, "question_id": sample_id, "condition": condition})
            predictions.setdefault(condition, {})[sample_id] = (
                answer
            )
            parsed_rows += 1
            if parsed_rows > parse_row_limit:
                raise SubmissionValidationError(
                    "too_many_submission_rows",
                    f"The file contains more response rows than this benchmark accepts ({maximum_rows:,}). Use the current template and include exactly one row for each required question and condition.",
                    line_number=lineno,
                    max_rows=maximum_rows,
                    rows_seen=parsed_rows,
                )
            records.append({
                "row_index": parsed_rows,
                "line_number": lineno,
                "question_id": sample_id,
                "condition": condition,
                "answer": answer,
            })

        if parsed_rows == 0:
            raise SubmissionValidationError(
                "empty_submission_file",
                "The JSONL file contains no response rows. Start from the downloaded template and add one final answer for every question_id.",
                row_count=0,
            )
        if empty_answers:
            examples = empty_answers[:5]
            example_text = ", ".join(
                f"{item['question_id']} (line {item['line_number']})" for item in examples
            )
            remainder = len(empty_answers) - len(examples)
            if remainder:
                example_text += f", and {remainder} more"
            raise SubmissionValidationError(
                "empty_sample_outputs",
                f"The file contains {len(empty_answers)} row(s) with an empty answer: {example_text}. Provide the model's final output for every row; blank answers cannot be scored.",
                count=len(empty_answers),
                examples=examples,
            )
        missing_conditions = [
            condition for condition in self.allowed_conditions if not predictions.get(condition)
        ]
        if missing_conditions:
            if self.task_id == "spatial":
                message = (
                    "The spatial output file is missing required condition(s): "
                    f"{', '.join(missing_conditions)}. Run the official harness to completion and "
                    "upload its unchanged spatial_reasoning_submission.zip package; all six conditions are required."
                )
            else:
                message = (
                    "The file has no 'standard' condition responses. Include every question_id "
                    "with condition 'standard'."
                )
            raise SubmissionValidationError(
                "missing_required_conditions",
                message,
                conditions_present=sorted(predictions),
                missing_conditions=missing_conditions,
            )
        return predictions, {}, records

    def _format_sample_ids(self, sample_ids) -> str:
        items = sorted(str(s) for s in sample_ids)
        preview = ", ".join(items[:5])
        if len(items) > 5:
            preview += f", ... (+{len(items) - 5} more)"
        return preview

    def _validate_predictions(self, predictions: Dict[str, Dict[str, str]]) -> None:
        missing_conditions = [
            condition for condition in self.allowed_conditions if not predictions.get(condition)
        ]
        if missing_conditions:
            raise SubmissionValidationError(
                "missing_required_conditions",
                f"The submission is missing required condition(s): {', '.join(missing_conditions)}.",
                conditions_present=sorted(predictions),
                missing_conditions=missing_conditions,
            )

        for condition in self.allowed_conditions:
            gt_ids = self._expected_ids(condition)
            condition_preds = predictions.get(condition) or {}
            unknown = set(condition_preds.keys()) - gt_ids
            missing = gt_ids - set(condition_preds.keys())
            if unknown and missing:
                raise SubmissionValidationError(
                    "sample_id_coverage_mismatch",
                    f"Condition '{condition}' contains {len(unknown)} question_id(s) that are not expected for that condition and is missing {len(missing)} required output(s). This usually means the file was generated from the wrong benchmark version or was edited after the harness run. Unknown IDs include: {self._format_sample_ids(unknown)}. Missing IDs include: {self._format_sample_ids(missing)}.",
                    condition=condition,
                    unknown_count=len(unknown),
                    missing_count=len(missing),
                    unknown_question_ids=sorted(str(item) for item in unknown)[:20],
                    missing_question_ids=sorted(str(item) for item in missing)[:20],
                )
            if unknown:
                raise SubmissionValidationError(
                    "unknown_sample_ids",
                    f"Condition '{condition}' contains {len(unknown)} question_id(s) that do not belong to this benchmark: {self._format_sample_ids(unknown)}. Use the question IDs from the current downloaded template.",
                    condition=condition,
                    count=len(unknown),
                    question_ids=sorted(str(item) for item in unknown)[:20],
                )
            if missing:
                raise SubmissionValidationError(
                    "missing_sample_outputs",
                    f"Condition '{condition}' is missing {len(missing)} required model output(s): {self._format_sample_ids(missing)}. Add one JSONL row for every missing question_id; do not remove unanswered template rows.",
                    condition=condition,
                    count=len(missing),
                    question_ids=sorted(str(item) for item in missing)[:20],
                )

    # ------------------------------------------------------------------ score
    def _grade_condition(self, sample_id: str, pred, condition: str) -> bool:
        if condition.startswith("no_image_plus_"):
            return self._is_no_image_plus_correct(sample_id, pred, condition)
        return self._grade(
            pred,
            self._gold_for_condition(sample_id, condition),
            sample_id,
        )

    def _spatial_condition_result(
        self,
        condition: str,
        preds: Dict[str, str],
    ) -> Tuple[int, int, Dict[str, List[int]]]:
        """Score circular variants all-or-nothing, then aggregate by dataset."""
        group_states = {}
        for sample_id in self._expected_ids(condition):
            info = self.ground_truth[sample_id]
            dataset = str(info.get(self.group_by) or info.get("group") or "all")
            evaluation_group = str(info.get("evaluation_group") or sample_id)
            key = (dataset, evaluation_group)
            state = group_states.setdefault(key, {"all_correct": True, "variants": 0})
            state["all_correct"] = state["all_correct"] and self._grade_condition(
                sample_id,
                preds.get(sample_id, ""),
                condition,
            )
            state["variants"] += 1

        group_acc: Dict[str, List[int]] = {}
        overall_correct = 0
        for (dataset, _evaluation_group), state in group_states.items():
            is_correct = int(state["all_correct"])
            overall_correct += is_correct
            bucket = group_acc.setdefault(dataset, [0, 0])
            bucket[0] += is_correct
            bucket[1] += 1
        return overall_correct, len(group_states), group_acc

    def _condition_result(
        self,
        condition: str,
        preds: Dict[str, str],
    ) -> Tuple[int, int, Dict[str, List[int]]]:
        if self.task_id == "spatial":
            return self._spatial_condition_result(condition, preds)
        group_acc: Dict[str, List[int]] = {}
        correct = 0
        expected_ids = self._expected_ids(condition)
        for sample_id in expected_ids:
            info = self.ground_truth[sample_id]
            is_correct = int(
                self._grade_condition(sample_id, preds.get(sample_id, ""), condition)
            )
            correct += is_correct
            group = str(info.get(self.group_by) or info.get("group") or "all")
            bucket = group_acc.setdefault(group, [0, 0])
            bucket[0] += is_correct
            bucket[1] += 1
        return correct, len(expected_ids), group_acc

    def _condition_accuracy(
        self,
        condition: str,
        preds: Dict[str, str],
    ) -> Tuple[int, int]:
        correct, total, _groups = self._condition_result(condition, preds)
        return correct, total

    @staticmethod
    def _macro_from_counts(group_acc: Dict[str, List[int]]) -> float:
        values = [correct / total for correct, total in group_acc.values() if total]
        return sum(values) / len(values) if values else 0.0

    def score(self, file_path: Path, model_name: str,
              model_meta: Optional[Dict] = None) -> TaskScore:
        predictions, parsed_meta = self.parse_submission(file_path)
        return self.score_predictions(predictions, model_name, parsed_meta, model_meta)

    def score_submission_text(self, text: str, model_name: str,
                              model_meta: Optional[Dict] = None) -> TaskScore:
        predictions, parsed_meta = self.parse_submission_text(text)
        return self.score_predictions(predictions, model_name, parsed_meta, model_meta)

    def score_predictions(
        self,
        predictions: Dict[str, Dict[str, str]],
        model_name: str,
        parsed_meta: Optional[Dict] = None,
        model_meta: Optional[Dict] = None,
        submission_metadata: Optional[Dict] = None,
    ) -> TaskScore:
        method_counts: Dict[str, int] = {}
        token = self._method_counts.set(method_counts)
        try:
            return self._score_predictions_in_context(
                predictions,
                model_name,
                parsed_meta,
                model_meta,
                submission_metadata,
                method_counts,
            )
        finally:
            self._method_counts.reset(token)

    def _score_predictions_in_context(
        self,
        predictions: Dict[str, Dict[str, str]],
        model_name: str,
        parsed_meta: Optional[Dict],
        model_meta: Optional[Dict],
        submission_metadata: Optional[Dict],
        method_counts: Dict[str, int],
    ) -> TaskScore:
        self._validate_predictions(predictions)
        meta = {**(parsed_meta or {}), **(model_meta or {})}
        gt = self.ground_truth
        primary = predictions[self.primary_condition]

        group_acc: Dict[str, List[int]] = {}    # group -> [correct, total]
        analysis_acc: Dict[str, Dict[str, List[int]]] = {}
        overall_correct = overall_total = 0

        if self.task_id == "spatial":
            overall_correct, overall_total, group_acc = self._spatial_condition_result(
                self.primary_condition,
                primary,
            )
        else:
            for sid in self._expected_ids(self.primary_condition):
                info = gt[sid]
                is_correct = int(
                    self._grade_condition(
                        sid,
                        primary[sid],
                        self.primary_condition,
                    )
                )
                overall_total += 1
                overall_correct += is_correct
                group = str(info.get(self.group_by) or info.get("group") or "all")
                group_acc.setdefault(group, [0, 0])
                group_acc[group][0] += is_correct
                group_acc[group][1] += 1

                analysis_values = {}
                if self.task_id == "do_you_see_me":
                    dimension = str(info.get("dimension") or "").strip()
                    capability = str(info.get("capability") or group).strip()
                    analysis_values = {
                        "dimension": dimension,
                        "difficulty": info.get("difficulty"),
                        "task_variant": (
                            f"{dimension}:{capability}"
                            if dimension and capability
                            else None
                        ),
                    }
                elif self.task_id == "minds_eye":
                    capability = str(info.get("capability") or group)
                    analysis_values = {
                        "art": MINDS_EYE_ART_BY_CAPABILITY.get(capability),
                    }
                for axis, label in analysis_values.items():
                    if not label:
                        continue
                    bucket = analysis_acc.setdefault(axis, {}).setdefault(str(label), [0, 0])
                    bucket[0] += is_correct
                    bucket[1] += 1

        groups = {
            g: GroupResult(name=g, total_samples=t, correct_samples=c,
                           accuracy=(c / t if t else 0.0))
            for g, (c, t) in group_acc.items()
        }
        analysis = {
            axis: {
                label: GroupResult(
                    name=label,
                    total_samples=total,
                    correct_samples=correct,
                    accuracy=(correct / total if total else 0.0),
                )
                for label, (correct, total) in values.items()
            }
            for axis, values in analysis_acc.items()
        }

        # ``accuracy`` remains the sample-weighted micro result for auditing.
        # ``macro_accuracy`` is the benchmark's paper-aligned ranking score.
        micro = overall_correct / overall_total if overall_total else 0.0
        group_accs = [gr.accuracy for gr in groups.values()]
        score_method = self.task.get("score_method")
        spread_values = group_accs
        if self.task_id == "do_you_see_me":
            by_dimension: Dict[str, List[float]] = {}
            task_variants = analysis.get("task_variant", {})
            for label, result in task_variants.items():
                dimension = str(label).split(":", 1)[0]
                by_dimension.setdefault(dimension, []).append(result.accuracy)
            dimension_macros = [
                sum(values) / len(values)
                for values in by_dimension.values()
                if values
            ]
            macro = (
                sum(dimension_macros) / len(dimension_macros)
                if dimension_macros
                else (sum(group_accs) / len(group_accs) if group_accs else 0.0)
            )
            spread_values = [
                result.accuracy for result in task_variants.values()
            ] or group_accs
        else:
            macro = sum(group_accs) / len(group_accs) if group_accs else 0.0
        spread = (
            statistics.stdev(spread_values)
            if len(spread_values) > 1
            else 0.0
        )

        diagnostics = self._compute_diagnostics(predictions) if self.supports_diagnostics else None

        grading = {
            "method": self.grading_cfg.get("method"),
            "judge_model": self.grading_cfg.get("judge_model"),
            "paper": self.grading_cfg.get("paper"),
            "backend": (
                "harness_judge_plus_deterministic_verification"
                if self.task_id == "spatial"
                else "deterministic"
            ),
            "llm_graded": self.task_id == "spatial",
            "method_counts": dict(method_counts),
        }

        return TaskScore(
            task_id=self.task_id,
            submission_id=str(uuid.uuid4()),
            model_name=model_name,
            submitted_at=datetime.now(timezone.utc),
            accuracy=micro,
            macro_accuracy=macro,
            task_spread=spread,
            accuracy_std=spread,
            random_baseline=self._random_baseline(),
            score_method=score_method,
            total_samples=overall_total,
            correct_samples=overall_correct,
            groups=groups,
            analysis=analysis,
            diagnostics=diagnostics,
            grading=grading,
            model_meta=meta,
            metadata={
                "submission_format": "jsonl",
                **(submission_metadata or {}),
            },
        )

    def _compute_diagnostics(self, predictions: Dict[str, Dict[str, str]]) -> Optional[Diagnostics]:
        conditions = [c for c in self.allowed_conditions if predictions.get(c)]
        if self.task_id != "spatial" or not conditions:
            return None

        def macro(condition: str) -> float:
            _correct, _total, groups = self._condition_result(
                condition,
                predictions[condition],
            )
            return self._macro_from_counts(groups)

        standard_accuracy = macro("main_noncot")
        cot_accuracy = macro("main_cot")
        diag = Diagnostics(
            conditions_present=conditions,
            standard_accuracy=standard_accuracy,
            cot_accuracy=cot_accuracy,
            cot_delta=cot_accuracy - standard_accuracy,
            shortcut_score=macro("no_image_noncot"),
            shortcut_score_cot=macro("no_image_cot"),
            hallucination_resistance=macro("no_image_plus_noncot"),
            hallucination_resistance_cot=macro("no_image_plus_cot"),
        )
        return diag
