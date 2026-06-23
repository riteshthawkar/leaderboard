"""
Generic scorer for a single task submission (do_you_see_me / minds_eye / spatial).

Scores a submission file against the task's private ground truth and produces a
:class:`TaskScore` with overall + per-group accuracy and, for tasks that support
it (Task 3 spatial), the robustness diagnostics.

Scoring is deterministic (normalised exact / single-letter MCQ match). An
LLM-as-judge can be layered in later by overriding :meth:`match`.
"""

import csv
import json
import re
import statistics
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import TASKS, GRADING
from models.tasks import GroupResult, Diagnostics, TaskScore
from scoring.llm_grader import LLMGrader

_OPTIONAL_CONDITIONS = {"cot", "no_image", "no_image_plus"}
_ALL_CONDITIONS = {"standard"} | _OPTIONAL_CONDITIONS

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
    s = re.sub(r"^\s*(answer|option|final answer)\s*[:\-]?\s*", "", s)
    s = s.strip().strip("().,:;\"'")
    s = re.sub(r"\s+", " ", s)
    return s


def _is_cannot_determine(pred) -> bool:
    return _normalize(pred) in _CANNOT_DETERMINE


def _match(pred, gold) -> bool:
    np_, ng = _normalize(pred), _normalize(gold)
    if not np_:
        return False
    if np_ == ng:
        return True
    # single-letter MCQ answers: compare the leading token/letter
    if len(ng) == 1 and np_[:1] == ng:
        if re.fullmatch(r"[a-z]\b.*", np_) and np_.split(" ")[0] == ng:
            return True
    return False


class TaskScorer:
    """Scores submissions for one task against its private ground truth."""

    def __init__(self, task_id: str):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.ground_truth_file = Path(self.task["paths"]["ground_truth"])
        self.questions_file = Path(self.task["paths"]["questions"])
        self.group_by = self.task.get("group_by", "group")
        self.supports_diagnostics = bool(self.task.get("supports_diagnostics"))
        self._gt: Optional[Dict[str, dict]] = None
        self._questions: Optional[Dict[str, dict]] = None
        # Per-paper grading pipeline (LLM extractor / judge with deterministic
        # fallback). See config.GRADING and scoring/llm_grader.py.
        self.grading_cfg = GRADING.get(task_id, {"method": "extract", "answer_types": ["mcq"]})
        self.grader = LLMGrader(self.grading_cfg)
        self._method_counts: Dict[str, int] = {}

    @property
    def ground_truth(self) -> Dict[str, dict]:
        if self._gt is None:
            if not self.ground_truth_file.exists():
                raise FileNotFoundError(
                    f"Ground truth for task '{self.task_id}' not found: "
                    f"{self.ground_truth_file}. Run backend/build_tasks.py first."
                )
            with open(self.ground_truth_file, "r", encoding="utf-8") as f:
                self._gt = json.load(f)
        return self._gt

    @property
    def questions(self) -> Dict[str, dict]:
        """sample_id -> {question, options, format}; empty if no questions file.

        The question text and options are passed to the LLM extractor / judge so
        grading matches the source papers (which grade the raw response against
        the question + options).
        """
        if self._questions is None:
            self._questions = {}
            if self.questions_file.exists():
                try:
                    with open(self.questions_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    samples = data.get("samples", []) if isinstance(data, dict) else data
                    for s in samples or []:
                        if not isinstance(s, dict):
                            continue
                        sid = str(s.get("sample_id") or "")
                        if sid:
                            self._questions[sid] = {
                                "question": s.get("question", ""),
                                "options": s.get("options"),
                                "format": s.get("format", "mcq"),
                            }
                except Exception:
                    self._questions = {}
        return self._questions

    # ------------------------------------------------------------- grading
    def _answer_type(self, gold) -> str:
        """Pick the extractor target for a sample based on the configured types
        and the shape of the gold answer (numeric / single-letter MCQ / free)."""
        types = self.grading_cfg.get("answer_types", ["mcq"]) or ["mcq"]
        g = str(gold).strip()
        if "numeric" in types and re.fullmatch(r"-?\d+(?:\.\d+)?", g):
            return "numeric"
        if re.fullmatch(r"[A-Fa-f]", g):
            return "mcq"
        # Do-You-See-Me has free-response (non-letter / non-numeric) answers.
        return "free" if "numeric" in types else "mcq"

    def _grade(self, pred, gold, sid) -> bool:
        """Grade one prediction with the task's paper-faithful pipeline."""
        q = self.questions.get(str(sid), {})
        correct, method = self.grader.grade(
            pred, gold,
            question=q.get("question", ""),
            options=q.get("options"),
            answer_type=self._answer_type(gold),
        )
        self._method_counts[method] = self._method_counts.get(method, 0) + 1
        return correct

    def _random_baseline(self) -> Optional[float]:
        cfg = self.grading_cfg.get("random_baseline")
        if cfg is None:
            return None
        vals = []
        for q in self.questions.values():
            opts = q.get("options")
            if isinstance(opts, (list, tuple, dict)) and opts:
                vals.append(1.0 / len(opts))
        return sum(vals) / len(vals) if vals else float(cfg)

    # ------------------------------------------------------------------ parse
    def parse_submission(self, file_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict]:
        """Return (predictions_by_condition, model_meta)."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Submission file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            return self._parse_json(file_path)
        if suffix == ".csv":
            return self._parse_csv(file_path)
        raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")

    def _parse_json(self, file_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_meta: Dict = {}
        predictions: Dict[str, Dict[str, str]] = {}

        if isinstance(data, dict) and "predictions" in data:
            model_meta = data.get("model_meta") or data.get("model") or {}
            preds = data["predictions"] or {}
            if isinstance(preds, dict) and any(
                isinstance(v, dict) for v in preds.values()
            ) and set(preds.keys()) & _ALL_CONDITIONS:
                for cond, mapping in preds.items():
                    if isinstance(mapping, dict):
                        predictions[cond] = {str(k): str(v) for k, v in mapping.items()}
            else:
                predictions["standard"] = {str(k): str(v) for k, v in preds.items()}
        elif isinstance(data, dict):
            predictions["standard"] = {str(k): str(v) for k, v in data.items()}
        elif isinstance(data, list):
            std = {}
            for item in data:
                if isinstance(item, dict):
                    sid = item.get("sample_id")
                    pred = item.get("prediction", item.get("answer", item.get("response")))
                    if sid is not None:
                        std[str(sid)] = str(pred)
            predictions["standard"] = std
        else:
            raise ValueError("Unrecognised JSON submission structure")

        if not predictions.get("standard"):
            raise ValueError("Submission must include 'standard' predictions")
        if not self.supports_diagnostics:
            predictions = {"standard": predictions["standard"]}
        return predictions, model_meta

    def _parse_csv(self, file_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict]:
        predictions: Dict[str, Dict[str, str]] = {}
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            sid_col = cols.get("sample_id") or cols.get("id")
            pred_col = (cols.get("prediction") or cols.get("response")
                        or cols.get("answer") or cols.get("output"))
            cond_col = cols.get("condition") or cols.get("eval_condition")
            if not sid_col or not pred_col:
                raise ValueError("CSV must contain 'sample_id' and 'prediction' columns")
            for row in reader:
                sid = str(row[sid_col]).strip()
                pred = str(row[pred_col]).strip()
                cond = (str(row[cond_col]).strip().lower()
                        if cond_col and row.get(cond_col) else "standard")
                if not sid:
                    continue
                predictions.setdefault(cond, {})[sid] = pred

        if not predictions.get("standard"):
            raise ValueError("Submission must include 'standard' predictions")
        if not self.supports_diagnostics:
            predictions = {"standard": predictions["standard"]}
        return predictions, {}

    # ------------------------------------------------------------------ score
    def _condition_accuracy(self, preds: Dict[str, str]) -> Tuple[int, int]:
        gt = self.ground_truth
        correct = total = 0
        for sid, meta in gt.items():
            if sid not in preds:
                continue
            total += 1
            if self._grade(preds[sid], meta["answer"], sid):
                correct += 1
        return correct, total

    def score(self, file_path: Path, model_name: str,
              model_meta: Optional[Dict] = None) -> TaskScore:
        predictions, parsed_meta = self.parse_submission(file_path)
        meta = {**(parsed_meta or {}), **(model_meta or {})}
        gt = self.ground_truth
        standard = predictions["standard"]
        self._method_counts = {}

        group_acc: Dict[str, List[int]] = {}    # group -> [correct, total]
        overall_correct = overall_total = 0

        for sid, info in gt.items():
            if sid not in standard:
                continue
            is_correct = 1 if self._grade(standard[sid], info["answer"], sid) else 0
            overall_total += 1
            overall_correct += is_correct
            group = str(info.get(self.group_by) or info.get("group") or "all")
            group_acc.setdefault(group, [0, 0])
            group_acc[group][0] += is_correct
            group_acc[group][1] += 1

        groups = {
            g: GroupResult(name=g, total_samples=t, correct_samples=c,
                           accuracy=(c / t if t else 0.0))
            for g, (c, t) in group_acc.items()
        }

        # Micro accuracy (overall correct / overall samples) is kept as the
        # headline `accuracy` used for ranking and the VCI. The papers report a
        # macro mean +/- std. dev. across sub-tasks / datasets, plus (Mind's-Eye)
        # a random-choice baseline; we surface those alongside.
        micro = overall_correct / overall_total if overall_total else 0.0
        group_accs = [gr.accuracy for gr in groups.values()]
        macro = sum(group_accs) / len(group_accs) if group_accs else 0.0
        std = statistics.stdev(group_accs) if len(group_accs) > 1 else 0.0

        diagnostics = self._compute_diagnostics(predictions) if self.supports_diagnostics else None

        llm_used = any(m in ("llm_judge", "llm_extract") for m in self._method_counts)
        grading = {
            "method": self.grading_cfg.get("method"),
            "judge_model": self.grading_cfg.get("judge_model"),
            "paper": self.grading_cfg.get("paper"),
            "backend": self.grader._backend or "deterministic",
            "llm_graded": llm_used,
            "method_counts": dict(self._method_counts),
        }

        return TaskScore(
            task_id=self.task_id,
            submission_id=str(uuid.uuid4()),
            model_name=model_name,
            submitted_at=datetime.now(),
            accuracy=micro,
            macro_accuracy=macro,
            accuracy_std=std,
            random_baseline=self._random_baseline(),
            total_samples=overall_total,
            correct_samples=overall_correct,
            groups=groups,
            diagnostics=diagnostics,
            grading=grading,
            model_meta=meta,
            metadata={"submission_file": Path(file_path).name},
        )

    def _compute_diagnostics(self, predictions: Dict[str, Dict[str, str]]) -> Optional[Diagnostics]:
        conditions = [c for c in predictions if predictions[c]]
        if not (set(conditions) & _OPTIONAL_CONDITIONS):
            return None

        std_c, std_t = self._condition_accuracy(predictions["standard"])
        std_acc = std_c / std_t if std_t else 0.0
        diag = Diagnostics(conditions_present=conditions, standard_accuracy=std_acc)

        if predictions.get("cot"):
            cot_c, cot_t = self._condition_accuracy(predictions["cot"])
            cot_acc = cot_c / cot_t if cot_t else 0.0
            diag.cot_accuracy = cot_acc
            diag.cot_delta = cot_acc - std_acc

        if predictions.get("no_image"):
            ni_c, ni_t = self._condition_accuracy(predictions["no_image"])
            diag.shortcut_score = ni_c / ni_t if ni_t else 0.0

        if predictions.get("no_image_plus"):
            preds = predictions["no_image_plus"]
            correct = total = 0
            for sid in self.ground_truth:
                if sid not in preds:
                    continue
                total += 1
                if _is_cannot_determine(preds[sid]):
                    correct += 1
            diag.hallucination_resistance = correct / total if total else 0.0

        return diag
