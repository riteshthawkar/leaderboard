"""
Scoring engine for the unified Visual Cognition golden-set submission.

Scores a single submission file (covering every golden-set sample) and produces:
  * per-task / per-capability / per-layer accuracy
  * the Visual Cognition Index (VCI), a layer-weighted composite
  * robustness diagnostics (CoT-delta, shortcut, hallucination resistance)
    when the optional evaluation conditions are provided.
"""

import csv
import io
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from config import (
    GOLDEN_SET_GROUND_TRUTH_FILE,
    TASK_TAXONOMY,
    LAYER_LABELS,
    VCI_LAYER_WEIGHTS,
)
from models.golden import (
    GoldenTaskResult,
    GoldenCapabilityResult,
    GoldenLayerResult,
    GoldenDiagnostics,
    GoldenSubmissionScore,
)

# Phrases that count as a correct "cannot determine" response in No-Image++.
_CANNOT_DETERMINE = {
    "cannot determine", "cannot determine from the image", "cant determine",
    "can not determine", "indeterminate", "unknown", "not enough information",
    "insufficient information", "none of the above", "cannot be determined",
    "cannot tell", "no answer", "unanswerable",
}


def _normalize(value: str) -> str:
    """Normalise an answer for robust comparison."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    # strip surrounding option wrappers like "(a)", "a.", "option a", "answer: a"
    s = re.sub(r"^\s*(answer|option|final answer)\s*[:\-]?\s*", "", s)
    s = s.strip().strip("().,:;\"'")
    s = re.sub(r"\s+", " ", s)
    return s


def _is_cannot_determine(pred: str) -> bool:
    return _normalize(pred) in _CANNOT_DETERMINE


def _match(pred: str, gold: str) -> bool:
    np_, ng = _normalize(pred), _normalize(gold)
    if not np_:
        return False
    if np_ == ng:
        return True
    # single-letter MCQ answers: compare first token/letter
    if len(ng) == 1 and np_[:1] == ng:
        # only if the prediction is essentially just that letter
        if re.fullmatch(r"[a-z]\b.*", np_) and np_.split(" ")[0] == ng:
            return True
    return False


class GoldenSetScorer:
    """Scores submissions against the fixed golden set."""

    def __init__(self, ground_truth_file: Path = GOLDEN_SET_GROUND_TRUTH_FILE):
        self.ground_truth_file = Path(ground_truth_file)
        self._gt: Optional[Dict[str, dict]] = None

    @property
    def ground_truth(self) -> Dict[str, dict]:
        if self._gt is None:
            if not self.ground_truth_file.exists():
                raise FileNotFoundError(
                    f"Golden set ground truth not found: {self.ground_truth_file}. "
                    f"Run backend/build_golden_set.py first."
                )
            with open(self.ground_truth_file, "r", encoding="utf-8") as f:
                self._gt = json.load(f)
        return self._gt

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
            ) and set(preds.keys()) & {"standard", "cot", "no_image", "no_image_plus"}:
                # conditioned format: {condition: {sample_id: pred}}
                for cond, mapping in preds.items():
                    if isinstance(mapping, dict):
                        predictions[cond] = {str(k): str(v) for k, v in mapping.items()}
            else:
                # flat {sample_id: pred} under "predictions"
                predictions["standard"] = {str(k): str(v) for k, v in preds.items()}
        elif isinstance(data, dict):
            # bare {sample_id: pred}
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

        if "standard" not in predictions or not predictions["standard"]:
            raise ValueError("Submission must include 'standard' predictions")
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
                raise ValueError(
                    "CSV must contain 'sample_id' and 'prediction' columns"
                )
            for row in reader:
                sid = str(row[sid_col]).strip()
                pred = str(row[pred_col]).strip()
                cond = str(row[cond_col]).strip().lower() if cond_col and row.get(cond_col) else "standard"
                if not sid:
                    continue
                predictions.setdefault(cond, {})[sid] = pred

        if "standard" not in predictions or not predictions["standard"]:
            raise ValueError("Submission must include 'standard' predictions")
        return predictions, {}

    # ------------------------------------------------------------------ score
    def _condition_accuracy(self, preds: Dict[str, str]) -> Tuple[int, int]:
        """Accuracy of a condition's predictions against the standard answers."""
        gt = self.ground_truth
        correct = total = 0
        for sid, gold in gt.items():
            if sid not in preds:
                continue
            total += 1
            if _match(preds[sid], gold["answer"]):
                correct += 1
        return correct, total

    def score(self, file_path: Path, model_name: str) -> GoldenSubmissionScore:
        predictions, model_meta = self.parse_submission(file_path)
        gt = self.ground_truth
        standard = predictions["standard"]

        # Per-sample scoring of the standard condition.
        task_acc: Dict[str, List[int]] = {}        # key -> [correct, total]
        cap_acc: Dict[str, List[int]] = {}
        layer_acc: Dict[str, List[int]] = {}
        overall_correct = overall_total = 0

        for sid, meta in gt.items():
            if sid not in standard:
                continue
            is_correct = 1 if _match(standard[sid], meta["answer"]) else 0
            overall_total += 1
            overall_correct += is_correct

            tkey = f"{meta['benchmark']}/{meta['task']}"
            task_acc.setdefault(tkey, [0, 0])
            task_acc[tkey][0] += is_correct
            task_acc[tkey][1] += 1

            cap_acc.setdefault(meta["capability"], [0, 0])
            cap_acc[meta["capability"]][0] += is_correct
            cap_acc[meta["capability"]][1] += 1

            layer_acc.setdefault(meta["layer"], [0, 0])
            layer_acc[meta["layer"]][0] += is_correct
            layer_acc[meta["layer"]][1] += 1

        # Build result objects.
        task_results = {}
        for tkey, (c, t) in task_acc.items():
            capability, layer, _, _ = TASK_TAXONOMY.get(tkey, ("unknown", "unknown", "", ""))
            benchmark, task = tkey.split("/", 1)
            task_results[tkey] = GoldenTaskResult(
                task=task, benchmark=benchmark, capability=capability, layer=layer,
                total_samples=t, correct_samples=c,
                accuracy=(c / t if t else 0.0),
            )

        capability_results = {}
        for cap, (c, t) in cap_acc.items():
            layer = next((v[1] for v in TASK_TAXONOMY.values() if v[0] == cap), "unknown")
            capability_results[cap] = GoldenCapabilityResult(
                capability=cap, layer=layer, total_samples=t,
                correct_samples=c, accuracy=(c / t if t else 0.0),
            )

        layer_results = {}
        for layer, (c, t) in layer_acc.items():
            layer_results[layer] = GoldenLayerResult(
                layer=layer, label=LAYER_LABELS.get(layer, layer),
                total_samples=t, correct_samples=c,
                accuracy=(c / t if t else 0.0),
            )

        vci = self._compute_vci(layer_results)
        diagnostics = self._compute_diagnostics(predictions)

        return GoldenSubmissionScore(
            submission_id=str(uuid.uuid4()),
            model_name=model_name,
            submitted_at=datetime.now(),
            vci=vci,
            overall_accuracy=(overall_correct / overall_total if overall_total else 0.0),
            total_samples=overall_total,
            correct_samples=overall_correct,
            task_results=task_results,
            capability_results=capability_results,
            layer_results=layer_results,
            diagnostics=diagnostics,
            model_meta=model_meta,
            metadata={"submission_file": Path(file_path).name},
        )

    def _compute_vci(self, layer_results: Dict[str, GoldenLayerResult]) -> float:
        """Layer-weighted composite over the layers actually present."""
        present = {l: r for l, r in layer_results.items() if r.total_samples > 0}
        if not present:
            return 0.0
        weights = {l: VCI_LAYER_WEIGHTS.get(l, 0.0) for l in present}
        wsum = sum(weights.values())
        if wsum <= 0:  # fall back to equal weighting
            weights = {l: 1.0 for l in present}
            wsum = float(len(present))
        return sum(present[l].accuracy * w for l, w in weights.items()) / wsum

    def _compute_diagnostics(self, predictions: Dict[str, Dict[str, str]]) -> Optional[GoldenDiagnostics]:
        conditions = [c for c in predictions if predictions[c]]
        optional = [c for c in conditions if c != "standard"]
        if not optional:
            return None

        std_c, std_t = self._condition_accuracy(predictions["standard"])
        std_acc = std_c / std_t if std_t else 0.0
        diag = GoldenDiagnostics(conditions_present=conditions, standard_accuracy=std_acc)

        if "cot" in predictions and predictions["cot"]:
            cot_c, cot_t = self._condition_accuracy(predictions["cot"])
            cot_acc = cot_c / cot_t if cot_t else 0.0
            diag.cot_accuracy = cot_acc
            diag.cot_delta = cot_acc - std_acc

        if "no_image" in predictions and predictions["no_image"]:
            ni_c, ni_t = self._condition_accuracy(predictions["no_image"])
            diag.shortcut_score = ni_c / ni_t if ni_t else 0.0

        if "no_image_plus" in predictions and predictions["no_image_plus"]:
            preds = predictions["no_image_plus"]
            gt = self.ground_truth
            correct = total = 0
            for sid in gt:
                if sid not in preds:
                    continue
                total += 1
                if _is_cannot_determine(preds[sid]):
                    correct += 1
            diag.hallucination_resistance = correct / total if total else 0.0

        return diag
