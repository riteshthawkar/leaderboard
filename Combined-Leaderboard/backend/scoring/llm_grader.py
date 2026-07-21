"""
Legacy LLM answer extraction and judging utilities.

The production task submission path no longer uses this module. Public uploads
must contain final answers in JSONL, and ``scoring.task_scorer`` matches those
answers deterministically against private ground truth. This file is retained
for legacy utilities and offline tests that exercise the older extractor/judge
behavior.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import requests

# OpenAI chat-completions API base URL (fixed; only the key comes from .env).
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Phrases that count as a correct "cannot determine" response in no-image++.
CANNOT_DETERMINE_PHRASES = {
    "cannot determine", "cannot determine from the image", "cant determine",
    "can not determine", "indeterminate", "unknown", "not enough information",
    "insufficient information", "none of the above", "cannot be determined",
    "cannot tell", "no answer", "unanswerable",
}


def normalize(value) -> str:
    """Lower-case, strip answer prefixes / punctuation / extra whitespace."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^\s*(answer|option|final answer|the answer is)\s*[:\-]?\s*", "", s)
    s = s.strip().strip("().,:;\"'")
    s = re.sub(r"\s+", " ", s)
    return s


def deterministic_match(pred, gold) -> bool:
    """Normalised exact / single-letter MCQ match (the offline fallback)."""
    np_, ng = normalize(pred), normalize(gold)
    if not np_:
        return False
    if np_ == ng:
        return True
    if len(ng) == 1 and np_[:1] == ng:
        if re.fullmatch(r"[a-z]\b.*", np_) and np_.split(" ")[0] == ng:
            return True
    # numeric equality (free-response numeric answers, e.g. Do-You-See-Me)
    mp = re.search(r"-?\d+(?:\.\d+)?", np_)
    mg = re.search(r"-?\d+(?:\.\d+)?", ng)
    if mp and mg:
        try:
            return float(mp.group()) == float(mg.group())
        except ValueError:
            pass
    return False


def is_cannot_determine(pred) -> bool:
    return normalize(pred) in CANNOT_DETERMINE_PHRASES


def _format_options(options) -> str:
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in options.items())
    if isinstance(options, (list, tuple)):
        out = []
        for i, opt in enumerate(options):
            label = chr(ord("A") + i)
            out.append(f"{label}. {opt}")
        return "\n".join(out)
    return str(options)


class LLMGrader:
    """Single-judge LLM extractor / judge (GPT-4o) with a deterministic fallback.

    Parameters come from ``config.GRADING[task_id]``. The judge is GPT-4o served
    via the OpenAI chat-completions API; the only environment value used is the
    API key (``OPENAI_API_KEY``). The grader is lazy and stateless apart from a
    small "backend reachable" flag that disables further network attempts once a
    call has failed (so a misconfigured endpoint does not slow every sample).
    """

    def __init__(self, grading: Dict):
        self.method: str = grading.get("method", "extract")  # "extract" | "judge"
        self.judge_model: str = grading.get("judge_model", "gpt-4o")
        self.answer_types: List[str] = grading.get("answer_types", ["mcq"])
        self.paper: str = grading.get("paper", "")

        # Single shared judge: GPT-4o via the OpenAI chat-completions API.
        # Only the API key is read from the environment (.env); the endpoint is
        # fixed so grading is uniform across all tasks.
        self.base_url: str = OPENAI_BASE_URL
        self.api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
        self.timeout: int = int(os.getenv("GRADING_TIMEOUT", "60"))

        # "openai" when a key is available, otherwise "" (offline deterministic).
        self.backend: str = "openai"
        self._backend: str = "openai" if self.api_key else ""
        self._disabled = False  # set True after a failed call to stop retrying

    # ------------------------------------------------------------- backend
    @property
    def enabled(self) -> bool:
        return bool(self._backend) and not self._disabled

    def _chat(self, system: str, user: str) -> Optional[str]:
        """Send a single-turn chat completion; return content or None."""
        if not self.enabled:
            return None
        try:
            return self._chat_openai(system, user)
        except Exception:
            # One failure disables the backend for the rest of this submission
            # so we degrade gracefully to deterministic matching.
            self._disabled = True
        return None

    def _chat_openai(self, system: str, user: str) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.judge_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,  # greedy decoding (paper protocol)
            "max_tokens": 32,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    # ------------------------------------------------------------- grading
    def grade(self, prediction, gold, *, question: str = "", options=None,
              answer_type: str = "mcq") -> Tuple[bool, str]:
        """Return (is_correct, method_used).

        ``method_used`` is one of: ``llm_judge``, ``llm_extract`` or
        ``deterministic``.
        """
        if self.method == "judge":
            ok = self._judge(prediction, gold, question=question, options=options)
            if ok is not None:
                return ok, "llm_judge"
        else:  # extract
            canonical = self._extract(
                prediction, question=question, options=options,
                answer_type=answer_type,
            )
            if canonical is not None:
                return deterministic_match(canonical, gold), "llm_extract"
        # fallback
        return deterministic_match(prediction, gold), "deterministic"

    # ---- Do-You-See-Me / Mind's-Eye: LLM answer extractor ----
    def _extract(self, prediction, *, question: str, options,
                 answer_type: str) -> Optional[str]:
        if not self.enabled:
            return None
        text = str(prediction).strip()
        if not text:
            return ""
        opt_block = _format_options(options)
        if answer_type == "numeric":
            target = ("the single final numeric value the response settles on "
                      "(digits only, no units or words)")
        elif answer_type == "free":
            target = ("the concise final answer the response commits to (a short "
                      "word, phrase, number or option letter) - just the answer "
                      "itself, normalized to how it would appear in an answer key")
        else:
            target = ("ONLY the final selected multiple-choice option label "
                      "(e.g. A, B, C, D). If the response gives the option text "
                      "instead of a letter, return the matching letter. If there "
                      "is no clear final choice, return X")
        system = (
            "You are a strict answer-extraction module used to grade a "
            "multimodal model's response. You never solve the task yourself; "
            "you only copy out the final answer the response committed to."
        )
        user = (
            f"Question:\n{question or '(not provided)'}\n\n"
            + (f"Options:\n{opt_block}\n\n" if opt_block else "")
            + f"Model response:\n\"{text}\"\n\n"
            f"Extract {target}. Output only the extracted answer with no "
            f"explanation."
        )
        out = self._chat(system, user)
        if out is None:
            return None
        return out.strip()

    # ---- Spatial: LLM-as-judge ----
    def _judge(self, prediction, gold, *, question: str, options) -> Optional[bool]:
        if not self.enabled:
            return None
        text = str(prediction).strip()
        if not text:
            return False
        opt_block = _format_options(options)
        system = (
            "You are an impartial judge scoring a multimodal model's answer to "
            "a multiple-choice spatial-reasoning question. The model's final "
            "answer is restricted to the provided options. Decide whether the "
            "model's final selected option matches the correct answer."
        )
        user = (
            f"Question:\n{question or '(not provided)'}\n\n"
            + (f"Options:\n{opt_block}\n\n" if opt_block else "")
            + f"Correct answer: {gold}\n\n"
            f"Model response:\n\"{text}\"\n\n"
            "Did the model select the correct answer? Respond with exactly one "
            "word: 'correct' or 'incorrect'."
        )
        out = self._chat(system, user)
        if out is None:
            return None
        verdict = normalize(out)
        if verdict.startswith("correct") or verdict in {"yes", "true", "1"}:
            return True
        if verdict.startswith("incorrect") or verdict in {"no", "false", "0"}:
            return False
        # Unparseable verdict -> fall back rather than guess.
        return None
