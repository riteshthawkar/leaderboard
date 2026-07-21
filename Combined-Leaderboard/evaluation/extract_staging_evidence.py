"""Extract canonical visual answers once with the production v4 evidence contract."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from evaluation.build_production_visual_results import (
    FINAL_STATUSES,
    UNRESOLVED_STATUSES,
    ProductionBuildError,
    load_completed_audit,
)
from evaluation.common.visual_pipeline import (
    EvaluationPipelineError,
    export_submission,
    load_questions,
    read_diagnostics,
    write_diagnostics,
)
from evaluation.common.vllm_runner import INFERENCE_METHOD
from evaluation.extract_canonical_answers import (
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_REVISION,
    METHOD,
    candidate_category,
    candidate_key,
    extract_evidence_candidate,
    extractor_contract_sha256,
    load_audit_checkpoint,
    wait_for_extractor_clients,
)
from visual_answer_contract import (
    INVALID_FORMAT_TOKEN,
    UNRESOLVED_TOKEN,
    task_from_question_id,
)


TRACKS = ("do_you_see_me", "minds_eye")


def _endpoints(values: list[str]) -> list[str]:
    endpoints = [
        endpoint.strip().rstrip("/")
        for value in values
        for endpoint in value.split(",")
        if endpoint.strip()
    ]
    if not endpoints:
        raise EvaluationPipelineError("Provide at least one extractor endpoint.")
    return endpoints


def _load_inference_rows(
    path: Path,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_diagnostics([path])
    expected_ids = [str(question["question_id"]) for question in questions]
    actual_ids = [str(row.get("question_id") or "") for row in rows]
    if actual_ids != expected_ids:
        raise EvaluationPipelineError(
            f"{path} coverage/order mismatch: expected {len(expected_ids)} rows, "
            f"got {len(actual_ids)}."
        )
    for index, row in enumerate(rows, 1):
        output = row.get("output")
        if not isinstance(output, str) or not output.strip():
            raise EvaluationPipelineError(f"{path} row {index} has no response.")
        if row.get("error") or row.get("inference_error"):
            raise EvaluationPipelineError(
                f"{path} row {index} has an inference error."
            )
        output_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
        if (
            row.get("inference_method") != INFERENCE_METHOD
            or row.get("inference_output_sha256") != output_hash
        ):
            raise EvaluationPipelineError(
                f"{path} row {index} has invalid response provenance."
            )
    return rows


def _candidates(
    *,
    model_slug: str,
    track: str,
    questions: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for question, record in zip(questions, inference_rows, strict=True):
        question_id = str(question["question_id"])
        answer_type = str(question.get("answer_type") or record.get("answer_type") or "text")
        response = str(record["output"])
        candidates.append(
            {
                "model_slug": model_slug,
                "source_relative_dir": model_slug,
                "track": track,
                "question_id": question_id,
                "answer_type": answer_type,
                "task": str(question.get("task") or task_from_question_id(question_id)),
                "category": candidate_category(track, answer_type, record)
                or "contract_exact",
                "question": str(question.get("question") or ""),
                "response": response,
                "response_finish_reason": record.get("finish_reason"),
                "response_completion_tokens": record.get("completion_tokens"),
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "current_submission_answer": UNRESOLVED_TOKEN,
            }
        )
    return candidates


def _answer_from_evidence(row: dict[str, Any]) -> str:
    status = str(row["status"])
    if status == "committed":
        return str(row["answer"])
    if status == "invalid_format_committed":
        return INVALID_FORMAT_TOKEN
    if status in UNRESOLVED_STATUSES:
        return UNRESOLVED_TOKEN
    raise EvaluationPipelineError(f"Blocking v4 extractor status: {status}.")


def _write_final_artifacts(
    args: argparse.Namespace,
    questions: list[dict[str, Any]],
    inference_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    audit, contract = load_completed_audit(args.evidence, candidates)
    diagnostics = []
    for inference, candidate in zip(inference_rows, candidates, strict=True):
        evidence = audit[candidate_key(candidate)]
        diagnostic = dict(inference)
        diagnostic.update(
            {
                "answer_extraction_method": METHOD,
                "extractor_model": evidence["extractor_model"],
                "extractor_revision": evidence["extractor_revision"],
                "extractor_contract_sha256": contract,
                "extractor_output": str(evidence.get("extractor_output") or ""),
                "extractor_evidence": str(evidence.get("evidence") or ""),
                "extractor_verdict": str(evidence.get("extractor_verdict") or ""),
                "extractor_status": str(evidence["status"]),
                "extractor_finish_reason": evidence.get("finish_reason"),
                "extractor_completion_tokens": evidence.get("completion_tokens"),
                "extractor_source_diagnostics": args.source.name,
                "extractor_source_output_sha256": evidence["response_sha256"],
                "extractor_ground_truth_loaded": False,
                "extractor_ground_truth_supplied": False,
                "extracted_answer": _answer_from_evidence(evidence),
            }
        )
        if evidence.get("proposed_answer"):
            diagnostic["extractor_proposed_answer"] = str(evidence["proposed_answer"])
        if evidence.get("extractor_attempts"):
            diagnostic["extractor_attempts"] = list(evidence["extractor_attempts"])
        if evidence.get("terminal_fallback_method"):
            diagnostic["terminal_fallback_method"] = str(
                evidence["terminal_fallback_method"]
            )
            diagnostic["terminal_fallback_from_status"] = str(
                evidence.get("terminal_fallback_from_status") or ""
            )
        diagnostics.append(diagnostic)

    write_diagnostics(args.diagnostics, diagnostics, preserve_extra_fields=True)
    report = export_submission(
        diagnostics,
        questions,
        args.out,
        require_extracted_answers=True,
    )
    return {
        **report,
        "evidence_path": str(args.evidence),
        "evidence_sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
        "extractor_contract_sha256": contract,
        "status_counts": {
            status: sum(row["status"] == status for row in audit.values())
            for status in sorted(FINAL_STATUSES)
        },
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_questions(args.questions, args.track_config)
    inference_rows = _load_inference_rows(args.source, questions)
    candidates = _candidates(
        model_slug=args.model_slug,
        track=args.track,
        questions=questions,
        inference_rows=inference_rows,
    )
    contract = extractor_contract_sha256(args.model, args.max_tokens, args.revision)

    if args.evidence.is_file():
        try:
            return _write_final_artifacts(args, questions, inference_rows, candidates)
        except ProductionBuildError:
            pass

    endpoints = _endpoints(args.endpoint)
    clients = [
        AsyncOpenAI(
            base_url=endpoint,
            api_key=args.api_key,
            timeout=args.timeout,
            max_retries=0,
        )
        for endpoint in endpoints
    ]
    try:
        await wait_for_extractor_clients(
            clients,
            endpoints,
            args.model,
            args.endpoint_start_timeout,
        )
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        candidates_by_key = {candidate_key(candidate): candidate for candidate in candidates}
        existing, retry_history = load_audit_checkpoint(
            args.evidence,
            candidates_by_key,
            contract,
        )
        pending = [
            candidate
            for candidate in candidates
            if candidate_key(candidate) not in existing
        ]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for candidate in pending:
            queue.put_nowait(candidate)
        lock = asyncio.Lock()
        completed = 0

        async def worker(index: int) -> None:
            nonlocal completed
            client = clients[index % len(clients)]
            while True:
                try:
                    candidate = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    row = await extract_evidence_candidate(
                        client,
                        candidate,
                        model=args.model,
                        revision=args.revision,
                        max_tokens=args.max_tokens,
                        retries=args.retries,
                        extractor_contract=contract,
                        retry_history=retry_history.get(candidate_key(candidate)),
                    )
                    async with lock:
                        with args.evidence.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        completed += 1
                        if completed % args.report_every == 0 or completed == len(pending):
                            print(
                                f"[{args.track}] evidence {completed}/{len(pending)} new",
                                flush=True,
                            )
                finally:
                    queue.task_done()

        await asyncio.gather(
            *(worker(index) for index in range(min(args.concurrency, len(pending))))
        )
    finally:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)

    return _write_final_artifacts(args, questions, inference_rows, candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=DEFAULT_EXTRACTOR_MODEL)
    parser.add_argument("--revision", default=DEFAULT_EXTRACTOR_REVISION)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--endpoint-start-timeout", type=float, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=25)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.concurrency < 1 or args.retries < 0:
        parser.error("max-tokens and concurrency must be positive; retries cannot be negative")

    from evaluation.do_you_see_me.config import TRACK as DYS_TRACK
    from evaluation.minds_eye.config import TRACK as MINDS_EYE_TRACK

    args.track_config = {
        "do_you_see_me": DYS_TRACK,
        "minds_eye": MINDS_EYE_TRACK,
    }[args.track]
    return args


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(_run(args))
    except (EvaluationPipelineError, ProductionBuildError, OSError, RuntimeError) as exc:
        print(f"Evidence extraction failed: {exc}", flush=True)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
