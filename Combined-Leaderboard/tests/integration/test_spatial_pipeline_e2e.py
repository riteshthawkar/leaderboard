import asyncio
import base64
import hashlib
import importlib
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import build_server_bundle
import judge_track3
import run_track3_vllm
import spatial_contract
import spatial_submission
import submission_store


def _image_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (32, 64, 96)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _run_cli(monkeypatch, argv, function) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    result = function()
    if result is not None:
        asyncio.run(result)


def test_official_spatial_workflow_reaches_public_evidence_api(tmp_path, monkeypatch):
    # Import the application before swapping the submission store to an isolated
    # engine so its real startup migration is not accidentally skipped for tests
    # that reuse the module later in this process.
    web_app = importlib.import_module("web.app")
    lmudata = tmp_path / "LMUData"
    lmudata.mkdir()
    image = _image_b64()
    for dataset in spatial_contract.DATASETS:
        pd.DataFrame([{
            "index": "0",
            "image": image,
            "question": f"Synthetic question for {dataset}",
            "A": "left",
            "B": "right",
            "answer": "A",
        }]).to_csv(lmudata / f"{dataset}.tsv", sep="\t", index=False)

    ablation_path = tmp_path / "ablation_manifest.json"
    ablation_path.write_text(
        json.dumps({dataset: ["0"] for dataset in spatial_contract.DATASETS}),
        encoding="utf-8",
    )
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "base_noncot.txt").write_text(
        "Return the final option.", encoding="utf-8"
    )
    (prompts_dir / "cot_default.txt").write_text(
        "Reason, then return the final option.", encoding="utf-8"
    )
    monkeypatch.setattr(spatial_contract, "ABLATION_SAMPLES_PER_DATASET", 1)

    bundle_dir = tmp_path / "bundle"
    private_ground_truth = tmp_path / "private" / "ground_truth.json"
    _run_cli(
        monkeypatch,
        [
            "build_server_bundle.py",
            "--lmudata",
            str(lmudata),
            "--benchmark-version",
            "e2e-v1",
            "--output-dir",
            str(bundle_dir),
            "--ground-truth-output",
            str(private_ground_truth),
            "--ablation-manifest",
            str(ablation_path),
            "--prompts-dir",
            str(prompts_dir),
        ],
        build_server_bundle.main,
    )

    class FakeCompletions:
        calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            user_content = kwargs["messages"][-1]["content"]
            if isinstance(user_content, list):
                question = "\n".join(
                    part.get("text", "")
                    for part in user_content
                    if part.get("type") == "text"
                )
                answer = (
                    "Cannot determine from the image"
                    if "Cannot determine from the image" in question
                    else "A"
                )
            else:
                answer = "A"
            message = types.SimpleNamespace(content=answer)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)]
            )

    class FakeAsyncOpenAI:
        instances = []

        def __init__(self, **kwargs):
            self.config = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())
            self.instances.append(self)

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    results_dir = tmp_path / "results"
    _run_cli(
        monkeypatch,
        [
            "run_track3_vllm.py",
            "--lmudata",
            str(lmudata),
            "--benchmark-manifest",
            str(bundle_dir / "manifest.json"),
            "--ablation-manifest",
            str(ablation_path),
            "--prompts-dir",
            str(prompts_dir),
            "--endpoint-model",
            "served/synthetic-vlm",
            "--leaderboard-model-name",
            "Synthetic Public Model",
            "--endpoints",
            "https://model.invalid/v1",
            "--api-key",
            "model-secret",
            "--request-timeout-seconds",
            "7",
            "--max-tokens-noncot",
            "8",
            "--max-tokens-cot",
            "8",
            "--concurrency",
            "4",
            "--request-batch-size",
            "8",
            "--out",
            str(results_dir),
        ],
        run_track3_vllm.main,
    )
    _run_cli(
        monkeypatch,
        [
            "judge_track3.py",
            "--results-dir",
            str(results_dir),
            "--endpoints",
            "https://judge.invalid/v1",
            "--endpoint-model",
            "served/spatial-judge",
            "--api-key",
            "judge-secret",
            "--request-timeout-seconds",
            "9",
            "--concurrency",
            "4",
            "--batch-size",
            "8",
        ],
        judge_track3.main,
    )

    package_path = results_dir / spatial_submission.SPATIAL_SUBMISSION_ARCHIVE_NAME
    package_bytes = package_path.read_bytes()
    submission_bytes, run_manifest_bytes, report_bytes = (
        spatial_submission.read_spatial_submission_archive(package_bytes)
    )
    manifest_bytes = (bundle_dir / "manifest.json").read_bytes()
    template_bytes = (bundle_dir / "submission_template.jsonl").read_bytes()
    questions_bytes = (bundle_dir / "questions.jsonl").read_bytes()
    records, computed_report, _manifest = spatial_submission.parse_spatial_evidence(
        submission_bytes,
        manifest_bytes,
        template_bytes,
        questions_bytes,
    )
    report = spatial_submission.validate_spatial_report(
        report_bytes,
        "Synthetic Public Model",
        computed_report,
    )
    run_metadata = spatial_submission.validate_run_manifest(
        run_manifest_bytes,
        submission_bytes,
        report_bytes,
        "Synthetic Public Model",
        records,
        manifest_bytes,
    )
    score = spatial_submission.build_spatial_task_score(
        report,
        "Synthetic Public Model",
        {"organization": "Synthetic Lab"},
        run_metadata,
    )

    inference_manifest = json.loads(
        (results_dir / "inference_manifest.json").read_text(encoding="utf-8")
    )
    assert inference_manifest["model"] == {
        "name": "Synthetic Public Model",
        "endpoint_model": "served/synthetic-vlm",
    }
    assert {instance.config["timeout"] for instance in FakeAsyncOpenAI.instances} == {
        7.0,
        9.0,
    }
    assert all(
        instance.config["max_retries"] == 0
        for instance in FakeAsyncOpenAI.instances
    )

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(submission_store, "_engine", engine)
    monkeypatch.setattr(submission_store, "_Session", session_factory)
    monkeypatch.setattr(submission_store, "_DB_DRIVER", "sqlite")
    monkeypatch.setattr(submission_store, "_DB_PATH", None)
    submission_store.Base.metadata.create_all(engine)

    reservation = submission_store.try_consume_quota(
        "member@example.com",
        "spatial",
        "Synthetic Public Model",
        limit=2,
    )
    artifacts = [
        {
            "artifact_name": spatial_submission.SPATIAL_SUBMISSION_ARCHIVE_NAME,
            "media_type": "application/zip",
            "content": package_bytes,
        },
        {
            "artifact_name": spatial_submission.SPATIAL_SUBMISSION_MEMBER,
            "media_type": "application/x-ndjson",
            "content": submission_bytes,
        },
        {
            "artifact_name": spatial_submission.SPATIAL_MANIFEST_MEMBER,
            "media_type": "application/json",
            "content": run_manifest_bytes,
        },
        {
            "artifact_name": spatial_submission.SPATIAL_REPORT_MEMBER,
            "media_type": "application/json",
            "content": report_bytes,
        },
    ]
    submission_store.store_submission_answers(
        reservation.submission_id,
        score_submission_id=score.submission_id,
        file_sha256=hashlib.sha256(package_bytes).hexdigest(),
        records=records,
        model_meta=score.model_meta,
        score_json=score.to_dict(),
        artifacts=artifacts,
        spatial_contract={
            "manifest": manifest_bytes,
            "template": template_bytes,
            "questions": questions_bytes,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
    )
    submission_store.finalize_submission(reservation.submission_id, True)

    retained = submission_store.get_submission_for_rescore(score.submission_id)
    assert retained["spatial_contract"]["manifest"] == manifest_bytes
    assert retained["spatial_contract"]["benchmark_version"] == "e2e-v1"

    monkeypatch.setattr(
        web_app,
        "get_submission_for_rescore",
        submission_store.get_submission_for_rescore,
    )
    monkeypatch.setattr(
        web_app,
        "update_submission_score",
        submission_store.update_submission_score,
    )
    with web_app.app.test_request_context("/api/admin/rescore"):
        rescored, rescore_error = web_app._rescore_stored_submission(
            score.submission_id
        )
    assert rescore_error is None
    assert rescored[0].submission_id == score.submission_id
    assert rescored[0].macro_accuracy == score.macro_accuracy
    assert rescored[0].metadata["public_evidence"]["url"] == (
        f"/api/public/submissions/{score.submission_id}/evidence"
    )

    monkeypatch.setattr(
        web_app,
        "get_public_spatial_evidence",
        submission_store.get_public_spatial_evidence,
    )
    monkeypatch.setattr(
        web_app,
        "get_public_spatial_artifact",
        submission_store.get_public_spatial_artifact,
    )
    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as client:
        evidence_response = client.get(
            f"/api/public/submissions/{score.submission_id}/evidence"
        )
        archive_response = client.get(
            f"/api/public/submissions/{score.submission_id}/artifacts/"
            f"{spatial_submission.SPATIAL_SUBMISSION_ARCHIVE_NAME}"
        )

    assert evidence_response.status_code == 200
    assert {
        artifact["name"]
        for artifact in evidence_response.get_json()["artifacts"]
    } == set(spatial_submission.SPATIAL_PUBLIC_ARTIFACT_NAMES)
    assert archive_response.status_code == 200
    assert archive_response.data == package_bytes
    with zipfile.ZipFile(io.BytesIO(archive_response.data)) as archive:
        assert set(archive.namelist()) == set(spatial_submission.SPATIAL_ARCHIVE_MEMBERS)

    engine.dispose()
