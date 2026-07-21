import json
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


import submission_store  # noqa: E402


@pytest.fixture()
def isolated_submission_store(monkeypatch):
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
    try:
        yield submission_store
    finally:
        engine.dispose()


def _seed_scored_submission(store, *, owner="member@example.com"):
    with store._Session() as session:
        row = store.Submission(
            user_email=owner,
            task_id="do_you_see_me",
            model_name="Member Model",
            status="scored",
            score_submission_id="submission-owned-1",
            file_sha256="a" * 64,
            row_count=1,
            latest_score_json='{"accuracy":0.5,"total_samples":1}',
            moderation_status="visible",
            created_at=datetime.now(timezone.utc),
        )
        session.add(row)
        session.flush()
        session.add(store.SubmissionAnswer(
            submission_id=row.id,
            row_index=0,
            line_number=1,
            question_id="sample-1",
            condition="standard",
            raw_answer_text="A",
            answer_sha256="b" * 64,
            created_at=datetime.now(timezone.utc),
        ))
        session.commit()


def _spatial_contract():
    template = b'{"question_id":"q1","condition":"main_noncot","answer":""}\n'
    questions = b'{"question_id":"q1","dataset_key":"BLINK"}\n'
    manifest = json.dumps({
        "benchmark_version": "test-v1",
        "artifacts": {
            "submission_template": {"sha256": hashlib.sha256(template).hexdigest()},
            "questions": {"sha256": hashlib.sha256(questions).hexdigest()},
        },
    }, sort_keys=True).encode("utf-8")
    return {
        "manifest": manifest,
        "template": template,
        "questions": questions,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def test_owner_deletion_hides_history_export_and_preserves_quota(isolated_submission_store):
    store = isolated_submission_store
    _seed_scored_submission(store)

    assert store.delete_owned_submission("submission-owned-1", "other@example.com") is None
    deleted = store.delete_owned_submission("submission-owned-1", "MEMBER@example.com")

    assert deleted["moderation_status"] == "deleted"
    assert deleted["submission_export_url"] is None
    assert store.get_submission_export("submission-owned-1", "member@example.com") is None
    assert store.list_submissions(
        user_email="member@example.com",
        include_deleted=False,
    ) == []
    audit_rows = store.list_submissions(user_email="member@example.com")
    assert len(audit_rows) == 1
    assert audit_rows[0]["moderation_status"] == "deleted"
    assert store.quota_status("member@example.com", limit=3)["used"] == 1
    assert store.delete_owned_submission("submission-owned-1", "member@example.com") is None


def test_spatial_artifacts_are_stored_exactly_and_only_visible_while_public(
    isolated_submission_store,
):
    store = isolated_submission_store
    reservation = store.try_consume_quota(
        "member@example.com",
        "spatial",
        "Spatial Model",
        limit=2,
    )
    artifacts = [
        {
            "artifact_name": "spatial_reasoning_submission.zip",
            "media_type": "application/zip",
            "content": b"zip-evidence",
        },
        {
            "artifact_name": "submission.jsonl",
            "media_type": "application/x-ndjson",
            "content": b'{"question_id":"q1","answer":"A"}\n',
        },
        {
            "artifact_name": "run_manifest.json",
            "media_type": "application/json",
            "content": b'{"schema_version":"run-v2"}\n',
        },
        {
            "artifact_name": "leaderboard.json",
            "media_type": "application/json",
            "content": b'{"macro_accuracy":0.5}\n',
        },
    ]
    store.store_submission_answers(
        reservation.submission_id,
        score_submission_id="spatial-evidence-1",
        file_sha256=hashlib.sha256(b"zip-evidence").hexdigest(),
        records=[{
            "row_index": 1,
            "line_number": 1,
            "question_id": "q1",
            "condition": "main_noncot",
            "answer": "A",
        }],
        score_json={
            "accuracy": 0.5,
            "macro_accuracy": 0.5,
            "metadata": {"public_evidence": {"available": True}},
        },
        artifacts=artifacts,
        spatial_contract=_spatial_contract(),
    )
    store.finalize_submission(reservation.submission_id, True)

    evidence = store.get_public_spatial_evidence("spatial-evidence-1")
    stored_artifact = store.get_public_spatial_artifact(
        "spatial-evidence-1",
        "submission.jsonl",
    )

    assert {item["name"] for item in evidence["artifacts"]} == {
        artifact["artifact_name"] for artifact in artifacts
    }
    assert stored_artifact["content"] == artifacts[1]["content"]
    assert stored_artifact["sha256"] == hashlib.sha256(artifacts[1]["content"]).hexdigest()
    stored_for_rescore = store.get_submission_for_rescore("spatial-evidence-1")
    assert stored_for_rescore["spatial_contract"] == {
        **_spatial_contract(),
        "benchmark_version": "test-v1",
    }
    with store._Session() as session:
        assert session.query(store.SpatialBenchmarkContract).count() == 1

    store.set_moderation_status("spatial-evidence-1", "hidden")
    assert store.get_public_spatial_evidence("spatial-evidence-1") is None
    assert store.get_public_spatial_artifact(
        "spatial-evidence-1",
        "submission.jsonl",
    ) is None


def test_registered_model_has_stable_identity_and_global_name_claim(isolated_submission_store):
    store = isolated_submission_store
    meta = {
        "organization": "Example Lab",
        "access": "open_weights",
        "parameter_count": "7B",
        "base_model": "Example Base",
        "training_data": "Public and licensed multimodal data.",
        "paper_url": "https://example.com/paper",
    }

    model = store.create_registered_model(
        "Owner@Example.com",
        "  Model   A  ",
        meta,
    )

    assert model["model_id"].startswith("mdl_")
    assert model["model_name"] == "Model A"
    assert model["owner_email"] == "owner@example.com"
    assert store.get_owned_model(model["model_id"], "OWNER@example.com")["model_id"] == model["model_id"]
    assert store.get_owned_model(model["model_id"], "other@example.com") is None
    with pytest.raises(store.ModelNameConflictError):
        store.create_registered_model("other@example.com", "ｍｏｄｅｌ a", meta)


def test_quota_is_independent_for_each_benchmark(isolated_submission_store):
    store = isolated_submission_store
    first = store.try_consume_quota(
        "member@example.com",
        "do_you_see_me",
        "Model A",
        model_id="mdl_a",
        limit=1,
    )
    repeated = store.try_consume_quota(
        "member@example.com",
        "do_you_see_me",
        "Model A",
        model_id="mdl_a",
        limit=1,
    )
    second_track = store.try_consume_quota(
        "member@example.com",
        "minds_eye",
        "Model A",
        model_id="mdl_a",
        limit=1,
    )

    assert first.allowed is True
    assert repeated.allowed is False
    assert 1 <= repeated.retry_after <= int(store.RESERVATION_TIMEOUT.total_seconds())
    assert second_track.allowed is True
    status = store.quota_status("member@example.com", limit=1)
    assert status["limit"] == 3
    assert status["used"] == 2
    assert status["per_benchmark"]["do_you_see_me"]["remaining"] == 0
    assert status["per_benchmark"]["minds_eye"]["remaining"] == 0
    assert status["per_benchmark"]["spatial"]["remaining"] == 1


def test_abandoned_quota_reservation_is_expired_and_refunded(isolated_submission_store):
    store = isolated_submission_store
    first = store.try_consume_quota(
        "member@example.com",
        "do_you_see_me",
        "Model A",
        model_id="mdl_a",
        limit=1,
    )
    with store._Session() as session:
        row = session.get(store.Submission, first.submission_id)
        row.created_at = datetime.now(timezone.utc) - store.RESERVATION_TIMEOUT - timedelta(minutes=1)
        session.commit()

    assert store.quota_status(
        "member@example.com",
        task_id="do_you_see_me",
        limit=1,
    )["remaining"] == 1
    replacement = store.try_consume_quota(
        "member@example.com",
        "do_you_see_me",
        "Model A",
        model_id="mdl_a",
        limit=1,
    )

    assert replacement.allowed is True
    with store._Session() as session:
        assert session.get(store.Submission, first.submission_id).status == "failed"


def test_registered_model_lists_latest_results_by_track(isolated_submission_store):
    store = isolated_submission_store
    model = store.create_registered_model(
        "member@example.com",
        "Linked Model",
        {
            "organization": "Example Lab",
            "access": "closed",
            "base_model": "Example Base",
            "training_data": "Private multimodal data.",
        },
    )
    with store._Session() as session:
        for task_id, accuracy in (("do_you_see_me", 0.4), ("minds_eye", 0.6)):
            session.add(store.Submission(
                user_email="member@example.com",
                task_id=task_id,
                model_id=model["model_id"],
                model_name="Linked Model",
                status="scored",
                score_submission_id=f"submission-{task_id}",
                latest_score_json=f'{{"accuracy":{accuracy}}}',
                moderation_status="visible",
                created_at=datetime.now(timezone.utc),
            ))
        session.commit()

    rows = store.list_registered_models("MEMBER@example.com")

    assert len(rows) == 1
    assert rows[0]["model_id"] == model["model_id"]
    assert set(rows[0]["benchmarks"]) == {"do_you_see_me", "minds_eye"}
    assert rows[0]["benchmarks"]["minds_eye"]["accuracy"] == 0.6


def test_latest_visible_submission_falls_back_to_previous_run(isolated_submission_store):
    store = isolated_submission_store
    model = store.create_registered_model(
        "member@example.com",
        "Fallback Model",
        {
            "organization": "Example Lab",
            "access": "open_weights",
            "base_model": "Example Base",
            "training_data": "Public data.",
        },
    )
    now = datetime.now(timezone.utc)
    with store._Session() as session:
        session.add_all([
            store.Submission(
                user_email="member@example.com",
                task_id="do_you_see_me",
                model_id=model["model_id"],
                model_name="Fallback Model",
                status="scored",
                score_submission_id="submission-older-visible",
                latest_score_json=json.dumps({
                    "submission_id": "submission-older-visible",
                    "accuracy": 0.4,
                }),
                moderation_status="visible",
                created_at=now - timedelta(days=2),
            ),
            store.Submission(
                user_email="member@example.com",
                task_id="do_you_see_me",
                model_id=model["model_id"],
                model_name="Fallback Model",
                status="scored",
                score_submission_id="submission-newer-hidden",
                moderation_status="hidden",
                created_at=now,
            ),
        ])
        session.commit()

    assert store.latest_visible_scored_submission_id(
        model["model_id"],
        "do_you_see_me",
    ) == "submission-older-visible"
    assert store.latest_visible_scored_submission_ids() == [
        "submission-older-visible"
    ]
    fingerprints = store.latest_visible_scored_submission_fingerprints()
    assert set(fingerprints) == {"submission-older-visible"}
    assert len(fingerprints["submission-older-visible"]) == 64


def test_public_fingerprint_uses_authoritative_submission_metadata(isolated_submission_store):
    store = isolated_submission_store
    model = store.create_registered_model(
        "member@example.com",
        "Canonical Model",
        {
            "organization": "Example Lab",
            "access": "open_weights",
            "base_model": "Example Base",
            "training_data": "Public data.",
        },
    )
    current_meta = {
        "organization": "Example Lab",
        "method": "Current benchmark run",
    }
    stale_score = {
        "submission_id": "stale-submission-id",
        "model_id": None,
        "model_name": "Wrong model",
        "task_id": "minds_eye",
        "accuracy": 0.75,
        "model_meta": {"method": "Stale benchmark run"},
    }
    with store._Session() as session:
        session.add(store.Submission(
            user_email="member@example.com",
            task_id="do_you_see_me",
            model_id=model["model_id"],
            model_name="Canonical Model",
            status="scored",
            score_submission_id="submission-canonical",
            model_meta_json=json.dumps(current_meta),
            latest_score_json=json.dumps(stale_score),
            moderation_status="visible",
            created_at=datetime.now(timezone.utc),
        ))
        session.commit()

    expected_score = {
        **stale_score,
        "submission_id": "submission-canonical",
        "model_id": model["model_id"],
        "model_name": "Canonical Model",
        "task_id": "do_you_see_me",
        "model_meta": current_meta,
    }
    expected_canonical = json.dumps(
        expected_score,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stale_canonical = json.dumps(
        stale_score,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    fingerprint = store.latest_visible_scored_submission_fingerprints()[
        "submission-canonical"
    ]

    assert fingerprint == hashlib.sha256(expected_canonical.encode("utf-8")).hexdigest()
    assert fingerprint != hashlib.sha256(stale_canonical.encode("utf-8")).hexdigest()


def test_submission_integrity_detects_logical_corruption(isolated_submission_store):
    store = isolated_submission_store
    model = store.create_registered_model(
        "member@example.com",
        "Integrity Model",
        {
            "organization": "Example Lab",
            "access": "open_weights",
            "base_model": "Example Base",
            "training_data": "Public data.",
        },
    )
    with store._Session() as session:
        good = store.Submission(
            user_email="member@example.com",
            task_id="do_you_see_me",
            model_id=model["model_id"],
            model_name="Integrity Model",
            status="scored",
            score_submission_id="submission-good",
            row_count=1,
            latest_score_json=json.dumps({"accuracy": 0.5}),
            moderation_status="visible",
            created_at=datetime.now(timezone.utc),
        )
        corrupt = store.Submission(
            user_email="member@example.com",
            task_id="unknown-track",
            model_id=None,
            model_name="Corrupt Model",
            status="scored",
            score_submission_id=None,
            row_count=2,
            latest_score_json="[]",
            moderation_status="visible",
            created_at=datetime.now(timezone.utc),
        )
        session.add_all([good, corrupt])
        session.flush()
        session.add(store.SubmissionAnswer(
            submission_id=good.id,
            row_index=0,
            line_number=1,
            question_id="sample-1",
            condition="standard",
            raw_answer_text="A",
            answer_sha256="b" * 64,
            created_at=datetime.now(timezone.utc),
        ))
        session.commit()

    status = store.submission_integrity_status()

    assert status["healthy"] is False
    assert status["scored_submission_count"] == 2
    assert status["stored_answer_count"] == 1
    assert status["missing_model_id_count"] == 1
    assert status["missing_score_submission_id_count"] == 1
    assert status["malformed_score_payload_count"] == 1
    assert status["answer_row_count_mismatch_count"] == 1
    assert status["invalid_task_count"] == 1


def test_schema_migration_backfills_existing_benchmarks_to_one_model(tmp_path, monkeypatch):
    store = submission_store
    database = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE submissions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_email VARCHAR(255) NOT NULL, task_id VARCHAR(64) NOT NULL, "
            "model_name VARCHAR(255), status VARCHAR(16) NOT NULL, "
            "request_id VARCHAR(64), ip VARCHAR(64), created_at DATETIME NOT NULL)"
        )
        for task_id in ("do_you_see_me", "minds_eye"):
            connection.execute(
                store.text(
                    "INSERT INTO submissions "
                    "(user_email, task_id, model_name, status, created_at) "
                    "VALUES (:owner, :task_id, :model_name, 'scored', :created_at)"
                ),
                {
                    "owner": "member@example.com",
                    "task_id": task_id,
                    "model_name": "Model A",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    session_factory = sessionmaker(bind=engine)
    monkeypatch.setattr(store, "_engine", engine)
    monkeypatch.setattr(store, "_Session", session_factory)
    monkeypatch.setattr(store, "_DB_DRIVER", "sqlite")
    monkeypatch.setattr(store, "_DB_PATH", None)

    store.init_db()

    with engine.connect() as connection:
        model_rows = connection.execute(store.text(
            "SELECT id, owner_email, display_name FROM registered_models"
        )).mappings().all()
        submission_model_ids = connection.execute(store.text(
            "SELECT DISTINCT model_id FROM submissions"
        )).scalars().all()
        submission_columns = {
            column["name"]
            for column in store.inspect(connection).get_columns("submissions")
        }
        table_names = set(store.inspect(connection).get_table_names())
        schema_version = connection.execute(store.text(
            "SELECT version FROM app_schema_versions WHERE component = 'submissions'"
        )).scalar_one()
    assert len(model_rows) == 1
    assert model_rows[0]["owner_email"] == "member@example.com"
    assert model_rows[0]["display_name"] == "Model A"
    assert submission_model_ids == [model_rows[0]["id"]]
    assert "spatial_contract_sha256" in submission_columns
    assert "spatial_benchmark_contracts" in table_names
    assert schema_version == store.SUBMISSION_SCHEMA_VERSION
    engine.dispose()
