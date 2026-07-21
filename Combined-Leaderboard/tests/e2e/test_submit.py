"""Optional live end-to-end submission tests against a running server.

Run with a disposable server configured for local auth testing:
    LEADERBOARD_RUN_LIVE_TESTS=1 LEADERBOARD_TEST_BASE=http://127.0.0.1:5050 \
        python -m pytest tests/e2e/test_submit.py
"""

import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import requests


BASE = os.getenv("LEADERBOARD_TEST_BASE", "http://127.0.0.1:5050")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSWORD = "StrongPass123!QA"
NEW_PASSWORD = "ReplacementPass456!QA"


def _token_from_fragment(url, name):
    return parse_qs(urlparse(url).fragment)[name][0]


def _verified_session(prefix):
    session = requests.Session()
    email = f"{prefix}-{time.time_ns()}@example.com"
    response = session.post(
        f"{BASE}/api/auth/register",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert response.status_code == 201, response.text
    verify_url = response.json().get("dev_verify_url")
    if not verify_url:
        pytest.skip("live server must run with AUTH_DEV_MODE=true for this test")

    login_before_verify = requests.post(
        f"{BASE}/api/auth/login",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert login_before_verify.status_code == 403
    assert login_before_verify.json().get("code") == "unverified"

    verify = session.post(
        f"{BASE}/api/auth/verify",
        json={"token": _token_from_fragment(verify_url, "verify_token")},
        timeout=30,
    )
    assert verify.status_code == 200, verify.text
    csrf_token = verify.json().get("csrf_token")
    assert csrf_token
    return session, email, csrf_token


def _register_model(session, csrf_token, prefix):
    name = f"{prefix}-{time.time_ns()}"
    response = session.post(
        f"{BASE}/api/models",
        headers={"X-CSRF-Token": csrf_token},
        json={
            "model_name": name,
            "organization": "Live QA Lab",
            "access": "open",
            "parameter_count": "Test fixture",
            "base_model": "Live QA Base",
            "training_data": "Synthetic local QA data only; no production data used.",
            "paper_url": "",
        },
        timeout=30,
    )
    assert response.status_code == 201, response.text
    return response.json()["model"]


def _run_metadata():
    return {
        "method_description": " ".join(f"methodword{i}" for i in range(100)),
        "cot_used": "no",
        "prompt_template": "Answer with the final answer only.",
        "changes_from_previous": " ".join(f"changeword{i}" for i in range(50)),
    }


def _completed_submission(task_id, answer="A"):
    completed_rows = []
    template = PROJECT_ROOT / "tasks" / task_id / "submission_template.jsonl"
    with template.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["answer"] = answer
            completed_rows.append(json.dumps(row, separators=(",", ":")))
    return ("\n".join(completed_rows) + "\n").encode("utf-8")


@pytest.mark.skipif(
    os.getenv("LEADERBOARD_RUN_LIVE_TESTS") != "1",
    reason="set LEADERBOARD_RUN_LIVE_TESTS=1 to run live HTTP submission tests",
)
def test_live_legacy_submit_requires_auth():
    """The legacy submit endpoint remains protected in production mode."""
    response = requests.post(f"{BASE}/api/submit", timeout=30)

    assert response.status_code == 401
    assert response.json().get("code") == "auth_required"


@pytest.mark.skipif(
    os.getenv("LEADERBOARD_RUN_LIVE_TESTS") != "1",
    reason="set LEADERBOARD_RUN_LIVE_TESTS=1 to run live HTTP submission tests",
)
def test_live_jsonl_submission_flow():
    """Exercise stable model linkage, task submits, history, export, and delete."""
    session, _email, csrf_token = _verified_session("live-submit")
    model = _register_model(session, csrf_token, "Live-QA-Model")
    submissions = {}
    expected_counts = {"do_you_see_me": 4500, "minds_eye": 799}

    for task_id, expected_count in expected_counts.items():
        completed_submission = _completed_submission(task_id)
        submit = session.post(
            f"{BASE}/api/tasks/{task_id}/submit",
            headers={"X-CSRF-Token": csrf_token},
            data={
                "model_id": model["model_id"],
                "model_meta": json.dumps(_run_metadata()),
            },
            files={
                "file": (
                    f"{task_id}_responses.jsonl",
                    completed_submission,
                    "application/x-ndjson",
                )
            },
            timeout=180,
        )
        assert submit.status_code == 200, submit.text
        body = submit.json()
        assert body["success"] is True
        assert body["model_id"] == model["model_id"]
        assert body["total_samples"] == expected_count
        assert body["submission_export_url"].endswith("/export.jsonl")
        submissions[task_id] = body

    history = session.get(f"{BASE}/api/submissions/mine", timeout=30)
    assert history.status_code == 200
    rows = history.json()["submissions"]
    assert len(rows) == 2
    assert {row["model_id"] for row in rows} == {model["model_id"]}
    assert {row["task_id"] for row in rows} == set(expected_counts)

    for task_id, body in submissions.items():
        export = session.get(f"{BASE}{body['submission_export_url']}", timeout=60)
        assert export.status_code == 200
        assert len([line for line in export.text.splitlines() if line]) == expected_counts[task_id]

    leaderboard = requests.get(f"{BASE}/api/leaderboard/visual-cognition", timeout=30)
    assert leaderboard.status_code == 200
    matching = [
        row for row in leaderboard.json().get("leaderboard", [])
        if row.get("model_id") == model["model_id"]
    ]
    assert len(matching) == 1
    assert matching[0]["has_perception"] is True
    assert matching[0]["has_cognition"] is True
    assert matching[0]["perception_submission"] == submissions["do_you_see_me"]["submission_id"]
    assert matching[0]["cognition_submission"] == submissions["minds_eye"]["submission_id"]

    deleted = session.delete(
        f"{BASE}/api/submissions/{submissions['minds_eye']['submission_id']}",
        headers={"X-CSRF-Token": csrf_token},
        timeout=30,
    )
    assert deleted.status_code == 200, deleted.text
    history = session.get(f"{BASE}/api/submissions/mine", timeout=30)
    assert history.json()["count"] == 1


@pytest.mark.skipif(
    os.getenv("LEADERBOARD_RUN_LIVE_TESTS") != "1",
    reason="set LEADERBOARD_RUN_LIVE_TESTS=1 to run live HTTP submission tests",
)
def test_live_submission_validation_errors_are_propagated():
    """Verify actionable JSONL errors survive the full authenticated HTTP path."""
    session, _email, csrf_token = _verified_session("live-errors")
    model = _register_model(session, csrf_token, "Live-Error-QA-Model")
    cases = [
        (
            "malformed.jsonl",
            b'{"question_id":"s1","answer":"A"}\n{not json\n',
            "invalid_jsonl_syntax",
            {"line_number": 2},
        ),
        (
            "blank.jsonl",
            b'{"question_id":"s1","answer":""}\n',
            "empty_sample_outputs",
            {"count": 1},
        ),
        (
            "missing.jsonl",
            b'{"question_id":"t1_2d_shape_discrimination_easy_0000","answer":"A"}\n',
            "missing_sample_outputs",
            {"count": 4499},
        ),
        (
            "duplicate.jsonl",
            b'{"question_id":"s1","answer":"A"}\n'
            b'{"question_id":"s1","answer":"B"}\n',
            "duplicate_sample_output",
            {"line_number": 2},
        ),
        (
            "object-answer.jsonl",
            b'{"question_id":"s1","answer":{"choice":"A"}}\n',
            "invalid_answer_type",
            {"line_number": 1},
        ),
    ]

    for filename, content, expected_code, expected_details in cases:
        response = session.post(
            f"{BASE}/api/tasks/do_you_see_me/submit",
            headers={"X-CSRF-Token": csrf_token},
            data={
                "model_id": model["model_id"],
                "model_meta": json.dumps(_run_metadata()),
            },
            files={"file": (filename, content, "application/x-ndjson")},
            timeout=30,
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["code"] == expected_code
        assert body["validation"]["code"] == expected_code
        assert body["field_errors"]["file"].startswith("Correct the JSONL issue")
        for key, value in expected_details.items():
            assert body["validation"][key] == value

    wrong_extension = session.post(
        f"{BASE}/api/tasks/do_you_see_me/submit",
        headers={"X-CSRF-Token": csrf_token},
        data={
            "model_id": model["model_id"],
            "model_meta": json.dumps(_run_metadata()),
        },
        files={"file": ("responses.csv", b"question_id,answer\ns1,A\n", "text/csv")},
        timeout=30,
    )
    assert wrong_extension.status_code == 400
    assert wrong_extension.json()["code"] == "invalid_submission_file"


@pytest.mark.skipif(
    os.getenv("LEADERBOARD_RUN_LIVE_TESTS") != "1",
    reason="set LEADERBOARD_RUN_LIVE_TESTS=1 to run live HTTP submission tests",
)
def test_live_password_reset_revokes_the_existing_session():
    """Exercise forgot/reset, single-use tokens, and session revocation over HTTP."""
    session, email, _csrf_token = _verified_session("live-reset")
    requested = requests.post(
        f"{BASE}/api/auth/forgot-password",
        json={"email": email},
        timeout=30,
    )
    assert requested.status_code == 200, requested.text
    reset_url = requested.json().get("dev_reset_url")
    if not reset_url:
        pytest.skip("live server must run with AUTH_DEV_MODE=true for this test")
    token = _token_from_fragment(reset_url, "reset_token")

    reset = requests.post(
        f"{BASE}/api/auth/reset-password",
        json={"token": token, "password": NEW_PASSWORD},
        timeout=30,
    )
    assert reset.status_code == 200, reset.text
    assert session.get(f"{BASE}/api/auth/me", timeout=30).json()["authenticated"] is False

    reused = requests.post(
        f"{BASE}/api/auth/reset-password",
        json={"token": token, "password": PASSWORD},
        timeout=30,
    )
    assert reused.status_code == 400
    assert reused.json()["code"] == "invalid_reset_token"

    old_login = requests.post(
        f"{BASE}/api/auth/login",
        json={"email": email, "password": PASSWORD},
        timeout=30,
    )
    assert old_login.status_code == 401
    new_login = requests.post(
        f"{BASE}/api/auth/login",
        json={"email": email, "password": NEW_PASSWORD},
        timeout=30,
    )
    assert new_login.status_code == 200
    assert new_login.json().get("csrf_token")
