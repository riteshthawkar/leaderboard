"""Quick end-to-end submission test against the running server."""
import requests

BASE = "http://127.0.0.1:5000"

with open("sample_submission_correct.json", "rb") as f:
    files = {"file": ("sample_submission_correct.json", f, "application/json")}
    data = {
        "model_name": "Test-Model-VLM",
        "benchmark": "do_you_see_me",
        "task_name": "visual_spatial",
    }
    # API_TOKENS is unset -> empty token is accepted
    headers = {"Authorization": "Bearer "}
    resp = requests.post(f"{BASE}/api/submit", files=files, data=data, headers=headers)

print("Status:", resp.status_code)
print("Response:", resp.text)
