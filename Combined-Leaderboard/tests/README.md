# Test Suites

The Python test tree is separated by execution scope:

- `backend/` contains isolated API, auth, scoring, storage, and configuration tests.
- `evaluation/` validates benchmark harness contracts and packaging.
- `integration/` checks behavior spanning backend, evaluation, and deployment helpers.
- `e2e/` contains opt-in tests against a disposable running API.

Run the offline suite from the application root:

```bash
python -m pytest -q
```

Live tests are skipped unless explicitly enabled:

```bash
LEADERBOARD_RUN_LIVE_TESTS=1 \
LEADERBOARD_TEST_BASE=http://127.0.0.1:5050 \
python -m pytest -q tests/e2e
```
