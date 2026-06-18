# leaderboard

A web leaderboard that benchmarks Vision-Language Models (VLMs) across three
tasks in two sections: **Visual Cognition** (Do-You-See-Me perception +
Mind's-Eye imagery) and **Spatial Reasoning** (13 public spatial benchmarks,
accuracy + robustness diagnostics).

The application lives in [`Combined-Leaderboard/`](Combined-Leaderboard/).

## Quick start

```powershell
cd Combined-Leaderboard
pip install -r requirements.txt
$env:API_TOKENS = 'your-secret-token'
python backend\web\app.py          # Waitress on http://localhost:5000
```

## Documentation

See [HANDOVER.md](HANDOVER.md) for the full project state: architecture, how to
run, the three model harnesses, VLMEvalKit setup, verification status, and known
issues.

> Note: the `Do-You-See-Me/`, `Mind-s-Eye/`, and `env/` folders are intentionally
> excluded from this repository (separate third-party repos / local virtualenv).
