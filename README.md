# MS VISTA Leaderboard

The maintained application lives in [`Combined-Leaderboard`](Combined-Leaderboard/).
It contains the API-only Flask backend, React/Vite frontend, benchmark tasks,
evaluation harnesses, tests, and deployment documentation.

## Documentation

- [Application README](Combined-Leaderboard/README.md)
- [Architecture](Combined-Leaderboard/docs/architecture.md)
- [Deployment guide](Combined-Leaderboard/docs/deployment.md)
- [Deployment cost estimate](Combined-Leaderboard/docs/deployment-cost.md)

## Local runtime data

The example local configuration writes SQLite, backups, cache files, and logs
under the ignored `Combined-Leaderboard/.local-data/` directory. Private answer
keys, model outputs, and research results must remain outside source control.
Production deployments must use the persistent paths documented in the
[deployment guide](Combined-Leaderboard/docs/deployment.md).
