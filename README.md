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

The workspace root also contains ignored local runtime state such as SQLite
backups, leaderboard cache files, private model outputs, and development logs.
These files are not application source and must not be included in a deployment
image. The active SQLite database is stored under `Combined-Leaderboard/` in the
current local configuration.
