# Oracle AMD Micro testing deployment

This is a temporary testing profile for `VM.Standard.E2.1.Micro` (1 GB RAM).
It is not a production capacity recommendation or a high-availability setup.
The standard production profile and its 3.5 GiB RAM requirement are unchanged.

The profile reuses the production API, fallback frontend, Caddy, authentication,
rate limiting, health checks, persistent storage, backup verification and
rollback scripts. GitHub Pages remains the primary frontend. The small Nginx
fallback is retained for deployment smoke tests and recovery.

Limits: one Gunicorn worker/thread; 512 MiB API memory; 96 MiB Caddy memory;
32 MiB fallback frontend memory; 16 MiB general uploads; 10 MiB Track 3 archive;
12 MiB Track 3 multipart body; 32 MiB expanded Track 3 content. Concurrent or
larger submissions require upgrading the VM and using the production profile.
Swap is a crash buffer, not additional workload capacity.

1. Follow `../oci/README.md` for host bootstrap, reserved IP, TLS and private data.
2. Attach a separate 50 GB backup block volume; retain the 50 GB boot volume.
3. Use `../oci/production.env.example` as the protected `production.env` here.
   Set the actual API hostname, GitHub Pages URL and existing auth credentials.
   Do not import an outdated production database as authoritative test data.
4. Run `./check_host.sh`. It requires 800 MiB RAM, one logical CPU, 2 GiB swap,
   10 GiB free storage and separate ext4/XFS backup storage.
5. Build deployment images using the manual GitHub `Build deployment images`
   workflow. Download its image archive, verify SHA256, transfer it over SSH,
   and run `docker load`. Do not build images on the Micro VM.
6. Run `sudo ./deploy_release.sh <commit-sha>` and
   `sudo ./install_systemd_units.sh` from this directory.
7. Verify HTTPS readiness, backups, CORS, email login, Microsoft redirect,
   authenticated submission, logout, memory and a container restart before
   enabling the GitHub Pages deployment. Microsoft Entra must allow the new
   backend callback URL. A correct redirect alone is not a successful login.

Container limits are deliberately not silently increased after an OOM. Record
the failing payload and upgrade before testing larger workloads. Do not bypass
authentication or readiness checks to make a deployment appear healthy.

Release and rollback verification allow at most six complete smoke-check
attempts, five seconds apart, for cold startup on shared CPUs. All readiness,
privacy-related configuration, and backup checks must pass in one attempt;
exhausting the retries still fails the release. This is startup tolerance, not
a throughput guarantee or permission to ignore ongoing request timeouts.

## Resource upgrades without application changes

CPU/RAM upgrades on the same architecture do not require application source
changes or new images. Keep the reserved public IP/API hostname and the mounted
data/backup volumes. GitHub Pages, CORS, and Microsoft redirect configuration
remain unchanged while the API origin stays the same. Moving to ARM requires
compatible images, not a rewrite. Changing the API origin requires the Pages
build variable, CORS and identity-provider redirect URLs to be updated.

1. Verify a fresh mirrored backup and run `ms-vista-backup-verify.service` before
   the maintenance window. Keep a protected copy outside this VM. A separate
   attached volume alone does not protect against loss of the region/account.
2. Stop the application service during the provider's resize. Do not delete or
   reinitialize data, import a development DB, or detach the backup permanently.
3. After restart, run the standard `../oci/check_host.sh` to confirm at least
   3.5 GiB usable RAM, two CPUs, disk space, and separate backup storage.
4. Back up the protected `production.env`, then set resource overrides there.
   These example values are a conservative starting point for a 4 GiB VM with
   at least 1 GiB swap, not a throughput guarantee:

   ```dotenv
   API_MEMORY_LIMIT=2048m
   API_MEMORY_SWAP_LIMIT=2560m
   FRONTEND_MEMORY_LIMIT=64m
   FRONTEND_MEMORY_SWAP_LIMIT=96m
   CADDY_MEMORY_LIMIT=128m
   CADDY_MEMORY_SWAP_LIMIT=192m
   CADDY_GO_MEMORY_LIMIT=96MiB
   MICRO_GUNICORN_THREADS=4
   MICRO_MAX_CONTENT_LENGTH=52428800
   MICRO_MAX_FILE_SIZE_PER_SUBMISSION=52428800
   MICRO_MAX_SPATIAL_ARCHIVE_BYTES=33554432
   MICRO_MAX_SPATIAL_MULTIPART_BYTES=37748736
   MICRO_MAX_SPATIAL_SUBMISSION_BYTES=67108864
   ```

   The `MICRO_` prefix explicitly overrides this directory's testing defaults;
   general production example values cannot accidentally enlarge a Micro VM's
   upload limits. Keep `WEB_CONCURRENCY=1` for SQLite and the JSON result cache.
5. Run `sudo bash ./check_capacity.sh`, then redeploy the **same image tag** with
   `sudo ./deploy_release.sh <existing-tag>`. The release script checks capacity
   before changing a running container, and does not print the resolved secret
   environment. Defaults still preserve the original Micro limits.
6. Verify readiness, both rankings, sign-in, representative upload concurrency,
   memory/OOM counters, backups and a restore rehearsal. Roll back environment
   values if checks fail. Keep peak payloads bounded; swap is not real RAM.

Multiple API workers/replicas are not enabled by a larger VM. That requires a
separate migration of SQLite, result-cache publication, rate-limit state, and
evidence storage to shared services. The present single VM is not highly
available. It still needs owner-approved hosting, monitoring/alert delivery,
recovery objectives, and the external privacy approvals before production use.

## Initial published rankings

A fresh API database returns an empty leaderboard even when authentication and
readiness are healthy. Deployment images deliberately contain no private data.
Do not restore a development database over a live deployment with new accounts.

For an approved, previously published SQLite dataset, use
`scripts/seed_published_results.py` with Python 3.11 or later. Export from the
same backend schema version while the source result store is not being modified:

```sh
python3 scripts/seed_published_results.py export \
  --source-db /private/source/leaderboard.db \
  --source-cache /private/source/results/leaderboard_store.json \
  --destination /private/published-seed
```

The bundle includes only the latest visible scored run per model/task, model
metadata, retained answers, evidence artifacts, and their benchmark contracts.
It excludes accounts, login credentials/tokens, superseded runs, and unpublished
models. Scores and verification labels are preserved without reevaluation.
The cache must match the stored score fingerprints; answer and artifact hashes
are verified. Treat the bundle as private and transfer it over SSH, not Git.

Run the current backend's `submission_integrity_status()` and compare
`latest_visible_scored_submission_fingerprints()` with
`LeaderboardStore.public_submission_fingerprints()` against an isolated copy of
the exported database/cache before importing. Do not initialize the application
against the original bundle, which would change its checksums.

On the destination, dry-run first (result tables must be empty):

```sh
sudo python3 /private/seed_published_results.py import \
  --bundle /private/published-seed \
  --target-db /srv/ms-vista/data/leaderboard.db \
  --target-cache /srv/ms-vista/data/results/leaderboard_store.json \
  --backup-dir /private/published-seed-backups/unique-timestamp
```

During the actual import, hold `/run/lock/ms-vista-deployment.lock` to exclude
the watchdog and deployments. Stop the API, repeat the command with `--apply`,
and restart the API even if the command fails. Each apply requires a new backup
directory and takes a private SQLite snapshot before writing. Authentication
tables are never changed. Conflicting/nonempty results are refused; an exact
repeat is allowed to recover a cache publication interrupted after SQL commit.

Verify HTTPS readiness, nonempty public rankings, evidence downloads, and
unchanged live accounts. Create and verify a new mirrored application backup.
Historical accepted evidence can exceed this VM's new-upload limits; importing
it does not change those limits or make the Micro profile production capacity.
