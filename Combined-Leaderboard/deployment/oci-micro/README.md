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
