# Azure VM deployment

This topology runs the React frontend and Flask API as private containers behind
one Caddy HTTPS endpoint. SQLite remains single-instance and is stored under
`/srv/ms-vista/data`. Verified backup archives are mirrored to the separately
mounted `/mnt/ms-vista-backups` Azure File Share.

Follow [AZURE_PORTAL_SETUP.md](AZURE_PORTAL_SETUP.md) to create and verify each
Azure resource before completing the host-side setup below.

## Required Azure resources

- One Linux VM with at least 2 vCPUs and 4 GiB RAM.
- A managed data disk mounted at `/srv/ms-vista`.
- An Azure Storage File Share mounted at `/mnt/ms-vista-backups`.
- A static public IP with a DNS name or a custom DNS record.
- NSG inbound rules for TCP 22, 80, and 443. Restrict SSH to administrator IPs.
- Azure Communication Services Email and a Microsoft Entra app registration.

The API deliberately refuses `DEPLOYMENT_MODE=public` when `/data/backups` and
`/backup` are on the same filesystem.

## Host setup

Run `bootstrap_host.sh` once with `sudo`. It installs Docker Engine and Compose,
enables unattended security updates and fail2ban, creates a small swap file for
the 4 GiB VM, and opens only SSH, HTTP, and HTTPS in the host firewall.

Copy `production.env.example` to the ignored `production.env` file and replace
every hostname and credential. The Entra redirect URI is:

```text
https://<hostname>/api/auth/oauth/microsoft/callback
```

Private ground-truth files belong under:

```text
/srv/ms-vista/data/ground_truths/
```

They are mounted read-only inside the API container and are excluded from the
Docker build context.

## Start and verify

```bash
cd /srv/ms-vista/app/deployment/azure
TAG=prod-$(date -u +%Y%m%dT%H%M%SZ)
sudo env MS_VISTA_IMAGE_TAG="$TAG" \
  docker compose --env-file production.env build api frontend
sudo ./deploy_release.sh "$TAG"
curl -fsS https://<hostname>/api/health/live
curl -fsS https://<hostname>/api/readiness
```

`deploy_release.sh` verifies that both versioned images exist, backs up the
protected environment, waits for all containers to become healthy, runs the
full HTTPS production smoke test, rolls back and verifies the previous release
on failure, and records the deployed image IDs under
`/srv/ms-vista/deployment-manifests`. Environment backups and manifests retain
the 20 newest entries.

After the first successful deployment, install all host units:

```bash
sudo ./install_systemd_units.sh
```

`ms-vista.service` reconciles Compose on every boot and waits for healthy
containers. The one-minute watchdog tolerates two transient failures, then
performs one bounded recovery and enforces a 15-minute cooldown. The daily
backup verifier validates the newest Azure Files archive and restores it into
an isolated temporary directory without touching live data.

On this dedicated VM, `docker-ms-vista-storage.conf` keeps Docker from restoring
the containers before the data disk and Azure File Share mounts are ready.

Inspect the controls with:

```bash
systemctl status ms-vista.service ms-vista-watchdog.timer \
  ms-vista-backup-verify.timer
journalctl -u ms-vista-watchdog.service -u ms-vista-backup-verify.service
```

Before opening registration, complete the email, OAuth, backup, and restore
acceptance checks in `docs/deployment.md`.
