# Oracle Cloud Always Free deployment

This topology runs the MS VISTA API, a fallback copy of the React frontend, and
Caddy on one Oracle Cloud Infrastructure (OCI) Ampere A1 VM. The public React
client can remain on GitHub Pages and call this VM over HTTPS with bearer
authentication.

The intended pilot shape is `VM.Standard.A1.Flex` with 2 OCPUs and 12 GiB RAM.
It fits inside the Ampere A1 Always Free allowance and is sufficient for the
current single-worker SQLite service. This is a low-traffic, single-instance
deployment, not a high-availability architecture. OCI capacity for an Always
Free shape is not guaranteed.

## 1. Create a dedicated SSH key

Run this on the administrator workstation. Do not upload or commit the private
key.

```bash
ssh-keygen -t ed25519 -a 64 \
  -f ~/.ssh/ms-vista-oci \
  -C ms-vista-oci
chmod 600 ~/.ssh/ms-vista-oci
```

Upload only `~/.ssh/ms-vista-oci.pub` while creating the instance.

## 2. Create the OCI instance

Choose the account home region carefully because Always Free compute resources
must be created there.

1. Open **Compute > Instances > Create instance**.
2. Use a name such as `ms-vista-leaderboard`.
3. Select Canonical Ubuntu 24.04.
4. Select `VM.Standard.A1.Flex`, 2 OCPUs, and 12 GiB memory. Confirm that the
   console labels the selection **Always Free eligible** before creating it.
5. Create or select a VCN with a public subnet and assign a public IPv4 address.
6. Use a 50 GiB boot volume.
7. Add the public SSH key created above.
8. Leave confidential-computing and preemptible-instance options disabled.

Reserve the assigned public IP before publishing DNS so an instance stop or
replacement cannot silently change the API address.

## 3. Configure ingress

Add these stateful ingress rules to the subnet security list or the instance's
network security group. Keep the instance firewall enabled as a second layer.

| Source | Protocol | Destination port | Purpose |
|---|---|---:|---|
| administrator public IP `/32` | TCP | 22 | SSH administration |
| `0.0.0.0/0` | TCP | 80 | ACME validation and HTTPS redirect |
| `0.0.0.0/0` | TCP | 443 | HTTPS API and fallback frontend |
| `0.0.0.0/0` | UDP | 443 | Optional HTTP/3 |

Do not expose ports `5050`, `7860`, or `8080`; they remain on the private Docker
network. Add equivalent IPv6 rules only if IPv6 is intentionally enabled.

## 4. Attach the backup volume

Create a 50 GiB block volume in the same availability domain and attach it to
the instance as read/write, paravirtualized storage. This volume is separate
from the boot filesystem so the API's backup readiness checks can detect loss
of the backup mount.

For an additional recovery layer, create a custom block-volume backup policy
with one weekly incremental schedule retained for four weeks and assign it only
to this backup volume. Do not enable cross-region copy for the free deployment.
Four retained backups stay below the five-volume-backup Always Free allowance;
confirm the console still marks every resource Always Free eligible.

Connect to the instance. The default Ubuntu image user is `ubuntu`:

```bash
ssh -i ~/.ssh/ms-vista-oci ubuntu@<reserved-public-ip>
```

Identify the new, empty device before formatting it:

```bash
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS
ls -l /dev/oracleoci/
```

Set `BACKUP_DEVICE` to the attached empty volume. Never run `mkfs` against the
boot disk or a device that already contains data.

```bash
BACKUP_DEVICE=/dev/oracleoci/<attached-volume-device>
test -b "$BACKUP_DEVICE"
test -z "$(lsblk -n -o FSTYPE "$BACKUP_DEVICE")"
sudo mkfs.ext4 -L ms-vista-backups "$BACKUP_DEVICE"
sudo install -d -m 0700 /mnt/ms-vista-backups
BACKUP_UUID=$(sudo blkid -s UUID -o value "$BACKUP_DEVICE")
echo "UUID=$BACKUP_UUID /mnt/ms-vista-backups ext4 defaults,nofail,x-systemd.device-timeout=30 0 2" \
  | sudo tee -a /etc/fstab
sudo mount -a
findmnt /mnt/ms-vista-backups
```

## 5. Install the application

Clone the full repository so release manifests can record an exact Git commit.
The symlink keeps the provider-independent application path at
`/srv/ms-vista/app`.

```bash
sudo apt-get update
sudo apt-get install -y git
git clone https://github.com/riteshthawkar/leaderboard.git ~/leaderboard-bootstrap
sudo ~/leaderboard-bootstrap/Combined-Leaderboard/deployment/oci/bootstrap_host.sh
mv ~/leaderboard-bootstrap /srv/ms-vista/repository
ln -s /srv/ms-vista/repository/Combined-Leaderboard /srv/ms-vista/app
sudo chown -R ubuntu:ubuntu /mnt/ms-vista-backups
sudo chmod 0700 /mnt/ms-vista-backups
```

Log out and reconnect after bootstrap so the `docker` group change takes
effect. Then validate the host:

```bash
cd /srv/ms-vista/app/deployment/oci
./check_host.sh
```

## 6. Install protected state

Do not commit the production environment, private answer keys, database, or
backups. Copy them over SSH to a temporary directory and install them with
restricted permissions.

From the administrator workstation:

```bash
scp -i ~/.ssh/ms-vista-oci production.env \
  ubuntu@<reserved-public-ip>:/tmp/ms-vista-production.env
scp -i ~/.ssh/ms-vista-oci -r private-ground-truth \
  ubuntu@<reserved-public-ip>:/tmp/ms-vista-private-ground-truth
scp -i ~/.ssh/ms-vista-oci leaderboard.db \
  ubuntu@<reserved-public-ip>:/tmp/ms-vista-leaderboard.db
```

On the VM:

```bash
sudo install -m 0600 -o root -g root /tmp/ms-vista-production.env \
  /srv/ms-vista/app/deployment/oci/production.env
sudo rsync -a --delete /tmp/ms-vista-private-ground-truth/ \
  /srv/ms-vista/data/ground_truths/
sudo install -m 0600 -o ubuntu -g ubuntu /tmp/ms-vista-leaderboard.db \
  /srv/ms-vista/data/leaderboard.db
sudo chown -R ubuntu:ubuntu /srv/ms-vista/data
sudo chmod 0700 /srv/ms-vista/data /srv/ms-vista/data/ground_truths
rm -rf /tmp/ms-vista-production.env /tmp/ms-vista-private-ground-truth \
  /tmp/ms-vista-leaderboard.db
```

The local database may be older than the inaccessible Azure VM database. Do
not open registration until the migration source has been chosen explicitly;
otherwise newer Azure accounts or submissions could be lost.

## 7. Configure production variables

Start from `production.env.example`. At minimum replace:

- `MS_VISTA_DOMAIN`, `API_BASE_URL`, and `OAUTH_REDIRECT_BASE_URL` with the OCI
  API hostname.
- `ACME_EMAIL` and `ADMIN_EMAILS` with operational addresses.
- `SECRET_KEY` with the existing production secret when migrating sessions, or
  a new output from `python3 -c 'import secrets; print(secrets.token_hex(32))'`
  for a fresh deployment.
- Microsoft Entra client ID, secret, and tenant ID.
- Azure Communication Services Email connection string and the verified sender
  address. ACS Email works from OCI; no Azure VM identity is required in
  connection-string mode.

Keep these values for the GitHub Pages split deployment:

```env
FRONTEND_BASE_URL=https://riteshthawkar.github.io/leaderboard
CORS_ORIGINS=https://riteshthawkar.github.io,https://<oci-api-host>
AUTH_TRANSPORT=dual
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_SAMESITE=Lax
```

Register this exact Microsoft Entra redirect URI:

```text
https://<oci-api-host>/api/auth/oauth/microsoft/callback
```

For initial TLS, a temporary `<dashed-ip>.sslip.io` hostname can be used. Use a
controlled custom domain before treating the deployment as final production.

## 8. Build, deploy, and install recovery services

```bash
cd /srv/ms-vista/app/deployment/oci
TAG=prod-$(date -u +%Y%m%dT%H%M%SZ)
sudo env MS_VISTA_IMAGE_TAG="$TAG" \
  docker compose --env-file production.env build api frontend
git -C /srv/ms-vista/repository rev-parse HEAD \
  | sudo tee /srv/ms-vista/app/.release-commit >/dev/null
sudo chmod 0644 /srv/ms-vista/app/.release-commit
sudo ./deploy_release.sh "$TAG"
sudo ./install_systemd_units.sh
```

The release script validates images, waits for all services, runs the HTTPS
production smoke check, and restores the previous version if deployment fails.
The systemd units reconcile the stack after reboot, run a bounded watchdog, and
perform a daily restore test of the latest mirrored backup.

## 9. Verify and cut over GitHub Pages

```bash
curl -fsS https://<oci-api-host>/api/health/live
curl -fsS https://<oci-api-host>/api/readiness
python3 /srv/ms-vista/app/scripts/production_smoke.py \
  --api-url https://<oci-api-host> \
  --frontend-url https://riteshthawkar.github.io/leaderboard
sudo systemctl status ms-vista.service ms-vista-watchdog.timer \
  ms-vista-backup-verify.timer --no-pager
```

Complete one real Microsoft login, email verification, password reset, and
authenticated submission using controlled accounts. Then update the GitHub
Actions repository variable `PAGES_API_BASE_URL` to the OCI HTTPS origin and
set `PAGES_DEPLOY_ENABLED=true`.

## Operations

Update and deploy a new revision:

```bash
cd /srv/ms-vista/repository
git fetch origin
git checkout main
git pull --ff-only origin main
cd Combined-Leaderboard/deployment/oci
./check_host.sh
TAG=prod-$(date -u +%Y%m%dT%H%M%SZ)
sudo env MS_VISTA_IMAGE_TAG="$TAG" \
  docker compose --env-file production.env build api frontend
git -C /srv/ms-vista/repository rev-parse HEAD \
  | sudo tee /srv/ms-vista/app/.release-commit >/dev/null
sudo ./deploy_release.sh "$TAG"
```

Inspect status and logs:

```bash
cd /srv/ms-vista/app/deployment/oci
sudo docker compose --env-file production.env ps
sudo docker compose --env-file production.env logs --tail=200 api caddy
sudo journalctl -u ms-vista-watchdog.service \
  -u ms-vista-backup-verify.service --since today
```

Do not delete the boot or backup volume during an instance replacement. Before
any destructive infrastructure change, download a recently restore-tested
backup archive to a separate administrative location.

## Oracle references

- [Always Free resource limits](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)
- [Create the first Linux instance](https://docs.oracle.com/en-us/iaas/Content/Compute/tutorials/first-linux-instance/overview.htm)
- [Attach a block volume](https://docs.oracle.com/en-us/iaas/Content/Block/Tasks/attach-compute-volume-attachment.htm)
- [Create and assign a custom backup policy](https://docs.oracle.com/en-us/iaas/Content/Block/Tasks/create-bv-volume-backup-policy.htm)
