#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

DEPLOY_USER="${SUDO_USER:-azureuser}"
if ! id "${DEPLOY_USER}" >/dev/null 2>&1; then
  echo "Deployment user ${DEPLOY_USER} does not exist." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  cifs-utils \
  curl \
  fail2ban \
  git \
  gnupg \
  jq \
  rsync \
  sqlite3 \
  unattended-upgrades \
  ufw

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

ARCH="$(dpkg --print-architecture)"
. /etc/os-release
cat >/etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${VERSION_CODENAME}
Components: stable
Architectures: ${ARCH}
Signed-By: /etc/apt/keyrings/docker.gpg
EOF

apt-get update
apt-get install -y --no-install-recommends \
  containerd.io \
  docker-buildx-plugin \
  docker-ce \
  docker-ce-cli \
  docker-compose-plugin

install -d -m 0755 /etc/docker
cat >/etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "5"
  },
  "live-restore": true
}
EOF

usermod -aG docker "${DEPLOY_USER}"
install -d -o "${DEPLOY_USER}" -g "${DEPLOY_USER}" -m 0750 /srv/ms-vista
install -d -o "${DEPLOY_USER}" -g "${DEPLOY_USER}" -m 0700 \
  /srv/ms-vista/data \
  /srv/ms-vista/data/backups \
  /srv/ms-vista/data/ground_truths \
  /srv/ms-vista/data/logs \
  /srv/ms-vista/data/results \
  /mnt/ms-vista-backups

if [[ ! -e /swapfile ]]; then
  fallocate -l 2G /swapfile
  chmod 0600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >>/etc/fstab
fi
cat >/etc/sysctl.d/99-ms-vista.conf <<'EOF'
vm.swappiness=10
EOF
sysctl --system >/dev/null

cat >/etc/fail2ban/jail.d/sshd.local <<'EOF'
[sshd]
enabled = true
maxretry = 5
findtime = 10m
bantime = 1h
EOF

ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp
ufw --force enable

systemctl enable --now docker
systemctl enable --now fail2ban
systemctl enable --now unattended-upgrades

echo "Host bootstrap complete. Log out and back in before using Docker without sudo."
