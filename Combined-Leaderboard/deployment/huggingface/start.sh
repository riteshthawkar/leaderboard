#!/bin/sh
set -eu

fail() {
  printf '%s\n' "Hugging Face startup failed: $*" >&2
  exit 1
}

if [ -n "${SPACE_HOST:-}" ]; then
  case "$SPACE_HOST" in
    *[!A-Za-z0-9.-]*) fail "SPACE_HOST contains unsupported characters" ;;
  esac
  public_origin="https://${SPACE_HOST}"
  export FRONTEND_BASE_URL="${FRONTEND_BASE_URL:-$public_origin}"
  export API_BASE_URL="${API_BASE_URL:-$public_origin}"
  export OAUTH_REDIRECT_BASE_URL="${OAUTH_REDIRECT_BASE_URL:-$public_origin}"
  export CORS_ORIGINS="${CORS_ORIGINS:-$public_origin}"
  export PRIVACY_POLICY_URL="${PRIVACY_POLICY_URL:-$public_origin/privacy}"
fi

for variable in SECRET_KEY ADMIN_EMAILS GROUND_TRUTHS_HF_REPO HF_TOKEN FRONTEND_BASE_URL API_BASE_URL OAUTH_REDIRECT_BASE_URL CORS_ORIGINS PRIVACY_POLICY_URL
do
  eval "value=\${$variable:-}"
  [ -n "$value" ] || fail "$variable is required"
done

if [ -z "${ACS_CONNECTION_STRING:-}" ] || [ -z "${ACS_SENDER_ADDRESS:-}" ]; then
  if [ -z "${SMTP_HOST:-}" ] || [ -z "${SMTP_FROM:-}" ]; then
    fail "configure Azure Communication Services Email or SMTP"
  fi
fi

if [ "${HF_REQUIRE_PERSISTENT_VOLUMES:-true}" = "true" ]; then
  mountpoint -q /data || fail "attach a writable Hugging Face bucket at /data"
  mountpoint -q /backup || fail "attach a second writable Hugging Face bucket at /backup"
fi

umask 077
for directory in /data /data/results /data/backups /data/logs /data/ground_truths_cache /data/hf_cache /backup /tmp/ms-vista-client-body /tmp/ms-vista-proxy
do
  mkdir -p "$directory"
  test_file="$directory/.ms-vista-write-check"
  : >"$test_file" || fail "$directory is not writable by UID 1000"
  rm -f "$test_file"
done

exec /usr/bin/supervisord -c /home/user/app/deployment/huggingface/supervisord.conf
