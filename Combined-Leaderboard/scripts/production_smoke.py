"""External production smoke checks for the separately hosted API and frontend."""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


def _request(url: str, *, origin: str | None = None) -> tuple[int, dict, bytes]:
    headers = {"Accept": "application/json, text/html;q=0.9"}
    if origin:
        headers["Origin"] = origin
    request = Request(url, headers=headers)
    with urlopen(request, timeout=15) as response:
        return response.status, dict(response.headers.items()), response.read()


def _json(url: str, *, origin: str | None = None) -> tuple[int, dict, dict]:
    status, headers, body = _request(url, origin=origin)
    return status, headers, json.loads(body.decode("utf-8"))


def _origin(value: str) -> str:
    parsed = urlparse(value.rstrip("/"))
    return f"{parsed.scheme}://{parsed.netloc}"


def run(api_url: str, frontend_url: str, *, allow_http: bool, require_spatial: bool) -> dict:
    api_url = api_url.rstrip("/") + "/"
    frontend_url = frontend_url.rstrip("/") + "/"
    api_origin = _origin(api_url)
    frontend_origin = _origin(frontend_url)
    if not allow_http and (
        urlparse(api_origin).scheme != "https" or urlparse(frontend_origin).scheme != "https"
    ):
        raise RuntimeError("Production smoke checks require HTTPS origins. Use --allow-http only locally.")

    checks = {}
    live_status, _live_headers, live = _json(urljoin(api_url, "api/health/live"), origin=frontend_origin)
    checks["api_liveness"] = live_status == 200 and live.get("status") == "alive"

    readiness_status, readiness_headers, readiness = _json(
        urljoin(api_url, "api/readiness"),
        origin=frontend_origin,
    )
    deployment = readiness.get("details", {}).get("deployment", {})
    auth = readiness.get("details", {}).get("auth", {})
    backup = readiness.get("details", {}).get("backup", {})
    checks["api_readiness"] = readiness_status == 200 and readiness.get("status") == "healthy"
    checks["cors"] = readiness_headers.get("Access-Control-Allow-Origin") == frontend_origin
    checks["public_deployment"] = allow_http or deployment.get("public_deployment_ready") is True
    checks["verified_admin"] = allow_http or auth.get("admin_ready") is True
    checks["offsite_backup"] = allow_http or (
        backup.get("mirror_required") is True
        and backup.get("mirror_configured") is True
        and backup.get("mirror_separate_filesystem") is True
    )
    if require_spatial:
        checks["spatial_bundle"] = readiness.get("components", {}).get("spatial_bundle") == "healthy"

    providers_status, _provider_headers, providers = _json(
        urljoin(api_url, "api/auth/providers"),
        origin=frontend_origin,
    )
    provider_ids = {provider.get("id") for provider in providers.get("providers", [])}
    checks["microsoft_oauth_configured"] = providers_status == 200 and "microsoft" in provider_ids

    frontend_status, frontend_headers, _frontend_body = _request(frontend_url)
    normalized_headers = {key.lower(): value for key, value in frontend_headers.items()}
    checks["frontend_available"] = frontend_status == 200
    checks["frontend_csp"] = "content-security-policy" in normalized_headers
    checks["frontend_hsts"] = allow_http or "strict-transport-security" in normalized_headers
    checks["frontend_permissions_policy"] = "permissions-policy" in normalized_headers

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "passed" if not failed else "failed",
        "api_origin": api_origin,
        "frontend_origin": frontend_origin,
        "checks": checks,
        "failed_checks": failed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test an MS-VISTA production deployment.")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--frontend-url", required=True)
    parser.add_argument("--allow-http", action="store_true", help="Allow local HTTP origins.")
    parser.add_argument("--require-spatial", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(
            args.api_url,
            args.frontend_url,
            allow_http=args.allow_http,
            require_spatial=args.require_spatial,
        )
    except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as exc:
        result = {"status": "failed", "error": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
