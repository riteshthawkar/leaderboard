"""Offline backup verification and restore drill commands."""

import argparse
import io
import json
from pathlib import Path

try:
    from backup import restore_backup_archive, validate_backup_archive
except ImportError:  # pragma: no cover - package import fallback
    from .backup import restore_backup_archive, validate_backup_archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify or restore an MS-VISTA backup archive.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="Validate ZIP and SQLite integrity.")
    verify.add_argument("archive", type=Path)

    restore = subparsers.add_parser("restore", help="Unpack into an offline recovery directory.")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        payload = args.archive.expanduser().resolve().read_bytes()
        result = validate_backup_archive(io.BytesIO(payload))
        output = {
            "archive": args.archive.name,
            "sqlite_snapshots": result["sqlite_snapshots"],
            "zip_crc": "ok",
            "sqlite_quick_check": "ok",
        }
    else:
        output = restore_backup_archive(
            args.archive,
            args.destination,
            overwrite=args.overwrite,
        )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
