"""Configure scheduler ownership and manage explicit environment leases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from wind_forecast.scheduler_ownership import (
    SCHEDULER_OWNERS,
    acquire_scheduler_lease,
    configure_scheduler_owner,
    load_scheduler_owner,
    recover_scheduler_lease,
    release_scheduler_lease,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage one fail-closed scheduler owner per environment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scheduler-root", type=Path, required=True)
    common.add_argument("--environment-id", required=True)

    configure = subparsers.add_parser("configure", parents=[common])
    configure.add_argument("--owner", choices=SCHEDULER_OWNERS, required=True)
    configure.add_argument("--expected-generation", type=int, required=True)
    expected = configure.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-owner", choices=SCHEDULER_OWNERS)
    expected.add_argument("--expect-no-owner", action="store_true")
    configure.add_argument("--dry-run", action="store_true")

    acquire = subparsers.add_parser("acquire", parents=[common])
    acquire.add_argument("--scheduler", choices=SCHEDULER_OWNERS, required=True)
    acquire.add_argument("--workflow", required=True)
    acquire.add_argument("--run-id", required=True)

    verify = subparsers.add_parser("verify", parents=[common])
    verify.add_argument("--scheduler", choices=SCHEDULER_OWNERS, required=True)

    release = subparsers.add_parser("release", parents=[common])
    release.add_argument("--lease-id", required=True)

    recover = subparsers.add_parser("recover", parents=[common])
    recover.add_argument("--lease-id", required=True)
    recover.add_argument("--recovered-by", required=True)
    recover.add_argument("--note", required=True)
    recover.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "configure":
        result = configure_scheduler_owner(
            args.scheduler_root,
            args.environment_id,
            args.owner,
            expected_generation=args.expected_generation,
            expected_owner=None if args.expect_no_owner else args.expected_owner,
            dry_run=args.dry_run,
        )
        payload = {"status": "planned" if args.dry_run else "configured", **result.to_dict()}
    elif args.command == "verify":
        result = load_scheduler_owner(
            args.scheduler_root,
            args.environment_id,
        )
        if result is None or result.active_scheduler != args.scheduler:
            observed = None if result is None else result.active_scheduler
            raise RuntimeError(
                f"Scheduler owner is {observed!r}, not {args.scheduler!r}."
            )
        payload = {"status": "verified", **result.to_dict()}
    elif args.command == "acquire":
        result = acquire_scheduler_lease(
            args.scheduler_root,
            args.environment_id,
            args.scheduler,
            workflow=args.workflow,
            run_id=args.run_id,
        )
        payload = {"status": "acquired", **result.to_dict()}
    elif args.command == "release":
        release_scheduler_lease(
            args.scheduler_root,
            args.environment_id,
            args.lease_id,
        )
        payload = {"status": "released", "lease_id": args.lease_id}
    else:
        path = recover_scheduler_lease(
            args.scheduler_root,
            args.environment_id,
            args.lease_id,
            recovered_by=args.recovered_by,
            note=args.note,
            dry_run=args.dry_run,
        )
        payload = {
            "status": "planned" if args.dry_run else "recovered",
            "lease_id": args.lease_id,
            "recovery_path": None if path is None else str(path),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
