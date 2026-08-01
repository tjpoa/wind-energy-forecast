"""Sanitized manual command dispatcher for the operational projection."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from wind_forecast.config import (
    load_monitoring_store_config,
    load_operational_projection_database_config,
)
from wind_forecast.operational_projection_migrations import main as migration_main
from wind_forecast.operational_projection_projector import (
    ERROR_SCHEMA_VERSION,
    OperationalProjectionError,
    plan_projection,
    project_projection,
    resolve_source_git_commit,
    verify_projection,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage the dedicated operational PostgreSQL projection."
    )
    parser.add_argument(
        "command",
        choices=("migration-status", "migrate", "plan", "project", "verify"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command in {"migration-status", "migrate"}:
        return migration_main([args.command])
    try:
        role = "writer" if args.command == "project" else "reader"
        database = load_operational_projection_database_config(role)
        store = load_monitoring_store_config()
        source_git_commit = resolve_source_git_commit()
        if args.command == "plan":
            result = plan_projection(
                database.dsn,
                store.store_root,
                environment_id=database.environment_id,
                source_git_commit=source_git_commit,
            )
        elif args.command == "project":
            result = project_projection(
                database.dsn,
                store.store_root,
                environment_id=database.environment_id,
                source_git_commit=source_git_commit,
            )
        else:
            result = verify_projection(
                database.dsn,
                store.store_root,
                environment_id=database.environment_id,
                source_git_commit=source_git_commit,
            )
        print(json.dumps(result.summary(), sort_keys=True))
        return 0 if result.status not in {
            "missing",
            "stale",
            "mismatch",
            "incompatible",
        } else 1
    except OperationalProjectionError as exc:
        return _print_error(exc.code)
    except (OSError, ValueError):
        return _print_error("configuration_error")


def _print_error(code: str) -> int:
    print(
        json.dumps(
            {
                "schema_version": ERROR_SCHEMA_VERSION,
                "status": "failed",
                "error_code": code,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 1


__all__ = ["main", "parse_args"]
