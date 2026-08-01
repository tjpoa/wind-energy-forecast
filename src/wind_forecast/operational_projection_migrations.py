"""Checksummed, forward-only migrations for the operational projection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from wind_forecast.config import load_operational_projection_database_config
from wind_forecast.paths import project_root


MIGRATIONS_ROOT = project_root() / "operational_projection" / "migrations"
MIGRATION_PATTERN = re.compile(r"^(?P<version>[0-9]{4})_(?P<name>[a-z][a-z0-9_]*)\.sql$")
MIGRATION_LOCK_KEY = 7_318_041_256_746_211_919
STATUS_SCHEMA_VERSION = "wind_forecast.operational_projection.migration_status.v1"
ERROR_SCHEMA_VERSION = "wind_forecast.operational_projection.migration_error.v1"


class OperationalProjectionMigrationError(RuntimeError):
    """Base class for sanitized migration failures."""

    code = "migration_failed"


class MigrationDefinitionError(OperationalProjectionMigrationError):
    """Raised when bundled migration files do not form a valid sequence."""

    code = "invalid_migration_definition"


class MigrationIncompatibleError(OperationalProjectionMigrationError):
    """Raised when the database ledger is not supported by this checkout."""

    code = "incompatible_schema"


class MigrationChecksumError(OperationalProjectionMigrationError):
    """Raised when an applied migration has changed."""

    code = "migration_checksum_mismatch"


class MigrationDatabaseError(OperationalProjectionMigrationError):
    """Raised when PostgreSQL cannot safely complete the requested operation."""

    code = "database_unavailable"


@dataclass(frozen=True)
class Migration:
    """One immutable SQL migration discovered from the approved directory."""

    version: int
    name: str
    sha256: str
    sql: str


@dataclass(frozen=True)
class AppliedMigration:
    """One migration recorded in the database ledger."""

    version: int
    name: str
    sha256: str


@dataclass(frozen=True)
class MigrationStatus:
    """Sanitized comparison between local migrations and the database ledger."""

    state: str
    applied: tuple[AppliedMigration, ...]
    pending: tuple[Migration, ...]

    def summary(self, *, command: str, environment_id: str) -> dict[str, Any]:
        return {
            "schema_version": STATUS_SCHEMA_VERSION,
            "command": command,
            "status": self.state,
            "environment_id": environment_id,
            "applied_versions": [item.version for item in self.applied],
            "pending_versions": [item.version for item in self.pending],
        }


def discover_migrations(root: Path = MIGRATIONS_ROOT) -> tuple[Migration, ...]:
    """Load a contiguous migration sequence and checksum its original bytes."""
    try:
        paths = sorted(path for path in root.iterdir() if path.is_file())
    except OSError as exc:
        raise MigrationDefinitionError("Migration directory is unavailable.") from exc

    migrations: list[Migration] = []
    names: set[str] = set()
    for path in paths:
        match = MIGRATION_PATTERN.fullmatch(path.name)
        if match is None:
            raise MigrationDefinitionError("Migration filename is invalid.")
        version = int(match.group("version"))
        name = match.group("name")
        if name in names:
            raise MigrationDefinitionError("Migration name is duplicated.")
        try:
            payload = path.read_bytes()
            sql_text = payload.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise MigrationDefinitionError("Migration file is unreadable.") from exc
        if not sql_text.strip():
            raise MigrationDefinitionError("Migration file is empty.")
        migrations.append(
            Migration(
                version=version,
                name=name,
                sha256=sha256(payload).hexdigest(),
                sql=sql_text,
            )
        )
        names.add(name)

    expected_versions = list(range(1, len(migrations) + 1))
    if [item.version for item in migrations] != expected_versions:
        raise MigrationDefinitionError("Migration versions must be contiguous.")
    return tuple(migrations)


def migration_status(dsn: str) -> MigrationStatus:
    """Compare the database ledger with local migrations without writing."""
    migrations = discover_migrations()
    connection = _connect(dsn)
    try:
        applied = _read_applied(connection, read_only=True)
        return _compare(migrations, applied)
    finally:
        connection.close()


def migrate(dsn: str) -> tuple[MigrationStatus, tuple[int, ...]]:
    """Apply every pending migration under one session advisory lock."""
    migrations = discover_migrations()
    connection = _connect(dsn)
    locked = False
    applied_now: list[int] = []
    try:
        _acquire_lock(connection)
        locked = True
        initial = _compare(migrations, _read_applied(connection, read_only=True))
        for migration in initial.pending:
            _apply_migration(connection, migration)
            applied_now.append(migration.version)
        final = _compare(migrations, _read_applied(connection, read_only=True))
        return final, tuple(applied_now)
    except OperationalProjectionMigrationError:
        raise
    except Exception as exc:
        raise MigrationDatabaseError(
            "PostgreSQL could not complete the migration operation."
        ) from exc
    finally:
        if locked:
            _release_lock(connection)
        connection.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or apply the dedicated operational projection schema."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("migration-status")
    subparsers.add_parser("migrate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_operational_projection_database_config("migrator")
        if args.command == "migration-status":
            status = migration_status(config.dsn)
            payload = status.summary(
                command=args.command,
                environment_id=config.environment_id,
            )
        else:
            status, applied_now = migrate(config.dsn)
            payload = status.summary(
                command=args.command,
                environment_id=config.environment_id,
            )
            payload["applied_now"] = list(applied_now)
        print(json.dumps(payload, sort_keys=True))
        return 0
    except OperationalProjectionMigrationError as exc:
        return _print_error(exc.code)
    except (OSError, ValueError):
        return _print_error("configuration_error")


def _connect(dsn: str) -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise MigrationDatabaseError("PostgreSQL driver is unavailable.") from exc
    try:
        connection = psycopg.connect(
            dsn,
            autocommit=True,
            application_name="wind_forecast_projection_migrator",
            connect_timeout=5,
        )
        with connection.cursor() as cursor:
            cursor.execute("SET TIME ZONE 'UTC'")
            cursor.execute("SET statement_timeout = '30s'")
            cursor.execute("SET lock_timeout = '5s'")
        return connection
    except Exception as exc:
        raise MigrationDatabaseError("PostgreSQL is unavailable.") from exc


def _read_applied(connection: Any, *, read_only: bool) -> tuple[AppliedMigration, ...]:
    try:
        with connection.transaction():
            with connection.cursor() as cursor:
                if read_only:
                    cursor.execute("SET TRANSACTION READ ONLY")
                _set_migration_role(cursor)
                cursor.execute(
                    "SELECT to_regclass('operational_projection.schema_migration')"
                )
                if cursor.fetchone()[0] is None:
                    return ()
                cursor.execute(
                    "SELECT version, name, sha256 "
                    "FROM operational_projection.schema_migration "
                    "ORDER BY version"
                )
                return tuple(
                    AppliedMigration(int(version), str(name), str(checksum))
                    for version, name, checksum in cursor.fetchall()
                )
    except OperationalProjectionMigrationError:
        raise
    except Exception as exc:
        raise MigrationDatabaseError("Migration ledger is unavailable.") from exc


def _compare(
    migrations: tuple[Migration, ...],
    applied: tuple[AppliedMigration, ...],
) -> MigrationStatus:
    expected_prefix = migrations[: len(applied)]
    if [item.version for item in applied] != [item.version for item in expected_prefix]:
        raise MigrationIncompatibleError("Applied migration sequence is unsupported.")
    for recorded, local in zip(applied, expected_prefix, strict=True):
        if recorded.name != local.name:
            raise MigrationIncompatibleError("Applied migration identity is unsupported.")
        if recorded.sha256 != local.sha256:
            raise MigrationChecksumError("Applied migration checksum has changed.")
    pending = migrations[len(applied) :]
    return MigrationStatus(
        state="current" if not pending else "pending",
        applied=applied,
        pending=pending,
    )


def _apply_migration(connection: Any, migration: Migration) -> None:
    try:
        with connection.transaction():
            with connection.cursor() as cursor:
                _set_migration_role(cursor)
                cursor.execute(migration.sql, prepare=False)
                cursor.execute(
                    "INSERT INTO operational_projection.schema_migration "
                    "(version, name, sha256, applied_at_utc) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        migration.version,
                        migration.name,
                        migration.sha256,
                        datetime.now(timezone.utc),
                    ),
                )
    except Exception as exc:
        raise MigrationDatabaseError("Migration could not be applied.") from exc


def _set_migration_role(cursor: Any) -> None:
    cursor.execute("SET LOCAL ROLE wf_projection_owner")
    cursor.execute("SET LOCAL search_path TO operational_projection, pg_catalog")


def _acquire_lock(connection: Any) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
    except Exception as exc:
        raise MigrationDatabaseError("Migration lock is unavailable.") from exc


def _release_lock(connection: Any) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
    except Exception:
        pass


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


__all__ = [
    "AppliedMigration",
    "Migration",
    "MigrationChecksumError",
    "MigrationDatabaseError",
    "MigrationDefinitionError",
    "MigrationIncompatibleError",
    "MigrationStatus",
    "OperationalProjectionMigrationError",
    "discover_migrations",
    "main",
    "migrate",
    "migration_status",
    "parse_args",
]
