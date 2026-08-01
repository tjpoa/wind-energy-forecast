from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from wind_forecast.config import (
    OPERATIONAL_ENVIRONMENT_ID_ENV,
    OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV,
    OPERATIONAL_PROJECTION_READER_DSN_ENV,
    OPERATIONAL_PROJECTION_WRITER_DSN_ENV,
    load_operational_projection_database_config,
)
from wind_forecast import operational_projection_migrations as migrations
from wind_forecast import operational_projection_projector as projector


INTEGRATION_FLAG = "WIND_FORECAST_OPERATIONAL_PROJECTION_TEST_INTEGRATION"


def test_projection_config_loads_only_selected_role(monkeypatch) -> None:
    monkeypatch.setenv(OPERATIONAL_ENVIRONMENT_ID_ENV, "local")
    monkeypatch.setenv(OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV, "postgresql://migrator")
    monkeypatch.delenv(OPERATIONAL_PROJECTION_WRITER_DSN_ENV, raising=False)
    monkeypatch.delenv(OPERATIONAL_PROJECTION_READER_DSN_ENV, raising=False)

    config = load_operational_projection_database_config("migrator")

    assert config.environment_id == "local"
    assert config.role == "migrator"
    assert config.dsn == "postgresql://migrator"


@pytest.mark.parametrize("environment_id", ["", "staging", "LOCAL"])
def test_projection_config_rejects_non_local_environment(
    monkeypatch, environment_id: str
) -> None:
    monkeypatch.setenv(OPERATIONAL_ENVIRONMENT_ID_ENV, environment_id)
    monkeypatch.setenv(OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV, "secret-dsn")

    with pytest.raises(ValueError, match=OPERATIONAL_ENVIRONMENT_ID_ENV) as exc_info:
        load_operational_projection_database_config("migrator")

    assert "secret-dsn" not in str(exc_info.value)


def test_projection_config_rejects_missing_role_dsn(monkeypatch) -> None:
    monkeypatch.delenv(OPERATIONAL_PROJECTION_READER_DSN_ENV, raising=False)

    with pytest.raises(ValueError, match=OPERATIONAL_PROJECTION_READER_DSN_ENV):
        load_operational_projection_database_config("reader")


def test_imports_do_not_import_psycopg_or_connect() -> None:
    code = (
        "import sys; "
        "import wind_forecast.config; "
        "import wind_forecast.operational_projection_migrations; "
        "import wind_forecast.operational_projection_projector; "
        "assert 'psycopg' not in sys.modules; "
        "assert 'joblib' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _patch_empty_projector_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(projector, "load_monitoring_report_state", lambda _root: None)
    monkeypatch.setattr(projector, "load_active_alerts", lambda _root: {})
    monkeypatch.setattr(projector, "load_alert_history", lambda _root: [])
    monkeypatch.setattr(projector, "load_reporting_attempts", lambda _root: [])


def test_discover_migrations_is_contiguous_and_hashes_original_bytes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "0001_first.sql"
    second = tmp_path / "0002_second.sql"
    first.write_bytes(b"SELECT 1;\n")
    second.write_bytes(b"SELECT 2;\n")

    discovered = migrations.discover_migrations(tmp_path)

    assert [item.version for item in discovered] == [1, 2]
    assert discovered[0].sha256 == sha256(first.read_bytes()).hexdigest()
    assert discovered[1].sql == "SELECT 2;\n"


@pytest.mark.parametrize(
    "filenames",
    [
        ("0002_gap.sql",),
        ("0001_same.sql", "0002_same.sql"),
        ("0001_valid.sql", "README.txt"),
    ],
)
def test_discover_migrations_rejects_invalid_sequences(
    tmp_path: Path, filenames: tuple[str, ...]
) -> None:
    for filename in filenames:
        (tmp_path / filename).write_text("SELECT 1;\n", encoding="utf-8")

    with pytest.raises(migrations.MigrationDefinitionError):
        migrations.discover_migrations(tmp_path)


def test_compare_rejects_changed_checksum_and_future_version() -> None:
    local = (
        migrations.Migration(1, "first", "a" * 64, "SELECT 1;"),
        migrations.Migration(2, "second", "b" * 64, "SELECT 2;"),
    )

    with pytest.raises(migrations.MigrationChecksumError):
        migrations._compare(
            local,
            (migrations.AppliedMigration(1, "first", "c" * 64),),
        )
    with pytest.raises(migrations.MigrationIncompatibleError):
        migrations._compare(
            local,
            (
                migrations.AppliedMigration(1, "first", "a" * 64),
                migrations.AppliedMigration(2, "second", "b" * 64),
                migrations.AppliedMigration(3, "future", "c" * 64),
            ),
        )


def test_cli_failure_is_sanitized(monkeypatch, capsys) -> None:
    secret_dsn = "postgresql://user:do-not-print@example.test/database"
    monkeypatch.setenv(OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV, secret_dsn)

    def fail(_dsn: str) -> migrations.MigrationStatus:
        raise migrations.MigrationDatabaseError(f"raw failure for {secret_dsn}")

    monkeypatch.setattr(migrations, "migration_status", fail)

    assert migrations.main(["migration-status"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert captured.out == ""
    assert payload["error_code"] == "database_unavailable"
    assert secret_dsn not in captured.err
    assert "raw failure" not in captured.err


@pytest.fixture(scope="module")
def projection_dsns() -> dict[str, str]:
    if os.getenv(INTEGRATION_FLAG) != "1":
        pytest.skip("PostgreSQL integration test was not explicitly enabled.")
    variables = {
        "migrator": OPERATIONAL_PROJECTION_MIGRATOR_DSN_ENV,
        "writer": OPERATIONAL_PROJECTION_WRITER_DSN_ENV,
        "reader": OPERATIONAL_PROJECTION_READER_DSN_ENV,
    }
    values = {role: os.getenv(variable, "") for role, variable in variables.items()}
    if any(not value for value in values.values()):
        pytest.fail("Explicit integration mode requires all three test DSNs.")
    return values


def test_postgres_migrations_schema_roles_and_integrity(
    projection_dsns: dict[str, str],
) -> None:
    import psycopg
    from psycopg import errors

    migrator_dsn = projection_dsns["migrator"]
    initial = migrations.migration_status(migrator_dsn)
    assert initial.state == "pending"
    assert [item.version for item in initial.pending] == [1, 2]

    current, applied_now = migrations.migrate(migrator_dsn)
    assert current.state == "current"
    assert applied_now == (1, 2)
    repeated, repeated_now = migrations.migrate(migrator_dsn)
    assert repeated.state == "current"
    assert repeated_now == ()

    with psycopg.connect(migrator_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE wf_projection_owner")
            cursor.execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'operational_projection' "
                "AND table_type = 'BASE TABLE'"
            )
            assert cursor.fetchone()[0] == 15
            cursor.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'operational_projection'"
            )
            columns = cursor.fetchall()
            assert not any(data_type in {"json", "jsonb"} for _, data_type in columns)
            prohibited = ("path", "dsn", "secret", "password", "payload")
            assert not any(
                marker in column_name.lower()
                for column_name, _ in columns
                for marker in prohibited
            )
            cursor.execute(
                "SELECT count(*) FROM information_schema.role_table_grants "
                "WHERE table_schema = 'operational_projection' "
                "AND grantee = 'PUBLIC'"
            )
            assert cursor.fetchone()[0] == 0

    with psycopg.connect(projection_dsns["writer"]) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM operational_projection.schema_migration")
            assert cursor.fetchone()[0] == 2
            cursor.execute(
                "INSERT INTO operational_projection.projection_generation ("
                "generation_id, environment_id, contract_version, schema_version, "
                "projector_version, source_git_commit, source_set_sha256, "
                "evidence_record_count, generation_evidence_count, model_era_count, "
                "monitoring_report_count, quality_issue_count, monitoring_window_count, "
                "performance_metric_count, drift_measurement_count, alert_event_count, "
                "active_alert_snapshot_count, reporting_attempt_count, lineage_edge_count"
                ") VALUES ("
                "%s, 'local', 'contract-v1', 'schema-v1', 'projector-v1', %s, %s, "
                "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"
                ")",
                ("a" * 64, "b" * 40, "c" * 64),
            )
            cursor.execute(
                "UPDATE operational_projection.projection_generation "
                "SET ready_at_utc = CURRENT_TIMESTAMP WHERE generation_id = %s",
                ("a" * 64,),
            )
            cursor.execute(
                "INSERT INTO operational_projection.projection_head "
                "(environment_id, generation_id, published_at_utc) "
                "VALUES ('local', %s, CURRENT_TIMESTAMP)",
                ("a" * 64,),
            )
        connection.rollback()
        with connection.cursor() as cursor:
            with pytest.raises(errors.InsufficientPrivilege):
                cursor.execute("CREATE TABLE operational_projection.writer_forbidden (id int)")
        connection.rollback()
        with connection.cursor() as cursor:
            with pytest.raises(errors.InsufficientPrivilege):
                cursor.execute("DELETE FROM operational_projection.projection_generation")
        connection.rollback()

    with psycopg.connect(projection_dsns["reader"]) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM operational_projection.schema_migration")
            assert cursor.fetchone()[0] == 2
            with pytest.raises(errors.InsufficientPrivilege):
                cursor.execute(
                    "INSERT INTO operational_projection.schema_migration "
                    "(version, name, sha256, applied_at_utc) "
                    "VALUES (99, 'forbidden', %s, CURRENT_TIMESTAMP)",
                    ("f" * 64,),
                )
        connection.rollback()

    failed = migrations.Migration(
        version=3,
        name="forced_failure",
        sha256="f" * 64,
        sql=(
            "CREATE TABLE operational_projection.must_rollback (id integer); "
            "SELECT 1 / 0;"
        ),
    )
    connection = migrations._connect(migrator_dsn)
    try:
        with pytest.raises(migrations.MigrationDatabaseError):
            migrations._apply_migration(connection, failed)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('operational_projection.must_rollback')"
            )
            assert cursor.fetchone()[0] is None
    finally:
        connection.close()

    bundled = migrations.discover_migrations()
    with psycopg.connect(migrator_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE wf_projection_owner")
            cursor.execute(
                "UPDATE operational_projection.schema_migration "
                "SET sha256 = %s WHERE version = 2",
                ("0" * 64,),
            )
    try:
        with pytest.raises(migrations.MigrationChecksumError):
            migrations.migration_status(migrator_dsn)
    finally:
        with psycopg.connect(migrator_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET ROLE wf_projection_owner")
                cursor.execute(
                    "UPDATE operational_projection.schema_migration "
                    "SET sha256 = %s WHERE version = 2",
                    (bundled[1].sha256,),
                )

    with psycopg.connect(migrator_dsn, autocommit=True) as blocker:
        with blocker.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_lock(%s)",
                (migrations.MIGRATION_LOCK_KEY,),
            )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(migrations.migrate, migrator_dsn)
            time.sleep(0.2)
            assert not future.done()
            with blocker.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_unlock(%s)",
                    (migrations.MIGRATION_LOCK_KEY,),
                )
            serialized, serialized_now = future.result(timeout=5)
    assert serialized.state == "current"
    assert serialized_now == ()


def test_postgres_projector_publication_verification_and_rollback(
    projection_dsns: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    _patch_empty_projector_sources(monkeypatch)
    migrations.migrate(projection_dsns["migrator"])
    first_commit = "d" * 40
    second_commit = "e" * 40
    first_snapshot = projector.build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit=first_commit,
    )

    missing = projector.verify_projection(
        projection_dsns["reader"],
        tmp_path,
        environment_id="local",
        source_git_commit=first_commit,
    )
    assert missing.status == "missing"

    projected = projector.project_projection(
        projection_dsns["writer"],
        tmp_path,
        environment_id="local",
        source_git_commit=first_commit,
    )
    assert projected.status == "projected"
    assert projected.generation_id == first_snapshot.generation_id
    assert projector.verify_projection(
        projection_dsns["reader"],
        tmp_path,
        environment_id="local",
        source_git_commit=first_commit,
    ).status == "ready"
    assert projector.plan_projection(
        projection_dsns["reader"],
        tmp_path,
        environment_id="local",
        source_git_commit=first_commit,
    ).status == "no_op"
    assert projector.project_projection(
        projection_dsns["writer"],
        tmp_path,
        environment_id="local",
        source_git_commit=first_commit,
    ).status == "no_op"

    second_snapshot = projector.build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit=second_commit,
    )
    assert projector.verify_projection(
        projection_dsns["reader"],
        tmp_path,
        environment_id="local",
        source_git_commit=second_commit,
    ).status == "stale"

    with pytest.raises(projector.ProjectionDatabaseError):
        projector.project_projection(
            projection_dsns["writer"],
            tmp_path,
            environment_id="local",
            source_git_commit=second_commit,
            failure_hook=lambda stage: (
                (_ for _ in ()).throw(RuntimeError("forced failure"))
                if stage == "before_commit"
                else None
            ),
        )
    with psycopg.connect(projection_dsns["reader"]) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT generation_id FROM operational_projection.projection_head "
                "WHERE environment_id = 'local'"
            )
            assert cursor.fetchone()[0] == first_snapshot.generation_id
            cursor.execute(
                "SELECT count(*) FROM operational_projection.projection_generation "
                "WHERE generation_id = %s",
                (second_snapshot.generation_id,),
            )
            assert cursor.fetchone()[0] == 0

    assert projector.project_projection(
        projection_dsns["writer"],
        tmp_path,
        environment_id="local",
        source_git_commit=second_commit,
    ).status == "projected"
    assert projector.project_projection(
        projection_dsns["writer"],
        tmp_path,
        environment_id="local",
        source_git_commit=first_commit,
    ).status == "projected"

    with psycopg.connect(projection_dsns["migrator"], autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE wf_projection_owner")
            cursor.execute(
                "UPDATE operational_projection.projection_generation "
                "SET evidence_record_count = evidence_record_count + 1 "
                "WHERE generation_id = %s",
                (first_snapshot.generation_id,),
            )
    try:
        assert projector.verify_projection(
            projection_dsns["reader"],
            tmp_path,
            environment_id="local",
            source_git_commit=first_commit,
        ).status == "mismatch"
    finally:
        with psycopg.connect(projection_dsns["migrator"], autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET ROLE wf_projection_owner")
                cursor.execute(
                    "UPDATE operational_projection.projection_generation "
                    "SET evidence_record_count = 0 WHERE generation_id = %s",
                    (first_snapshot.generation_id,),
                )

    bundled = migrations.discover_migrations()
    with psycopg.connect(projection_dsns["migrator"], autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE wf_projection_owner")
            cursor.execute(
                "UPDATE operational_projection.schema_migration SET sha256 = %s "
                "WHERE version = 2",
                ("0" * 64,),
            )
    try:
        assert projector.verify_projection(
            projection_dsns["reader"],
            tmp_path,
            environment_id="local",
            source_git_commit=first_commit,
        ).status == "incompatible"
    finally:
        with psycopg.connect(projection_dsns["migrator"], autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET ROLE wf_projection_owner")
                cursor.execute(
                    "UPDATE operational_projection.schema_migration SET sha256 = %s "
                    "WHERE version = 2",
                    (bundled[1].sha256,),
                )

    from test_operational_projection_projector import _patch_sources

    _patch_sources(monkeypatch)
    full = projector.project_projection(
        projection_dsns["writer"],
        tmp_path,
        environment_id="local",
        source_git_commit="1" * 40,
    )
    assert full.status == "projected"
    assert full.counts["monitoring_report_count"] == 1
    assert full.counts["performance_metric_count"] == 5
    assert full.counts["drift_measurement_count"] == 2
    assert projector.verify_projection(
        projection_dsns["reader"],
        tmp_path,
        environment_id="local",
        source_git_commit="1" * 40,
    ).status == "ready"
    with psycopg.connect(projection_dsns["reader"]) as connection:
        with connection.cursor() as cursor:
            for table, expected in (
                ("model_era", 1),
                ("monitoring_report", 1),
                ("quality_issue", 1),
                ("monitoring_window", 2),
                ("performance_metric", 5),
                ("drift_measurement", 2),
                ("alert_event", 1),
                ("active_alert_snapshot", 1),
                ("reporting_attempt", 1),
                ("lineage_edge", 7),
            ):
                cursor.execute(
                    f"SELECT count(*) FROM operational_projection.{table}"
                )
                assert cursor.fetchone()[0] == expected

    with psycopg.connect(projection_dsns["migrator"], autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE wf_projection_owner")
            cursor.execute(
                "SELECT evidence_record_id FROM "
                "operational_projection.generation_evidence "
                "WHERE generation_id = %s ORDER BY evidence_record_id LIMIT 1",
                (full.generation_id,),
            )
            evidence_record_id = cursor.fetchone()[0]
            cursor.execute(
                "DELETE FROM operational_projection.generation_evidence "
                "WHERE generation_id = %s AND evidence_record_id = %s",
                (full.generation_id, evidence_record_id),
            )
    try:
        assert projector.verify_projection(
            projection_dsns["reader"],
            tmp_path,
            environment_id="local",
            source_git_commit="1" * 40,
        ).status == "mismatch"
    finally:
        with psycopg.connect(projection_dsns["migrator"], autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET ROLE wf_projection_owner")
                cursor.execute(
                    "INSERT INTO operational_projection.generation_evidence "
                    "(generation_id, evidence_record_id) VALUES (%s, %s)",
                    (full.generation_id, evidence_record_id),
                )

    with psycopg.connect(projection_dsns["migrator"], autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE wf_projection_owner")
            cursor.execute(
                "UPDATE operational_projection.projection_generation "
                "SET source_set_sha256 = %s WHERE generation_id = %s",
                ("0" * 64, full.generation_id),
            )
    try:
        assert projector.verify_projection(
            projection_dsns["reader"],
            tmp_path,
            environment_id="local",
            source_git_commit="1" * 40,
        ).status == "mismatch"
    finally:
        with psycopg.connect(projection_dsns["migrator"], autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET ROLE wf_projection_owner")
                cursor.execute(
                    "UPDATE operational_projection.projection_generation "
                    "SET source_set_sha256 = %s WHERE generation_id = %s",
                    (full_snapshot_source_set(tmp_path, monkeypatch), full.generation_id),
                )


def full_snapshot_source_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    from test_operational_projection_projector import _patch_sources

    _patch_sources(monkeypatch)
    return projector.build_projection_snapshot(
        tmp_path,
        environment_id="local",
        source_git_commit="1" * 40,
    ).manifest.source_set_sha256


def test_postgres_projectors_serialize_on_environment_lock(
    projection_dsns: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    _patch_empty_projector_sources(monkeypatch)
    migrations.migrate(projection_dsns["migrator"])
    with psycopg.connect(projection_dsns["writer"], autocommit=True) as blocker:
        with blocker.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_lock(%s)",
                (projector.PROJECTION_LOCK_KEY,),
            )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                projector.project_projection,
                projection_dsns["writer"],
                tmp_path,
                environment_id="local",
                source_git_commit="f" * 40,
            )
            time.sleep(0.2)
            assert not future.done()
            with blocker.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_unlock(%s)",
                    (projector.PROJECTION_LOCK_KEY,),
                )
            result = future.result(timeout=5)
    assert result.status == "projected"
