CREATE SCHEMA operational_projection AUTHORIZATION wf_projection_owner;

REVOKE ALL ON SCHEMA operational_projection FROM PUBLIC;

CREATE TABLE operational_projection.schema_migration (
    version integer PRIMARY KEY CHECK (version > 0),
    name text NOT NULL UNIQUE CHECK (name ~ '^[a-z][a-z0-9_]*$'),
    sha256 character(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    applied_at_utc timestamp with time zone NOT NULL
);

REVOKE ALL ON TABLE operational_projection.schema_migration FROM PUBLIC;
GRANT USAGE ON SCHEMA operational_projection
    TO wf_projection_migrator, wf_projection_writer, wf_projection_reader;
GRANT SELECT ON TABLE operational_projection.schema_migration
    TO wf_projection_writer, wf_projection_reader;
