\set ON_ERROR_STOP on

\getenv projection_database POSTGRES_DB
\getenv migrator_password WIND_FORECAST_OPERATIONAL_PROJECTION_MIGRATOR_PASSWORD
\getenv writer_password WIND_FORECAST_OPERATIONAL_PROJECTION_WRITER_PASSWORD
\getenv reader_password WIND_FORECAST_OPERATIONAL_PROJECTION_READER_PASSWORD

SELECT 'CREATE ROLE wf_projection_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION'
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'wf_projection_owner')
\gexec

SELECT format(
    'CREATE ROLE wf_projection_migrator LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT',
    :'migrator_password'
)
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'wf_projection_migrator')
\gexec

SELECT format(
    'CREATE ROLE wf_projection_writer LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT',
    :'writer_password'
)
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'wf_projection_writer')
\gexec

SELECT format(
    'CREATE ROLE wf_projection_reader LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT',
    :'reader_password'
)
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'wf_projection_reader')
\gexec

ALTER ROLE wf_projection_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
ALTER ROLE wf_projection_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT;
ALTER ROLE wf_projection_writer LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT;
ALTER ROLE wf_projection_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT;

SELECT format('ALTER ROLE wf_projection_migrator PASSWORD %L', :'migrator_password')
\gexec
SELECT format('ALTER ROLE wf_projection_writer PASSWORD %L', :'writer_password')
\gexec
SELECT format('ALTER ROLE wf_projection_reader PASSWORD %L', :'reader_password')
\gexec

GRANT wf_projection_owner TO wf_projection_migrator;

SELECT format('ALTER DATABASE %I OWNER TO wf_projection_owner', :'projection_database')
\gexec
SELECT format('REVOKE ALL ON DATABASE %I FROM PUBLIC', :'projection_database')
\gexec
SELECT format(
    'GRANT CONNECT ON DATABASE %I TO wf_projection_migrator, wf_projection_writer, wf_projection_reader',
    :'projection_database'
)
\gexec

REVOKE ALL ON SCHEMA public FROM PUBLIC;
