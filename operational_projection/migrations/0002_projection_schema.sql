CREATE TABLE operational_projection.projection_generation (
    generation_id character(64) PRIMARY KEY
        CHECK (generation_id ~ '^[0-9a-f]{64}$'),
    environment_id text NOT NULL CHECK (environment_id = 'local'),
    contract_version text NOT NULL CHECK (length(contract_version) BETWEEN 1 AND 128),
    schema_version text NOT NULL CHECK (length(schema_version) BETWEEN 1 AND 128),
    projector_version text NOT NULL CHECK (length(projector_version) BETWEEN 1 AND 128),
    source_git_commit text NOT NULL
        CHECK (source_git_commit ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    source_set_sha256 character(64) NOT NULL
        CHECK (source_set_sha256 ~ '^[0-9a-f]{64}$'),
    evidence_record_count bigint NOT NULL CHECK (evidence_record_count >= 0),
    generation_evidence_count bigint NOT NULL CHECK (generation_evidence_count >= 0),
    model_era_count bigint NOT NULL CHECK (model_era_count >= 0),
    monitoring_report_count bigint NOT NULL CHECK (monitoring_report_count >= 0),
    quality_issue_count bigint NOT NULL CHECK (quality_issue_count >= 0),
    monitoring_window_count bigint NOT NULL CHECK (monitoring_window_count >= 0),
    performance_metric_count bigint NOT NULL CHECK (performance_metric_count >= 0),
    drift_measurement_count bigint NOT NULL CHECK (drift_measurement_count >= 0),
    alert_event_count bigint NOT NULL CHECK (alert_event_count >= 0),
    active_alert_snapshot_count bigint NOT NULL CHECK (active_alert_snapshot_count >= 0),
    reporting_attempt_count bigint NOT NULL CHECK (reporting_attempt_count >= 0),
    lineage_edge_count bigint NOT NULL CHECK (lineage_edge_count >= 0),
    ready_at_utc timestamp with time zone
);

CREATE TABLE operational_projection.projection_head (
    environment_id text PRIMARY KEY CHECK (environment_id = 'local'),
    generation_id character(64) NOT NULL
        REFERENCES operational_projection.projection_generation(generation_id),
    published_at_utc timestamp with time zone NOT NULL
);

CREATE TABLE operational_projection.evidence_record (
    evidence_record_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    domain text NOT NULL CHECK (length(domain) BETWEEN 1 AND 128),
    source_kind text NOT NULL CHECK (length(source_kind) BETWEEN 1 AND 128),
    schema_version text NOT NULL CHECK (length(schema_version) BETWEEN 1 AND 128),
    record_id text NOT NULL CHECK (length(record_id) BETWEEN 1 AND 512),
    sha256 character(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    effective_at text NOT NULL CHECK (length(effective_at) BETWEEN 1 AND 128),
    observed_at_utc timestamp with time zone,
    UNIQUE (domain, source_kind, schema_version, record_id, sha256)
);

CREATE TABLE operational_projection.generation_evidence (
    generation_id character(64) NOT NULL
        REFERENCES operational_projection.projection_generation(generation_id),
    evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    PRIMARY KEY (generation_id, evidence_record_id)
);

CREATE TABLE operational_projection.model_era (
    model_era_id text PRIMARY KEY CHECK (length(model_era_id) BETWEEN 1 AND 512),
    evidence_record_id bigint NOT NULL UNIQUE
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    association_kind text NOT NULL
        CHECK (association_kind IN ('active_deployment', 'bootstrap_adopted')),
    deployment_id text NOT NULL CHECK (length(deployment_id) BETWEEN 1 AND 512),
    deployment_generation bigint NOT NULL CHECK (deployment_generation >= 0),
    registered_model_name text NOT NULL
        CHECK (length(registered_model_name) BETWEEN 1 AND 512),
    model_version text NOT NULL CHECK (length(model_version) BETWEEN 1 AND 128),
    fit_cutoff date NOT NULL,
    activation_cutoff date NOT NULL,
    bundle_sha256 character(64) NOT NULL CHECK (bundle_sha256 ~ '^[0-9a-f]{64}$'),
    model_sha256 character(64) NOT NULL CHECK (model_sha256 ~ '^[0-9a-f]{64}$'),
    dataset_sha256 character(64) NOT NULL CHECK (dataset_sha256 ~ '^[0-9a-f]{64}$'),
    feature_schema_sha256 character(64) NOT NULL
        CHECK (feature_schema_sha256 ~ '^[0-9a-f]{64}$'),
    calibration_sha256 character(64) NOT NULL
        CHECK (calibration_sha256 ~ '^[0-9a-f]{64}$'),
    ledger_sha256 character(64) NOT NULL CHECK (ledger_sha256 ~ '^[0-9a-f]{64}$'),
    calibration_id text NOT NULL CHECK (length(calibration_id) BETWEEN 1 AND 512),
    reference_id text NOT NULL CHECK (length(reference_id) BETWEEN 1 AND 512),
    CHECK (fit_cutoff <= activation_cutoff)
);

CREATE TABLE operational_projection.monitoring_report (
    report_id text PRIMARY KEY CHECK (length(report_id) BETWEEN 1 AND 512),
    evidence_record_id bigint NOT NULL UNIQUE
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    reporting_run_id text NOT NULL UNIQUE
        CHECK (length(reporting_run_id) BETWEEN 1 AND 512),
    created_at_utc timestamp with time zone NOT NULL,
    through_date date NOT NULL,
    source_run_id text NOT NULL CHECK (length(source_run_id) BETWEEN 1 AND 512),
    source_status text NOT NULL CHECK (source_status IN ('succeeded', 'no_op', 'failed')),
    calibration_id text NOT NULL CHECK (length(calibration_id) BETWEEN 1 AND 512),
    reference_id text NOT NULL CHECK (length(reference_id) BETWEEN 1 AND 512),
    policy_sha256 character(64) NOT NULL CHECK (policy_sha256 ~ '^[0-9a-f]{64}$'),
    quality_status text NOT NULL
        CHECK (quality_status IN ('available', 'not_available', 'succeeded', 'failed')),
    batch_status text NOT NULL CHECK (batch_status IN ('succeeded', 'no_op', 'failed')),
    verdict text NOT NULL CHECK (verdict IN ('PASS', 'FAIL', 'not_available')),
    watermark_date date,
    watermark_age_days integer CHECK (watermark_age_days IS NULL OR watermark_age_days >= 0),
    objective_days integer CHECK (objective_days IS NULL OR objective_days >= 0),
    late_days integer CHECK (late_days IS NULL OR late_days >= 0),
    objective_missed boolean NOT NULL,
    unresolved_late_date_count integer NOT NULL CHECK (unresolved_late_date_count >= 0),
    date_count integer NOT NULL CHECK (date_count >= 0),
    ren_complete_count integer NOT NULL CHECK (ren_complete_count >= 0),
    era5_complete_count integer NOT NULL CHECK (era5_complete_count >= 0),
    integration_ready_count integer NOT NULL CHECK (integration_ready_count >= 0),
    feature_ready_count integer NOT NULL CHECK (feature_ready_count >= 0),
    model_era_id text REFERENCES operational_projection.model_era(model_era_id)
);

CREATE TABLE operational_projection.quality_issue (
    report_id text NOT NULL
        REFERENCES operational_projection.monitoring_report(report_id),
    position integer NOT NULL CHECK (position >= 0),
    evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    code text NOT NULL CHECK (length(code) BETWEEN 1 AND 256),
    severity text NOT NULL CHECK (severity IN ('warning', 'critical')),
    PRIMARY KEY (report_id, position)
);

CREATE TABLE operational_projection.monitoring_window (
    report_id text NOT NULL
        REFERENCES operational_projection.monitoring_report(report_id),
    window_days integer NOT NULL CHECK (window_days IN (30, 90)),
    evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    status text NOT NULL CHECK (status IN ('available', 'not_available')),
    sample_count integer NOT NULL CHECK (sample_count >= 0),
    coverage_ratio double precision,
    coverage_severity text NOT NULL
        CHECK (coverage_severity IN ('ok', 'warning', 'critical', 'not_available')),
    minimum_samples integer NOT NULL CHECK (minimum_samples >= 0),
    calendar_start date,
    calendar_end date,
    PRIMARY KEY (report_id, window_days),
    CHECK (coverage_ratio IS NULL OR (
        coverage_ratio >= 0 AND coverage_ratio <= 1
        AND coverage_ratio::text NOT IN ('NaN', 'Infinity', '-Infinity')
    )),
    CHECK (calendar_start IS NULL OR calendar_end IS NULL OR calendar_start <= calendar_end)
);

CREATE TABLE operational_projection.performance_metric (
    report_id text NOT NULL,
    window_days integer NOT NULL,
    evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    metric_name text NOT NULL CHECK (length(metric_name) BETWEEN 1 AND 128),
    value double precision,
    value_status text NOT NULL
        CHECK (value_status IN ('available', 'insufficient_data', 'constant_target', 'not_available')),
    severity text NOT NULL
        CHECK (severity IN ('ok', 'warning', 'critical', 'not_available')),
    warning_threshold double precision,
    critical_threshold double precision,
    direction text NOT NULL CHECK (direction IN ('upper', 'lower')),
    unit_or_scale text NOT NULL CHECK (length(unit_or_scale) BETWEEN 1 AND 256),
    PRIMARY KEY (report_id, window_days, metric_name),
    FOREIGN KEY (report_id, window_days)
        REFERENCES operational_projection.monitoring_window(report_id, window_days),
    CHECK (value IS NULL OR value::text NOT IN ('NaN', 'Infinity', '-Infinity')),
    CHECK (warning_threshold IS NULL OR warning_threshold::text NOT IN ('NaN', 'Infinity', '-Infinity')),
    CHECK (critical_threshold IS NULL OR critical_threshold::text NOT IN ('NaN', 'Infinity', '-Infinity'))
);

CREATE TABLE operational_projection.drift_measurement (
    report_id text NOT NULL,
    window_days integer NOT NULL,
    position integer NOT NULL CHECK (position >= 0),
    evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    feature text NOT NULL CHECK (length(feature) BETWEEN 1 AND 256),
    comparator text NOT NULL CHECK (length(comparator) BETWEEN 1 AND 128),
    detector text NOT NULL CHECK (length(detector) BETWEEN 1 AND 128),
    value double precision NOT NULL,
    severity text NOT NULL CHECK (severity IN ('ok', 'warning', 'critical')),
    warning_threshold double precision NOT NULL,
    critical_threshold double precision NOT NULL,
    direction text NOT NULL CHECK (direction IN ('upper', 'lower')),
    PRIMARY KEY (report_id, window_days, position),
    FOREIGN KEY (report_id, window_days)
        REFERENCES operational_projection.monitoring_window(report_id, window_days),
    CHECK (value::text NOT IN ('NaN', 'Infinity', '-Infinity')),
    CHECK (warning_threshold::text NOT IN ('NaN', 'Infinity', '-Infinity')),
    CHECK (critical_threshold::text NOT IN ('NaN', 'Infinity', '-Infinity'))
);

CREATE TABLE operational_projection.alert_event (
    alert_event_id text PRIMARY KEY CHECK (length(alert_event_id) BETWEEN 1 AND 512),
    evidence_record_id bigint NOT NULL UNIQUE
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    rule_id text NOT NULL CHECK (length(rule_id) BETWEEN 1 AND 512),
    through_date date NOT NULL,
    event_type text NOT NULL CHECK (event_type IN ('opened', 'escalated', 'resolved')),
    severity text NOT NULL CHECK (severity IN ('ok', 'warning', 'critical')),
    previous_alert_event_id text
        REFERENCES operational_projection.alert_event(alert_event_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE operational_projection.active_alert_snapshot (
    generation_id character(64) NOT NULL
        REFERENCES operational_projection.projection_generation(generation_id),
    rule_id text NOT NULL CHECK (length(rule_id) BETWEEN 1 AND 512),
    evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    alert_event_id text NOT NULL
        REFERENCES operational_projection.alert_event(alert_event_id),
    PRIMARY KEY (generation_id, rule_id)
);

CREATE TABLE operational_projection.reporting_attempt (
    reporting_run_id text PRIMARY KEY CHECK (length(reporting_run_id) BETWEEN 1 AND 512),
    evidence_record_id bigint NOT NULL UNIQUE
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    attempted_at_utc timestamp with time zone NOT NULL,
    through_date date NOT NULL,
    source_run_id text NOT NULL CHECK (length(source_run_id) BETWEEN 1 AND 512),
    source_status text NOT NULL CHECK (source_status IN ('succeeded', 'no_op', 'failed')),
    status text NOT NULL CHECK (status IN ('succeeded', 'failed', 'in_progress')),
    report_id text UNIQUE REFERENCES operational_projection.monitoring_report(report_id),
    active_alert_count integer NOT NULL CHECK (active_alert_count >= 0),
    failure_at_utc timestamp with time zone,
    failure_type text CHECK (failure_type IS NULL OR length(failure_type) BETWEEN 1 AND 256),
    failure_message text CHECK (failure_message IS NULL OR length(failure_message) BETWEEN 1 AND 1024),
    CHECK ((status = 'failed') = (failure_at_utc IS NOT NULL)),
    CHECK ((status = 'failed') = (failure_type IS NOT NULL)),
    CHECK ((status = 'failed') = (failure_message IS NOT NULL)),
    CHECK ((status = 'succeeded') = (report_id IS NOT NULL))
);

CREATE TABLE operational_projection.lineage_edge (
    generation_id character(64) NOT NULL
        REFERENCES operational_projection.projection_generation(generation_id),
    edge_type text NOT NULL CHECK (length(edge_type) BETWEEN 1 AND 128),
    source_evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    target_evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    position integer NOT NULL CHECK (position >= 0),
    evidence_record_id bigint NOT NULL
        REFERENCES operational_projection.evidence_record(evidence_record_id),
    PRIMARY KEY (
        generation_id,
        edge_type,
        source_evidence_record_id,
        target_evidence_record_id,
        position
    )
);

CREATE INDEX monitoring_report_latest_idx
    ON operational_projection.monitoring_report (through_date DESC, report_id);
CREATE INDEX monitoring_report_source_run_idx
    ON operational_projection.monitoring_report (source_run_id);
CREATE INDEX reporting_attempt_latest_idx
    ON operational_projection.reporting_attempt (attempted_at_utc DESC, reporting_run_id);
CREATE INDEX alert_event_date_idx
    ON operational_projection.alert_event (through_date DESC, alert_event_id);
CREATE INDEX alert_event_rule_idx
    ON operational_projection.alert_event (rule_id, through_date DESC, alert_event_id);
CREATE INDEX generation_evidence_evidence_idx
    ON operational_projection.generation_evidence (evidence_record_id, generation_id);
CREATE INDEX lineage_edge_source_idx
    ON operational_projection.lineage_edge (generation_id, source_evidence_record_id, position);
CREATE INDEX lineage_edge_target_idx
    ON operational_projection.lineage_edge (generation_id, target_evidence_record_id, position);

REVOKE ALL ON ALL TABLES IN SCHEMA operational_projection FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA operational_projection FROM PUBLIC;

GRANT SELECT ON ALL TABLES IN SCHEMA operational_projection
    TO wf_projection_writer, wf_projection_reader;
GRANT INSERT ON TABLE
    operational_projection.projection_generation,
    operational_projection.evidence_record,
    operational_projection.generation_evidence,
    operational_projection.model_era,
    operational_projection.monitoring_report,
    operational_projection.quality_issue,
    operational_projection.monitoring_window,
    operational_projection.performance_metric,
    operational_projection.drift_measurement,
    operational_projection.alert_event,
    operational_projection.active_alert_snapshot,
    operational_projection.reporting_attempt,
    operational_projection.lineage_edge
TO wf_projection_writer;
GRANT INSERT, UPDATE ON TABLE operational_projection.projection_head
    TO wf_projection_writer;
GRANT UPDATE (ready_at_utc) ON TABLE operational_projection.projection_generation
    TO wf_projection_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA operational_projection
    TO wf_projection_writer;

ALTER DEFAULT PRIVILEGES FOR ROLE wf_projection_owner IN SCHEMA operational_projection
    REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE wf_projection_owner IN SCHEMA operational_projection
    REVOKE ALL ON SEQUENCES FROM PUBLIC;
