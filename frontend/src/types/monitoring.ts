export type MonitoringState = "empty" | "available";
export type MonitoringSeverity =
  | "not_available"
  | "ok"
  | "warning"
  | "critical";

export interface MonitoringRun {
  readonly run_id: string;
  readonly attempted_at_utc: string;
  readonly through_date: string | null;
  readonly source_pipeline_run_id: string | null;
  readonly source_pipeline_status: string | null;
  readonly status: "succeeded" | "failed" | "in_progress";
  readonly report_id: string | null;
  readonly active_alert_count: number;
  readonly failure: {
    readonly failed_at_utc: string | null;
    readonly error_type: string | null;
    readonly message: string;
  } | null;
}

export interface MonitoringAlert {
  readonly alert_event_id: string;
  readonly rule_id: string;
  readonly through_date: string;
  readonly event_type: "opened" | "escalated" | "resolved";
  readonly severity: MonitoringSeverity;
  readonly previous_alert_event_id: string | null;
}

export interface MonitoringMetric {
  readonly metric: string;
  readonly label: string;
  readonly value: number | null;
  readonly status: string;
  readonly severity: MonitoringSeverity;
  readonly warning: number;
  readonly critical: number;
  readonly direction: "upper" | "lower";
}

export interface MonitoringDrift {
  readonly feature: string;
  readonly comparator: "global" | "seasonal";
  readonly detector: "normalized_wasserstein" | "ks_statistic";
  readonly value: number;
  readonly severity: MonitoringSeverity;
  readonly threshold: number;
  readonly threshold_ratio: number;
}

export interface MonitoringWindow {
  readonly window_days: number;
  readonly status: "available" | "insufficient_data" | "not_available";
  readonly sample_count: number;
  readonly minimum_samples: number | null;
  readonly calendar_start: string | null;
  readonly calendar_end: string | null;
  readonly coverage_ratio: number | null;
  readonly coverage_severity: MonitoringSeverity | null;
  readonly performance: readonly MonitoringMetric[];
  readonly top_drift: readonly MonitoringDrift[];
}

export interface MonitoringReport {
  readonly report_id: string;
  readonly reporting_run_id: string;
  readonly created_at_utc: string;
  readonly as_of_date: string;
  readonly source_pipeline: {
    readonly run_id: string;
    readonly status: string;
  };
  readonly freshness: {
    readonly status:
      | "within_objective"
      | "behind_objective"
      | "late"
      | "unknown";
    readonly watermark_date: string | null;
    readonly objective_at: string | null;
    readonly late_at: string | null;
    readonly timezone: "Europe/Lisbon";
    readonly objective_days: number;
    readonly late_days: number;
  };
  readonly model: {
    readonly snapshot_id: string | null;
    readonly checksum: string;
    readonly model_type: string | null;
    readonly dataset_version: string | null;
    readonly dataset_checksum: string;
    readonly transformation_version: string;
    readonly status: "selected_not_promoted";
  } | null;
  readonly windows: {
    readonly "30": MonitoringWindow;
    readonly "90": MonitoringWindow;
  };
  readonly active_alerts: readonly MonitoringAlert[];
  readonly target_scale: "sum_of_15_minute_MW_observations";
}

export interface MonitoringLatestResponse {
  readonly state: MonitoringState;
  readonly mode: "retrospective_historical_batch_not_real_time";
  readonly served_at_utc: string;
  readonly message: string | null;
  readonly latest_attempt: MonitoringRun | null;
  readonly report: MonitoringReport | null;
}

export interface MonitoringHistoryResponse {
  readonly state: MonitoringState;
  readonly mode: "retrospective_historical_batch_not_real_time";
  readonly runs: MonitoringPage<MonitoringRun>;
  readonly alerts: MonitoringPage<MonitoringAlert>;
}

export interface MonitoringPage<T> {
  readonly items: readonly T[];
  readonly total: number;
  readonly limit: number;
  readonly offset: number;
}

export interface MonitoringRunResponse {
  readonly state: "available";
  readonly mode: "retrospective_historical_batch_not_real_time";
  readonly run: MonitoringRun;
  readonly report: MonitoringReport | null;
}
