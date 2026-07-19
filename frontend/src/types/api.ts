export type ApiConnectionState =
  | "not-connected"
  | "connecting"
  | "connected"
  | "unavailable";

export interface ApiStatusProps {
  readonly baseUrl: string | null;
  readonly status: ApiConnectionState;
}

export interface PerformanceInterval {
  readonly requested_start_date: string | null;
  readonly requested_end_date: string | null;
  readonly available_start_date: string;
  readonly available_end_date: string;
  readonly returned_start_date: string;
  readonly returned_end_date: string;
}

export interface PerformanceMetrics {
  readonly r2: number | null;
  readonly mae: number;
  readonly rmse: number;
  readonly mape_percent: number;
}

export interface PerformanceResultInfo {
  readonly model_type: string;
  readonly seed: number;
  readonly test_fraction: number;
  readonly dataset_version: string | null;
  readonly evaluation_start_date: string;
  readonly evaluation_end_date: string;
  readonly artifact_metrics: PerformanceMetrics | null;
}

export interface PerformanceObservation {
  readonly date: string;
  readonly actual: number;
  readonly predicted: number;
  readonly error: number;
  readonly absolute_error: number;
}

export interface PerformanceResponse {
  readonly interval: PerformanceInterval;
  readonly observation_count: number;
  readonly metrics: PerformanceMetrics;
  readonly result: PerformanceResultInfo | null;
  readonly observations: readonly PerformanceObservation[];
}
