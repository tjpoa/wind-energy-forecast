export type ApiConnectionState =
  | "not-connected"
  | "connected"
  | "unavailable";

export interface ApiStatusProps {
  readonly baseUrl: string | null;
  readonly status: ApiConnectionState;
}
