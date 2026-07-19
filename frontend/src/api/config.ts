export interface ApiConfig {
  readonly baseUrl: string | null;
}

function normalizeBaseUrl(value: string | undefined): string | null {
  const trimmedValue = value?.trim();

  if (!trimmedValue) {
    return null;
  }

  return trimmedValue.replace(/\/+$/, "");
}

export const apiConfig: ApiConfig = Object.freeze({
  baseUrl: normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL),
});
