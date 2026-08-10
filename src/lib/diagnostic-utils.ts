const LOG_TEXT_LENGTH_LIMIT = 12_000;
const REDACTED = "[REDACTED]";

/** Sanitize diagnostic text before it reaches console/telemetry. */
export function sanitizeDiagnosticText(value: string): string {
  return value
    .replace(
      /([?&](?:token|access_token|refresh_token|api[_-]?key|authorization|password|secret|signature)=)[^&\s]+/gi,
      `$1${REDACTED}`,
    )
    .replace(/(Bearer\s+)[A-Za-z0-9._~+\/-]+=*/gi, `$1${REDACTED}`)
    .slice(0, LOG_TEXT_LENGTH_LIMIT);
}
