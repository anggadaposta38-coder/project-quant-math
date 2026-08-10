import { describe, expect, it } from "vitest";

import { describeError } from "./error-capture";
import { sanitizeDiagnosticText } from "./diagnostic-utils";

describe("diagnostic safety", () => {
  it("redacts credential-like query parameters and bearer tokens", () => {
    const input = "https://example.test/data?apiKey=super-secret&ok=1 Authorization: Bearer abc.def.ghi";
    const output = sanitizeDiagnosticText(input);

    expect(output).not.toContain("super-secret");
    expect(output).not.toContain("abc.def.ghi");
    expect(output).toContain("apiKey=[REDACTED]");
    expect(output).toContain("Bearer [REDACTED]");
  });

  it("keeps useful error message and status while sanitizing", () => {
    const error = Object.assign(new Error("request failed?token=secret"), { statusCode: 503 });
    const output = describeError(error);

    expect(output).toContain("status 503");
    expect(output).toContain("[REDACTED]");
    expect(output).not.toContain("secret");
  });
});
