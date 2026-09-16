import { describe, expect, it } from "vitest";

import { runEventSchema } from "./api";

describe("run event contract", () => {
  it("accepts dense versioned events", () => {
    expect(
      runEventSchema.parse({
        schema_version: 1,
        run_id: "run-1",
        seq: 1,
        kind: "run.started",
        created_at: "2026-08-29T00:00:00Z",
        payload: {},
      }),
    ).toMatchObject({ seq: 1, kind: "run.started" });
  });

  it("rejects non-positive sequence numbers", () => {
    expect(() =>
      runEventSchema.parse({
        schema_version: 1,
        run_id: "run-1",
        seq: 0,
        kind: "run.started",
        created_at: "2026-08-29T00:00:00Z",
        payload: {},
      }),
    ).toThrow();
  });
});
