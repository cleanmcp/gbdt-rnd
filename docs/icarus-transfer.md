# Mono-icarus transfer boundary

The engine is not a second chat agent and must not train inside Mono-icarus's
ReAct retry loop.

## What transfers

- generated strict JSON Schemas;
- immutable product/ICP/goal/signal-policy versions;
- corpus and model artifact hashes;
- `recommend` request/output;
- dense resumable run events;
- React trace and account-inspector presentation;
- point-in-time fixtures and contract tests.

## Future call

The eventual Mono worker should make one bounded call equivalent to:

```text
recommend(
  workspace_id,
  icp_version_id,
  goal_version_id,
  as_of,
  capacity,
  idempotency_key
) -> recommendations
```

Mono remains responsible for authentication, workspace authorization, active ICP
selection, quotas, billing, delivery, and workspace outcomes. The Python engine
remains responsible for evidence processing, feature/model versions, scoring,
simulation, and model artifacts.

## Patterns mirrored from current main

The local protocol intentionally mirrors, without importing:

- frozen versioned JSON and canonical hashes from
  `Mono-icarus/record/run/contracts.ts`;
- queued/running/terminal-once run states from
  `Mono-icarus/record/schema/run.ts`;
- dense append-only SSE cursor behavior from
  `Mono-icarus/surface/stream/run-sse.ts`;
- corpus/model/version pinning from
  `Mono-icarus/record/schema/icp-version.ts`;
- strict capability contract hashing from
  `Mono-icarus/run/agent/capability-registry.ts`.

Do not reuse Mono's current `run.run` table: it is constrained to chat and
requires conversation/message parents. A later approved amendment must add the
lead-generation execution owner.

LinkedIn assumptions are excluded. Current Mono main retired the unactivated
LinkedIn account foundation in migration `0049_little_roulette.sql`; a
replacement requires a separately proven provider path and approved amendment.
