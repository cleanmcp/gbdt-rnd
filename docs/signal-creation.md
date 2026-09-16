# Adding a signal

New signals must enter through a declaration in [`signals/`](../signals/).
Do not add a source-specific branch to the feature builder.

## 1. State the hypothesis

Document:

- what real-world object or event the signal represents;
- why its trajectory may matter for a named product/ICP;
- whether it indicates fit, timing, value, risk, or corroboration;
- what evidence would prove it unhelpful.

Signal meaning is never universally positive. Product-specific relevance belongs
in a versioned signal policy, not the source adapter.

## 2. Preserve identity

Every normalized event needs:

- `source_id`;
- the source's own `source_record_id`;
- a conservative `signal_instance_id`;
- a resolved internal `account_id`;
- evidence showing how the source record was linked.

Company names are labels, not stable identities. Do not merge two companies on
name alone. Keep ambiguous matches separate or route them for review.

## 3. Choose one shape

| Shape               | Examples                            | Shared descriptors                        |
| ------------------- | ----------------------------------- | ----------------------------------------- |
| `numeric_series`    | TRI production ratio, injury counts | latest, max, mean, delta, slope           |
| `state_machine`     | loan lifecycle, permits             | state, transitions, dwell, terminal state |
| `event_burst`       | contract awards, review activity    | count, strength, recency, burst           |
| `document_versions` | jobs, SOP/public-page changes       | edits, field changes, reposts             |

High-volume atomic events should normally become a time series or burst. Do not
create one LLM StoryCard per page view, click, or social post.

## 4. Declare the contract

Example:

```yaml
schema_version: 1
signal_type: machine_maintenance
version: 1
description: A machine maintenance lifecycle.
shape: state_machine
source_ids: [maintenance_feed]
durable: true
identity_fields: [machine_id]
state_field: status
terminal_states: [resolved]
material_kinds: [opened, escalated, resolved]
numeric_features:
  - field: downtime_hours
    unit: hours
    reducers: [latest, max, sum]
default_half_life_days: 30
applicability_naics_prefixes: ["31", "32", "33"]
```

The registry hashes every validated declaration. Runs freeze the resulting
manifest.

## 5. Write the source adapter

Adapters perform only:

1. field mapping;
2. source-record anchoring;
3. conservative account and instance identity;
4. timestamp normalization;
5. raw-to-normalized strength conversion.

They must retain:

- `occurred_at` — when the underlying event happened;
- `delivered_at` — when the source published/delivered it;
- `ingested_at` — when this system received it;
- `available_at` — earliest time a historical run may use it.

If publication time is estimated, store that fact in the payload and never
present the backtest as exact.

## 6. Define material change

Story semantics are recomputed only when a declared material event or descriptor
changes. Clock-only changes such as age increasing by one day do not change the
semantic hash.

## 7. Add policy and sequence hypotheses

Add the signal to the relevant product/ICP policy with:

- applicability rules;
- initial cold-start weight;
- half-life;
- whether a human pinned the signal;
- ordered sequence rules where justified.

The current deployed model continues to work when a new signal appears. Generic
counts/recency can contribute immediately; signal-specific predictive value
begins only after a versioned retrain has enough outcome evidence.

## 8. Required tests

- same source row normalizes identically twice;
- duplicate ingest is idempotent;
- future observations are invisible before `available_at`;
- missing is distinct from not applicable;
- semantic hash changes on material state and not on clock-only age;
- signal works through the generic FactSheet and feature paths;
- account/instance identity ambiguity fails closed;
- new feature schema cannot silently alter an existing model artifact.
