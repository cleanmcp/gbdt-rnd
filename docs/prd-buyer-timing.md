# PRD — Who would buy, and when

**Status:** draft for review · **Owner:** Clarissa · **Date:** 2026-09-01
**Depends on:** [`docs/ablations.md`](ablations.md) (harness), HANDOFF Priorities 3–4

## 1. The question this product must answer

> For a named product and ICP, which accounts will become customers within a
> given window, and in which window — so that a capacity-limited team contacts
> the right accounts at the right time.

Everything the engine does today answers a *proxy* of this question with
public data ("will a public event happen"). The first ablation run showed that
the current public proxy can be answered from source coverage alone
([`docs/ablations.md`](ablations.md), "Architecture — what the arms actually
measure"), so it cannot stand in for the real question. The real question needs
the real answer key: this document specifies that answer key, the data it
requires, how it is evaluated, and what the harness must add.

## 2. Users and the decisions they make

| User | Decision | What they need from the model |
|---|---|---|
| SDR / AE | Which K accounts to work this week | A ranked list of size K with an expected conversion rate the team can plan against |
| Sales lead | When to re-check an account that is not ready | A "wait until" window with the reason (which signal is expected to mature) |
| Product / research | Whether the signal architecture adds anything over a flat firmographic table | The same arms and controls as today's harness, scored on real outcomes |

Capacity K is a first-class input (today's `capacity` in `ExperimentSpec`);
every quality metric is reported at K, not only as an area under a curve.

## 3. Definitions

**Product** — a versioned product fixture (today `sidekick-v1`). Labels are
product-specific; a second product means a second label column, not a new
harness.

**Eligible account at `as_of`** — an account in the ICP universe that is not
already a customer of the product at `as_of` and whose features come only from
events with `available_at ≤ as_of`.

**Outcome ladder** (each step is a separate label; later steps imply earlier ones):

1. contacted → 2. replied (positive / negative / none) → 3. meeting →
4. opportunity → 5. closed-won (**buy**) → 6. active / churned.

**Buy label at horizon H** — `closed_won_at ∈ (as_of, as_of + H]` for an
eligible account. Horizons: 90, 180, 365 days. 180 is the primary horizon
(matches `GoalSpec.horizon_days`).

**When label** — the window in which the close occurs: `0–90`, `90–180`,
`180–365`, `>365 or never`. Reported both as a 4-class outcome and as the
survival-style quantity "probability of closing by each horizon".

**Customer-similarity label (interim, Stage 1)** — the account became a
customer at *any* later date. Weaker than the buy label (no contacted
non-converters), but available as soon as a customer list exists.

## 4. Data the team must supply

### 4.1 Customer list (Stage 1) — required

```csv
company_name,domain,customer_since,closed_won_at,first_contact_at,product,deal_size_usd,status,employee_count,naics_code,city,state,zip
```

Minimum: `company_name`, `domain`, `customer_since`. Everything else raises
what can be measured: `closed_won_at` enables the *when* label,
`first_contact_at` bounds the leakage window, `status` separates active from
churned, `city/state/zip` lifts the match rate.

### 4.2 Outreach log (Stages 2–4) — required for response, purchase, and timing

One row per contact attempt:

```csv
account_key,contact_at,channel,campaign_id,message_version,delivered,reply_class,replied_at,meeting_at,opportunity_at,closed_won_at,deal_size_usd,assignment_rule
```

`assignment_rule` records how the contact delay was chosen (e.g. `random_0_14d`,
`rep_choice`). Randomized delays are what make the action question (Stage 4)
answerable; without them only response and purchase *prediction* are possible,
not contact-timing *policy*.

**Contacted non-converters are the most valuable rows in this file.** A
customer list alone tells the model who bought; the outreach log tells it who
was asked and said no.

### 4.3 Matching to the public corpus

Accounts are matched to the SBA/OSHA corpus by the existing precision-first
rule (normalized legal name + city + state + ZIP), extended with domain when the
public source carries one. The harness must report the match rate and the
unmatched rows; a match rate below ~60% is itself a finding (the ICP universe
is not covered by public sources) and blocks Stage 1 claims.

## 5. Leakage boundary and control construction

- Features use only events with `available_at ≤ as_of`; labels use
  `closed_won_at` (or `replied_at`, …) strictly after `as_of`.
- Accounts already customers at `as_of` are excluded from that row.
- `first_contact_at` must be `> as_of` for the buy label to count as a
  prediction rather than a description of an in-flight deal.
- **Matched controls share the customer's source-coverage pattern.** For every
  customer row, sample controls at the same `as_of` with the same NAICS
  3-digit, state, employee bucket, and — critically — the same set of covering
  sources (SBA-only, OSHA-only, both). Without this the model learns "looks
  like an account in our CRM" from coverage alone, exactly the shortcut the
  public proxy fell into.
- The customer list must not be joined to any public source that could carry
  the purchase itself (press releases, case studies) until Stage 3 review.

## 6. Evaluation protocol

**Backtrace grid.** For each customer, snapshots at `T−180`, `T−90`, `T−30`
relative to `closed_won_at`, plus the regular quarterly grid used by the
engine. Controls are sampled at the same dates.

**Split.** Point-in-time by date with an account holdout, as today. Random
splits are reported only as a bridge to public figures.

**Arms.** The same `B1_raw_latest … B7_story_tags` and every `C_*` control,
unchanged, so results are directly comparable with the public-proxy runs.
`C_missingness_only` must sit at chance under matched controls; if it does not,
the controls are wrong.

**Metrics.**

| Question | Metric | Reported as |
|---|---|---|
| Who | PR-AUC, precision@K (K = weekly capacity), lift over matched-control base rate | point + 95% bootstrap CI, 3 seeds |
| Who | Brier score and reliability table | needed before any probability is shown to a user |
| When | 4-window accuracy; probability-of-close-by-horizon calibration; concordance index on time-to-close | same |
| Action (Stage 4) | IPW / DR policy value of "contact now vs wait", uplift@K, Qini | requires randomized assignment |

**Acceptance criteria (proposed, to be agreed):**

1. precision@50 ≥ 2× the matched-control base rate at H = 180, with the CI
   excluding 1×, on the point-in-time split, across 3 seeds.
2. `C_shuffled_labels` at chance and `C_missingness_only` within its CI of
   chance in the same run.
3. B6 (full architecture) beats B1 (flat latest values) by a margin whose CI
   excludes zero — otherwise the architecture has not earned its complexity and
   the decision layer should run on B1.
4. For *when*: probability-of-close-by-180-days calibrated within ±5 pp in
   every reliability bucket that holds ≥ 30 accounts.

**Minimum data for a first read.** ≥ 100 matched customers for Stage 1;
≥ 300 for the four timing windows; ≥ 200 positive replies for Stage 2.
Below these, the report is labelled exploratory and no threshold is promoted
to the decision layer.

## 7. Harness changes — the fourth surface

`signal-engine ablate customers` with:

```
--customers data/private/customers.csv      required, Stage 1
--outreach  data/private/outreach.csv       optional, Stages 2–4
--product   sidekick-v1
--horizons  90,180,365
--controls-per-customer 20
--label buy|reply|meeting|similarity        default buy when outreach exists, else similarity
--arms / --models / --seeds / --bootstrap   as today
```

Components, in the order they run:

1. **Loader and validator** — schema check, date parsing, duplicate domains,
   customers with `closed_won_at < first_contact_at` flagged.
2. **Matcher** — reuse the corpus identity rule; write a match audit
   (`matched`, `ambiguous`, `unmatched`) with counts and examples.
3. **Label builder** — per `(account, as_of)`: eligibility, buy/when/reply
   labels per horizon, computed once from the pristine files and injected into
   every arm (same invariant as `pristine_labels`).
4. **Matched-control sampler** — coverage-aware stratified sampling described
   in §5, seeded.
5. **Evaluation** — reuse `evaluate_cell`, arms, masks, bootstrap; add the
   *when* metrics and reliability table.
6. **Report** — same JSON shape (`cells`, `aggregates`) plus `matchAudit`,
   `controlBalance` (covariate balance customers vs controls), and the
   acceptance-criteria checklist evaluated automatically.

Private inputs live under a new gitignored `data/private/`; reports never
contain company names, only account ids and aggregates.

## 8. Staged roadmap

| Stage | Question answered | Needs | Output |
|---|---|---|---|
| 0 — now | Does the public proxy test anything? | done (this week) | proxy redefined; fair architecture run on GPU |
| 1 | Do customers look different from matched controls before they bought? | customer CSV | customer-similarity score, labelled as such |
| 2 | Who replies / takes a meeting? | outreach log | response model; first real precision@K |
| 3 | Who buys, and in which window? | ≥ 300 closed-won with dates | buy + when models; calibrated probabilities to the decision layer |
| 4 | Contact now or wait? | randomized contact delays | action policy with IPW/DR value; re-enable EV decisions |

Action simulation and dollar EV stay disabled until Stage 3 numbers pass the
acceptance criteria — the same rule the handoff already states.

## 9. Risks

- **Low match rate** between CRM names and public identities; mitigated by
  domain matching and by reporting the rate before any modelling.
- **Survivorship / CRM bias**: customers are companies someone already found;
  matched controls reduce but do not remove this.
- **Population identification** via source coverage, exactly as in the public
  proxy; guarded by coverage-matched controls and the `C_missingness_only`
  acceptance check.
- **Small N**: a first customer list may be under 100; the harness reports
  exploratory status rather than silently producing wide-interval numbers.
- **Label leakage through public sources** that mention the purchase; reviewed
  in Stage 3.

## 10. Open questions

1. Which product versions and which CRM export are in scope for Stage 1, and
   who owns the export?
2. Can contacted non-converters be exported from the CRM (Mono-icarus
   delivery/outcome tables) with dates?
3. Is a randomized contact-delay experiment acceptable operationally (a
   fraction of accounts contacted 0–14 days later than the rep would choose)?
   Without it, Stage 4 cannot be evaluated honestly.
4. What is the weekly capacity K per rep, so precision@K is reported at the
   number the team actually uses?
