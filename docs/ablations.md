# Ablation harness

`signal-engine ablate …` answers the Priority 1/2 questions in the handoff with one
evaluation core (fit → score → metrics → test-set bootstrap) applied to three surfaces.
Nothing here writes to DuckDB; every cell records the rows, prevalence, feature count,
runtime evidence, and a 95% bootstrap interval so a number can be traced to its
configuration. Implementation: [`src/signal_engine/ablations.py`](../src/signal_engine/ablations.py)
and [`src/signal_engine/uplift.py`](../src/signal_engine/uplift.py).

```powershell
uv run signal-engine ablate list          # every key below with a one-line description
```

## 1. Model × protocol × feature family (real outcomes)

```powershell
uv run --env-file .env.local signal-engine ablate outcomes `
  --datasets uci-bank,olist `
  --models positive-rate,logistic,random-forest,xgboost,catboost,lightgbm-default,lightgbm,lightgbm-full-train,lightgbm-regularized,tabpfn `
  --protocols point_in_time,point_in_time_recent,random_stratified,random_late_period `
  --masks none,drop:macro,drop:calendar,drop:prior_contact,only:prior_contact `
  --controls none,shuffled_labels `
  --seeds 3 --bootstrap 300
```

- **Models** — positive-rate (random targeting), logistic, random forest, XGBoost,
  CatBoost, LightGBM defaults, the engine LightGBM (newest 20% of training held out
  for Platt), the same LightGBM trained on every row, a shallow regularized LightGBM,
  and TabPFN under the active profile. XGBoost/CatBoost need `uv sync --extra baselines`.
- **Protocols** — the engine's chronological split, the same test with only the most
  recent third of training, a seeded stratified random holdout (the public protocol),
  and a random holdout restricted to the test era (drift vs era).
- **Masks** — named feature families per dataset (`uci-bank`: macro, calendar,
  prior_contact, client, channel; `olist`: origin, landing_page, calendar).
- **Controls** — `shuffled_labels` (must fall to chance) and `shuffled_as_of`
  (destroys chronology before a time-based split).

Reports: `data/manifests/ablations/<dataset>-ablation-<suffix>.json` with `cells`
(one per configuration × seed) and `aggregates` (mean/sd across seeds).

## 2. Uplift meta-learners (Hillstrom)

```powershell
uv run --env-file .env.local signal-engine ablate uplift `
  --learners random,response-propensity,s-learner,t-learner,x-learner,dr-learner,r-learner `
  --bases lightgbm,lightgbm-regularized,logistic,random-forest,xgboost,catboost,tabpfn `
  --protocols hashed_holdout,random_stratified --seeds 3 --bootstrap 300
```

Metrics per cell: sklift-style normalized Qini and AUUC (perfect-model normalization),
unscaled Qini coefficient, uplift@10/20/30%, IPW policy value for "treat if τ>0" and
"treat top 20%", plus IPW contact-all / contact-none references and the test ATE.
Bootstrap intervals cover normalized Qini, uplift@20, and the top-20% IPW value.
Propensity is the training treatment share (randomized design). DR- and R-learners
cross-fit their nuisance models (2 folds); the R-learner needs a base learner that
accepts `sample_weight` (not logistic, not TabPFN). Causal Forest and Criteo are not
implemented yet.

By default the whole hashed holdout (~12.8k rows) is the test set; the original
benchmark's 4,000-row cap leaves ~35 conversions and makes Qini swing by ±0.2 between
learners, which the first smoke run demonstrated.

## 3. Architecture layers (public manufacturing corpus)

```powershell
uv run --env-file .env.local signal-engine ablate architecture `
  --max-accounts 10000 --label-mode new_instance --population sba_history `
  --models positive-rate,logistic,lightgbm,lightgbm-full-train,tabpfn --seeds 3
```

Arms add one layer at a time on identical accounts, dates (the six-date training grid
runs use), labels, and splits:

| Arm | Adds |
|---|---|
| `B1_raw_latest` | latest source values + account size |
| `B2_pit_recency` | point-in-time lifecycle/recency descriptors, instance counts |
| `B3_factsheet` | max/mean/sum/delta/slope reducers, state/stage facts |
| `B4_decay` | stage-aware decayed scores, account roll-ups |
| `B5_sequences` | cross-signal sequence features |
| `B6_icp` | ICP conditioning (today's engine feature set) |
| `B7_story_tags` | deterministic StoryCard phase tags (flag-gated feature layer) |

Controls (at the B6 level): `C_shuffled_labels`, `C_shuffled_timestamps` (per-account
permutation of pre-history timestamps; labels provably unchanged), `C_no_sequences`,
`C_sequence_reversed`, `C_latest_only`, `C_no_decay`, `C_no_stage`, `C_no_sba`,
`C_no_osha`, and `C_missingness_only` (only which sources cover the account).

Labels are computed once from the pristine corpus and injected into every arm.
`--label-mode any_event` reproduces the engine rule (any SBA/USAspending/job event in
180 days, including later stages of an existing loan); `new_instance` counts only the
first event of a previously unseen instance. `--population` restricts rows to
`sba_history` (accounts with an SBA event already visible at `as_of`) or `linked`
(accounts with both SBA and OSHA events). `--horizon-days` overrides the 180-day
label window for the ablation only.

Sample size matters for the restricted populations: with `new_instance` labels and
`sba_history`, a new loan within 180 days occurs in only ~0.4% of (account, date)
rows (6,000 accounts → 3,484 training rows with 12 positives and 554 test rows with
3). Cells whose test partition has a single class are reported as `degenerate`, not
`ok`. Use the full corpus on the GPU box (`ablate export-corpus --max-accounts 130000`)
and/or `--horizon-days 365` before reading those arms.

For remote runs, export a compact corpus first and upload it:

```powershell
uv run signal-engine ablate export-corpus --max-accounts 10000
uv run --env-file .env.local signal-engine lambda run --approve-billable-launch `
  --upload data/cache/ablation-corpus-10000.json.gz `
  --remote "ablate architecture --corpus data/cache/ablation-corpus-10000.json.gz --max-accounts 10000 --label-mode new_instance --population sba_history --seeds 3"
```

Any `ablate …` or `data …` command can run on the GPU via `lambda run --remote`
(repeatable — several commands run in order on one instance);
`--profile gpu` and an `--out-dir` under `data/manifests/remote` are appended when
absent, and the fetched reports land under `data/manifests/ablations/`.

## First smoke findings (2026-09-01, laptop, single seed — directional only)

**UCI Bank, split-control decomposition (same duration-free features):**

| Protocol | logistic | LightGBM (engine cfg, all rows) | LightGBM (engine cfg, as benchmarked) |
|---|---|---|---|
| random stratified 5-fold | 79.2% | 80.2% | — |
| random within test era | 78.9% | 80.5% | — |
| chronological (engine) | 47.7% | 63.8% | 53.7% (raw 46.3%, flipped by Platt) |
| chronological, recent third only | 46.6% | 65.7% (shallow: 69.1%) | — |

Dropping the five macro time-proxy features lifts chronological logistic regression
from 47.7% to 69.9%. TabPFN at full context scores 75.8% on the chronological split.
Conclusion: protocol explains the whole GBDT gap to public numbers; the engine's
time-ordered Platt holdout inverted the model on UCI and should be replaced.

**Olist (point-in-time):** shallow regularized LightGBM 66.3% > logistic 64.6% >
random forest 62.4% ≈ TabPFN-600 62.2% > engine LightGBM 58.1%; `shuffled_labels`
puts every model at 43–50%; `drop:origin` costs 5–15 points, so acquisition origin is
most of the signal.

**Hillstrom (full holdout, LightGBM base):** random Qini −0.004 (sanity), T-learner
0.101 / uplift@20 +0.53 pp, response-propensity 0.051 / +0.97 pp, DR-learner −0.044.

**Architecture (1,500 accounts, `new_instance` labels, all accounts):**
`C_missingness_only` — six features saying which sources cover the account — scores
ROC-AUC 99.3%, identical to the full `B6_icp` architecture and to `C_no_sba`. The
unrestricted public proxy label is population identification (SBA-covered accounts lack
OSHA coverage and vice versa), not timing. The full-scale section below is the fair
version.

## Full-scale results (2026-09-01, GPU job 20260901T043846Z-32cf4a, 3 seeds)

**Architecture — fair quiz** (`--population sba_history --label-mode new_instance
--horizon-days 365`, 28,939 accounts, 81,356 / 11,327 train / test rows, 1,112 / 114
positives; ROC-AUC mean ± sd across seeds):

| Arm | logistic | engine LightGBM | regularized LightGBM | TabPFN (20k ctx) |
|---|---|---|---|---|
| positive-rate | 49.3 ± 3.5 | — | — | — |
| B1_raw_latest | 66.4 | 64.6 ± 0.1 | 69.7 ± 0.2 | 66.6 ± 0.3 |
| B2_pit_recency | **75.7** | **71.4 ± 0.6** | **75.0 ± 0.1** | 69.8 ± 4.0 |
| B3_factsheet | 74.2 | 70.5 ± 0.3 | 74.8 | 67.4 ± 5.2 |
| B4_decay | 75.1 | 71.5 ± 0.3 | 75.2 ± 0.1 | 71.2 ± 2.4 |
| B5_sequences | 75.3 | 71.3 ± 0.4 | 75.1 ± 0.1 | 70.0 ± 2.6 |
| B6_icp | 75.2 | 71.3 ± 0.4 | 75.1 | 71.8 ± 0.9 |
| B7_story_tags | 75.8 | 71.6 ± 0.4 | 75.4 ± 0.1 | 71.3 ± 2.5 |
| C_latest_only | 76.9 | 70.8 ± 0.5 | 75.1 ± 0.2 | 71.1 ± 1.3 |
| C_no_decay | 74.3 | 71.3 ± 0.8 | 75.2 ± 0.1 | 69.9 ± 1.8 |
| C_no_sequences | 75.1 | 71.3 ± 0.4 | 75.1 ± 0.2 | 73.0 ± 1.3 |
| C_sequence_reversed | 75.2 | 71.4 ± 0.4 | 75.1 ± 0.1 | 66.7 ± 5.4 |
| C_no_stage | 74.7 | 71.8 ± 0.2 | 75.0 | 71.8 ± 1.0 |
| C_no_sba | 71.0 | 70.3 ± 0.1 | 73.6 | 71.5 ± 0.2 |
| C_no_osha | 76.2 | 71.0 ± 0.1 | 75.1 ± 0.1 | 73.2 ± 0.1 |
| C_shuffled_timestamps | 74.5 ± 0.9 | 70.4 ± 0.7 | 75.3 ± 0.1 | 73.1 ± 0.2 |
| C_missingness_only | 55.6 | 55.2 | 54.0 | 54.6 ± 0.6 |
| C_shuffled_labels | 48.0 ± 9.6 | 47.7 ± 5.8 | 47.4 ± 5.4 | 50.4 ± 2.8 |

Reading: the controls pass (labels shuffled → chance; coverage only → 55%), the
task has real signal (75% vs 49%), and **the entire gain over raw latest values
comes from B2's point-in-time lifecycle/recency descriptors**. Reducers, decay,
sequences, ICP conditioning, and story tags each move the number by ≤ 1 point,
inside seed noise; removing them costs nothing. PR-AUC is 2.6–3.0% against a 1.0%
base rate (≈2.7× lift); precision@100 is 2–5%. Logistic regression and the
shallow regularized LightGBM beat the engine LightGBM by ~4 points and TabPFN
(20k-row context of an 81k-row partition) by ~3–4, with TabPFN showing up to ±5
points of seed variance from context subsampling.

**UCI Bank grid** (ROC-AUC, 3 seeds): chronological — TabPFN 75.7 ± 0.2, engine
LightGBM (fixed) 64.7 ± 0.7, regularized LightGBM 64.7 ± 1.2, random forest
63.3 ± 0.5, LightGBM defaults 58.5, old holdout LightGBM 47.5 ± 0.9, logistic
47.7; random stratified — TabPFN 81.2, LightGBM 81.0, regularized 80.9, random
forest 80.3, logistic 79.8, old holdout LightGBM 64.1 ± 2.0.

**Olist grid** (ROC-AUC, 3 seeds): chronological — regularized LightGBM
66.5 ± 0.2, logistic 64.6, random forest 63.9 ± 1.3, TabPFN 63.5 ± 0.2, engine
LightGBM 60.1 ± 0.9, old holdout 57.1; random — regularized 68.5, random forest
68.0, logistic 67.6, TabPFN 67.3, LightGBM 66.6.

**Hillstrom uplift** (normalized Qini mean ± sd, 12,822-row holdout, holdout ATE
+0.47 pp): S-learner/LightGBM 0.125 ± 0.008, X-learner/LightGBM 0.105 ± 0.036,
T-learner/LightGBM 0.099 ± 0.007, X/logistic 0.076, T/TabPFN 0.071, X/TabPFN
0.070, R/LightGBM 0.068, DR/logistic 0.066, T/logistic 0.061, random 0.016 ± 0.113,
response-propensity −0.003 to −0.078, DR/TabPFN −0.096. Every 95% bootstrap
interval includes zero; LightGBM-based S/X/T learners are the consistent leaders.

XGBoost and CatBoost were `unavailable` in this run (the instance bootstrap did
not install the `baselines` extra); the bootstrap now installs it.

## Caveats

- Bootstrap intervals are over test rows only; seed variance is reported separately as
  the across-seed standard deviation in `aggregates`.
- TabPFN under the laptop profile still uses 600 rows; use `--profile gpu` via Lambda
  for full-context cells.
- The heuristic engine scorer is not in the grid because it reads only four engine
  features and degrades to a constant in most arms.
