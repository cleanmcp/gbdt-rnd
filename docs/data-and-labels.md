# Data acquisition and labels

## R&D boundary

The first vertical slice uses free public downloads/APIs only:

- no LinkedIn;
- no paid/private datasets;
- no automated SAM.gov scraping;
- no live network access in tests;
- no LLM calls without an explicit key and per-run budget.

Official bulk files are preferred to browser automation. Browser discovery is
acceptable only to locate an official download; a reproducible downloader and
checksum manifest must own the corpus.

## First corpus

1. **SBA 7(a)/504 FOIA** — quarterly public CSVs; financing lifecycle and amount.
2. **EPA TRI** — annual facility reports; production/operational trajectory.
3. **OSHA ITA** — annual establishment injury/illness summaries; workforce and
   operational-pressure trajectory.
4. **USAspending** — later free/API source for award and renewal sequences.

The first three intentionally overrepresent financed, regulated, or reporting
establishments. Evaluation must report coverage by NAICS, geography, and source.

The current SBA FOIA files do not expose a documented loan-number column. The
adapter therefore preserves `LocationID` and derives a reproducible source-row
anchor from program, location, borrower match key, approval date, amount, and
lender. It marks `loan_id_is_source_native=false`; this must not be presented as
an SBA-issued loan identifier.

Cross-source account linkage is deliberately conservative: normalized legal
name plus city, state, and five-digit ZIP must agree. Legal suffixes and
punctuation are ignored; fuzzy name-only merges are forbidden. This sacrifices
recall to prevent sequences from being assembled across different companies.

## Frozen snapshot acquired on 2026-08-29

The checked-in manifest `data/manifests/public-sidekick-v1.json` covers five
official files (two SBA and three OSHA). Raw bytes remain gitignored.

- 130,226 normalized manufacturing accounts;
- 264,804 normalized lifecycle/annual events;
- 55,048 accounts with multiple OSHA reporting years;
- 1,245 conservatively linked accounts with both SBA and OSHA signal types.

A 500-account point-in-time smoke benchmark completed over the real corpus:
calibrated LightGBM reached PR-AUC 0.150 and precision@5 0.20 for the explicitly
non-production `public_future_event` proxy. The heuristic baseline reached
PR-AUC 0.014 and precision@5 0.00. These results prove the execution and
temporal-evaluation path; they do not estimate Sidekick purchase propensity.

## Source-time rules

Backtests use `available_at`, not the date printed on the underlying event.

- SBA FOIA is quarterly and typically available after quarter end.
- TRI annual files can be revised; a frozen download is one historical view.
- OSHA annual data is published after the reporting year.

Where historical publication snapshots are unavailable, an estimated
availability timestamp is stored and the UI labels the result accordingly.

## Labels

Signal sequences can produce an evidence-backed Sidekick opportunity hypothesis:

```text
financing → production change → safety/training pressure → frontline hiring
```

They cannot validate themselves. Training a model against a label calculated
from the same sequence would only teach the model to reproduce the hand-written
rule.

The harness therefore keeps two separate R&D benchmarks:

### Synthetic oracle

A generated manufacturing world plants known latent opportunity and timing
relationships. This proves that feature extraction, sequence detection, model
training, and simulation can recover a known truth.

### Public future-event proxy

Point-in-time account state predicts a later observable operational event, such
as new financing, a contract award, or frontline hiring. This tests real noise,
missingness, identity, and temporal generalization. It is not Sidekick purchase
propensity.

### Production outcome

Only contact, reply, meeting, opportunity, and closed-won outcomes tied to the
exact product/ICP/goal/action can calibrate real propensity. Those outcomes
remain a later Icarus integration.

## Data commands

```powershell
# Show current official resources.
uv run signal-engine data discover --sources sba,osha,epa_tri

# Download official bytes and produce a content-hashed manifest.
uv run signal-engine data download --sources sba

# Import an SBA CSV after inspecting the manifest.
uv run signal-engine data import-sba path\to\file.csv
```

Raw bytes and derived databases stay outside Git. A tiny deterministic fixture
keeps local tests and the UI repeatable at zero marginal cost.
