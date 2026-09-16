# Signal Simulation Engine R&D

An end-to-end research harness for testing whether longitudinal microsignals and
their sequences can identify manufacturers with a timely, evidence-backed
opportunity for [Sidekick](https://textside.com/).

This repository deliberately separates:

- **world evidence** — source-anchored events, FactSheets, and account state;
- **customer meaning** — versioned product, ICP, goal, and signal policy;
- **prediction** — heuristic, LightGBM, and optional TabPFN scorers;
- **decision** — contact-now / wait / never expected-value simulation;
- **presentation** — evidence-locked StoryCards for delivered accounts only.

An R&D opportunity score is not a probability that an account will buy Sidekick.
Only real outreach outcomes can calibrate that claim.

## Stack

- Python, FastAPI, Pydantic, DuckDB, Polars
- LightGBM and an optional TabPFN benchmark adapter
- Next.js 16, React 19, TypeScript, Tailwind 4
- pytest, Vitest, Playwright

## Start locally

```powershell
uv sync --all-extras --dev
pnpm install
uv run signal-engine serve --reload
pnpm dev
```

Open `http://127.0.0.1:3100`. The API is at `http://127.0.0.1:8100`.

Run a complete fixture experiment without the UI:

```powershell
uv run signal-engine demo
```

## Public data

Discover current official resources without downloading:

```powershell
uv run signal-engine data discover --sources sba,osha,epa_tri
```

Download a frozen SBA corpus and write a checksum manifest:

```powershell
uv run signal-engine data download --sources sba
```

Public imports write `data/public.duckdb`. Run the API against it with:

```powershell
$env:SIGNAL_ENGINE_CORPUS_MODE = "public"
uv run signal-engine serve
```

Raw and processed data are gitignored. Manifests, schemas, source declarations,
and small deterministic fixtures are committed. Tests never access the network.

SAM.gov is excluded from automated acquisition because its terms prohibit
scraping and require approved API/system-account access. Paid/private sources
and LinkedIn are outside this R&D data boundary.

The EPA TRI downloader targets EPA's documented national CSV endpoint. On the
initial 2026-08-29 acquisition attempt that endpoint returned HTTP 500, so the
failure is recorded rather than replaced with scraped or guessed data.

TabPFN is installed as an optional benchmark. Its current model checkpoint
requires accepting PriorLabs terms interactively; until that is done and
`TABPFN_ENABLE=1` is set, runs report TabPFN as unavailable and continue with
the heuristic and calibrated LightGBM scorers.

## Contracts and extension

Export language-neutral JSON Schemas:

```powershell
uv run signal-engine export-schemas
```

Read:

- [`docs/HANDOFF.md`](docs/HANDOFF.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/signal-creation.md`](docs/signal-creation.md)
- [`docs/data-and-labels.md`](docs/data-and-labels.md)
- [`docs/icarus-transfer.md`](docs/icarus-transfer.md)

## Verification

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
pnpm typecheck
pnpm test:web
pnpm build
```

## GPU benchmarks on Lambda Cloud

The local TabPFN benchmark is capped at 600 CPU rows. To run the Olist, UCI
Bank, and Hillstrom outcome benchmarks with full-partition TabPFN on CUDA:

```powershell
uv run --env-file .env.local signal-engine lambda verify        # read-only
uv run --env-file .env.local signal-engine lambda run --dry-run # plan only
uv run --env-file .env.local signal-engine lambda run --approve-billable-launch
```

Reports land in `data/manifests/*-gpu-v1.json`; launch, checksum, termination,
and cost evidence in `artifacts/lambda/<job-id>/`. See
[`docs/lambda-gpu-runner.md`](docs/lambda-gpu-runner.md).

## Ablations

`signal-engine ablate outcomes|uplift|architecture` runs model baselines
(logistic, random forest, XGBoost, CatBoost, LightGBM variants, TabPFN), split
protocols, feature-family masks, negative controls, S/T/X/DR/R uplift learners,
and one-layer-at-a-time architecture arms with seeds and bootstrap intervals.
Any of them can run on the GPU via `lambda run --remote "ablate ..."`. See
[`docs/ablations.md`](docs/ablations.md).
