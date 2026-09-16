# Lambda Cloud GPU runner

The local TabPFN configuration is a laptop compromise: 600 training rows, two
estimators, CPU. The GPU runner moves the three real-outcome benchmarks (Olist,
UCI Bank Marketing, Hillstrom) onto a Lambda Cloud GPU where TabPFN sees the
full point-in-time training partition (capped at 50,000 rows) with eight
estimators on CUDA. Data, labels, splits, leakage boundaries, and LightGBM
settings are identical to the laptop run; only the TabPFN execution profile
changes, and the profile is recorded inside every report.

Implementation: [`src/signal_engine/lambda_runner.py`](../src/signal_engine/lambda_runner.py).
Profiles: [`src/signal_engine/outcome_benchmarks.py`](../src/signal_engine/outcome_benchmarks.py)
(`LAPTOP_PROFILE`, `GPU_PROFILE`).

## Safety model

- **Read-only until approved.** `lambda verify` and `lambda run --dry-run` only
  list keys, instances, and capacity. `lambda run` refuses to launch without
  `--approve-billable-launch`.
- **Spend and time ceilings.** `--max-hours` (default 1.5) bounds the job; the
  plan is rejected if `price/hour × max-hours` exceeds `--max-usd` (default 6).
  A watchdog thread terminates the instance at the ceiling even if the SSH
  session hangs, and the remote benchmark itself runs under `timeout`.
- **Termination in `finally`.** Success, failure, `Ctrl+C`, and timeouts all
  terminate the instance and poll until Lambda reports `terminated`. The only
  exception is the explicit `--keep-instance-on-failure` flag, which prints a
  loud warning; `lambda cleanup --yes` reaps any instance named
  `signal-engine-*`.
- **Evidence.** `artifacts/lambda/<job-id>/job.json` records the plan, instance
  id (written the moment it exists), status transitions, bootstrap outcome,
  artifact SHA-256s (verified against remote `sha256sum`), termination
  response, and an estimated cost. `runner.log` and `remote-benchmark.log`
  hold the streamed output.
- **No secrets in the bundle.** Only `src/`, `configs/`, `signals/`,
  `contracts/`, `pyproject.toml`, `uv.lock`, `.python-version`, and `README.md`
  are shipped. `.env*`, `data/`, `.venv`, and caches are excluded. The remote
  `.env.remote` (mode 600) contains only `TABPFN_TOKEN`, `TABPFN_ENABLE=1`, and
  `TABPFN_NO_BROWSER=1`; the Lambda and OpenAI keys never leave this machine.
- **SSH keys.** Lambda requires a registered key. The runner generates an
  ed25519 pair under `artifacts/lambda/ssh/` (gitignored) and registers it as
  `signal-engine-<hostname>`, or reuses `LAMBDA_SSH_KEY_NAME` +
  `LAMBDA_SSH_PRIVATE_KEY_PATH` if set. Host keys are stored per job in
  `artifacts/lambda/<job-id>/known_hosts`.

## Environment

```env
LAMBDA_API_KEY=                       # required; full account access, billable
LAMBDA_API_BASE_URL=https://cloud.lambda.ai/api/v1
LAMBDA_REGION_NAME=                   # optional; else first US region with capacity
LAMBDA_INSTANCE_TYPE_NAME=            # optional; else cheapest 1-GPU >= 40 GB with capacity
LAMBDA_SSH_KEY_NAME=                  # optional; reuse an existing Lambda key
LAMBDA_SSH_PRIVATE_KEY_PATH=          # optional; private half of that key
TABPFN_TOKEN=                         # forwarded so the instance can download TabPFN weights
```

The Lambda API sits behind Cloudflare, which rejects Python's default
`urllib` User-Agent with a 403 "browser signature banned". The client sends
`signal-engine-lambda-runner/0.1 (python-httpx)` instead.

## Commands

```powershell
# 1. Read-only: key validity, registered SSH keys, live instances, capacity, plan preview.
uv run --env-file .env.local signal-engine lambda verify

# 2. Print the exact plan (type, region, price, ceiling) without launching.
uv run --env-file .env.local signal-engine lambda run --dry-run

# 3. Billable. Launch, bootstrap, run the three benchmarks, fetch, terminate.
uv run --env-file .env.local signal-engine lambda run --approve-billable-launch

# Optional overrides
#   --instance-type gpu_1x_h100_pcie --region us-west-3
#   --max-hours 1 --max-usd 4
#   --datasets uci-bank,hillstrom
#   --tabpfn-max-rows 20000 --tabpfn-estimators 4
#   --reuse-instance <id> --skip-bootstrap   (iterate on a still-running instance)

# 4. Afterwards
uv run --env-file .env.local signal-engine lambda status
uv run --env-file .env.local signal-engine lambda cleanup --yes   # only if something is left
```

The same benchmark entry point runs locally, which is how the remote command is
exercised before spending money:

```powershell
uv run --env-file .env.local signal-engine data benchmark-outcomes --profile laptop --out-dir artifacts/dryrun
```

## What happens during `lambda run`

1. Select an instance type/region with capacity and validate the ceiling.
2. Ensure an SSH key is registered; launch `signal-engine-<job-id>`; record the id.
3. Wait for `active` + public IP, then for SSH (cloud-init needs a minute to
   install the key).
4. Upload the source bundle; install `uv`; `uv sync --frozen --no-dev --extra tabpfn`
   on Python 3.13 (this pulls the CUDA torch wheels and is the slowest step).
5. Verify `torch.cuda.is_available()`. If the locked torch build cannot see the
   driver, reinstall torch from the cu128 index once; if CUDA is still missing
   the job aborts rather than paying GPU prices for a CPU run.
6. Upload the three dataset files and `.env.remote`; warm up TabPFN so the v3
   checkpoint downloads via the PriorLabs token. If the download fails, the
   local checkpoint from `%APPDATA%\tabpfn` is uploaded instead.
7. Run `signal-engine data benchmark-outcomes --profile gpu ... --suffix gpu-v1`
   under `timeout`, streaming output locally.
8. `sha256sum` the reports remotely, download them, verify hashes, and publish
   them to `data/manifests/*-gpu-v1.json` next to the laptop reports.
9. Terminate, poll to `terminated`, and write the cost estimate.

## Reading the results

Each report gains `executionProfile`, `hardware` (GPU name, VRAM, versions),
and `modelRuntimes` (rows actually used, estimators, device, every OOM retry).
When TabPFN runs out of GPU memory it halves the training context (down to a
floor of 4,000 rows) and records each attempt instead of silently degrading.

Compare `*-v1.json` (laptop) with `*-gpu-v1.json` (GPU) on the same metrics;
the split and test rows are identical, so differences are attributable to the
TabPFN context size and ensemble, not to the data.

Expected duration is roughly 15–25 minutes end to end (boot 2–5 min,
environment sync 3–6 min, benchmarks 5–10 min), i.e. well under $1 on an A100
40 GB at $1.99/h.

## First run (2026-09-01)

Job `20260901T011845Z-157c13`: `gpu_1x_a100_sxm4` in us-east-1, launched
01:18:52Z, active 01:22:35Z, benchmarks 01:24:16–01:25:10Z, termination
requested 01:25:31Z — 6.7 minutes billed, ≈$0.22. The driver was 570.148.08,
so the cu128 fallback ran (torch 2.11.0+cu128). TabPFN used 2,842 / 22,103 /
34,158+17,020 training rows for Olist / UCI / Hillstrom with no OOM retries.
Headline: UCI Bank TabPFN ROC-AUC 60.4% → 75.8%, PR-AUC 29.0% → 45.3%.

## Known limits

- LightGBM stays on CPU on the instance; a CUDA LightGBM build is not worth
  the compile time for 22k–50k rows.
- cloud-init `user_data` is supported by the API but unused; bootstrap runs
  over SSH so every step is logged and retryable.
- The runner is single-job; run one `lambda run` at a time.
