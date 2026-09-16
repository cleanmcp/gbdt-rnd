"""Developer CLI for data acquisition, schema export, experiments, and serving."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from .ablations import (
    ARCHITECTURE_ARMS,
    CONTROLS,
    DEFAULT_ARCHITECTURE_ARMS,
    DEFAULT_ARCHITECTURE_MODELS,
    DEFAULT_BASE_LEARNERS,
    DEFAULT_MODELS,
    DEFAULT_UPLIFT_LEARNERS,
    FEATURE_FAMILIES,
    MODEL_SPECS,
    PROTOCOLS,
    export_manufacturing_corpus,
    run_architecture_ablation,
    run_outcome_ablation,
    run_uplift_ablation,
)
from .config import load_decision_context
from .contracts import (
    AccountRecord,
    AccountSnapshot,
    BenchmarkResult,
    CorpusManifest,
    ExperimentRun,
    ExperimentSpec,
    FactSheet,
    NormalizedEvent,
    PortfolioWeight,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
    RunEvent,
    StoryCard,
)
from .data_sources import discover_resources, download_public_corpus
from .lambda_runner import (
    DATASET_FILES,
    LambdaBenchmarkJob,
    LambdaClient,
    LambdaRunnerError,
    LambdaSettings,
    LaunchNotApprovedError,
    LaunchPlan,
    default_tabpfn_cache_dir,
    find_runner_instances,
    new_job_id,
    select_offer,
    verify_environment,
)
from .normalizers import normalize_epa_tri, normalize_osha_ita, normalize_sba
from .outcome_benchmarks import (
    resolve_profile,
    run_hillstrom_benchmark,
    run_olist_benchmark,
    run_outcome_benchmarks,
    run_uci_bank_benchmark,
)
from .runs import build_experiment_spec, default_coordinator
from .signal_registry import SignalDefinition
from .store import ExperimentStore

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
lambda_app = typer.Typer(
    no_args_is_help=True,
    help="Run the outcome benchmarks on a Lambda Cloud GPU with launch/terminate evidence.",
)
ablate_app = typer.Typer(
    no_args_is_help=True,
    help="Model, protocol, feature-family, and architecture-layer ablations with bootstrap CIs.",
)
app.add_typer(data_app, name="data")
app.add_typer(lambda_app, name="lambda")
app.add_typer(ablate_app, name="ablate")
ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = ROOT / "data" / "manifests"
ABLATION_DIR = MANIFESTS_DIR / "ablations"
LAMBDA_ARTIFACTS = ROOT / "artifacts" / "lambda"


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8100,
    reload: bool = False,
) -> None:
    import uvicorn

    uvicorn.run("signal_engine.api:app", host=host, port=port, reload=reload)


@app.command()
def demo(
    capacity: int = 25,
    as_of: str = "2026-06-30",
    max_accounts: int = 5_000,
) -> None:
    async def execute() -> None:
        coordinator = default_coordinator(ROOT)
        context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
        spec = build_experiment_spec(
            context,
            as_of=datetime.fromisoformat(as_of).replace(tzinfo=UTC),
            corpus_version=(
                "public-sidekick-v1"
                if os.getenv("SIGNAL_ENGINE_CORPUS_MODE") == "public"
                else "fixture-manufacturing-v1"
            ),
            capacity=capacity,
            max_accounts=max_accounts,
            signal_contracts=coordinator.registry.manifest(),
        )
        run, created = coordinator.request(spec)
        if created:
            await coordinator.execute(run.run_id)
        result = coordinator.store.run(run.run_id)
        recommendations = coordinator.store.recommendations(run.run_id)
        typer.echo(
            json.dumps(
                {
                    "run": result.model_dump(mode="json") if result else None,
                    "recommendations": [
                        recommendation.model_dump(mode="json")
                        for recommendation in recommendations[:5]
                    ],
                },
                indent=2,
            )
        )
        coordinator.store.close()

    asyncio.run(execute())


@data_app.command("discover")
def discover(
    sources: str = "sba,osha,epa_tri",
) -> None:
    resources = asyncio.run(discover_resources(tuple(sources.split(","))))
    typer.echo(
        json.dumps(
            [
                {
                    "sourceId": resource.source_id,
                    "url": resource.url,
                    "filename": resource.filename,
                    "license": resource.license,
                }
                for resource in resources
            ],
            indent=2,
        )
    )


@data_app.command("download")
def download(
    sources: str = "sba",
    corpus_version: str = "public-manufacturing-v1",
) -> None:
    manifest = asyncio.run(
        download_public_corpus(
            tuple(sources.split(",")),
            ROOT / "data" / "raw",
            corpus_version=corpus_version,
        )
    )
    manifest_dir = ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_dir / f"{corpus_version}.json"
    target.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(str(target))


@data_app.command("import-sba")
def import_sba(
    path: Path,
    ingested_at: str | None = None,
) -> None:
    timestamp = (
        datetime.fromisoformat(ingested_at).astimezone(UTC) if ingested_at else datetime.now(UTC)
    )
    accounts, events = normalize_sba(path, ingested_at=timestamp)
    typer.echo(json.dumps(_import_records(accounts, events)))


def _import_records(
    accounts: list[AccountRecord],
    events: list[NormalizedEvent],
) -> dict[str, int]:
    configured_path = os.getenv("SIGNAL_ENGINE_DB_PATH")
    store = ExperimentStore(
        Path(configured_path) if configured_path else ROOT / "data" / "public.duckdb"
    )
    store.upsert_accounts(accounts)
    created = store.append_events(events)
    store.close()
    return {
        "accounts": len(accounts),
        "events": len(events),
        "insertedEvents": created,
    }


@data_app.command("import-osha")
def import_osha(path: Path, ingested_at: str | None = None) -> None:
    timestamp = (
        datetime.fromisoformat(ingested_at).astimezone(UTC) if ingested_at else datetime.now(UTC)
    )
    accounts, events = normalize_osha_ita(
        path,
        ingested_at=timestamp,
        work_dir=ROOT / "data" / "cache",
    )
    typer.echo(json.dumps(_import_records(accounts, events)))


@data_app.command("import-epa-tri")
def import_epa_tri(path: Path, ingested_at: str | None = None) -> None:
    timestamp = (
        datetime.fromisoformat(ingested_at).astimezone(UTC) if ingested_at else datetime.now(UTC)
    )
    accounts, events = normalize_epa_tri(
        path,
        ingested_at=timestamp,
        work_dir=ROOT / "data" / "cache",
    )
    typer.echo(json.dumps(_import_records(accounts, events)))


@data_app.command("benchmark-olist")
def benchmark_olist(
    path: Path = ROOT / "data" / "raw" / "olist-marketing-funnel.zip",
) -> None:
    report = run_olist_benchmark(
        path,
        ROOT / "data" / "manifests" / "olist-outcome-benchmark-v1.json",
    )
    typer.echo(json.dumps(report, indent=2))


@data_app.command("benchmark-uci-bank")
def benchmark_uci_bank(
    path: Path = ROOT / "data" / "raw" / "uci-bank-marketing.zip",
) -> None:
    report = run_uci_bank_benchmark(
        path,
        ROOT / "data" / "manifests" / "uci-bank-outcome-benchmark-v1.json",
    )
    typer.echo(json.dumps(report, indent=2))


@data_app.command("benchmark-hillstrom")
def benchmark_hillstrom(
    path: Path = ROOT / "data" / "raw" / "hillstrom-email.csv.gz",
) -> None:
    report = run_hillstrom_benchmark(
        path,
        ROOT / "data" / "manifests" / "hillstrom-uplift-benchmark-v1.json",
    )
    typer.echo(json.dumps(report, indent=2))


@data_app.command("benchmark-outcomes")
def benchmark_outcomes(
    profile: str = typer.Option(
        "laptop", help="laptop = 600-row CPU TabPFN; gpu = full-partition CUDA TabPFN."
    ),
    datasets: str = typer.Option("olist,uci-bank,hillstrom", help="Comma-separated dataset keys."),
    out_dir: Annotated[Path, typer.Option(help="Where reports are written.")] = MANIFESTS_DIR,
    suffix: str | None = typer.Option(
        None, help="Report filename suffix (default v1 for laptop, gpu-v1 for gpu)."
    ),
    tabpfn_max_rows: int | None = typer.Option(None, help="Override the TabPFN training-row cap."),
    tabpfn_estimators: int | None = typer.Option(None, help="Override TabPFN n_estimators."),
    tabpfn_device: str | None = typer.Option(None, help="Override TabPFN device (cpu/cuda/auto)."),
    seed: int = 7,
) -> None:
    """Run Olist, UCI Bank, and Hillstrom under one execution profile and print a summary."""
    resolved = resolve_profile(
        profile,
        tabpfn_max_rows=tabpfn_max_rows,
        tabpfn_estimators=tabpfn_estimators,
        tabpfn_device=tabpfn_device,
    )
    summary = run_outcome_benchmarks(
        ROOT,
        out_dir,
        profile=resolved,
        datasets=tuple(part.strip() for part in datasets.split(",") if part.strip()),
        random_seed=seed,
        report_suffix=suffix or ("gpu-v1" if profile == "gpu" else "v1"),
        log=typer.echo,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _ablation_profile(
    profile: str,
    tabpfn_max_rows: int | None,
    tabpfn_estimators: int | None,
    tabpfn_device: str | None,
) -> tuple[Any, str]:
    resolved = resolve_profile(
        profile,
        tabpfn_max_rows=tabpfn_max_rows,
        tabpfn_estimators=tabpfn_estimators,
        tabpfn_device=tabpfn_device,
    )
    return resolved, ("gpu-v1" if profile == "gpu" else "v1")


@ablate_app.command("list")
def ablate_list() -> None:
    """Print every model, protocol, control, feature family, arm, and uplift learner key."""
    typer.echo(
        json.dumps(
            {
                "models": {key: spec.description for key, spec in MODEL_SPECS.items()},
                "protocols": {key: spec.description for key, spec in PROTOCOLS.items()},
                "controls": CONTROLS,
                "featureFamilies": {
                    dataset: sorted(families) for dataset, families in FEATURE_FAMILIES.items()
                },
                "architectureArms": {
                    key: arm.description for key, arm in ARCHITECTURE_ARMS.items()
                },
                "upliftLearners": list(DEFAULT_UPLIFT_LEARNERS),
                "upliftBaseLearners": list(DEFAULT_BASE_LEARNERS),
            },
            indent=2,
        )
    )


@ablate_app.command("outcomes")
def ablate_outcomes(
    datasets: str = "uci-bank,olist",
    models: str = ",".join(DEFAULT_MODELS),
    protocols: str = "point_in_time,random_stratified",
    masks: Annotated[
        str, typer.Option(help="Comma list of none | drop:<fam>[+fam] | only:<fam>.")
    ] = "none",
    controls: str = "none",
    seeds: int = 1,
    bootstrap: int = 200,
    profile: str = "laptop",
    out_dir: Annotated[Path, typer.Option(help="Report directory.")] = ABLATION_DIR,
    suffix: str | None = None,
    tabpfn_max_rows: int | None = None,
    tabpfn_estimators: int | None = None,
    tabpfn_device: str | None = None,
    seed: int = 7,
) -> None:
    """Model x protocol x feature-family x control grid on the real-outcome datasets."""
    resolved, default_suffix = _ablation_profile(
        profile, tabpfn_max_rows, tabpfn_estimators, tabpfn_device
    )
    summary = run_outcome_ablation(
        ROOT,
        out_dir,
        datasets=_csv(datasets),
        models=_csv(models),
        protocols=_csv(protocols),
        masks=_csv(masks),
        controls=_csv(controls),
        seeds=seeds,
        base_seed=seed,
        bootstrap_samples=bootstrap,
        profile=resolved,
        report_suffix=suffix or default_suffix,
        log=typer.echo,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@ablate_app.command("uplift")
def ablate_uplift(
    learners: str = ",".join(DEFAULT_UPLIFT_LEARNERS),
    bases: str = ",".join(DEFAULT_BASE_LEARNERS),
    masks: str = "none",
    protocols: str = "hashed_holdout",
    seeds: int = 1,
    bootstrap: int = 200,
    test_cap: Annotated[
        int, typer.Option(help="Cap on test rows; 0 keeps the whole holdout (~12.8k rows).")
    ] = 0,
    profile: str = "laptop",
    out_dir: Annotated[Path, typer.Option(help="Report directory.")] = ABLATION_DIR,
    suffix: str | None = None,
    tabpfn_max_rows: int | None = None,
    tabpfn_estimators: int | None = None,
    tabpfn_device: str | None = None,
    seed: int = 7,
) -> None:
    """Hillstrom meta-learner x base-learner grid with Qini, AUUC, uplift@k, IPW value."""
    resolved, default_suffix = _ablation_profile(
        profile, tabpfn_max_rows, tabpfn_estimators, tabpfn_device
    )
    summary = run_uplift_ablation(
        ROOT,
        out_dir,
        learners=_csv(learners),
        bases=_csv(bases),
        masks=_csv(masks),
        protocols=_csv(protocols),
        seeds=seeds,
        base_seed=seed,
        bootstrap_samples=bootstrap,
        test_cap=test_cap,
        profile=resolved,
        report_suffix=suffix or default_suffix,
        log=typer.echo,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@ablate_app.command("export-corpus")
def ablate_export_corpus(
    max_accounts: int = 10_000,
    db_path: Path | None = None,
    target: Path | None = None,
) -> None:
    """Write a compact accounts+events extract so the architecture ablation can run remotely."""
    written = export_manufacturing_corpus(
        db_path or ROOT / "data" / "public.duckdb",
        max_accounts,
        target or ROOT / "data" / "cache" / f"ablation-corpus-{max_accounts}.json.gz",
    )
    typer.echo(json.dumps(written, indent=2))


@ablate_app.command("architecture")
def ablate_architecture(
    arms: str = ",".join(DEFAULT_ARCHITECTURE_ARMS),
    models: str = ",".join(DEFAULT_ARCHITECTURE_MODELS),
    max_accounts: int = 10_000,
    corpus: Annotated[
        Path | None,
        typer.Option(help="data/public.duckdb (default) or an `ablate export-corpus` extract."),
    ] = None,
    seeds: int = 1,
    bootstrap: int = 200,
    precision_k: int = 100,
    label_mode: Annotated[
        str, typer.Option(help="any_event (engine parity) or new_instance (first event only).")
    ] = "any_event",
    population: Annotated[
        str, typer.Option(help="all | sba_history (existing borrowers only) | linked.")
    ] = "all",
    horizon_days: Annotated[
        int | None, typer.Option(help="Label horizon override (engine goal uses 180).")
    ] = None,
    profile: str = "laptop",
    out_dir: Annotated[Path, typer.Option(help="Report directory.")] = ABLATION_DIR,
    suffix: str | None = None,
    tabpfn_max_rows: int | None = None,
    tabpfn_estimators: int | None = None,
    tabpfn_device: str | None = None,
    seed: int = 7,
) -> None:
    """Add one feature layer at a time (B1 raw ... B7 story tags) plus negative controls."""
    resolved, default_suffix = _ablation_profile(
        profile, tabpfn_max_rows, tabpfn_estimators, tabpfn_device
    )
    summary = run_architecture_ablation(
        ROOT,
        out_dir,
        corpus_path=corpus,
        max_accounts=max_accounts,
        arms=_csv(arms),
        models=_csv(models),
        seeds=seeds,
        base_seed=seed,
        bootstrap_samples=bootstrap,
        precision_k=precision_k,
        profile=resolved,
        report_suffix=suffix or f"{label_mode}-{population}-{default_suffix}",
        label_mode=label_mode,
        population=population,
        horizon_days=horizon_days,
        log=typer.echo,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


def _lambda_context() -> tuple[LambdaSettings, LambdaClient]:
    try:
        settings = LambdaSettings.from_env()
    except LambdaRunnerError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    return settings, LambdaClient(settings)


def _dataset_keys(datasets: str) -> tuple[str, ...]:
    keys = tuple(part.strip() for part in datasets.split(",") if part.strip())
    unknown = sorted(set(keys) - set(DATASET_FILES))
    if unknown:
        typer.echo(f"unknown datasets {unknown}; choose from {sorted(DATASET_FILES)}", err=True)
        raise typer.Exit(code=2)
    return keys


@lambda_app.command("verify")
def lambda_verify(
    instance_type: str | None = typer.Option(None, help="Force an instance type name."),
    region: str | None = typer.Option(None, help="Force a region name."),
    min_vram_gb: int = typer.Option(40, help="Minimum GPU memory when auto-selecting."),
    max_hours: float = typer.Option(1.5, help="Hard wall-clock ceiling for a run."),
    max_usd: float = typer.Option(6.0, help="Spend ceiling = price/hour x max-hours."),
    datasets: str = "olist,uci-bank,hillstrom",
) -> None:
    """Read-only: validate the API key, list capacity, and preview the launch plan."""
    settings, client = _lambda_context()
    try:
        result = verify_environment(
            settings,
            client,
            root=ROOT,
            instance_type_name=instance_type,
            region_name=region,
            min_vram_gb=min_vram_gb,
            max_hours=max_hours,
            max_usd=max_usd,
            datasets=_dataset_keys(datasets),
            tabpfn_token_present=bool(os.getenv("TABPFN_TOKEN")),
        )
    finally:
        client.close()
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@lambda_app.command("run")
def lambda_run(
    approve_billable_launch: bool = typer.Option(
        False,
        "--approve-billable-launch",
        help="Required. Confirms you accept the billable instance launch shown in the plan.",
    ),
    instance_type: str | None = typer.Option(None, help="Instance type name (else auto-select)."),
    region: str | None = typer.Option(
        None, help="Region name (else first US region with capacity)."
    ),
    min_vram_gb: int = typer.Option(40, help="Minimum GPU memory when auto-selecting."),
    max_hours: float = typer.Option(1.5, help="Hard ceiling; the watchdog terminates at this age."),
    max_usd: float = typer.Option(6.0, help="Refuse plans whose price x max-hours exceeds this."),
    datasets: str = "olist,uci-bank,hillstrom",
    tabpfn_max_rows: int = typer.Option(50_000, help="TabPFN training-row cap on the GPU."),
    tabpfn_estimators: int = typer.Option(8, help="TabPFN ensemble size on the GPU."),
    seed: int = 7,
    suffix: str = typer.Option("gpu-v1", help="Report filename suffix for the GPU run."),
    reuse_instance: str | None = typer.Option(None, help="Attach to an existing instance id."),
    skip_bootstrap: bool = typer.Option(False, help="Skip environment setup (reused instance)."),
    keep_instance_on_failure: bool = typer.Option(
        False, help="Do NOT terminate on failure (keeps billing; use lambda cleanup)."
    ),
    dry_run: bool = typer.Option(False, help="Print the plan and exit without launching."),
    remote: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "signal-engine command(s) to run remotely instead of the default benchmark, "
                "e.g. --remote 'ablate outcomes --datasets uci-bank --seeds 3'. Repeatable; "
                "commands run in order on one instance. --profile gpu and an --out-dir under "
                "data/manifests/remote are appended when absent."
            )
        ),
    ] = None,
    upload: Annotated[
        list[str] | None,
        typer.Option(help="Extra repo-relative files to upload (e.g. an export-corpus extract)."),
    ] = None,
) -> None:
    """Launch a GPU instance, run the benchmarks, fetch reports, and terminate it."""
    uploads = tuple(upload or ())
    settings, client = _lambda_context()
    dataset_keys = _dataset_keys(datasets)
    remote_commands: list[tuple[str, ...]] = []
    for command in remote or []:
        parts = shlex.split(command)
        if not parts or parts[0] not in {"data", "ablate"}:
            typer.echo("--remote commands must start with 'data' or 'ablate'", err=True)
            raise typer.Exit(code=2)
        if "--profile" not in parts:
            parts += ["--profile", "gpu"]
        if "--out-dir" not in parts:
            parts += [
                "--out-dir",
                "data/manifests/remote/ablations"
                if parts[0] == "ablate"
                else "data/manifests/remote",
            ]
        remote_commands.append(tuple(parts))
    for relative in uploads:
        if not (ROOT / relative).exists():
            typer.echo(f"upload not found: {relative}", err=True)
            raise typer.Exit(code=2)
    try:
        offer, region_name = select_offer(
            client.instance_types(),
            instance_type_name=instance_type or settings.instance_type_name,
            region_name=region or settings.region_name,
            min_vram_gb=min_vram_gb,
        )
        plan = LaunchPlan(
            job_id=new_job_id(),
            offer=offer,
            region_name=region_name,
            max_hours=max_hours,
            max_usd=max_usd,
            datasets=dataset_keys,
            report_suffix=suffix,
            tabpfn_max_rows=tabpfn_max_rows,
            tabpfn_estimators=tabpfn_estimators,
            random_seed=seed,
            min_vram_gb=min_vram_gb,
            remote_commands=tuple(remote_commands),
            uploads=uploads,
        )
        plan.validate()
        typer.echo(json.dumps({"plan": plan.describe()}, indent=2, sort_keys=True))
        if dry_run:
            return
        job = LambdaBenchmarkJob(
            settings,
            client,
            root=ROOT,
            artifacts_dir=LAMBDA_ARTIFACTS,
            echo=typer.echo,
            tabpfn_token=os.getenv("TABPFN_TOKEN"),
            local_tabpfn_cache=default_tabpfn_cache_dir(),
        )
        try:
            record = job.run(
                plan,
                approve_billable_launch=approve_billable_launch,
                reuse_instance_id=reuse_instance,
                skip_bootstrap=skip_bootstrap,
                keep_instance_on_failure=keep_instance_on_failure,
            )
        except LaunchNotApprovedError as error:
            typer.echo(f"\n{error}", err=True)
            raise typer.Exit(code=2) from error
    except LambdaRunnerError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        client.close()
    typer.echo(
        json.dumps(
            {
                "jobId": record["job_id"],
                "status": record["status"],
                "instance": record["instance"],
                "termination": record["termination"],
                "artifacts": record["artifacts"],
                "evidence": (LAMBDA_ARTIFACTS / plan.job_id / "job.json").as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )


@lambda_app.command("status")
def lambda_status() -> None:
    """List live Lambda instances on the account and local job records."""
    _, client = _lambda_context()
    try:
        instances = client.instances()
    finally:
        client.close()
    jobs = []
    for record_path in sorted(LAMBDA_ARTIFACTS.glob("*/job.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        jobs.append(
            {
                "jobId": record.get("job_id"),
                "status": record.get("status"),
                "instanceId": record.get("instance", {}).get("id"),
                "estimatedCostUsd": record.get("termination", {}).get("estimatedCostUsd"),
                "terminationStatus": record.get("termination", {}).get("finalStatus"),
                "error": record.get("error"),
            }
        )
    typer.echo(
        json.dumps(
            {
                "instances": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "status": item.get("status"),
                        "type": item.get("instance_type", {}).get("name"),
                        "region": item.get("region", {}).get("name"),
                        "ip": item.get("ip"),
                    }
                    for item in instances
                ],
                "jobs": jobs,
            },
            indent=2,
            sort_keys=True,
        )
    )


@lambda_app.command("cleanup")
def lambda_cleanup(
    yes: bool = typer.Option(False, "--yes", help="Actually terminate; otherwise just list."),
) -> None:
    """Terminate any live instance this runner launched (name prefix signal-engine-)."""
    _, client = _lambda_context()
    try:
        stray = find_runner_instances(client)
        if not stray:
            typer.echo("no live signal-engine instances")
            return
        for item in stray:
            typer.echo(f"{item.get('id')} {item.get('name')} status={item.get('status')}")
        if not yes:
            typer.echo("re-run with --yes to terminate the instances above")
            raise typer.Exit(code=2)
        terminated = client.terminate([str(item["id"]) for item in stray])
        typer.echo(json.dumps({"terminated": [item.get("id") for item in terminated]}, indent=2))
    finally:
        client.close()


@app.command("export-schemas")
def export_schemas() -> None:
    models = (
        AccountRecord,
        AccountSnapshot,
        BenchmarkResult,
        CorpusManifest,
        ExperimentRun,
        ExperimentSpec,
        FactSheet,
        NormalizedEvent,
        PortfolioWeight,
        RecommendRequest,
        RecommendResponse,
        Recommendation,
        RunEvent,
        SignalDefinition,
        StoryCard,
    )
    target_dir = ROOT / "contracts" / "generated"
    target_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        name = "".join(
            f"-{character.lower()}" if character.isupper() else character
            for character in model.__name__
        ).lstrip("-")
        target = target_dir / f"{name}.schema.json"
        target.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    typer.echo(f"wrote {len(models)} schemas to {target_dir}")
