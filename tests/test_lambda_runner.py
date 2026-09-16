import json
import tarfile
from pathlib import Path

import httpx
import pytest

from signal_engine.lambda_runner import (
    InstanceOffer,
    LambdaApiError,
    LambdaBenchmarkJob,
    LambdaCapacityError,
    LambdaClient,
    LambdaSettings,
    LaunchNotApprovedError,
    LaunchPlan,
    SpendCeilingError,
    build_source_bundle,
    remote_upload_target,
    select_offer,
)
from signal_engine.outcome_benchmarks import (
    GPU_PROFILE,
    LAPTOP_PROFILE,
    resolve_profile,
    summarize_report,
)


def _offer(
    name: str, gpu: str, cents: int, regions: tuple[str, ...], gpus: int = 1
) -> InstanceOffer:
    return InstanceOffer(
        name=name,
        description=name,
        gpu_description=gpu,
        price_cents_per_hour=cents,
        gpus=gpus,
        vcpus=30,
        memory_gib=200,
        storage_gib=512,
        architecture="x86_64",
        regions=regions,
    )


OFFERS = [
    _offer("gpu_1x_a10", "A10 (24 GB PCIe)", 129, ("us-east-1", "us-west-1")),
    _offer("gpu_1x_a6000", "A6000 (48 GB)", 109, ()),
    _offer("gpu_1x_a100_sxm4", "A100 (40 GB SXM4)", 199, ("asia-south-1", "us-east-1")),
    _offer("gpu_1x_h100_pcie", "H100 (80 GB PCIe)", 329, ("us-west-3",)),
    _offer("gpu_8x_a100", "A100 (40 GB SXM4)", 1592, ("us-east-1",), gpus=8),
]


def test_select_offer_prefers_cheapest_available_single_gpu_with_enough_vram() -> None:
    offer, region = select_offer(OFFERS, min_vram_gb=40)
    assert offer.name == "gpu_1x_a100_sxm4"  # the 48 GB A6000 has no capacity
    assert region == "us-east-1"  # US regions preferred over asia-south-1


def test_select_offer_relaxes_to_24gb_when_nothing_larger_is_available() -> None:
    offer, _ = select_offer(OFFERS[:1], min_vram_gb=40)
    assert offer.name == "gpu_1x_a10"


def test_select_offer_honours_explicit_type_and_region() -> None:
    offer, region = select_offer(
        OFFERS, instance_type_name="gpu_1x_h100_pcie", region_name="us-west-3"
    )
    assert (offer.name, region) == ("gpu_1x_h100_pcie", "us-west-3")
    with pytest.raises(LambdaCapacityError):
        select_offer(OFFERS, instance_type_name="gpu_1x_h100_pcie", region_name="us-east-1")
    with pytest.raises(LambdaCapacityError):
        select_offer(OFFERS, instance_type_name="gpu_1x_a6000")


def test_remote_upload_target_is_posix_regardless_of_host_separators() -> None:
    remote_path, parent = remote_upload_target(
        "/home/ubuntu/signal-engine", "data\\cache\\ablation-corpus-full.json.gz"
    )
    assert remote_path == "/home/ubuntu/signal-engine/data/cache/ablation-corpus-full.json.gz"
    assert parent == "/home/ubuntu/signal-engine/data/cache"
    assert "\\" not in remote_path and "\\" not in parent
    remote_path, parent = remote_upload_target("/home/ubuntu/signal-engine", "data/raw/x.zip")
    assert (remote_path, parent) == (
        "/home/ubuntu/signal-engine/data/raw/x.zip",
        "/home/ubuntu/signal-engine/data/raw",
    )


def test_launch_plan_enforces_spend_ceiling() -> None:
    plan = LaunchPlan(
        job_id="test",
        offer=OFFERS[3],
        region_name="us-west-3",
        max_hours=2,
        max_usd=6,
        datasets=("olist",),
    )
    assert plan.ceiling_usd == pytest.approx(6.58)
    with pytest.raises(SpendCeilingError):
        plan.validate()
    LaunchPlan(
        job_id="test",
        offer=OFFERS[2],
        region_name="us-east-1",
        max_hours=1.5,
        max_usd=6,
        datasets=("olist", "uci-bank"),
    ).validate()


def test_source_bundle_excludes_secrets_data_and_caches(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "src" / "pkg" / "module.py").write_text("x = 1\n")
    (root / "src" / "pkg" / "__pycache__" / "module.cpython-313.pyc").write_bytes(b"\x00")
    (root / "src" / "pkg" / ".env.local").write_text("SECRET=1\n")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "big.zip").write_bytes(b"\x00" * 10)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "uv.lock").write_text("version = 1\n")
    manifest = build_source_bundle(root, tmp_path / "bundle.tar.gz")
    members = set(manifest["members"])  # type: ignore[arg-type]
    assert "src/pkg/module.py" in members
    assert "pyproject.toml" in members
    assert "uv.lock" in members
    assert not any("__pycache__" in member or member.endswith(".pyc") for member in members)
    assert not any(".env" in member for member in members)
    assert not any(member.startswith("data") for member in members)
    with tarfile.open(tmp_path / "bundle.tar.gz") as archive:
        assert all(info.uid == 0 for info in archive.getmembers())


def _client(handler) -> LambdaClient:  # type: ignore[no-untyped-def]
    settings = LambdaSettings(api_key="secret_test", base_url="https://cloud.example/api/v1")
    return LambdaClient(settings, transport=httpx.MockTransport(handler))


def test_client_sends_bearer_and_user_agent_and_parses_offers() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        seen["ua"] = request.headers["User-Agent"]
        assert request.url.path == "/api/v1/instance-types"
        return httpx.Response(
            200,
            json={
                "data": {
                    "gpu_1x_a10": {
                        "instance_type": {
                            "name": "gpu_1x_a10",
                            "description": "1x A10",
                            "gpu_description": "A10 (24 GB PCIe)",
                            "price_cents_per_hour": 129,
                            "specs": {
                                "vcpus": 30,
                                "memory_gib": 200,
                                "storage_gib": 1400,
                                "gpus": 1,
                            },
                            "architecture": "x86_64",
                        },
                        "regions_with_capacity_available": [
                            {"name": "us-east-1", "description": "Virginia, USA"}
                        ],
                    }
                }
            },
        )

    client = _client(handler)
    offers = client.instance_types()
    assert seen["auth"] == "Bearer secret_test"
    assert seen["ua"].startswith("signal-engine-lambda-runner/")
    assert offers[0].vram_gb == 24
    assert offers[0].regions == ("us-east-1",)
    assert offers[0].price_usd_per_hour == pytest.approx(1.29)


def test_client_maps_json_and_non_json_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ssh-keys"):
            return httpx.Response(
                401,
                json={
                    "error": {
                        "code": "global/invalid-api-key",
                        "message": "API key is invalid",
                        "suggestion": "Create a new key",
                    }
                },
            )
        return httpx.Response(
            403, text="<html>blocked</html>", headers={"content-type": "text/html"}
        )

    client = _client(handler)
    with pytest.raises(LambdaApiError) as json_error:
        client.ssh_keys()
    assert json_error.value.code == "global/invalid-api-key"
    assert json_error.value.status == 401
    with pytest.raises(LambdaApiError) as html_error:
        client.instances()
    assert html_error.value.code == "http/403"


def test_client_launch_posts_expected_body_and_maps_capacity_errors() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if body["instance_type_name"] == "gpu_1x_h100_pcie":
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "instance-operations/launch/insufficient-capacity",
                        "message": "Not enough capacity",
                    }
                },
            )
        return httpx.Response(200, json={"data": {"instance_ids": ["inst-1"]}})

    client = _client(handler)
    assert (
        client.launch(
            region_name="us-east-1",
            instance_type_name="gpu_1x_a100_sxm4",
            ssh_key_name="signal-engine-laptop",
            name="signal-engine-job",
        )
        == "inst-1"
    )
    assert calls[0] == {
        "region_name": "us-east-1",
        "instance_type_name": "gpu_1x_a100_sxm4",
        "ssh_key_names": ["signal-engine-laptop"],
        "name": "signal-engine-job",
    }
    with pytest.raises(LambdaCapacityError):
        client.launch(
            region_name="us-west-3",
            instance_type_name="gpu_1x_h100_pcie",
            ssh_key_name="k",
            name="n",
        )


def test_job_refuses_to_launch_without_explicit_approval(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no API call expected, got {request.method} {request.url}")

    settings = LambdaSettings(api_key="secret_test", base_url="https://cloud.example/api/v1")
    job = LambdaBenchmarkJob(
        settings,
        LambdaClient(settings, transport=httpx.MockTransport(handler)),
        root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        echo=lambda _: None,
    )
    plan = LaunchPlan(
        job_id="unapproved",
        offer=OFFERS[2],
        region_name="us-east-1",
        max_hours=1,
        max_usd=6,
        datasets=("olist",),
    )
    with pytest.raises(LaunchNotApprovedError):
        job.run(plan, approve_billable_launch=False)
    assert not (tmp_path / "artifacts" / "unapproved").exists()


def test_profiles_and_overrides_are_recorded() -> None:
    assert LAPTOP_PROFILE.tabpfn.max_training_rows == 600
    assert GPU_PROFILE.tabpfn.device == "cuda"
    assert GPU_PROFILE.tabpfn.ignore_pretraining_limits is True
    custom = resolve_profile("gpu", tabpfn_max_rows=20_000, tabpfn_estimators=4)
    assert custom.name == "gpu-full-custom"
    assert custom.tabpfn.max_training_rows == 20_000
    assert custom.tabpfn.n_estimators == 4
    assert resolve_profile("gpu", tabpfn_max_rows=0).tabpfn.max_training_rows is None
    assert resolve_profile("laptop") is LAPTOP_PROFILE
    with pytest.raises(ValueError):
        resolve_profile("tpu")


def test_summarize_report_reads_runtime_rows_used() -> None:
    report = {
        "benchmarks": [
            {
                "model_id": "tabpfn-local-v2",
                "status": "ok",
                "pr_auc": 0.29,
                "roc_auc": 0.60,
                "precision_at_k": 0.25,
                "brier_score": 0.21,
                "train_rows": 22103,
                "fit_seconds": 5.0,
                "score_seconds": 8.0,
                "detail": None,
            }
        ],
        "modelRuntimes": {"tabpfn-local-v2": {"trainingRowsUsed": 600}},
    }
    rows = summarize_report(report)
    assert rows[0]["trainRowsUsed"] == 600
    assert rows[0]["trainRows"] == 22103
