"""Run the real-outcome benchmarks on a Lambda Cloud GPU instance with recorded evidence.

The runner is deliberately conservative:

- every API call is read-only until ``approve_billable_launch=True`` is passed explicitly;
- the instance id is written to ``artifacts/lambda/<job>/job.json`` the moment it exists;
- termination runs in a ``finally`` block, a watchdog thread enforces the time ceiling
  even if the SSH session hangs, and ``lambda cleanup`` can reap anything left behind;
- launch, bootstrap, execution, artifact checksums, termination, and estimated cost are
  all recorded so a benchmark number can be traced back to the machine that produced it.
"""

from __future__ import annotations

import io
import json
import os
import platform
import posixpath
import re
import shlex
import socket
import tarfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .hashing import sha256_file

DEFAULT_BASE_URL = "https://cloud.lambda.ai/api/v1"
USER_AGENT = "signal-engine-lambda-runner/0.1 (python-httpx)"
INSTANCE_NAME_PREFIX = "signal-engine-"
REMOTE_USER = "ubuntu"
REMOTE_PROJECT_DIR = "signal-engine"
REMOTE_BUNDLE_PATH = "/tmp/signal-engine-src.tar.gz"
DATASET_FILES: dict[str, str] = {
    "olist": "olist-marketing-funnel.zip",
    "uci-bank": "uci-bank-marketing.zip",
    "hillstrom": "hillstrom-email.csv.gz",
}
BUNDLE_ROOTS: tuple[str, ...] = (
    "src",
    "configs",
    "signals",
    "contracts",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
)
BUNDLE_EXCLUDED_DIRS = {"__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache"}
BUNDLE_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".duckdb", ".wal")
TERMINAL_STATUSES = {"terminated", "terminating", "preempted"}


class LambdaRunnerError(RuntimeError):
    """Base error for the runner."""


class LaunchNotApprovedError(LambdaRunnerError):
    """Raised when a billable launch is attempted without explicit approval."""


class SpendCeilingError(LambdaRunnerError):
    """Raised when the plan cannot fit under the configured spend ceiling."""


class LambdaApiError(LambdaRunnerError):
    def __init__(self, status: int, code: str, message: str, suggestion: str | None = None):
        detail = f"Lambda API {status} {code}: {message}"
        if suggestion:
            detail += f" ({suggestion})"
        super().__init__(detail)
        self.status = status
        self.code = code
        self.api_message = message
        self.suggestion = suggestion


class LambdaCapacityError(LambdaRunnerError):
    """Raised when no instance type/region satisfies the request right now."""


@dataclass(frozen=True)
class LambdaSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    region_name: str | None = None
    instance_type_name: str | None = None
    ssh_key_name: str | None = None
    ssh_private_key_path: Path | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> LambdaSettings:
        env = os.environ if environ is None else environ
        api_key = env.get("LAMBDA_API_KEY", "").strip()
        if not api_key:
            raise LambdaRunnerError(
                "LAMBDA_API_KEY is not set; add it to .env.local and run with "
                "`uv run --env-file .env.local ...`"
            )
        key_path = env.get("LAMBDA_SSH_PRIVATE_KEY_PATH", "").strip()
        return cls(
            api_key=api_key,
            base_url=(env.get("LAMBDA_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            region_name=env.get("LAMBDA_REGION_NAME") or None,
            instance_type_name=env.get("LAMBDA_INSTANCE_TYPE_NAME") or None,
            ssh_key_name=env.get("LAMBDA_SSH_KEY_NAME") or None,
            ssh_private_key_path=Path(key_path) if key_path else None,
        )


@dataclass(frozen=True)
class InstanceOffer:
    name: str
    description: str
    gpu_description: str
    price_cents_per_hour: int
    gpus: int
    vcpus: int
    memory_gib: int
    storage_gib: int
    architecture: str
    regions: tuple[str, ...]

    @property
    def price_usd_per_hour(self) -> float:
        return self.price_cents_per_hour / 100

    @property
    def vram_gb(self) -> int | None:
        match = re.search(r"(\d+)\s*GB", self.gpu_description or "")
        return int(match.group(1)) if match else None

    @property
    def available(self) -> bool:
        return bool(self.regions)

    @classmethod
    def from_api(cls, item: Mapping[str, Any]) -> InstanceOffer:
        instance_type = item["instance_type"]
        specs = instance_type.get("specs", {})
        return cls(
            name=str(instance_type["name"]),
            description=str(instance_type.get("description", "")),
            gpu_description=str(instance_type.get("gpu_description", "")),
            price_cents_per_hour=int(instance_type["price_cents_per_hour"]),
            gpus=int(specs.get("gpus", 0)),
            vcpus=int(specs.get("vcpus", 0)),
            memory_gib=int(specs.get("memory_gib", 0)),
            storage_gib=int(specs.get("storage_gib", 0)),
            architecture=str(instance_type.get("architecture", "x86_64")),
            regions=tuple(
                str(region["name"]) for region in item.get("regions_with_capacity_available", [])
            ),
        )

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "gpu": self.gpu_description,
            "vramGb": self.vram_gb,
            "gpus": self.gpus,
            "vcpus": self.vcpus,
            "memoryGib": self.memory_gib,
            "storageGib": self.storage_gib,
            "architecture": self.architecture,
            "priceUsdPerHour": self.price_usd_per_hour,
            "regionsWithCapacity": list(self.regions),
        }


class LambdaClient:
    """Thin, typed wrapper over the Lambda Cloud public API v1."""

    def __init__(
        self,
        settings: LambdaSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ):
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            timeout=httpx.Timeout(timeout, connect=20.0),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = self._client.request(method, path, json=payload)
        content_type = response.headers.get("content-type", "")
        body: Any = None
        if content_type.startswith("application/json"):
            try:
                body = response.json()
            except ValueError:
                body = None
        if response.is_success:
            return body.get("data") if isinstance(body, dict) else body
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            error = body["error"]
            raise LambdaApiError(
                response.status_code,
                str(error.get("code", "unknown")),
                str(error.get("message", "")),
                error.get("suggestion"),
            )
        text = body.get("title") if isinstance(body, dict) else response.text
        raise LambdaApiError(
            response.status_code,
            f"http/{response.status_code}",
            str(text or "")[:300] or "request rejected without a JSON error body",
        )

    # Read-only -----------------------------------------------------------------

    def instance_types(self) -> list[InstanceOffer]:
        data = self._request("GET", "/instance-types")
        return sorted(
            (InstanceOffer.from_api(item) for item in data.values()),
            key=lambda offer: (offer.price_cents_per_hour, offer.name),
        )

    def ssh_keys(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/ssh-keys"))

    def instances(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/instances"))

    def instance(self, instance_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/instances/{instance_id}"))

    # Mutations -------------------------------------------------------------------

    def add_ssh_key(self, name: str, public_key: str) -> dict[str, Any]:
        return dict(self._request("POST", "/ssh-keys", {"name": name, "public_key": public_key}))

    def delete_ssh_key(self, key_id: str) -> None:
        self._request("DELETE", f"/ssh-keys/{key_id}")

    def launch(
        self,
        *,
        region_name: str,
        instance_type_name: str,
        ssh_key_name: str,
        name: str,
        user_data: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "region_name": region_name,
            "instance_type_name": instance_type_name,
            "ssh_key_names": [ssh_key_name],
            "name": name[:64],
        }
        if user_data:
            payload["user_data"] = user_data
        try:
            data = self._request("POST", "/instance-operations/launch", payload)
        except LambdaApiError as error:
            if "insufficient-capacity" in error.code:
                raise LambdaCapacityError(
                    f"{instance_type_name} in {region_name} has no capacity right now: "
                    f"{error.api_message}"
                ) from error
            raise
        ids = data.get("instance_ids", [])
        if not ids:
            raise LambdaRunnerError(f"launch returned no instance ids: {data}")
        return str(ids[0])

    def terminate(self, instance_ids: Sequence[str]) -> list[dict[str, Any]]:
        data = self._request(
            "POST", "/instance-operations/terminate", {"instance_ids": list(instance_ids)}
        )
        return list(data.get("terminated_instances", []))


def select_offer(
    offers: Iterable[InstanceOffer],
    *,
    instance_type_name: str | None = None,
    region_name: str | None = None,
    min_vram_gb: int = 40,
    max_price_usd_per_hour: float | None = None,
    architecture: str = "x86_64",
) -> tuple[InstanceOffer, str]:
    """Pick an instance type and region that has capacity now.

    An explicit type wins; otherwise the cheapest single-GPU type with at least
    ``min_vram_gb`` is chosen, relaxing to 24 GB when nothing larger is available.
    """
    listed = list(offers)
    if instance_type_name:
        matches = [offer for offer in listed if offer.name == instance_type_name]
        if not matches:
            raise LambdaCapacityError(f"unknown instance type {instance_type_name!r}")
        offer = matches[0]
        if not offer.available:
            raise LambdaCapacityError(f"{offer.name} has no capacity in any region right now")
        return offer, _pick_region(offer, region_name)

    def candidates(vram_floor: int) -> list[InstanceOffer]:
        return [
            offer
            for offer in listed
            if offer.available
            and offer.gpus == 1
            and offer.architecture == architecture
            and (offer.vram_gb or 0) >= vram_floor
            and (
                max_price_usd_per_hour is None or offer.price_usd_per_hour <= max_price_usd_per_hour
            )
            and (region_name is None or region_name in offer.regions)
        ]

    for floor in sorted({min_vram_gb, 24}, reverse=True):
        found = sorted(candidates(floor), key=lambda offer: offer.price_cents_per_hour)
        if found:
            return found[0], _pick_region(found[0], region_name)
    available = ", ".join(
        f"{offer.name}[{','.join(offer.regions)}]" for offer in listed if offer.available
    )
    raise LambdaCapacityError(
        "no single-GPU instance with capacity matches the request right now; "
        f"currently available: {available or 'nothing'}"
    )


def _pick_region(offer: InstanceOffer, preferred: str | None) -> str:
    if preferred:
        if preferred not in offer.regions:
            raise LambdaCapacityError(
                f"{offer.name} has no capacity in {preferred}; available regions: "
                f"{', '.join(offer.regions) or 'none'}"
            )
        return preferred
    us_regions = [region for region in offer.regions if region.startswith("us-")]
    return (us_regions or list(offer.regions))[0]


@dataclass(frozen=True)
class LaunchPlan:
    job_id: str
    offer: InstanceOffer
    region_name: str
    max_hours: float
    max_usd: float
    datasets: tuple[str, ...]
    report_suffix: str = "gpu-v1"
    tabpfn_max_rows: int = 50_000
    tabpfn_estimators: int = 8
    random_seed: int = 7
    min_vram_gb: int = 40
    remote_commands: tuple[tuple[str, ...], ...] = ()
    uploads: tuple[str, ...] = ()

    @property
    def commands(self) -> tuple[tuple[str, ...], ...]:
        """Argument vectors after ``signal-engine``, executed in order on the instance."""
        if self.remote_commands:
            return self.remote_commands
        return (
            (
                "data",
                "benchmark-outcomes",
                "--profile",
                "gpu",
                "--datasets",
                ",".join(self.datasets),
                "--out-dir",
                "data/manifests/remote",
                "--suffix",
                self.report_suffix,
                "--tabpfn-max-rows",
                str(self.tabpfn_max_rows),
                "--tabpfn-estimators",
                str(self.tabpfn_estimators),
                "--seed",
                str(self.random_seed),
            ),
        )

    @property
    def ceiling_usd(self) -> float:
        return round(self.offer.price_usd_per_hour * self.max_hours, 2)

    @property
    def instance_name(self) -> str:
        return f"{INSTANCE_NAME_PREFIX}{self.job_id}"

    def validate(self) -> None:
        if self.max_hours <= 0 or self.max_usd <= 0:
            raise SpendCeilingError("max_hours and max_usd must both be positive")
        if self.ceiling_usd > self.max_usd:
            raise SpendCeilingError(
                f"{self.offer.name} at ${self.offer.price_usd_per_hour:.2f}/h for "
                f"{self.max_hours:g} h can cost ${self.ceiling_usd:.2f}, above the "
                f"${self.max_usd:.2f} ceiling; lower --max-hours, raise --max-usd, or choose a "
                "cheaper --instance-type"
            )
        unknown = sorted(set(self.datasets) - set(DATASET_FILES))
        if unknown:
            raise LambdaRunnerError(
                f"unknown datasets {unknown}; choose from {sorted(DATASET_FILES)}"
            )

    def describe(self) -> dict[str, object]:
        return {
            "jobId": self.job_id,
            "instanceName": self.instance_name,
            "instanceType": self.offer.describe(),
            "regionName": self.region_name,
            "maxHours": self.max_hours,
            "maxUsd": self.max_usd,
            "ceilingUsd": self.ceiling_usd,
            "datasets": list(self.datasets),
            "reportSuffix": self.report_suffix,
            "tabpfnMaxRows": self.tabpfn_max_rows,
            "tabpfnEstimators": self.tabpfn_estimators,
            "randomSeed": self.random_seed,
            "remoteCommands": ["signal-engine " + " ".join(args) for args in self.commands],
            "uploads": list(self.uploads),
        }


def new_job_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "host"


# SSH keys --------------------------------------------------------------------------


def generate_ed25519_keypair(private_path: Path, comment: str) -> str:
    """Create an OpenSSH ed25519 key pair locally; returns the public key line."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key = ed25519.Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.OpenSSH,
        serialization.PublicFormat.OpenSSH,
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(private_bytes)
    os.chmod(private_path, 0o600)
    public_line = f"{public_bytes.decode()} {comment}"
    private_path.with_suffix(".pub").write_text(public_line + "\n", encoding="utf-8")
    return public_line


def _public_key_material(line: str) -> str:
    parts = line.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else line.strip()


def ensure_ssh_key(
    client: LambdaClient,
    settings: LambdaSettings,
    key_dir: Path,
    log: Callable[[str], None],
) -> tuple[str, Path]:
    """Make sure Lambda knows a public key whose private half is on this machine."""
    key_name = settings.ssh_key_name or f"{INSTANCE_NAME_PREFIX}{_slug(platform.node())}"
    private_path = settings.ssh_private_key_path or key_dir / key_name
    public_path = private_path.with_suffix(".pub")
    registered = {str(key["name"]): key for key in client.ssh_keys()}
    if private_path.exists():
        if not public_path.exists():
            raise LambdaRunnerError(f"{private_path} exists but {public_path} is missing")
        public_line = public_path.read_text(encoding="utf-8").strip()
        if key_name in registered:
            remote_material = _public_key_material(str(registered[key_name]["public_key"]))
            if remote_material != _public_key_material(public_line):
                raise LambdaRunnerError(
                    f"Lambda SSH key {key_name!r} does not match the local key at {private_path}; "
                    "set LAMBDA_SSH_KEY_NAME to a new name or delete the stale key in Lambda"
                )
            log(f"[ssh] reusing registered key {key_name}")
            return key_name, private_path
        client.add_ssh_key(key_name, public_line)
        log(f"[ssh] registered existing local key as {key_name}")
        return key_name, private_path
    if key_name in registered:
        raise LambdaRunnerError(
            f"Lambda already has an SSH key named {key_name!r} but no private key exists at "
            f"{private_path}; set LAMBDA_SSH_PRIVATE_KEY_PATH or choose another LAMBDA_SSH_KEY_NAME"
        )
    public_line = generate_ed25519_keypair(private_path, comment=key_name)
    client.add_ssh_key(key_name, public_line)
    log(f"[ssh] generated ed25519 key at {private_path} and registered it as {key_name}")
    return key_name, private_path


# Source bundle ---------------------------------------------------------------------


def _bundle_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in BUNDLE_EXCLUDED_DIRS for part in parts):
        return None
    if info.name.endswith(BUNDLE_EXCLUDED_SUFFIXES):
        return None
    if Path(info.name).name.startswith(".env"):
        return None
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def build_source_bundle(root: Path, target: Path) -> dict[str, object]:
    """Tar the code needed to run benchmarks remotely; never data, secrets, or venvs."""
    target.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []
    with tarfile.open(target, "w:gz") as archive:
        for entry in BUNDLE_ROOTS:
            path = root / entry
            if not path.exists():
                continue
            archive.add(path, arcname=entry, filter=_bundle_filter)
            included.append(entry)
    with tarfile.open(target, "r:gz") as archive:
        members = archive.getnames()
    return {
        "path": target.as_posix(),
        "roots": included,
        "files": len(members),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "members": members,
    }


def remote_upload_target(remote_project: str, relative: str) -> tuple[str, str]:
    """Remote file path and its parent for a repo-relative upload, always POSIX.

    Remote paths must never go through ``pathlib.Path`` on Windows: its ``parent`` renders
    with backslashes, which made ``mkdir -p`` create a junk directory and the SFTP put fail.
    """
    remote_path = posixpath.join(remote_project, Path(relative).as_posix())
    return remote_path, posixpath.dirname(remote_path)


# Remote shell ----------------------------------------------------------------------


class RemoteShell:
    """Minimal paramiko session with streamed output and deadline enforcement."""

    def __init__(
        self,
        host: str,
        private_key_path: Path,
        known_hosts_path: Path,
        log: Callable[[str], None],
        username: str = REMOTE_USER,
    ):
        self.host = host
        self.username = username
        self.private_key_path = private_key_path
        self.known_hosts_path = known_hosts_path
        self.log = log
        self._client: Any = None

    def connect(self, *, deadline: float, retry_seconds: float = 10.0) -> None:
        import paramiko

        key = paramiko.Ed25519Key.from_private_key_file(str(self.private_key_path))
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    self.host,
                    username=self.username,
                    pkey=key,
                    timeout=20,
                    banner_timeout=45,
                    auth_timeout=45,
                    look_for_keys=False,
                    allow_agent=False,
                )
            except (OSError, paramiko.SSHException) as error:
                last_error = error
                client.close()
                self.log(f"[ssh] {self.host} not ready ({type(error).__name__}); retrying")
                time.sleep(retry_seconds)
                continue
            self.known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
            client.save_host_keys(str(self.known_hosts_path))
            self._client = client
            self.log(f"[ssh] connected to {self.username}@{self.host}")
            return
        raise TimeoutError(f"could not open SSH to {self.host}: {last_error}")

    def run(
        self,
        command: str,
        *,
        deadline: float,
        label: str,
        check: bool = True,
    ) -> tuple[int, list[str]]:
        if self._client is None:
            raise RuntimeError("remote shell is not connected")
        transport = self._client.get_transport()
        assert transport is not None
        channel = transport.open_session()
        channel.set_combine_stderr(True)
        channel.settimeout(15.0)
        channel.exec_command(f"bash -lc {shlex.quote(command)}")
        tail: list[str] = []
        pending = ""
        while True:
            if time.monotonic() > deadline:
                channel.close()
                raise TimeoutError(f"{label} exceeded the job deadline")
            try:
                chunk = channel.recv(65536)
            except TimeoutError:
                if channel.exit_status_ready() and not channel.recv_ready():
                    break
                continue
            if not chunk:
                break
            pending += chunk.decode("utf-8", errors="replace")
            *lines, pending = pending.split("\n")
            for line in lines:
                text = line.rstrip("\r")
                self.log(f"[{label}] {text}")
                tail.append(text)
                del tail[:-200]
        if pending.strip():
            self.log(f"[{label}] {pending.rstrip()}")
            tail.append(pending.rstrip())
        exit_status = int(channel.recv_exit_status())
        channel.close()
        if check and exit_status != 0:
            raise LambdaRunnerError(
                f"{label} failed with exit status {exit_status}; last output: "
                + " | ".join(tail[-5:])
            )
        return exit_status, tail

    def put(self, local: Path, remote: str) -> None:
        sftp = self._client.open_sftp()
        try:
            sftp.put(str(local), remote)
        finally:
            sftp.close()

    def put_text(self, text: str, remote: str, mode: int = 0o644) -> None:
        sftp = self._client.open_sftp()
        try:
            with sftp.file(remote, "w") as handle:
                handle.write(text)
            sftp.chmod(remote, mode)
        finally:
            sftp.close()

    def get(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp = self._client.open_sftp()
        try:
            sftp.get(remote, str(local))
        finally:
            sftp.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def wait_for_port(host: str, port: int, *, deadline: float, log: Callable[[str], None]) -> None:
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"{host}:{port} did not accept connections before the deadline")


# Remote scripts --------------------------------------------------------------------

BOOTSTRAP_SCRIPT = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"
echo "[bootstrap] host=$(hostname) kernel=$(uname -r) user=$(id -un)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || echo "[bootstrap] nvidia-smi unavailable"
if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
echo "[bootstrap] uv $(uv --version)"
if ! ldconfig -p | grep -q libgomp.so.1; then
  echo "[bootstrap] installing libgomp1 for LightGBM"
  (sudo -n apt-get update -qq && sudo -n apt-get install -y -qq libgomp1) \
    || echo "[bootstrap] libgomp1 install failed"
fi
mkdir -p "$HOME/__PROJECT__/data/raw" "$HOME/__PROJECT__/data/manifests/remote"
cd "$HOME/__PROJECT__"
tar -xzf __BUNDLE__
echo "[bootstrap] syncing Python 3.13 environment from uv.lock (torch CUDA wheels; slowest step)"
uv sync --frozen --no-dev --extra tabpfn --extra baselines --quiet
uv run --no-sync python - <<'PY'
import platform
import torch
ok = torch.cuda.is_available()
print(f"[bootstrap] python={platform.python_version()} torch={torch.__version__}")
print(f"[bootstrap] cuda_build={torch.version.cuda} cuda_available={ok}")
if ok:
    props = torch.cuda.get_device_properties(0)
    print(f"[bootstrap] gpu={torch.cuda.get_device_name(0)}")
    print(f"[bootstrap] vram_gib={props.total_memory / 1024**3:.1f}")
PY
echo "[bootstrap] done"
"""

CUDA_CHECK_SCRIPT = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/__PROJECT__"
uv run --no-sync python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"
"""

CUDA_FALLBACK_SCRIPT = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/__PROJECT__"
echo "[cuda-fallback] locked torch build cannot see the GPU; driver is:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader || true
echo "[cuda-fallback] reinstalling torch from the CUDA 12.8 wheel index"
uv pip install --python .venv --reinstall --quiet torch \
  --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python - <<'PY'
import torch
print(f"[cuda-fallback] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY
"""

WARMUP_SCRIPT = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/__PROJECT__"
set -a; . ./.env.remote; set +a
uv run --no-sync python - <<'PY'
import numpy as np
from tabpfn import TabPFNClassifier
rng = np.random.default_rng(0)
X = rng.random((64, 4))
y = (X[:, 0] > 0.5).astype(int)
model = TabPFNClassifier(n_estimators=1, device="cuda")
model.fit(X, y)
shape = model.predict_proba(X[:2]).shape
device = getattr(model, "devices_", None) or getattr(model, "device_", "?")
print(f"[warmup] tabpfn weights ready; device={device} proba={shape}")
PY
"""

EXECUTE_SCRIPT = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/__PROJECT__"
set -a; . ./.env.remote; set +a
mkdir -p data/manifests/remote
timeout --signal=TERM --kill-after=60 __TIMEOUT__ \
  bash remote-commands.sh 2>&1 \
  | tee remote-benchmark.log
"""

REMOTE_COMMANDS_HEADER = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/__PROJECT__"
set -a; . ./.env.remote; set +a
"""

CHECKSUM_SCRIPT = r"""
set -euo pipefail
cd "$HOME/__PROJECT__/data/manifests/remote"
find . -type f -name '*.json' -print0 | sort -z | xargs -0 sha256sum
"""


def _render(script: str, **values: str) -> str:
    rendered = script.replace("__PROJECT__", REMOTE_PROJECT_DIR).replace(
        "__BUNDLE__", REMOTE_BUNDLE_PATH
    )
    for key, value in values.items():
        rendered = rendered.replace(f"__{key}__", value)
    return rendered.strip() + "\n"


# Job -------------------------------------------------------------------------------


@dataclass
class JobRecord:
    job_id: str
    plan: dict[str, object]
    status: str = "planned"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    events: list[dict[str, object]] = field(default_factory=list)
    instance: dict[str, object] = field(default_factory=dict)
    bundle: dict[str, object] = field(default_factory=dict)
    artifacts: list[dict[str, object]] = field(default_factory=list)
    termination: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    watchdog_fired: bool = False

    def event(self, kind: str, **payload: object) -> None:
        self.events.append({"at": datetime.now(UTC).isoformat(), "kind": kind, **payload})


class LambdaBenchmarkJob:
    def __init__(
        self,
        settings: LambdaSettings,
        client: LambdaClient,
        *,
        root: Path,
        artifacts_dir: Path,
        echo: Callable[[str], None] = print,
        tabpfn_token: str | None = None,
        local_tabpfn_cache: Path | None = None,
    ):
        self.settings = settings
        self.client = client
        self.root = root
        self.artifacts_dir = artifacts_dir
        self.echo = echo
        self.tabpfn_token = tabpfn_token
        self.local_tabpfn_cache = local_tabpfn_cache
        self._log_handle: io.TextIOBase | None = None
        self._record: JobRecord | None = None
        self._record_path: Path | None = None

    # Logging / persistence -----------------------------------------------------

    def log(self, message: str) -> None:
        stamped = f"{datetime.now(UTC).strftime('%H:%M:%S')} {message}"
        self.echo(stamped)
        if self._log_handle is not None:
            self._log_handle.write(stamped + "\n")
            self._log_handle.flush()

    def _save(self) -> None:
        if self._record is None or self._record_path is None:
            return
        self._record_path.write_text(
            json.dumps(asdict(self._record), indent=2, sort_keys=True), encoding="utf-8"
        )

    # Main flow ------------------------------------------------------------------

    def run(
        self,
        plan: LaunchPlan,
        *,
        approve_billable_launch: bool,
        reuse_instance_id: str | None = None,
        skip_bootstrap: bool = False,
        keep_instance_on_failure: bool = False,
    ) -> dict[str, object]:
        plan.validate()
        if not approve_billable_launch:
            raise LaunchNotApprovedError(
                "launching a Lambda instance is billable; re-run with "
                "--approve-billable-launch after reviewing the plan"
            )
        job_dir = self.artifacts_dir / plan.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self._record = JobRecord(job_id=plan.job_id, plan=plan.describe())
        self._record_path = job_dir / "job.json"
        self._log_handle = (job_dir / "runner.log").open("a", encoding="utf-8")
        record = self._record
        started = time.monotonic()
        deadline = started + plan.max_hours * 3600
        instance_id: str | None = None
        watchdog: threading.Timer | None = None
        shell: RemoteShell | None = None
        failed = False
        try:
            key_name, key_path = ensure_ssh_key(
                self.client, self.settings, self.artifacts_dir / "ssh", self.log
            )
            record.event("ssh_key_ready", keyName=key_name)
            self._save()

            if reuse_instance_id:
                instance_id = reuse_instance_id
                self.log(f"[launch] reusing instance {instance_id}")
                record.event("instance_reused", instanceId=instance_id)
            else:
                self.log(
                    f"[launch] {plan.offer.name} ({plan.offer.gpu_description}) in "
                    f"{plan.region_name} at ${plan.offer.price_usd_per_hour:.2f}/h; "
                    f"ceiling ${plan.ceiling_usd:.2f} over {plan.max_hours:g} h"
                )
                instance_id = self.client.launch(
                    region_name=plan.region_name,
                    instance_type_name=plan.offer.name,
                    ssh_key_name=key_name,
                    name=plan.instance_name,
                )
                record.event("instance_launched", instanceId=instance_id)
            record.instance = {
                "id": instance_id,
                "launchedAt": datetime.now(UTC).isoformat(),
                "instanceType": plan.offer.name,
                "regionName": plan.region_name,
                "priceUsdPerHour": plan.offer.price_usd_per_hour,
            }
            record.status = "launched"
            self._save()

            watchdog = threading.Timer(
                plan.max_hours * 3600, self._watchdog_fire, args=(instance_id,)
            )
            watchdog.daemon = True
            watchdog.start()

            ip = self._wait_for_active(instance_id, deadline)
            record.instance["ip"] = ip
            self._save()
            wait_for_port(ip, 22, deadline=deadline, log=self.log)
            shell = RemoteShell(ip, key_path, job_dir / "known_hosts", self.log)
            shell.connect(deadline=deadline)

            if not skip_bootstrap:
                self._bootstrap(shell, plan, job_dir, deadline)
                record.status = "bootstrapped"
                record.event("bootstrap_complete")
                self._save()

            remaining = deadline - time.monotonic() - 300
            if remaining < 120:
                raise TimeoutError("less than two minutes remain for the benchmark itself")
            try:
                self._execute(shell, plan, int(remaining), deadline)
            except Exception as error:
                # A late step failing (or the remote timeout firing) must not discard the
                # reports earlier steps already wrote: fetch what exists, then re-raise.
                record.event("benchmark_failed", error=f"{type(error).__name__}: {error}")
                self.log("[execute] failed; fetching any reports written so far")
                try:
                    self._fetch(shell, plan, job_dir, min(deadline, time.monotonic() + 600))
                    record.event("partial_artifacts_fetched", count=len(record.artifacts))
                except Exception as fetch_error:
                    self.log(f"[fetch] partial fetch failed: {fetch_error}")
                raise
            record.status = "executed"
            record.event("benchmark_complete")
            self._save()

            self._fetch(shell, plan, job_dir, deadline)
            record.status = "completed"
            record.event("artifacts_fetched", count=len(record.artifacts))
        except BaseException as error:
            failed = True
            record.status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
            record.error = f"{type(error).__name__}: {error}"
            record.event("failed", error=record.error)
            self.log(f"[job] {record.status}: {record.error}")
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if shell is not None:
                shell.close()
            if instance_id is not None:
                if failed and keep_instance_on_failure:
                    self.log(
                        f"[terminate] SKIPPED (--keep-instance-on-failure); {instance_id} IS STILL "
                        "BILLING. Run `signal-engine lambda cleanup --yes` when done."
                    )
                    record.termination = {"skipped": True, "reason": "keep_instance_on_failure"}
                else:
                    self._terminate(instance_id)
            record.events.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "kind": "finished",
                    "elapsedSeconds": round(time.monotonic() - started, 1),
                }
            )
            self._save()
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
        return asdict(record)

    # Phases ---------------------------------------------------------------------

    def _wait_for_active(self, instance_id: str, deadline: float) -> str:
        last_status = None
        while time.monotonic() < deadline:
            instance = self.client.instance(instance_id)
            status = str(instance.get("status"))
            if status != last_status:
                self.log(f"[launch] instance {instance_id} status={status}")
                last_status = status
            if status == "active" and instance.get("ip"):
                assert self._record is not None
                self._record.instance.update(
                    {
                        "hostname": instance.get("hostname"),
                        "activeAt": datetime.now(UTC).isoformat(),
                        "image": instance.get("image"),
                    }
                )
                return str(instance["ip"])
            if status in TERMINAL_STATUSES or status == "unhealthy":
                raise LambdaRunnerError(f"instance {instance_id} entered status {status}")
            time.sleep(10)
        raise TimeoutError(f"instance {instance_id} did not become active before the deadline")

    def _bootstrap(
        self, shell: RemoteShell, plan: LaunchPlan, job_dir: Path, deadline: float
    ) -> None:
        assert self._record is not None
        bundle_path = job_dir / "signal-engine-src.tar.gz"
        bundle = build_source_bundle(self.root, bundle_path)
        self._record.bundle = {key: value for key, value in bundle.items() if key != "members"}
        self.log(
            f"[bundle] {bundle['files']} files, {bundle_path.stat().st_size / 1024:.0f} KiB, "
            f"sha256 {str(bundle['sha256'])[:12]}"
        )
        shell.put(bundle_path, REMOTE_BUNDLE_PATH)
        shell.run(_render(BOOTSTRAP_SCRIPT), deadline=deadline, label="bootstrap")

        status, _ = shell.run(
            _render(CUDA_CHECK_SCRIPT), deadline=deadline, label="cuda", check=False
        )
        if status != 0:
            self.log("[cuda] locked torch build cannot use the GPU; trying the cu128 wheel")
            shell.run(_render(CUDA_FALLBACK_SCRIPT), deadline=deadline, label="cuda-fallback")
            status, _ = shell.run(
                _render(CUDA_CHECK_SCRIPT), deadline=deadline, label="cuda", check=False
            )
            if status != 0:
                raise LambdaRunnerError(
                    "CUDA is not available to torch on the instance even after the fallback; "
                    "refusing to spend GPU money on a CPU run"
                )
        self._record.event("cuda_verified")

        remote_home = f"/home/{REMOTE_USER}"
        remote_project = f"{remote_home}/{REMOTE_PROJECT_DIR}"
        for key in plan.datasets:
            local = self.root / "data" / "raw" / DATASET_FILES[key]
            if not local.exists():
                raise LambdaRunnerError(f"dataset file missing locally: {local}")
            shell.put(local, f"{remote_project}/data/raw/{DATASET_FILES[key]}")
            self.log(f"[upload] {local.name} ({local.stat().st_size / 1024:.0f} KiB)")
        for relative in plan.uploads:
            local = self.root / relative
            if not local.is_file():
                raise LambdaRunnerError(f"upload missing locally: {local}")
            remote_path, remote_parent = remote_upload_target(remote_project, relative)
            shell.run(f"mkdir -p {shlex.quote(remote_parent)}", deadline=deadline, label="upload")
            shell.put(local, remote_path)
            self.log(f"[upload] {relative} ({local.stat().st_size / 1024**2:.1f} MiB)")

        env_lines = ["TABPFN_ENABLE=1", "TABPFN_NO_BROWSER=1", "TABPFN_ALLOW_CPU_LARGE_DATASET=0"]
        if self.tabpfn_token:
            env_lines.append(f"TABPFN_TOKEN={self.tabpfn_token}")
        shell.put_text("\n".join(env_lines) + "\n", f"{remote_project}/.env.remote", mode=0o600)

        status, tail = shell.run(
            _render(WARMUP_SCRIPT), deadline=deadline, label="warmup", check=False
        )
        if status != 0:
            checkpoints = (
                sorted(self.local_tabpfn_cache.glob("*.ckpt")) if self.local_tabpfn_cache else []
            )
            if not checkpoints:
                raise LambdaRunnerError(
                    "TabPFN could not download its weights on the instance and no local "
                    "checkpoint is available to upload: " + " | ".join(tail[-3:])
                )
            shell.run(f"mkdir -p {remote_home}/.cache/tabpfn", deadline=deadline, label="warmup")
            for checkpoint in checkpoints:
                size_mib = checkpoint.stat().st_size / 1024**2
                self.log(f"[upload] TabPFN checkpoint {checkpoint.name} ({size_mib:.0f} MiB)")
                shell.put(checkpoint, f"{remote_home}/.cache/tabpfn/{checkpoint.name}")
            shell.run(_render(WARMUP_SCRIPT), deadline=deadline, label="warmup")
            self._record.event("tabpfn_weights_uploaded", files=[c.name for c in checkpoints])
        else:
            self._record.event("tabpfn_weights_downloaded")

    def _execute(
        self, shell: RemoteShell, plan: LaunchPlan, timeout_seconds: int, deadline: float
    ) -> None:
        lines = [_render(REMOTE_COMMANDS_HEADER).rstrip("\n")]
        for index, args in enumerate(plan.commands, start=1):
            rendered = " ".join(shlex.quote(a) for a in args)
            lines.append(
                f'echo "[remote] step {index}/{len(plan.commands)}: signal-engine {rendered}"'
            )
            lines.append(f"uv run --no-sync signal-engine {rendered}")
        remote_project = f"/home/{REMOTE_USER}/{REMOTE_PROJECT_DIR}"
        shell.put_text("\n".join(lines) + "\n", f"{remote_project}/remote-commands.sh", mode=0o755)
        script = _render(EXECUTE_SCRIPT, TIMEOUT=str(timeout_seconds))
        self.log(
            f"[execute] remote timeout {timeout_seconds}s; {len(plan.commands)} command(s): "
            + " ; ".join("signal-engine " + " ".join(args) for args in plan.commands)
        )
        shell.run(script, deadline=deadline, label="benchmark")

    def _fetch(self, shell: RemoteShell, plan: LaunchPlan, job_dir: Path, deadline: float) -> None:
        assert self._record is not None
        _, lines = shell.run(_render(CHECKSUM_SCRIPT), deadline=deadline, label="checksum")
        remote_project = f"/home/{REMOTE_USER}/{REMOTE_PROJECT_DIR}"
        reports_dir = job_dir / "reports"
        published_dir = self.root / "data" / "manifests"
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].endswith(".json"):
                continue
            remote_sha, relative = parts
            name = relative.strip().removeprefix("./")
            if ".." in Path(name).parts:
                raise LambdaRunnerError(f"refusing suspicious remote artifact path {name!r}")
            local = reports_dir / name
            shell.get(f"{remote_project}/data/manifests/remote/{name}", local)
            local_sha = sha256_file(local)
            if local_sha != remote_sha:
                raise LambdaRunnerError(
                    f"checksum mismatch for {name}: remote {remote_sha} local {local_sha}"
                )
            published = published_dir / name
            published.parent.mkdir(parents=True, exist_ok=True)
            published.write_bytes(local.read_bytes())
            self._record.artifacts.append(
                {
                    "name": name,
                    "sha256": local_sha,
                    "path": local.as_posix(),
                    "publishedPath": published.as_posix(),
                }
            )
            self.log(f"[fetch] {name} sha256 {local_sha[:12]} -> {published}")
        shell.get(f"{remote_project}/remote-benchmark.log", job_dir / "remote-benchmark.log")
        self._record.artifacts.append(
            {"name": "remote-benchmark.log", "path": (job_dir / "remote-benchmark.log").as_posix()}
        )

    def _terminate(self, instance_id: str) -> None:
        assert self._record is not None
        record = self._record
        attempts = 0
        response: list[dict[str, Any]] = []
        while attempts < 5:
            attempts += 1
            try:
                response = self.client.terminate([instance_id])
                break
            except LambdaApiError as error:
                if error.status == 404:
                    self.log(f"[terminate] instance {instance_id} already gone")
                    break
                self.log(f"[terminate] attempt {attempts} failed: {error}; retrying")
                time.sleep(10)
        final_status = None
        wait_until = time.monotonic() + 300
        while time.monotonic() < wait_until:
            try:
                final_status = str(self.client.instance(instance_id).get("status"))
            except LambdaApiError as error:
                final_status = "terminated" if error.status == 404 else f"unknown ({error.code})"
                break
            if final_status in TERMINAL_STATUSES:
                break
            time.sleep(10)
        launched = record.instance.get("launchedAt")
        billed_hours = None
        if isinstance(launched, str):
            billed_hours = (
                datetime.now(UTC) - datetime.fromisoformat(launched)
            ).total_seconds() / 3600
        price = record.instance.get("priceUsdPerHour")
        estimated = (
            round(float(price) * billed_hours, 3)
            if isinstance(price, int | float) and billed_hours
            else None
        )
        record.termination = {
            "requestedAt": datetime.now(UTC).isoformat(),
            "attempts": attempts,
            "apiResponseIds": [str(item.get("id")) for item in response],
            "finalStatus": final_status,
            "billedHoursEstimate": round(billed_hours, 3) if billed_hours else None,
            "estimatedCostUsd": estimated,
            "note": "estimate = list price x wall time from launch to termination request",
        }
        cost_text = f"${estimated:.2f}" if estimated is not None else "n/a"
        self.log(
            f"[terminate] instance {instance_id} final status {final_status}; "
            f"~{(billed_hours or 0) * 60:.0f} min, est. {cost_text}"
        )
        if final_status not in TERMINAL_STATUSES:
            self.log(
                "[terminate] WARNING: termination not confirmed; run `signal-engine lambda status` "
                "and `lambda cleanup --yes` to make sure nothing is still billing"
            )

    def _watchdog_fire(self, instance_id: str) -> None:
        self.log(
            "[watchdog] time ceiling reached; terminating the instance from the watchdog thread"
        )
        if self._record is not None:
            self._record.watchdog_fired = True
        try:
            self.client.terminate([instance_id])
        except Exception as error:  # the watchdog must never raise
            self.log(f"[watchdog] terminate failed: {error}")


# Read-only verification --------------------------------------------------------------


def verify_environment(
    settings: LambdaSettings,
    client: LambdaClient,
    *,
    root: Path,
    instance_type_name: str | None,
    region_name: str | None,
    min_vram_gb: int,
    max_hours: float,
    max_usd: float,
    datasets: Sequence[str],
    tabpfn_token_present: bool,
) -> dict[str, object]:
    """Everything the runner needs, checked without launching anything."""
    result: dict[str, object] = {"baseUrl": settings.base_url}
    try:
        keys = client.ssh_keys()
        result["apiKey"] = {"valid": True, "sshKeysRegistered": [str(k["name"]) for k in keys]}
    except LambdaApiError as error:
        result["apiKey"] = {"valid": False, "error": str(error)}
        return result
    instances = client.instances()
    result["instances"] = [
        {
            "id": item.get("id"),
            "name": item.get("name"),
            "status": item.get("status"),
            "type": item.get("instance_type", {}).get("name"),
            "region": item.get("region", {}).get("name"),
        }
        for item in instances
    ]
    offers = client.instance_types()
    result["offersWithCapacity"] = [offer.describe() for offer in offers if offer.available]
    try:
        offer, region = select_offer(
            offers,
            instance_type_name=instance_type_name or settings.instance_type_name,
            region_name=region_name or settings.region_name,
            min_vram_gb=min_vram_gb,
        )
        plan = LaunchPlan(
            job_id="preview",
            offer=offer,
            region_name=region,
            max_hours=max_hours,
            max_usd=max_usd,
            datasets=tuple(datasets),
            min_vram_gb=min_vram_gb,
        )
        try:
            plan.validate()
            result["plan"] = {**plan.describe(), "fitsCeiling": True}
        except (SpendCeilingError, LambdaRunnerError) as error:
            result["plan"] = {**plan.describe(), "fitsCeiling": False, "error": str(error)}
    except LambdaCapacityError as error:
        result["plan"] = {"error": str(error)}
    result["datasets"] = {
        key: (root / "data" / "raw" / DATASET_FILES[key]).exists()
        for key in datasets
        if key in DATASET_FILES
    }
    result["tabpfnTokenPresent"] = tabpfn_token_present
    try:
        import paramiko  # noqa: F401

        result["paramiko"] = True
    except ImportError:
        result["paramiko"] = False
    return result


def find_runner_instances(client: LambdaClient) -> list[dict[str, Any]]:
    return [
        item
        for item in client.instances()
        if str(item.get("name") or "").startswith(INSTANCE_NAME_PREFIX)
        and str(item.get("status")) not in TERMINAL_STATUSES
    ]


def default_tabpfn_cache_dir() -> Path | None:
    candidates = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "tabpfn")
    candidates.append(Path.home() / ".cache" / "tabpfn")
    candidates.append(Path.home() / "Library" / "Caches" / "tabpfn")
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.ckpt")):
            return candidate
    return None
