"""FastAPI experiment harness with resumable run-event SSE."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import load_decision_context
from .contracts import ExperimentSpec, RecommendRequest
from .runs import RunCoordinator, build_experiment_spec, default_coordinator

ROOT = Path(__file__).resolve().parents[2]


class ChatRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4_000)


def _coordinator(request: Request) -> RunCoordinator:
    return cast(RunCoordinator, request.app.state.coordinator)


def _chat_to_spec(message: str, signal_contracts: dict[str, str]) -> ExperimentSpec:
    context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    public_mode = os.getenv("SIGNAL_ENGINE_CORPUS_MODE") == "public"
    lower = message.lower()
    model_ids = tuple(
        model_id
        for keyword, model_id in (
            ("heuristic", "heuristic-v1"),
            ("lightgbm", "lightgbm-v1"),
            ("tabpfn", "tabpfn-local-v2"),
        )
        if keyword in lower
    ) or ("heuristic-v1", "lightgbm-v1", "tabpfn-local-v2")
    capacity_match = re.search(r"(?:top|return|capacity)\s+(\d{1,4})", lower)
    capacity = int(capacity_match.group(1)) if capacity_match else 25
    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
    as_of = (
        datetime.fromisoformat(date_match.group(1)).replace(tzinfo=UTC)
        if date_match
        else datetime(2026, 6, 30, tzinfo=UTC)
    )
    return build_experiment_spec(
        context,
        as_of=as_of,
        corpus_version="public-sidekick-v1" if public_mode else "fixture-manufacturing-v1",
        capacity=capacity,
        max_accounts=1_000 if public_mode else 5_000,
        model_ids=model_ids,
        max_llm_story_cards=min(capacity, 5),
        signal_contracts=signal_contracts,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    coordinator = default_coordinator(ROOT)
    app.state.coordinator = coordinator
    yield
    coordinator.store.close()


app = FastAPI(
    title="Signal Simulation Engine R&D",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3100", "http://127.0.0.1:3100"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "signal-engine", "schemaVersion": 1}


@app.get("/v1/context")
def context(request: Request) -> dict[str, object]:
    coordinator = _coordinator(request)
    account_count, event_count = coordinator.store.corpus_counts()
    decision_context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    return {
        "schemaVersion": 1,
        "product": decision_context.product,
        "icp": decision_context.icp,
        "goal": decision_context.goal,
        "signalPolicy": decision_context.signal_policy,
        "signals": [
            {
                **definition.model_dump(mode="json"),
                "contract_hash": definition.contract_hash,
            }
            for definition in coordinator.registry.all()
        ],
        "accounts": account_count,
        "events": event_count,
    }


@app.post("/v1/chat/runs", status_code=202)
async def create_chat_run(body: ChatRunRequest, request: Request) -> dict[str, object]:
    coordinator = _coordinator(request)
    spec = _chat_to_spec(body.message, coordinator.registry.manifest())
    run, created = coordinator.request(spec)
    if created:
        coordinator.start(run.run_id)
    return {
        "schemaVersion": 1,
        "created": created,
        "run": run,
        "resolvedSpec": spec,
    }


@app.post("/v1/experiments", status_code=202)
async def create_experiment(spec: ExperimentSpec, request: Request) -> dict[str, object]:
    coordinator = _coordinator(request)
    run, created = coordinator.request(spec)
    if created:
        coordinator.start(run.run_id)
    return {"schemaVersion": 1, "created": created, "run": run}


@app.post("/v1/recommend", status_code=202)
async def recommend(body: RecommendRequest, request: Request) -> dict[str, object]:
    coordinator = _coordinator(request)
    context = load_decision_context(ROOT / "configs" / "side-manufacturing.yaml")
    public_mode = os.getenv("SIGNAL_ENGINE_CORPUS_MODE") == "public"
    if body.icp_version_id != context.icp.icp_version:
        raise HTTPException(409, "requested ICP version is not available")
    if body.goal_version_id != context.goal.goal_version:
        raise HTTPException(409, "requested goal version is not available")
    spec = build_experiment_spec(
        context,
        as_of=body.as_of,
        corpus_version="public-sidekick-v1" if public_mode else "fixture-manufacturing-v1",
        capacity=body.capacity,
        max_accounts=1_000 if public_mode else 5_000,
        max_llm_story_cards=min(body.capacity, 5),
        signal_contracts=coordinator.registry.manifest(),
    ).model_copy(update={"experiment_id": body.idempotency_key})
    run, created = coordinator.request(spec)
    if created:
        coordinator.start(run.run_id)
    return {
        "schemaVersion": 1,
        "created": created,
        "runId": run.run_id,
        "state": run.state,
    }


@app.get("/v1/experiments")
def list_experiments(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "runs": _coordinator(request).store.list_runs(limit),
    }


@app.get("/v1/experiments/{run_id}")
def read_experiment(run_id: str, request: Request) -> dict[str, object]:
    coordinator = _coordinator(request)
    run = coordinator.store.run(run_id)
    if run is None:
        raise HTTPException(404, "experiment not found")
    recommendations = coordinator.store.recommendations(run_id)
    accounts = coordinator.store.accounts_by_ids(
        [recommendation.account_id for recommendation in recommendations]
    )
    presented_recommendations = []
    for recommendation in recommendations:
        account = accounts.get(recommendation.account_id)
        presented_recommendations.append(
            {
                **recommendation.model_dump(mode="json"),
                "account_name": account.name if account else None,
                "naics_code": account.naics_code if account else None,
                "state": account.state if account else None,
            }
        )
    return {
        "schemaVersion": 1,
        "run": run,
        "benchmarks": coordinator.store.benchmarks(run_id),
        "recommendations": presented_recommendations,
    }


@app.get("/v1/experiments/{run_id}/events")
def read_events(
    run_id: str,
    request: Request,
    after: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, object]:
    coordinator = _coordinator(request)
    if coordinator.store.run(run_id) is None:
        raise HTTPException(404, "experiment not found")
    return {
        "schemaVersion": 1,
        "events": coordinator.store.run_events(run_id, after),
        "terminal": coordinator.store.is_terminal(run_id),
    }


@app.get("/v1/experiments/{run_id}/stream")
def stream_events(
    run_id: str,
    request: Request,
    after: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    coordinator = _coordinator(request)
    if coordinator.store.run(run_id) is None:
        raise HTTPException(404, "experiment not found")

    async def generate() -> AsyncIterator[str]:
        cursor = after
        heartbeat = 0
        while True:
            events = coordinator.store.run_events(run_id, cursor)
            for event in events:
                cursor = event.seq
                payload = event.model_dump(mode="json")
                yield (
                    f"id: {event.seq}\n"
                    "event: run-event\n"
                    f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                )
            if coordinator.store.is_terminal(run_id) and not events:
                return
            heartbeat += 1
            if heartbeat % 30 == 0:
                yield ":hb\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/accounts/{account_id}")
def read_account(account_id: str, request: Request) -> dict[str, object]:
    coordinator = _coordinator(request)
    account = coordinator.store.account(account_id)
    if account is None:
        raise HTTPException(404, "account not found")
    return {
        "schemaVersion": 1,
        "account": account,
        "events": coordinator.store.events(account_id=account_id),
        "factsheets": coordinator.store.factsheets(account_id=account_id),
    }
