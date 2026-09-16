"""Local DuckDB authority for frozen data, experiment runs, traces, and caches."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import duckdb
from pydantic import BaseModel, ValidationError

from .contracts import (
    AccountRecord,
    BenchmarkResult,
    ExperimentRun,
    FactSheet,
    NormalizedEvent,
    Recommendation,
    RunEvent,
    RunState,
    StoryCard,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _json(model: BaseModel | dict[str, Any]) -> str:
    if isinstance(model, BaseModel):
        return model.model_dump_json()
    return json.dumps(model, separators=(",", ":"), sort_keys=True)


class ExperimentStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = duckdb.connect(str(path))
        self.lock = threading.RLock()
        self._migrate()

    def _migrate(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS account (
              account_id VARCHAR PRIMARY KEY,
              body JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS normalized_event (
              event_id VARCHAR PRIMARY KEY,
              account_id VARCHAR NOT NULL,
              signal_instance_id VARCHAR NOT NULL,
              signal_type VARCHAR NOT NULL,
              available_at TIMESTAMPTZ NOT NULL,
              body JSON NOT NULL
            );
            CREATE INDEX IF NOT EXISTS event_account_time_idx
              ON normalized_event(account_id, available_at);
            CREATE TABLE IF NOT EXISTS factsheet (
              factsheet_id VARCHAR PRIMARY KEY,
              account_id VARCHAR NOT NULL,
              signal_instance_id VARCHAR NOT NULL,
              signal_type VARCHAR NOT NULL,
              as_of TIMESTAMPTZ NOT NULL,
              semantic_hash VARCHAR NOT NULL,
              body JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS story_card (
              story_card_id VARCHAR PRIMARY KEY,
              scope VARCHAR NOT NULL,
              subject_id VARCHAR NOT NULL,
              semantic_hash VARCHAR NOT NULL,
              generator VARCHAR NOT NULL,
              body JSON NOT NULL,
              UNIQUE(scope, subject_id, semantic_hash, generator)
            );
            CREATE TABLE IF NOT EXISTS experiment_run (
              run_id VARCHAR PRIMARY KEY,
              state VARCHAR NOT NULL,
              spec_hash VARCHAR NOT NULL,
              body JSON NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_event (
              run_id VARCHAR NOT NULL,
              seq INTEGER NOT NULL,
              kind VARCHAR NOT NULL,
              body JSON NOT NULL,
              PRIMARY KEY(run_id, seq)
            );
            CREATE TABLE IF NOT EXISTS recommendation (
              run_id VARCHAR NOT NULL,
              account_id VARCHAR NOT NULL,
              rank INTEGER NOT NULL,
              body JSON NOT NULL,
              PRIMARY KEY(run_id, account_id)
            );
            CREATE TABLE IF NOT EXISTS benchmark (
              run_id VARCHAR NOT NULL,
              model_id VARCHAR NOT NULL,
              body JSON NOT NULL,
              PRIMARY KEY(run_id, model_id)
            );
            """
        )

    def close(self) -> None:
        self.connection.close()

    def upsert_account(self, account: AccountRecord) -> None:
        with self.lock:
            existing_row = self.connection.execute(
                "SELECT body::VARCHAR FROM account WHERE account_id = ?",
                [account.account_id],
            ).fetchone()
            if existing_row:
                existing = AccountRecord.model_validate_json(existing_row[0])
                account = account.model_copy(
                    update={
                        "employee_count": account.employee_count
                        if account.employee_count is not None
                        else existing.employee_count,
                        "source_identifiers": {
                            **existing.source_identifiers,
                            **account.source_identifiers,
                        },
                        "attributes": {**existing.attributes, **account.attributes},
                    }
                )
            self.connection.execute(
                "INSERT OR REPLACE INTO account VALUES (?, ?::JSON)",
                [account.account_id, _json(account)],
            )

    def upsert_accounts(self, accounts: list[AccountRecord]) -> None:
        if not accounts:
            return
        with self.lock:
            existing = {account.account_id: account for account in self.accounts()}
            merged = []
            for account in accounts:
                previous = existing.get(account.account_id)
                if previous:
                    account = account.model_copy(
                        update={
                            "employee_count": (
                                account.employee_count
                                if account.employee_count is not None
                                else previous.employee_count
                            ),
                            "source_identifiers": {
                                **previous.source_identifiers,
                                **account.source_identifiers,
                            },
                            "attributes": {
                                **previous.attributes,
                                **account.attributes,
                            },
                        }
                    )
                existing[account.account_id] = account
                merged.append([account.account_id, _json(account)])
            for offset in range(0, len(merged), 1_000):
                self.connection.executemany(
                    """
                    INSERT INTO account VALUES (?, ?::JSON)
                    ON CONFLICT(account_id) DO UPDATE SET body = excluded.body
                    """,
                    merged[offset : offset + 1_000],
                )

    def accounts(self) -> list[AccountRecord]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT body::VARCHAR FROM account ORDER BY account_id"
            ).fetchall()
        return [AccountRecord.model_validate_json(row[0]) for row in rows]

    def account(self, account_id: str) -> AccountRecord | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT body::VARCHAR FROM account WHERE account_id = ?",
                [account_id],
            ).fetchone()
        return AccountRecord.model_validate_json(row[0]) if row else None

    def accounts_by_ids(self, account_ids: list[str]) -> dict[str, AccountRecord]:
        if not account_ids:
            return {}
        placeholders = ", ".join("?" for _ in account_ids)
        with self.lock:
            rows = self.connection.execute(
                f"SELECT body::VARCHAR FROM account WHERE account_id IN ({placeholders})",
                account_ids,
            ).fetchall()
        accounts = [AccountRecord.model_validate_json(row[0]) for row in rows]
        return {account.account_id: account for account in accounts}

    def corpus_counts(self) -> tuple[int, int]:
        with self.lock:
            row = self.connection.execute(
                """
                SELECT
                  (SELECT count(*) FROM account),
                  (SELECT count(*) FROM normalized_event)
                """
            ).fetchone()
        assert row is not None
        return int(row[0]), int(row[1])

    def append_event(self, event: NormalizedEvent) -> bool:
        with self.lock:
            exists = self.connection.execute(
                "SELECT 1 FROM normalized_event WHERE event_id = ?", [event.event_id]
            ).fetchone()
            if exists:
                return False
            self.connection.execute(
                """
                INSERT INTO normalized_event
                VALUES (?, ?, ?, ?, ?, ?::JSON)
                """,
                [
                    event.event_id,
                    event.account_id,
                    event.signal_instance_id,
                    event.signal_type,
                    event.available_at,
                    _json(event),
                ],
            )
            return True

    def append_events(self, events: list[NormalizedEvent]) -> int:
        if not events:
            return 0
        with self.lock:
            before_row = self.connection.execute("SELECT count(*) FROM normalized_event").fetchone()
            assert before_row is not None
            before = int(before_row[0])
            values = [
                [
                    event.event_id,
                    event.account_id,
                    event.signal_instance_id,
                    event.signal_type,
                    event.available_at,
                    _json(event),
                ]
                for event in events
            ]
            for offset in range(0, len(values), 1_000):
                self.connection.executemany(
                    """
                    INSERT INTO normalized_event
                    VALUES (?, ?, ?, ?, ?, ?::JSON)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    values[offset : offset + 1_000],
                )
            after_row = self.connection.execute("SELECT count(*) FROM normalized_event").fetchone()
            assert after_row is not None
            after = int(after_row[0])
            return after - before

    def events(
        self,
        as_of: datetime | None = None,
        account_id: str | None = None,
    ) -> list[NormalizedEvent]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if as_of is not None:
            clauses.append("available_at <= ?")
            parameters.append(as_of)
        if account_id is not None:
            clauses.append("account_id = ?")
            parameters.append(account_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.lock:
            rows = self.connection.execute(
                f"""
                SELECT body::VARCHAR FROM normalized_event
                {where}
                ORDER BY available_at, event_id
                """,
                parameters,
            ).fetchall()
        return [NormalizedEvent.model_validate_json(row[0]) for row in rows]

    def save_factsheet(self, sheet: FactSheet) -> None:
        with self.lock:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO factsheet
                VALUES (?, ?, ?, ?, ?, ?, ?::JSON)
                """,
                [
                    sheet.factsheet_id,
                    sheet.account_id,
                    sheet.signal_instance_id,
                    sheet.signal_type,
                    sheet.as_of,
                    sheet.semantic_hash,
                    _json(sheet),
                ],
            )

    def factsheets(
        self,
        as_of: datetime | None = None,
        account_id: str | None = None,
    ) -> list[FactSheet]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if as_of is not None:
            clauses.append("as_of <= ?")
            parameters.append(as_of)
        if account_id is not None:
            clauses.append("account_id = ?")
            parameters.append(account_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.lock:
            rows = self.connection.execute(
                f"""
                SELECT body::VARCHAR FROM factsheet
                {where}
                QUALIFY row_number() OVER (
                  PARTITION BY signal_instance_id ORDER BY as_of DESC
                ) = 1
                ORDER BY account_id, signal_instance_id
                """,
                parameters,
            ).fetchall()
        return [FactSheet.model_validate_json(row[0]) for row in rows]

    def story_card(
        self, scope: str, subject_id: str, semantic_hash: str, generator: str
    ) -> StoryCard | None:
        with self.lock:
            row = self.connection.execute(
                """
                SELECT body::VARCHAR FROM story_card
                WHERE scope = ? AND subject_id = ? AND semantic_hash = ? AND generator = ?
                """,
                [scope, subject_id, semantic_hash, generator],
            ).fetchone()
        return StoryCard.model_validate_json(row[0]) if row else None

    def save_story_card(self, card: StoryCard) -> None:
        with self.lock:
            self.connection.execute(
                """
                INSERT INTO story_card
                VALUES (?, ?, ?, ?, ?, ?::JSON)
                ON CONFLICT(story_card_id) DO UPDATE SET
                  scope = excluded.scope,
                  subject_id = excluded.subject_id,
                  semantic_hash = excluded.semantic_hash,
                  generator = excluded.generator,
                  body = excluded.body
                """,
                [
                    card.story_card_id,
                    card.scope,
                    card.subject_id,
                    card.semantic_hash,
                    card.generator,
                    _json(card),
                ],
            )

    def create_run(self, run: ExperimentRun) -> bool:
        with self.lock:
            exists = self.connection.execute(
                "SELECT 1 FROM experiment_run WHERE spec_hash = ?", [run.spec_hash]
            ).fetchone()
            if exists:
                return False
            self.connection.execute(
                "INSERT INTO experiment_run VALUES (?, ?, ?, ?::JSON)",
                [run.run_id, run.state, run.spec_hash, _json(run)],
            )
            return True

    def save_run(self, run: ExperimentRun) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO experiment_run VALUES (?, ?, ?, ?::JSON)",
                [run.run_id, run.state, run.spec_hash, _json(run)],
            )

    def run(self, run_id: str) -> ExperimentRun | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT body::VARCHAR FROM experiment_run WHERE run_id = ?", [run_id]
            ).fetchone()
        return ExperimentRun.model_validate_json(row[0]) if row else None

    def run_by_spec_hash(self, spec_hash: str) -> ExperimentRun | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT body::VARCHAR FROM experiment_run WHERE spec_hash = ?",
                [spec_hash],
            ).fetchone()
        return ExperimentRun.model_validate_json(row[0]) if row else None

    def list_runs(self, limit: int = 30) -> list[ExperimentRun]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT body::VARCHAR FROM experiment_run
                ORDER BY json_extract_string(body, '$.created_at') DESC LIMIT ?
                """,
                [limit],
            ).fetchall()
        runs = []
        for row in rows:
            try:
                runs.append(ExperimentRun.model_validate_json(row[0]))
            except ValidationError:
                continue
        return runs

    def running_runs(self) -> list[ExperimentRun]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT body::VARCHAR FROM experiment_run
                WHERE state = 'running'
                ORDER BY json_extract_string(body, '$.created_at')
                """
            ).fetchall()
        return [ExperimentRun.model_validate_json(row[0]) for row in rows]

    def append_run_event(self, run_id: str, kind: str, payload: dict[str, Any]) -> RunEvent:
        with self.lock:
            row = self.connection.execute(
                "SELECT coalesce(max(seq), 0) + 1 FROM run_event WHERE run_id = ?",
                [run_id],
            ).fetchone()
            assert row is not None
            event = RunEvent(
                run_id=run_id,
                seq=int(row[0]),
                kind=kind,
                created_at=datetime.now(UTC),
                payload=payload,
            )
            self.connection.execute(
                "INSERT INTO run_event VALUES (?, ?, ?, ?::JSON)",
                [run_id, event.seq, event.kind, _json(event)],
            )
            return event

    def run_events(self, run_id: str, after: int = 0, limit: int = 200) -> list[RunEvent]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT body::VARCHAR FROM run_event
                WHERE run_id = ? AND seq > ?
                ORDER BY seq LIMIT ?
                """,
                [run_id, after, limit],
            ).fetchall()
        return [RunEvent.model_validate_json(row[0]) for row in rows]

    def save_recommendations(self, run_id: str, recommendations: list[Recommendation]) -> None:
        with self.lock:
            for recommendation in recommendations:
                self.connection.execute(
                    "INSERT OR REPLACE INTO recommendation VALUES (?, ?, ?, ?::JSON)",
                    [
                        run_id,
                        recommendation.account_id,
                        recommendation.rank,
                        _json(recommendation),
                    ],
                )

    def save_benchmark(self, run_id: str, result: BenchmarkResult) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO benchmark VALUES (?, ?, ?::JSON)",
                [run_id, result.model_id, _json(result)],
            )

    def benchmarks(self, run_id: str) -> list[BenchmarkResult]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT body::VARCHAR FROM benchmark
                WHERE run_id = ? ORDER BY model_id
                """,
                [run_id],
            ).fetchall()
        return [BenchmarkResult.model_validate_json(row[0]) for row in rows]

    def recommendations(self, run_id: str) -> list[Recommendation]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT body::VARCHAR FROM recommendation
                WHERE run_id = ? ORDER BY rank
                """,
                [run_id],
            ).fetchall()
        return [Recommendation.model_validate_json(row[0]) for row in rows]

    def is_terminal(self, run_id: str) -> bool:
        run = self.run(run_id)
        return bool(
            run
            and run.state
            in {
                RunState.COMPLETED,
                RunState.FAILED,
                RunState.CANCELLED,
                RunState.INDETERMINATE,
            }
        )
