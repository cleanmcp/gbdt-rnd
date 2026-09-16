"""Deterministic manufacturing corpus used for offline tests and UI demos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from .contracts import AccountRecord, NormalizedEvent, SignalShape


@dataclass(frozen=True)
class FixtureCorpus:
    accounts: list[AccountRecord]
    events: list[NormalizedEvent]
    latent_opportunity: dict[str, float]


def _event(
    *,
    account_id: str,
    instance_id: str,
    signal_type: str,
    source_id: str,
    kind: str,
    occurred_at: datetime,
    strength: float,
    payload: dict[str, object],
    shape: SignalShape,
) -> NormalizedEvent:
    available_at = occurred_at + timedelta(days=21)
    event_id = f"{instance_id}:{kind}:{occurred_at.date().isoformat()}"
    return NormalizedEvent(
        event_id=event_id,
        account_id=account_id,
        signal_instance_id=instance_id,
        signal_type=signal_type,
        source_id=source_id,
        source_record_id=instance_id,
        shape=shape,
        kind=kind,
        occurred_at=occurred_at,
        delivered_at=available_at,
        ingested_at=available_at,
        available_at=available_at,
        strength=float(np.clip(strength, 0, 1)),
        payload=payload,
    )


def build_fixture_corpus(size: int = 180, seed: int = 7) -> FixtureCorpus:
    rng = np.random.default_rng(seed)
    accounts: list[AccountRecord] = []
    events: list[NormalizedEvent] = []
    latent_opportunity: dict[str, float] = {}
    naics_options = ("311", "321", "325", "326", "332", "333", "334", "336")
    states = ("OH", "MI", "PA", "TX", "WI", "NC", "IN", "GA")
    base_date = datetime(2023, 1, 1, tzinfo=UTC)

    for index in range(size):
        account_id = f"fixture-manufacturer-{index:04d}"
        latent = float(rng.beta(2.2, 3.0))
        latent_opportunity[account_id] = latent
        employee_count = int(max(12, rng.lognormal(5.1 + latent * 0.7, 0.7)))
        naics = str(rng.choice(naics_options))
        state = str(rng.choice(states))
        accounts.append(
            AccountRecord(
                account_id=account_id,
                name=f"{state} Industrial Works {index:04d}",
                naics_code=naics,
                city=f"Plant City {index % 29}",
                state=state,
                employee_count=employee_count,
                source_identifiers={"fixture": account_id},
                attributes={
                    "frontline_share": round(float(rng.uniform(0.45, 0.9)), 3),
                    "facility_count": int(rng.integers(1, 8)),
                },
            )
        )

        production = float(rng.uniform(0.75, 1.05))
        for year in (2023, 2024, 2025):
            production *= 1 + float(rng.normal(latent * 0.13 - 0.025, 0.07))
            occurred = datetime(year, 7, 1, tzinfo=UTC)
            events.append(
                _event(
                    account_id=account_id,
                    instance_id=f"tri:{account_id}",
                    signal_type="epa_tri_report",
                    source_id="epa_tri",
                    kind="annual_report",
                    occurred_at=occurred,
                    strength=min(abs(production - 1) + latent * 0.4, 1),
                    shape=SignalShape.NUMERIC_SERIES,
                    payload={
                        "tri_facility_id": f"TRI-{index:06d}",
                        "facility_name": f"{state} Industrial Works {index:04d}",
                        "naics_code": naics,
                        "state": state,
                        "reporting_year": year,
                        "production_ratio": round(production, 4),
                        "total_releases_lb": round(max(0, rng.lognormal(7.5, 1.0) * production), 2),
                    },
                )
            )

        if rng.random() < 0.25 + latent * 0.55:
            approved = base_date + timedelta(days=int(rng.integers(250, 950)))
            loan_amount = float(rng.lognormal(13.2 + latent, 0.7))
            loan_id = f"SBA-{index:06d}"
            events.append(
                _event(
                    account_id=account_id,
                    instance_id=f"sba_loan:{loan_id}",
                    signal_type="sba_loan",
                    source_id="sba_7a_504",
                    kind="approved",
                    occurred_at=approved,
                    strength=min(np.log1p(loan_amount) / np.log1p(5_000_000), 1),
                    shape=SignalShape.STATE_MACHINE,
                    payload={
                        "loan_id": loan_id,
                        "borrower_name": f"{state} Industrial Works {index:04d}",
                        "naics_code": naics,
                        "state": state,
                        "status": "approved",
                        "loan_amount_usd": round(loan_amount, 2),
                    },
                )
            )
            events.append(
                _event(
                    account_id=account_id,
                    instance_id=f"sba_loan:{loan_id}",
                    signal_type="sba_loan",
                    source_id="sba_7a_504",
                    kind="disbursed",
                    occurred_at=approved + timedelta(days=int(rng.integers(20, 90))),
                    strength=min(np.log1p(loan_amount) / np.log1p(5_000_000), 1),
                    shape=SignalShape.STATE_MACHINE,
                    payload={
                        "loan_id": loan_id,
                        "borrower_name": f"{state} Industrial Works {index:04d}",
                        "naics_code": naics,
                        "state": state,
                        "status": "disbursed",
                        "loan_amount_usd": round(loan_amount, 2),
                    },
                )
            )

        for year in (2024, 2025):
            employees = max(10, int(employee_count * (0.83 + 0.08 * (year - 2024))))
            pressure = max(0.0, latent + float(rng.normal(0, 0.22)))
            total_cases = int(rng.poisson(max(0.5, employees / 90 * (0.6 + pressure))))
            occurred = datetime(year, 12, 31, tzinfo=UTC)
            events.append(
                _event(
                    account_id=account_id,
                    instance_id=f"osha:{account_id}",
                    signal_type="osha_injury_summary",
                    source_id="osha_ita",
                    kind="annual_report",
                    occurred_at=occurred,
                    strength=min(total_cases / max(employees / 40, 1), 1),
                    shape=SignalShape.NUMERIC_SERIES,
                    payload={
                        "establishment_id": f"OSHA-{index:06d}",
                        "establishment_name": f"{state} Industrial Works {index:04d}",
                        "naics_code": naics,
                        "state": state,
                        "reporting_year": year,
                        "total_recordable_cases": total_cases,
                        "days_away_cases": int(total_cases * rng.uniform(0.2, 0.7)),
                        "average_annual_employees": employees,
                    },
                )
            )

        if rng.random() < latent * 0.75:
            awarded = base_date + timedelta(days=int(rng.integers(700, 1120)))
            award_id = f"AWARD-{index:06d}"
            events.append(
                _event(
                    account_id=account_id,
                    instance_id=f"award:{award_id}",
                    signal_type="usaspending_award",
                    source_id="usaspending",
                    kind="awarded",
                    occurred_at=awarded,
                    strength=latent,
                    shape=SignalShape.EVENT_BURST,
                    payload={
                        "award_id": award_id,
                        "recipient_name": f"{state} Industrial Works {index:04d}",
                        "recipient_uei": f"UEI{index:09d}",
                        "naics_code": naics,
                        "action_type": "awarded",
                        "obligation_usd": round(float(rng.lognormal(13.5, 0.9)), 2),
                    },
                )
            )

        if rng.random() < 0.15 + latent * 0.55:
            created = base_date + timedelta(days=int(rng.integers(650, 1050)))
            job_id = f"JOB-{index:06d}"
            base_payload = {
                "job_id": job_id,
                "title": "Maintenance Technician",
                "location": f"Plant City {index % 29}, {state}",
                "status": "open",
                "minimum_experience_years": 6,
            }
            events.append(
                _event(
                    account_id=account_id,
                    instance_id=f"job:{job_id}",
                    signal_type="job_posting",
                    source_id="fixture_jobs",
                    kind="created",
                    occurred_at=created,
                    strength=0.5 + latent * 0.3,
                    shape=SignalShape.DOCUMENT_VERSIONS,
                    payload=base_payload,
                )
            )
            if latent > 0.5:
                events.append(
                    _event(
                        account_id=account_id,
                        instance_id=f"job:{job_id}",
                        signal_type="job_posting",
                        source_id="fixture_jobs",
                        kind="reposted",
                        occurred_at=created + timedelta(days=90),
                        strength=0.85,
                        shape=SignalShape.DOCUMENT_VERSIONS,
                        payload={
                            **base_payload,
                            "title": "Maintenance Technician",
                            "minimum_experience_years": 3,
                        },
                    )
                )
    return FixtureCorpus(accounts, events, latent_opportunity)
