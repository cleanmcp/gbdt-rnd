"""Source adapters that preserve anchors and source-time semantics."""

from __future__ import annotations

import math
import re
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .contracts import AccountRecord, NormalizedEvent, SignalShape
from .hashing import sha256_json


def _column(frame: pl.DataFrame, *candidates: str) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in frame.columns}
    for candidate in candidates:
        match = normalized.get(re.sub(r"[^a-z0-9]", "", candidate.lower()))
        if match:
            return match
    return None


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_business_name(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9 ]", " ", value.upper())
    tokens = [
        token
        for token in normalized.split()
        if token
        not in {
            "INC",
            "INCORPORATED",
            "LLC",
            "LTD",
            "LIMITED",
            "CORP",
            "CORPORATION",
            "CO",
            "COMPANY",
        }
    ]
    return " ".join(tokens)


def _account_match_hash(name: str, city: str, state: str, postal: str) -> str:
    return sha256_json(
        {
            "name": _normalize_business_name(name),
            "city": re.sub(r"[^A-Z0-9]", "", city.upper()),
            "state": state.upper(),
            "postal": re.sub(r"[^0-9]", "", postal)[:5],
        }
    )


def _number(value: Any) -> float | None:
    try:
        number = float(str(value).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_datetime(value: Any) -> datetime | None:
    raw = _text(value)
    if not raw:
        return None
    for pattern in (
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _quarterly_available_at(occurred_at: datetime) -> datetime:
    quarter_end_month = ((occurred_at.month - 1) // 3 + 1) * 3
    if quarter_end_month == 12:
        quarter_end = datetime(occurred_at.year + 1, 1, 1, tzinfo=UTC)
    else:
        quarter_end = datetime(occurred_at.year, quarter_end_month + 1, 1, tzinfo=UTC)
    return quarter_end + timedelta(days=31)


def _read_table(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".csv":
        return pl.read_csv(
            path,
            infer_schema_length=10_000,
            ignore_errors=True,
            truncate_ragged_lines=True,
            encoding="utf8-lossy",
        )
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"unsupported normalized input: {path}")


def _tables_in(path: Path, work_dir: Path) -> list[Path]:
    if path.suffix.lower() != ".zip":
        return [path]
    target = work_dir / path.stem
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        safe_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and Path(member.filename).suffix.lower() in {".csv", ".parquet"}
            and ".." not in Path(member.filename).parts
        ]
        for member in safe_members:
            archive.extract(member, target)
    return [target / member.filename for member in safe_members]


def normalize_sba(
    path: Path,
    *,
    ingested_at: datetime,
) -> tuple[list[AccountRecord], list[NormalizedEvent]]:
    frame = _read_table(path)
    name_col = _column(frame, "BorrName", "BorrowerName", "borrower_name")
    city_col = _column(frame, "BorrCity", "BorrowerCity", "borrower_city")
    state_col = _column(frame, "BorrState", "BorrowerState", "borrower_state")
    zip_col = _column(frame, "BorrZip", "BorrowerZip", "borrower_zip")
    street_col = _column(frame, "BorrStreet", "BorrowerStreet", "borrower_street")
    naics_col = _column(frame, "NAICS", "NaicsCode", "naics_code")
    approval_col = _column(frame, "ApprovalDate", "approval_date")
    disbursement_col = _column(frame, "FirstDisbursementDate", "first_disbursement_date")
    amount_col = _column(
        frame,
        "GrossApproval",
        "GrossApprovalAmount",
        "SBAGuaranteedApproval",
        "loan_amount",
    )
    loan_col = _column(frame, "LoanNumber", "LoanNmb", "LoanNum", "loan_number", "loan_id")
    location_col = _column(frame, "LocationID", "location_id")
    program_col = _column(frame, "Program", "program")
    lender_col = _column(frame, "BankName", "Lender", "lender_name")
    status_col = _column(frame, "LoanStatus", "loan_status")
    paid_col = _column(frame, "PaidInFullDate", "paid_in_full_date")
    charged_off_col = _column(frame, "ChargeOffDate", "charge_off_date")
    jobs_col = _column(frame, "JobsSupported", "jobs_supported")
    term_col = _column(frame, "TermInMonths", "term_in_months")
    if not all((name_col, state_col, naics_col, approval_col)):
        raise ValueError(
            "SBA schema missing required borrower name, state, NAICS, or approval date"
        )
    assert name_col and state_col and naics_col and approval_col

    accounts: dict[str, AccountRecord] = {}
    events: list[NormalizedEvent] = []
    for row in frame.iter_rows(named=True):
        naics = _text(row.get(naics_col))
        if not naics.startswith(("31", "32", "33")):
            continue
        approval_at = _parse_datetime(row.get(approval_col))
        if approval_at is None:
            continue
        name = _text(row.get(name_col))
        city = _text(row.get(city_col)) if city_col else ""
        state = _text(row.get(state_col))
        street = _text(row.get(street_col)) if street_col else ""
        postal = _text(row.get(zip_col)) if zip_col else ""
        exact_entity_key = _account_match_hash(name, city, state, postal)
        account_id = f"acct_match_{exact_entity_key[:20]}"
        raw_loan_id = _text(row.get(loan_col)) if loan_col else ""
        source_location_id = _text(row.get(location_col)) if location_col else ""
        source_program = _text(row.get(program_col)) if program_col else ""
        amount_raw = row.get(amount_col) if amount_col else 0
        try:
            amount = max(float(str(amount_raw).replace(",", "").replace("$", "")), 0)
        except ValueError:
            amount = 0
        loan_anchor = raw_loan_id or sha256_json(
            {
                "borrower_exact_key": exact_entity_key,
                "approval_date": approval_at.date(),
                "amount": amount,
                "lender": _text(row.get(lender_col)) if lender_col else "",
                "location_id": source_location_id,
                "program": source_program,
            }
        )
        accounts[account_id] = AccountRecord(
            account_id=account_id,
            name=name,
            naics_code=naics,
            city=city or None,
            state=state or None,
            source_identifiers={
                "sba_borrower_match_hash": exact_entity_key,
            },
            attributes={
                "street": street or None,
                "postal_code": postal or None,
            },
        )
        available_at = min(_quarterly_available_at(approval_at), ingested_at)
        strength = min(math.log1p(amount) / math.log1p(5_000_000), 1.0)
        common_payload = {
            "borrower_name": name,
            "city": city,
            "state": state,
            "postal_code": postal,
            "naics_code": naics,
            "loan_amount_usd": amount,
            "loan_id": loan_anchor,
            "loan_id_is_source_native": bool(raw_loan_id),
            "source_location_id": source_location_id,
            "program": source_program,
            "availability_is_estimated": True,
            "lender_name": _text(row.get(lender_col)) if lender_col else "",
            "jobs_supported": _number(row.get(jobs_col)) if jobs_col else None,
            "term_months": _number(row.get(term_col)) if term_col else None,
        }
        instance_id = f"sba_loan:{loan_anchor}"
        events.append(
            NormalizedEvent(
                event_id=f"{instance_id}:approved",
                account_id=account_id,
                signal_instance_id=instance_id,
                signal_type="sba_loan",
                source_id="sba_7a_504",
                source_record_id=loan_anchor,
                shape=SignalShape.STATE_MACHINE,
                kind="approved",
                occurred_at=approval_at,
                delivered_at=available_at,
                ingested_at=ingested_at,
                available_at=available_at,
                strength=strength,
                payload={**common_payload, "status": "approved"},
            )
        )
        disbursed_at = _parse_datetime(row.get(disbursement_col)) if disbursement_col else None
        if disbursed_at:
            disbursed_available = min(_quarterly_available_at(disbursed_at), ingested_at)
            events.append(
                NormalizedEvent(
                    event_id=f"{instance_id}:disbursed",
                    account_id=account_id,
                    signal_instance_id=instance_id,
                    signal_type="sba_loan",
                    source_id="sba_7a_504",
                    source_record_id=loan_anchor,
                    shape=SignalShape.STATE_MACHINE,
                    kind="disbursed",
                    occurred_at=disbursed_at,
                    delivered_at=disbursed_available,
                    ingested_at=ingested_at,
                    available_at=disbursed_available,
                    strength=strength,
                    payload={**common_payload, "status": "disbursed"},
                )
            )
        source_status = _text(row.get(status_col)).upper().replace(" ", "") if status_col else ""
        final_kind: str | None = None
        final_at: datetime | None = None
        if source_status in {"PIF", "PAIDINFULL"}:
            final_kind = "paid_in_full"
            final_at = _parse_datetime(row.get(paid_col)) if paid_col else None
        elif source_status in {"CHGOFF", "CHARGEDOFF"}:
            final_kind = "charged_off"
            final_at = _parse_datetime(row.get(charged_off_col)) if charged_off_col else None
        if final_kind and final_at:
            final_available = min(_quarterly_available_at(final_at), ingested_at)
            events.append(
                NormalizedEvent(
                    event_id=f"{instance_id}:{final_kind}",
                    account_id=account_id,
                    signal_instance_id=instance_id,
                    signal_type="sba_loan",
                    source_id="sba_7a_504",
                    source_record_id=loan_anchor,
                    shape=SignalShape.STATE_MACHINE,
                    kind=final_kind,
                    occurred_at=final_at,
                    delivered_at=final_available,
                    ingested_at=ingested_at,
                    available_at=final_available,
                    strength=strength,
                    payload={**common_payload, "status": final_kind},
                )
            )
    return list(accounts.values()), events


def normalize_osha_ita(
    path: Path,
    *,
    ingested_at: datetime,
    work_dir: Path,
) -> tuple[list[AccountRecord], list[NormalizedEvent]]:
    frames = [_read_table(table) for table in _tables_in(path, work_dir)]
    frame = pl.concat(frames, how="diagonal_relaxed")
    establishment_col = _column(frame, "establishment_name", "establishmentname", "company_name")
    id_col = _column(frame, "establishment_id", "establishmentid")
    street_col = _column(frame, "street_address", "streetaddress", "address")
    city_col = _column(frame, "city", "city_name")
    state_col = _column(frame, "state", "state_code")
    zip_col = _column(frame, "zip_code", "zipcode", "zip")
    naics_col = _column(frame, "naics_code", "naics")
    year_col = _column(frame, "year", "reporting_year", "fiscal_year", "year_filing_for")
    employees_col = _column(
        frame,
        "annual_average_employees",
        "average_annual_employees",
        "annualaverageemployees",
    )
    total_cases_col = _column(
        frame, "total_recordable_cases", "total_cases", "totalrecordablecases"
    )
    dafw_col = _column(frame, "total_dafw_cases", "days_away_cases", "totaldafwcases")
    djtr_col = _column(frame, "total_djtr_cases", "totaldjtrcases")
    other_cases_col = _column(frame, "total_other_cases", "totalothercases")
    if not all((establishment_col, state_col, naics_col, year_col, employees_col)):
        raise ValueError("OSHA ITA schema missing establishment, state, NAICS, year, or employees")
    assert establishment_col and state_col and naics_col and year_col and employees_col

    accounts: dict[str, AccountRecord] = {}
    events: list[NormalizedEvent] = []
    for row in frame.iter_rows(named=True):
        naics = _text(row.get(naics_col))
        if not naics.startswith(("31", "32", "33")):
            continue
        name = _text(row.get(establishment_col))
        state = _text(row.get(state_col))
        city = _text(row.get(city_col)) if city_col else ""
        street = _text(row.get(street_col)) if street_col else ""
        postal = _text(row.get(zip_col)) if zip_col else ""
        try:
            year = int(float(str(row.get(year_col))))
            employees = max(int(float(str(row.get(employees_col) or 0))), 0)
        except ValueError:
            continue
        exact_entity_key = _account_match_hash(name, city, state, postal)
        account_id = f"acct_match_{exact_entity_key[:20]}"
        source_establishment_id = _text(row.get(id_col)) if id_col else exact_entity_key
        dafw_cases = max(_number(row.get(dafw_col)) or 0, 0) if dafw_col else 0
        if total_cases_col:
            total_cases = max(_number(row.get(total_cases_col)) or 0, 0)
        else:
            total_cases = sum(
                max(_number(row.get(column)) or 0, 0)
                for column in (dafw_col, djtr_col, other_cases_col)
                if column
            )
        occurred_at = datetime(year, 12, 31, tzinfo=UTC)
        available_at = min(datetime(year + 1, 3, 15, tzinfo=UTC), ingested_at)
        strength = min(total_cases / max(employees * 0.08, 1), 1)
        accounts[account_id] = AccountRecord(
            account_id=account_id,
            name=name,
            naics_code=naics,
            city=city or None,
            state=state or None,
            employee_count=employees,
            source_identifiers={"osha_establishment_id": source_establishment_id},
            attributes={"street": street or None, "postal_code": postal or None},
        )
        instance_id = f"osha:{source_establishment_id}"
        source_record_id = f"{source_establishment_id}:{year}"
        events.append(
            NormalizedEvent(
                event_id=f"{instance_id}:annual_report:{year}",
                account_id=account_id,
                signal_instance_id=instance_id,
                signal_type="osha_injury_summary",
                source_id="osha_ita",
                source_record_id=source_record_id,
                shape=SignalShape.NUMERIC_SERIES,
                kind="annual_report",
                occurred_at=occurred_at,
                delivered_at=available_at,
                ingested_at=ingested_at,
                available_at=available_at,
                strength=strength,
                payload={
                    "establishment_id": source_establishment_id,
                    "establishment_name": name,
                    "naics_code": naics,
                    "state": state,
                    "reporting_year": year,
                    "total_recordable_cases": total_cases,
                    "days_away_cases": dafw_cases,
                    "average_annual_employees": employees,
                    "availability_is_estimated": True,
                },
            )
        )
    return list(accounts.values()), events


def normalize_epa_tri(
    path: Path,
    *,
    ingested_at: datetime,
    work_dir: Path,
) -> tuple[list[AccountRecord], list[NormalizedEvent]]:
    frames = [_read_table(table) for table in _tables_in(path, work_dir)]
    frame = pl.concat(frames, how="diagonal_relaxed")
    facility_id_col = _column(frame, "TRI_FACILITY_ID", "facility_id")
    facility_col = _column(frame, "FACILITY_NAME", "facility_name")
    street_col = _column(frame, "STREET_ADDRESS", "street_address")
    city_col = _column(frame, "CITY_NAME", "city")
    state_col = _column(frame, "ST", "state", "state_code")
    zip_col = _column(frame, "ZIP_CODE", "zip")
    naics_col = _column(frame, "PRIMARY_NAICS_CODE", "naics_code", "naics")
    year_col = _column(frame, "REPORTING_YEAR", "year")
    production_col = _column(
        frame, "PRODUCTION_OR_ACTIVITY", "production_ratio", "production_index"
    )
    releases_col = _column(
        frame,
        "TOTAL_RELEASES",
        "total_releases",
        "on_and_off_site_reported_disposed_or_other_releases",
    )
    if not all((facility_id_col, facility_col, state_col, naics_col, year_col)):
        raise ValueError("EPA TRI schema missing facility id/name, state, NAICS, or year")
    assert facility_id_col and facility_col and state_col and naics_col and year_col

    aggregates: dict[tuple[str, int], dict[str, Any]] = {}
    for row in frame.iter_rows(named=True):
        naics = _text(row.get(naics_col))
        if not naics.startswith(("31", "32", "33")):
            continue
        try:
            year = int(float(str(row.get(year_col))))
        except ValueError:
            continue
        facility_id = _text(row.get(facility_id_col))
        key = (facility_id, year)
        bucket = aggregates.setdefault(
            key,
            {
                "facility_name": _text(row.get(facility_col)),
                "street": _text(row.get(street_col)) if street_col else "",
                "city": _text(row.get(city_col)) if city_col else "",
                "state": _text(row.get(state_col)),
                "postal": _text(row.get(zip_col)) if zip_col else "",
                "naics": naics,
                "production": [],
                "releases": 0.0,
            },
        )
        production = _number(row.get(production_col)) if production_col else None
        releases = _number(row.get(releases_col)) if releases_col else None
        if production is not None:
            bucket["production"].append(production)
        if releases is not None:
            bucket["releases"] += max(releases, 0)

    accounts: dict[str, AccountRecord] = {}
    events: list[NormalizedEvent] = []
    for (facility_id, year), bucket in aggregates.items():
        exact_entity_key = _account_match_hash(
            bucket["facility_name"],
            bucket["city"],
            bucket["state"],
            bucket["postal"],
        )
        has_address = bool(bucket["city"] and bucket["state"] and bucket["postal"])
        account_id = (
            f"acct_match_{exact_entity_key[:20]}"
            if has_address
            else f"acct_tri_{facility_id.lower()}"
        )
        production_ratio = (
            sum(bucket["production"]) / len(bucket["production"]) if bucket["production"] else 1.0
        )
        releases = bucket["releases"]
        occurred_at = datetime(year, 12, 31, tzinfo=UTC)
        available_at = min(datetime(year + 1, 11, 15, tzinfo=UTC), ingested_at)
        strength = min(abs(production_ratio - 1) + math.log1p(releases) / 30, 1)
        accounts[account_id] = AccountRecord(
            account_id=account_id,
            name=bucket["facility_name"],
            naics_code=bucket["naics"],
            city=bucket["city"] or None,
            state=bucket["state"] or None,
            source_identifiers={
                "tri_facility_id": facility_id,
                "account_match_hash": exact_entity_key,
            },
            attributes={
                "street": bucket["street"] or None,
                "postal_code": bucket["postal"] or None,
            },
        )
        instance_id = f"tri:{facility_id}"
        events.append(
            NormalizedEvent(
                event_id=f"{instance_id}:annual_report:{year}",
                account_id=account_id,
                signal_instance_id=instance_id,
                signal_type="epa_tri_report",
                source_id="epa_tri",
                source_record_id=f"{facility_id}:{year}",
                shape=SignalShape.NUMERIC_SERIES,
                kind="annual_report",
                occurred_at=occurred_at,
                delivered_at=available_at,
                ingested_at=ingested_at,
                available_at=available_at,
                strength=strength,
                payload={
                    "tri_facility_id": facility_id,
                    "facility_name": bucket["facility_name"],
                    "naics_code": bucket["naics"],
                    "state": bucket["state"],
                    "reporting_year": year,
                    "production_ratio": round(production_ratio, 6),
                    "total_releases_lb": round(releases, 3),
                    "availability_is_estimated": True,
                },
            )
        )
    return list(accounts.values()), events
