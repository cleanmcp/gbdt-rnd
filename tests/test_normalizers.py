from datetime import UTC, datetime
from pathlib import Path

from signal_engine.normalizers import normalize_osha_ita, normalize_sba


def test_sba_adapter_filters_manufacturing_and_keeps_lifecycle(tmp_path: Path) -> None:
    source = tmp_path / "sba.csv"
    source.write_text(
        "\n".join(
            [
                '"AsOfDate","Program","LocationID","BorrName","BorrStreet","BorrCity",'
                '"BorrState","BorrZip","BankName","GrossApproval","ApprovalDate",'
                '"FirstDisbursementDate","NaicsCode","LoanStatus","PaidInFullDate",'
                '"JobsSupported","TermInMonths"',
                '"2026-06-30","7A","001","Acme Press","10 Mill Rd","Akron","OH",'
                '"44301","Example Bank","750000","2024-01-03","2024-02-10",'
                '"333249","PIF","2026-01-05","18","120"',
                '"2026-06-30","7A","002","Not Manufacturing","1 Main","Akron","OH",'
                '"44301","Example Bank","100000","2024-01-03","2024-02-10",'
                '"541511","PIF","2026-01-05","3","60"',
            ]
        ),
        encoding="utf-8",
    )
    accounts, events = normalize_sba(
        source,
        ingested_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    assert len(accounts) == 1
    assert accounts[0].naics_code == "333249"
    assert [event.kind for event in events] == [
        "approved",
        "disbursed",
        "paid_in_full",
    ]
    assert all(event.payload["loan_id_is_source_native"] is False for event in events)
    assert all(event.available_at <= event.ingested_at for event in events)


def test_cross_source_account_key_tolerates_legal_suffix_and_street_format(
    tmp_path: Path,
) -> None:
    sba = tmp_path / "sba.csv"
    sba.write_text(
        "\n".join(
            [
                '"BorrName","BorrStreet","BorrCity","BorrState","BorrZip",'
                '"GrossApproval","ApprovalDate","NaicsCode"',
                '"Acme Press, Inc.","10 Mill Road","Akron","OH","44301-1000",'
                '"750000","2024-01-03","333249"',
            ]
        ),
        encoding="utf-8",
    )
    osha = tmp_path / "osha.csv"
    osha.write_text(
        "\n".join(
            [
                "establishment_name,establishment_id,street_address,city,state,"
                "zip_code,naics_code,annual_average_employees,total_dafw_cases,"
                "total_djtr_cases,total_other_cases,year_filing_for",
                "ACME PRESS,99,10 MILL RD,Akron,OH,44301,333249,75,2,1,0,2025",
            ]
        ),
        encoding="utf-8",
    )
    ingested_at = datetime(2026, 8, 28, tzinfo=UTC)
    sba_accounts, _ = normalize_sba(sba, ingested_at=ingested_at)
    osha_accounts, _ = normalize_osha_ita(
        osha,
        ingested_at=ingested_at,
        work_dir=tmp_path / "cache",
    )
    assert sba_accounts[0].account_id == osha_accounts[0].account_id
