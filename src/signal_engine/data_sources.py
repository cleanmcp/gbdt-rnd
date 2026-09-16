"""Reproducible acquisition of free public manufacturing datasets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from .contracts import CorpusManifest, CorpusSourceManifest
from .hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class Resource:
    source_id: str
    landing_page: str
    url: str
    filename: str
    license: str


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


async def discover_sba(client: httpx.AsyncClient) -> list[Resource]:
    landing = "https://data.sba.gov/dataset/7a-504-foia"
    response = await client.get(landing)
    response.raise_for_status()
    parser = _LinkParser()
    parser.feed(response.text)
    wanted = []
    for href, name in parser.links:
        normalized = f"{name} {href}".lower().replace(" ", "")
        if ".csv" not in normalized:
            continue
        if ("7(a)" in name and "fy2020-present" in normalized) or (
            "504" in name and "fy2010-present" in normalized
        ):
            url = urljoin(landing, href)
            wanted.append(
                Resource(
                    source_id="sba_7a_504",
                    landing_page=landing,
                    url=url,
                    filename=Path(urlparse(url).path).name or f"{sha256_json(url)[:12]}.csv",
                    license="U.S. Government Works",
                )
            )
    if not wanted:
        raise RuntimeError("SBA catalog no longer exposes the expected current CSV files")
    return wanted


async def _discover_page_links(
    client: httpx.AsyncClient,
    *,
    source_id: str,
    landing_page: str,
    license_name: str,
    include: tuple[str, ...],
    extensions: tuple[str, ...] = (".csv", ".zip", ".xlsx"),
) -> list[Resource]:
    response = await client.get(landing_page)
    response.raise_for_status()
    parser = _LinkParser()
    parser.feed(response.text)
    discovered = []
    for href, text in parser.links:
        absolute = urljoin(landing_page, href)
        searchable = f"{text} {href}".lower()
        path = urlparse(absolute).path.lower()
        if all(fragment.lower() in searchable for fragment in include) and any(
            extension in path for extension in extensions
        ):
            discovered.append(
                Resource(
                    source_id=source_id,
                    landing_page=landing_page,
                    url=absolute,
                    filename=Path(urlparse(absolute).path).name,
                    license=license_name,
                )
            )
    return list({resource.url: resource for resource in discovered}.values())


async def discover_osha_ita(
    client: httpx.AsyncClient,
    years: tuple[int, ...] = (2023, 2024, 2025),
) -> list[Resource]:
    all_resources: list[Resource] = []
    for year in years:
        resources = await _discover_page_links(
            client,
            source_id="osha_ita",
            landing_page="https://www.osha.gov/Establishment-Specific-Injury-and-Illness-Data",
            license_name="U.S. Government Works",
            include=(str(year), "summary"),
        )
        if not resources:
            resources = await _discover_page_links(
                client,
                source_id="osha_ita",
                landing_page="https://www.osha.gov/itadata",
                license_name="U.S. Government Works",
                include=(str(year), "summary"),
            )
        if not resources:
            raise RuntimeError(f"OSHA page exposed no {year} summary download")
        all_resources.append(resources[0])
    return all_resources


async def discover_epa_tri(client: httpx.AsyncClient, year: int = 2024) -> list[Resource]:
    del client
    landing = (
        "https://www.epa.gov/toxics-release-inventory-tri-program/"
        "tri-basic-data-files-calendar-years-1987-present"
    )
    return [
        Resource(
            source_id="epa_tri",
            landing_page=landing,
            url=(
                f"https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/{year}_US/csv"
            ),
            filename=f"epa_tri_basic_{year}_US.csv",
            license="U.S. Government Works",
        )
    ]


async def discover_resources(
    source_ids: tuple[str, ...],
    *,
    osha_years: tuple[int, ...] = (2023, 2024, 2025),
    tri_year: int = 2024,
) -> list[Resource]:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=180),
        headers={"User-Agent": "signal-engine-rnd/0.1 public-data-research"},
    ) as client:
        resources: list[Resource] = []
        for source_id in source_ids:
            if source_id == "sba":
                resources.extend(await discover_sba(client))
            elif source_id == "osha":
                resources.extend(await discover_osha_ita(client, osha_years))
            elif source_id == "epa_tri":
                resources.extend(await discover_epa_tri(client, tri_year))
            else:
                raise ValueError(f"unknown public source: {source_id}")
        return resources


async def _download_one(
    client: httpx.AsyncClient,
    resource: Resource,
    target_dir: Path,
) -> CorpusSourceManifest:
    source_dir = target_dir / resource.source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    target = source_dir / resource.filename
    temporary = target.with_suffix(target.suffix + ".partial")
    if not target.exists():
        for attempt in range(5):
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                async with client.stream("GET", resource.url, headers=headers) as response:
                    response.raise_for_status()
                    append = bool(offset and response.status_code == 206)
                    with temporary.open("ab" if append else "wb") as handle:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            handle.write(chunk)
                temporary.replace(target)
                break
            except (httpx.HTTPError, OSError):
                if attempt == 4:
                    raise
                await asyncio.sleep(2**attempt)
    retrieved_at = datetime.fromtimestamp(target.stat().st_mtime, UTC)
    try:
        local_path = target.relative_to(Path.cwd()).as_posix()
    except ValueError:
        local_path = target.as_posix()
    return CorpusSourceManifest.model_validate(
        {
            "source_id": resource.source_id,
            "landing_page": resource.landing_page,
            "retrieval_url": resource.url,
            "license": resource.license,
            "retrieved_at": retrieved_at,
            "content_sha256": sha256_file(target),
            "local_path": local_path,
            "bytes": target.stat().st_size,
        }
    )


async def download_public_corpus(
    source_ids: tuple[str, ...],
    target_dir: Path,
    *,
    corpus_version: str,
) -> CorpusManifest:
    resources = await discover_resources(source_ids)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(60, read=600),
        headers={"User-Agent": "signal-engine-rnd/0.1 public-data-research"},
    ) as client:
        manifests = []
        for resource in resources:
            manifests.append(await _download_one(client, resource, target_dir))
    created_at = datetime.now(UTC)
    payload = {
        "corpus_version": corpus_version,
        "sources": [
            {
                "source_id": manifest.source_id,
                "retrieval_url": str(manifest.retrieval_url),
                "license": manifest.license,
                "content_sha256": manifest.content_sha256,
                "bytes": manifest.bytes,
            }
            for manifest in manifests
        ],
    }
    return CorpusManifest(
        corpus_version=corpus_version,
        created_at=created_at,
        sources=tuple(manifests),
        manifest_hash=sha256_json(payload),
    )
