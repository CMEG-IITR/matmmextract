"""
paper_pipeline.openalex.fetcher
=================================
Fetch paper metadata from the OpenAlex API, filtered by publisher,
license, and optional keyword/topic constraints.

Why OpenAlex instead of Scopus CSVs?
-------------------------------------
Scopus exports require manual download and have no license metadata.
OpenAlex is free, has an API, exposes CC license data directly, and
covers >250M works.  The output CSV is intentionally shaped to match
what the rest of the pipeline expects (``DOI``, ``Publisher``,
``Open Access``, ``Title`` columns).

Key filters supported
---------------------
- ``publisher``  — publisher display name substring (e.g. ``"Elsevier"``)
- ``license``    — OA license type: ``"cc-by"``, ``"cc-by-nc"``, etc.
                   Pass ``None`` to skip license filtering.
- ``is_oa``      — restrict to open-access works (default True)
- ``from_year``  — publication year lower bound
- ``to_year``    — publication year upper bound
- ``keywords``   — list of keyword strings (ANDed together as concept search)
- ``topics``     — list of OpenAlex topic/concept IDs or display names

OpenAlex API reference: https://docs.openalex.org/api-entities/works
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE = "https://api.openalex.org/works"

# Fields we request from the API — keeps responses small
_SELECT_FIELDS = ",".join([
    "id", "doi", "title", "publication_year",
    "primary_location", "open_access",
    "authorships", "concepts",
    "cited_by_count", "type",
])

# Map from our friendly license names to OpenAlex filter values
LICENSE_MAP: dict[str, str] = {
    "cc-by":        "cc-by",
    "cc by":        "cc-by",
    "cc-by-nc":     "cc-by-nc",
    "cc-by-sa":     "cc-by-sa",
    "cc-by-nd":     "cc-by-nd",
    "cc-by-nc-sa":  "cc-by-nc-sa",
    "cc-by-nc-nd":  "cc-by-nc-nd",
    "public-domain": "public-domain",
}

# Column names that match what the rest of the pipeline expects
_OUTPUT_COLUMNS = [
    "DOI", "Title", "Year", "Publisher", "Journal",
    "Open Access", "License", "OA URL",
    "Authors", "Cited By Count", "Type",
    "Concepts", "openalex_id",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_fetched: int = 0
    output_csv: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_filter(
    publisher: str | None,
    license_: str | None,
    is_oa: bool,
    from_year: int | None,
    to_year: int | None,
    keywords: list[str] | None,
) -> str:
    parts: list[str] = []

    if is_oa:
        parts.append("is_oa:true")

    # Publisher filtering is done locally after download.

    if license_:
        normalized = LICENSE_MAP.get(
            license_.lower().strip(),
            license_.lower().strip()
        )
        parts.append(f"primary_location.license:{normalized}")

    if from_year and to_year:
        parts.append(f"publication_year:{from_year}-{to_year}")
    elif from_year:
        parts.append(f"publication_year:>{from_year - 1}")
    elif to_year:
        parts.append(f"publication_year:<{to_year + 1}")

    return ",".join(parts)


def _parse_work(work: dict) -> dict:
    """Flatten one OpenAlex work record into a row dict."""
    doi = work.get("doi") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    publisher = source.get("host_organization_name") or source.get("display_name") or ""
    journal = source.get("display_name") or ""
    oa_url = loc.get("landing_page_url") or loc.get("pdf_url") or ""

    oa = work.get("open_access") or {}
    oa_status = oa.get("oa_status") or ""
    license_ = loc.get("license") or oa.get("license") or ""

    # Map OA status to the "Open Access" column format used by Scopus exports
    oa_label_map = {
        "gold":   "All Open Access",
        "green":  "All Open Access; Green Open Access",
        "bronze": "All Open Access; Bronze Open Access",
        "hybrid": "All Open Access",
        "closed": "Closed",
    }
    oa_label = oa_label_map.get(oa_status, oa_status)

    authors = "; ".join(
        a.get("author", {}).get("display_name", "")
        for a in (work.get("authorships") or [])[:10]
    )

    concepts = "; ".join(
        c.get("display_name", "")
        for c in sorted(
            work.get("concepts") or [],
            key=lambda c: -c.get("score", 0),
        )[:8]
    )

    return {
        "DOI":            doi,
        "Title":          work.get("title") or "",
        "Year":           work.get("publication_year") or "",
        "Publisher":      publisher,
        "Journal":        journal,
        "Open Access":    oa_label,
        "License":        license_,
        "OA URL":         oa_url,
        "Authors":        authors,
        "Cited By Count": work.get("cited_by_count") or 0,
        "Type":           work.get("type") or "",
        "Concepts":       concepts,
        "openalex_id":    work.get("id") or "",
    }


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def fetch(
    publisher: str | None = None,
    license_: str | None = "cc-by",
    is_oa: bool = True,
    from_year: int | None = None,
    to_year: int | None = None,
    keywords: list[str] | None = None,
    topics: list[str] | None = None,
    max_results: int = 1000,
    output_csv: str | Path | None = None,
    email: str | None = None,
    api_key: str | None = None,
    per_page: int = 200,
    verbose: bool = True,
) -> FetchResult:
    """Fetch works from OpenAlex matching the given filters.

    Parameters
    ----------
    publisher:
        Publisher display name substring, e.g. ``"Elsevier"``,
        ``"Springer"``, ``"Wiley"``.  Case-insensitive partial match.
    license_:
        OA license filter.  Common values: ``"cc-by"``, ``"cc-by-nc"``,
        ``"cc-by-nc-nd"``.  Pass ``None`` to skip.
    is_oa:
        Restrict to open-access works (default ``True``).
    from_year / to_year:
        Publication year range (inclusive).
    keywords:
        List of concept/keyword strings to filter by (ANDed).
        e.g. ``["alloy", "microstructure"]``
    topics:
        List of OpenAlex concept display names or IDs to filter by.
        e.g. ``["Materials Science", "C2780799074"]``
        Applied as an additional ``concepts.display_name.search`` filter.
    max_results:
        Maximum number of works to return.
    output_csv:
        If provided, save the DataFrame as CSV.
    email:
        Your email address for the OpenAlex polite pool (appended to User-Agent).
    api_key:
        Free OpenAlex API key (get one at https://openalex.org/settings/api).
        Required for reliable access; passed as ``?api_key=`` query parameter.
    per_page:
        Results per API page (max 200).
    verbose:
        Print progress.

    Returns
    -------
    FetchResult
        ``.df`` has the same column structure as a Scopus CSV export so
        it can be passed directly to
        :func:`~paper_pipeline.preprocess.pipeline.filter_by_publisher`.

    Examples
    --------
    >>> from paper_pipeline.openalex.fetcher import fetch
    >>> result = fetch(publisher="Elsevier", license_="cc-by",
    ...                keywords=["titanium alloy"], max_results=500)
    >>> df = result.df
    >>> print(df[["DOI", "Title", "License"]].head())

    >>> # Springer, any OA license, materials science, 2020-2024
    >>> result = fetch(
    ...     publisher="Springer",
    ...     license_=None,
    ...     keywords=["nickel alloy", "microstructure"],
    ...     from_year=2020, to_year=2024,
    ...     max_results=2000,
    ... )
    """
    per_page = min(per_page, 200)
    all_keywords = list(keywords or []) + list(topics or [])
    search_query = " ".join(all_keywords).strip()

    filter_str = _build_filter(
        publisher=publisher,
        license_=license_,
        is_oa=is_oa,
        from_year=from_year,
        to_year=to_year,
        keywords=None,
    )

    headers: dict[str, str] = {"User-Agent": "paper_pipeline/1.0"}
    if email:
        headers["User-Agent"] += f" (mailto:{email})"

    params: dict = {
        "filter":   filter_str,
        "select":   _SELECT_FIELDS,
        "per-page": per_page,
        "cursor":   "*",
        "sort":     "cited_by_count:desc",
    }
    if api_key:
        params["api_key"] = api_key

    if search_query:
        params["search"] = search_query

    if verbose:
        print(f"[openalex] filter: {filter_str}")
        print(f"[openalex] max_results={max_results}  per_page={per_page}")

    rows: list[dict] = []
    page = 0

    while len(rows) < max_results:
        try:
            resp = requests.get(_BASE, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            if verbose:
                print(f"[openalex] request error: {exc}  — retrying in 5s")
            time.sleep(5)
            continue

        if resp.status_code == 429:
            if verbose:
                print("[openalex] rate limited — sleeping 30s")
            time.sleep(30)
            continue

        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenAlex API error {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for work in results:
            if len(rows) >= max_results:
                break
            parsed = _parse_work(work)
            if publisher and publisher.lower() not in parsed["Publisher"].lower():
                continue
            if keywords:
                haystack = (parsed["Title"] + " " + parsed["Concepts"]).lower()
                if not any(kw.lower() in haystack for kw in keywords):
                    continue
            rows.append(parsed)

        page += 1
        if verbose:
            print(f"[openalex] page {page} — {len(rows)} / {max_results} fetched")

        # Cursor-based pagination
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor

        time.sleep(0.1)  # polite pool: 10 req/s

    df = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)

    # if publisher:
    #     df = df[
    #         df["Publisher"]
    #         .fillna("")
    #         .str.contains(
    #             publisher,
    #             case=False,
    #             na=False,
    #             regex=False,
    #         )
    #     ]
    #
    # if keywords:
    #     for kw in keywords:
    #         mask = (
    #             df["Title"].fillna("")
    #             + " "
    #             + df["Concepts"].fillna("")
    #         ).str.contains(
    #             kw,
    #             case=False,
    #             na=False,
    #             regex=False,
    #         )
    #         df = df[mask]

    result = FetchResult(df=df, total_fetched=len(df))

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        result.output_csv = str(output_csv)
        if verbose:
            print(f"[openalex] saved {len(df)} rows → {output_csv}")

    if verbose:
        print(f"[openalex] done — {len(df)} works fetched")

    return result


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def fetch_elsevier(
    license_: str | None = "cc-by",
    keywords: list[str] | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
    max_results: int = 1000,
    output_csv: str | Path | None = None,
    email: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> FetchResult:
    """Fetch Elsevier works.  Shorthand for ``fetch(publisher="Elsevier", ...)``."""
    return fetch(
        publisher="Elsevier",
        license_=license_,
        keywords=keywords,
        from_year=from_year,
        to_year=to_year,
        max_results=max_results,
        output_csv=output_csv,
        email=email,
        api_key=api_key,
        verbose=verbose,
    )


def fetch_springer(
    license_: str | None = "cc-by",
    keywords: list[str] | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
    max_results: int = 1000,
    output_csv: str | Path | None = None,
    email: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> FetchResult:
    """Fetch Springer works.  Shorthand for ``fetch(publisher="Springer", ...)``."""
    return fetch(
        publisher="Springer",
        license_=license_,
        keywords=keywords,
        from_year=from_year,
        to_year=to_year,
        max_results=max_results,
        output_csv=output_csv,
        email=email,
        api_key=api_key,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch papers from OpenAlex by publisher / license."
    )
    p.add_argument("--publisher",    default=None,
                   help="Publisher name substring, e.g. 'Elsevier'")
    p.add_argument("--license",      default="cc-by",
                   help="OA license filter, e.g. cc-by, cc-by-nc (default: cc-by)")
    p.add_argument("--no-oa",        action="store_true",
                   help="Do not restrict to open-access works")
    p.add_argument("--from-year",    type=int, default=None)
    p.add_argument("--to-year",      type=int, default=None)
    p.add_argument("--keywords",     nargs="+", default=None,
                   help="One or more keyword strings (ANDed)")
    p.add_argument("--max-results",  type=int, default=1000)
    p.add_argument("--output-csv",   required=True)
    p.add_argument("--email",        default=None,
                   help="Your email for OpenAlex polite pool (optional)")
    p.add_argument("--api-key",      default=None,
                   help="Free OpenAlex API key (see https://openalex.org/settings/api)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    fetch(
        publisher=args.publisher,
        license_=None if args.license.lower() == "none" else args.license,
        is_oa=not args.no_oa,
        from_year=args.from_year,
        to_year=args.to_year,
        keywords=args.keywords,
        max_results=args.max_results,
        output_csv=args.output_csv,
        email=args.email,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
