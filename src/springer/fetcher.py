"""
paper_pipeline.springer.fetcher
=================================
Fetch full-text JATS XML from the Springer Nature API for a list of DOIs.

Ported from ``springer.py`` — logic unchanged, already clean.
Minor additions:
  - ``fetch_all()`` callable for programmatic use (not just CLI)
  - Imports shared ``doi_to_filename``, ``load_set``, ``append_line``

Environment variables
---------------------
SPRINGER_API_KEY      Springer API key
SPRINGER_API_METRIC   Full-text TDM API metric (omit when using --oa)
"""

from __future__ import annotations

import argparse
import os
import re
import random
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from paper_pipeline.shared.doi_utils import doi_to_filename, load_set, append_line


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

FULLTEXT_URL = "https://spdi.public.springernature.app/xmldata/jats"
OPEN_ACCESS_URL = "https://api.springernature.com/openaccess/jats"

DEFAULT_INPUT_CSV = "springer_papers.csv"
DEFAULT_OUTPUT_CSV = "springer_with_xml_paths.csv"
DEFAULT_OUTPUT_DIR = "alloys_springer"
DEFAULT_PROCESSED_FILE = "processed_dois_springer.txt"
DEFAULT_FAILED_FILE = "failed_dois_springer.txt"
DEFAULT_MAX_PER_RUN = 1000
DEFAULT_SLEEP_MIN = 2.0
DEFAULT_SLEEP_MAX = 5.0
DEFAULT_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    success: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    output_csv: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_api_key(api_key: str, api_metric: str | None) -> str:
    """Append metric to key when needed (full-text TDM endpoint only)."""
    if api_metric and "/" not in api_key:
        return f"{api_key}/{api_metric}"
    return api_key


def _response_has_records(xml_bytes: bytes) -> bool:
    """Return True when the XML response contains at least one article record."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return False
    records = root.find(".//records")
    if records is None:
        return False
    return any(child.tag != "result" for child in list(records))


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def fetch_fulltext_xml(
    doi: str,
    api_key: str,
    endpoint: str = FULLTEXT_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[bytes | None, str | None]:
    """Fetch JATS XML for *doi*.

    Parameters
    ----------
    doi:
        DOI string.
    api_key:
        Full API key string (already combined with metric if needed).
    endpoint:
        API endpoint URL.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    (content, None)   on success
    (None, error_str) on failure
    """
    params = {"q": f"doi:{doi}", "api_key": api_key}
    headers = {
        "Accept": "application/xml",
        "User-Agent": "springer-xml-downloader/1.0",
    }
    response = requests.get(endpoint, params=params, headers=headers, timeout=timeout)

    if response.status_code != 200:
        return None, f"HTTP_{response.status_code}"

    content_type = response.headers.get("content-type", "")
    if "xml" not in content_type.lower() and not response.content.lstrip().startswith(b"<"):
        return None, "NON_XML_RESPONSE"

    if not _response_has_records(response.content):
        return None, "NO_XML_RECORDS"

    return response.content, None


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_all(
    df: pd.DataFrame,
    api_key: str,
    api_metric: str | None = None,
    use_open_access: bool = False,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    processed_file: str | Path = DEFAULT_PROCESSED_FILE,
    failed_file: str | Path = DEFAULT_FAILED_FILE,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    max_per_run: int = DEFAULT_MAX_PER_RUN,
    sleep_min: float = DEFAULT_SLEEP_MIN,
    sleep_max: float = DEFAULT_SLEEP_MAX,
    doi_col: str = "DOI",
    verbose: bool = True,
) -> FetchResult:
    """Fetch Springer JATS XML for every DOI in *df*.

    Parameters
    ----------
    df:
        Input DataFrame with a ``DOI`` column (or override with *doi_col*).
    api_key:
        Springer API key.
    api_metric:
        Full-text TDM metric (ignored when *use_open_access* is ``True``).
    use_open_access:
        Use the Open Access JATS endpoint instead of the full-text TDM one.
    output_dir:
        Directory to write XML files.
    processed_file / failed_file:
        Resume-support files.
    output_csv:
        Written on completion with an added ``XML_File`` column.
    max_per_run:
        Stop after this many successes.
    sleep_min / sleep_max:
        Random sleep range between requests.
    doi_col:
        Name of the DOI column in *df*.
    verbose:
        Print progress to stdout.

    Returns
    -------
    FetchResult
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_api_key = api_key if use_open_access else _build_api_key(api_key, api_metric)
    endpoint = OPEN_ACCESS_URL if use_open_access else FULLTEXT_URL

    if "XML_File" not in df.columns:
        df["XML_File"] = None

    processed_dois = load_set(processed_file)
    result = FetchResult()
    success_count = 0

    for i, row in df.iterrows():
        if success_count >= max_per_run:
            if verbose:
                print(f"Reached max_per_run = {max_per_run}")
            break

        doi = str(row[doi_col]).strip()
        if not doi or doi.lower() == "nan":
            continue
        # Strip full URL prefix if present (e.g. "https://doi.org/10.1007/...")
        doi = re.sub(r"^https?://doi\.org/", "", doi)
        if doi in processed_dois:
            continue

        if verbose:
            print(f"Processing: {doi}")

        filepath = output_dir / doi_to_filename(doi)

        try:
            xml, error = fetch_fulltext_xml(doi, full_api_key, endpoint)
            if error:
                append_line(failed_file, f"{doi}\t{error}")
                result.failed.append((doi, error))
                if verbose:
                    print(f"  Failed: {error}")
                continue

            filepath.write_bytes(xml)
            df.at[i, "XML_File"] = str(filepath)
            append_line(processed_file, doi)
            processed_dois.add(doi)
            result.success.append(doi)
            success_count += 1

            if verbose:
                print(f"  Saved ({success_count}/{max_per_run}): {filepath.name}")

        except Exception as exc:
            reason = f"EXCEPTION:{exc}"
            append_line(failed_file, f"{doi}\t{reason}")
            result.failed.append((doi, reason))
            if verbose:
                print(f"  Exception: {exc}")

        time.sleep(random.uniform(sleep_min, sleep_max))

    df.to_csv(output_csv, index=False)
    result.output_csv = str(output_csv)
    if verbose:
        print(f"\nSaved: {output_csv}  (success={len(result.success)}, failed={len(result.failed)})")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Springer JATS XML files.")
    p.add_argument("--input", default=DEFAULT_INPUT_CSV)
    p.add_argument("--output", default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--processed-file", default=DEFAULT_PROCESSED_FILE)
    p.add_argument("--failed-file", default=DEFAULT_FAILED_FILE)
    p.add_argument("--max-per-run", type=int, default=DEFAULT_MAX_PER_RUN)
    p.add_argument("--api-key", default=os.getenv("SPRINGER_API_KEY"))
    p.add_argument("--api-metric", default=os.getenv("SPRINGER_API_METRIC"))
    p.add_argument("--oa", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.api_key:
        raise SystemExit("Missing Springer API key. Set SPRINGER_API_KEY or pass --api-key.")

    df = pd.read_csv(args.input)
    fetch_all(
        df=df,
        api_key=args.api_key,
        api_metric=args.api_metric,
        use_open_access=args.oa,
        output_dir=args.output_dir,
        processed_file=args.processed_file,
        failed_file=args.failed_file,
        output_csv=args.output,
        max_per_run=args.max_per_run,
    )


if __name__ == "__main__":
    main()