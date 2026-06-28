"""
matmmextract.elsevier.fetcher
================================
Fetch full-text XML from the Elsevier API for a list of DOIs.

Ported from ``elsevier.py``:
- API credentials read from environment variables (never hardcoded)
- Everything wrapped in ``fetch_all()`` / ``main()`` — safe to import
- Resume support via ``processed_dois_elsevier.txt`` / ``failed_dois_elsevier.txt``

Environment variables
---------------------
ELSEVIER_API_KEY      Your Elsevier API key
ELSEVIER_INST_TOKEN   Your institutional token
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from matmmextract.shared.doi_utils import doi_to_filename, load_set, append_line


# ---------------------------------------------------------------------------
# Config / defaults
# ---------------------------------------------------------------------------

_API_BASE = "https://api.elsevier.com/content/article"
DEFAULT_OUTPUT_DIR = "alloys_elsevier"
DEFAULT_PROCESSED_FILE = "processed_dois_elsevier.txt"
DEFAULT_FAILED_FILE = "failed_dois_elsevier.txt"
DEFAULT_OUTPUT_CSV = "elsevier_with_xml_paths.csv"
DEFAULT_MAX_PER_RUN = 1000
DEFAULT_SLEEP_MIN = 2.0
DEFAULT_SLEEP_MAX = 5.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    success: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)   # (doi, reason)
    output_csv: str = ""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _headers(api_key: str, inst_token: str, accept: str) -> dict:
    return {
        "X-ELS-APIKey": api_key,
        "X-ELS-Insttoken": inst_token,
        "Accept": accept,
    }


def get_pii_from_doi(doi: str, api_key: str, inst_token: str) -> str | None:
    """Return the PII for *doi*, or ``None`` on failure."""
    url = f"{_API_BASE}/doi/{doi}"
    r = requests.get(url, headers=_headers(api_key, inst_token, "application/json"), timeout=30)
    if r.status_code == 200:
        try:
            return r.json()["full-text-retrieval-response"]["coredata"]["pii"]
        except Exception:
            return None
    return None


def fetch_fulltext_xml(
    pii: str, api_key: str, inst_token: str
) -> tuple[bytes | None, str | None]:
    """Return (xml_bytes, None) on success or (None, error_reason) on failure."""
    url = f"{_API_BASE}/pii/{pii}"
    r = requests.get(
        url,
        headers=_headers(api_key, inst_token, "application/xml"),
        params={"view": "FULL"},
        timeout=60,
    )
    if r.status_code == 200:
        return r.content, None
    return None, f"HTTP_{r.status_code}"


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_all(
    df: pd.DataFrame,
    api_key: str,
    inst_token: str,
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
    """Fetch Elsevier full-text XML for every DOI in *df*.

    Parameters
    ----------
    df:
        Input DataFrame with a ``DOI`` column (or override with *doi_col*).
    api_key:
        Elsevier API key.
    inst_token:
        Elsevier institutional token.
    output_dir:
        Directory to write XML files into.
    processed_file:
        Newline-delimited file of already-processed DOIs (resume support).
    failed_file:
        Newline-delimited file of failed DOIs + reasons.
    output_csv:
        CSV written on completion with an added ``XML_File`` column.
    max_per_run:
        Stop after this many successful fetches (API rate-limit safety).
    sleep_min / sleep_max:
        Random sleep range (seconds) between requests.
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
        # Strip full URL prefix if present (e.g. "https://doi.org/10.1016/...")
        doi = re.sub(r"^https?://doi\.org/", "", doi)
        if doi in processed_dois:
            if verbose:
                print(f"  Skipping (already processed): {doi}")
            continue

        if verbose:
            print(f"Processing: {doi}")

        filepath = output_dir / doi_to_filename(doi)

        try:
            pii = get_pii_from_doi(doi, api_key, inst_token)
            if not pii:
                reason = "PII_NOT_FOUND"
                append_line(failed_file, f"{doi}\t{reason}")
                result.failed.append((doi, reason))
                if verbose:
                    print(f"  Failed: {reason}")
                continue

            xml, xml_error = fetch_fulltext_xml(pii, api_key, inst_token)
            if not xml:
                reason = xml_error or "XML_FETCH_FAILED"
                append_line(failed_file, f"{doi}\t{reason}")
                result.failed.append((doi, reason))
                if verbose:
                    print(f"  Failed: {reason} (pii={pii})")
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
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Elsevier full-text XML files.")
    p.add_argument("--input", required=True, help="CSV with a DOI column.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--processed-file", default=DEFAULT_PROCESSED_FILE)
    p.add_argument("--failed-file", default=DEFAULT_FAILED_FILE)
    p.add_argument("--max-per-run", type=int, default=DEFAULT_MAX_PER_RUN)
    p.add_argument("--api-key", default=os.getenv("ELSEVIER_API_KEY"))
    p.add_argument("--inst-token", default=os.getenv("ELSEVIER_INST_TOKEN"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.api_key:
        raise SystemExit("Missing Elsevier API key. Set ELSEVIER_API_KEY or pass --api-key.")
    if not args.inst_token:
        raise SystemExit("Missing Elsevier inst token. Set ELSEVIER_INST_TOKEN or pass --inst-token.")

    df = pd.read_csv(args.input)
    fetch_all(
        df=df,
        api_key=args.api_key,
        inst_token=args.inst_token,
        output_dir=args.output_dir,
        processed_file=args.processed_file,
        failed_file=args.failed_file,
        output_csv=args.output_csv,
        max_per_run=args.max_per_run,
    )


if __name__ == "__main__":
    main()