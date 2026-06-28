"""
matmmextract.shared.downloader
==================================
Threaded image download engine shared by the Elsevier and Springer
download scripts.

The two original scripts (``elsevier_download_image.py`` and
``download_springer_img.py``) had largely parallel structures.
This module provides:

* ``next_name_allocator``  — resume-safe sequential image naming
                             (ported from the Springer script; replaces the
                             hardcoded ``img_counter = 13101`` in Elsevier)
* ``load_download_log``    — read a previous log to skip already-done URLs
* ``append_log``           — append rows to the CSV log incrementally
* ``DownloadResult``       — typed result from a single download attempt
* ``download_one``         — download one URL with retries
* ``run_downloads``        — orchestrate a full threaded batch

Each publisher supplies its own ``build_candidate_urls`` callable and
``request_headers`` dict; everything else is shared.
"""

from __future__ import annotations

import csv
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests



@dataclass
class DownloadResult:
    row_index: Any
    image_url: str
    status: str                   # "success" | "failed" | "empty_url" | "skipped"
    downloaded_image_name: str
    local_path: str
    attempted_url: str
    error: str = ""


def next_name_allocator(
    output_dir: str | Path,
    df: pd.DataFrame,
    name_prefix: str,
) -> Callable[[str], str]:
    """Return a closure that allocates the next unique image filename.

    Scans *output_dir* for existing files matching ``<name_prefix><N>.*``
    and the ``downloaded_image_name`` column of *df* so that resumed runs
    never overwrite previous downloads.

    Parameters
    ----------
    output_dir:
        Directory where images will be saved.
    df:
        DataFrame that may already have a ``downloaded_image_name`` column
        from a previous run.
    name_prefix:
        Prefix string, e.g. ``"alloy_img"`` or ``"img"``.

    Returns
    -------
    Callable[[str], str]
        ``allocate(ext)`` → unique filename string (e.g. ``"alloy_img42.jpg"``).
    """
    used: set[str] = set()
    max_num = 0

    for path in Path(output_dir).glob(f"{name_prefix}*.*"):
        used.add(path.name)
        m = re.match(rf"^{re.escape(name_prefix)}(\d+)\.", path.name)
        if m:
            max_num = max(max_num, int(m.group(1)))

    if "downloaded_image_name" in df.columns:
        for val in df["downloaded_image_name"].dropna().astype(str):
            name = val.strip()
            if not name:
                continue
            used.add(name)
            m = re.match(rf"^{re.escape(name_prefix)}(\d+)\.", name)
            if m:
                max_num = max(max_num, int(m.group(1)))

    counter = max_num + 1

    def allocate(ext: str) -> str:
        nonlocal counter
        while True:
            name = f"{name_prefix}{counter}{ext}"
            counter += 1
            if name not in used:
                used.add(name)
                return name

    return allocate


def load_download_log(log_file: str | Path) -> dict[str, dict]:
    """Load a previous download log CSV into a ``{image_url: result}`` dict.

    Only ``"success"`` rows whose local file still exists are returned.

    Parameters
    ----------
    log_file:
        Path to the CSV log written by previous runs.

    Returns
    -------
    dict
        Keys are ``image_url`` strings; values are dicts with keys
        ``download_status``, ``downloaded_image_name``,
        ``downloaded_image_path``, ``attempted_url``.
    """
    log_path = Path(log_file)
    if not log_path.exists():
        return {}

    lookup: dict[str, dict] = {}
    with open(log_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 4:
                continue

            image_url = row[0].strip()
            status = row[1].strip()
            # Support both 4-column and 6-column log formats
            if len(row) >= 6:
                local_name = row[2].strip()
                local_path = row[3].strip()
                attempted_url = row[4].strip()
            else:
                local_path = row[2].strip()
                attempted_url = row[3].strip()
                local_name = Path(local_path).name if local_path else ""

            if not image_url or status != "success" or not local_path:
                continue
            if not Path(local_path).exists():
                continue

            lookup[image_url] = {
                "download_status": status,
                "downloaded_image_name": local_name or Path(local_path).name,
                "downloaded_image_path": local_path,
                "attempted_url": attempted_url,
            }

    return lookup


def append_log(log_file: str | Path, rows: list[dict]) -> None:
    """Append *rows* to the CSV log, writing a header if the file is new.

    Parameters
    ----------
    log_file:
        Destination CSV path.
    rows:
        List of dicts; all must share the same keys.
    """
    if not rows:
        return
    log_path = Path(log_file)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    pd.DataFrame(rows).to_csv(log_path, mode="a", header=write_header, index=False)


def extension_from_url(url: str) -> str:
    """Infer a file extension from a URL, defaulting to ``.jpg``."""
    suffix = Path(urlparse(str(url)).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".tif", ".tiff"}:
        return suffix
    return ".jpg"


def download_one(
    index: Any,
    image_url: str,
    candidate_urls: list[str],
    local_path: Path,
    local_name: str,
    headers: dict,
    retries: int = 3,
    timeout: int = 90,
) -> DownloadResult:
    """Attempt to download one image, trying each candidate URL in order.

    Parameters
    ----------
    index:
        DataFrame row index (passed through to the result for later join).
    image_url:
        Original URL from the figure CSV (used as the lookup key in logs).
    candidate_urls:
        Ordered list of URLs to try (built by the caller's
        ``build_candidate_urls`` function).
    local_path:
        Full path where the file should be written on success.
    local_name:
        Bare filename (used in the result for the CSV column).
    headers:
        HTTP request headers (auth tokens, User-Agent, Accept, etc.).
    retries:
        Attempts per candidate URL before moving to the next.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    DownloadResult
    """
    last_error = "NO_URL"

    for url in candidate_urls:
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)

                if response.status_code == 429:
                    time.sleep(60 + attempt * 30)
                    continue

                if response.status_code != 200:
                    last_error = f"HTTP_{response.status_code}"
                    time.sleep(1 + attempt)
                    continue

                content_type = response.headers.get("content-type", "").lower()
                content = response.content
                is_svg = content.lstrip().startswith(b"<svg")

                if "image" not in content_type and not is_svg:
                    last_error = f"NON_IMAGE:{content_type}"
                    break  # wrong content-type: skip remaining attempts for this URL

                local_path.write_bytes(content)
                return DownloadResult(
                    row_index=index,
                    image_url=image_url,
                    status="success",
                    downloaded_image_name=local_name,
                    local_path=str(local_path),
                    attempted_url=url,
                )

            except Exception as exc:
                last_error = str(exc)
                time.sleep(2 + attempt * 2)

    return DownloadResult(
        row_index=index,
        image_url=image_url,
        status="failed",
        downloaded_image_name="",
        local_path="",
        attempted_url=" || ".join(candidate_urls),
        error=last_error,
    )


def run_downloads(
    df: pd.DataFrame,
    output_dir: str | Path,
    output_csv: str | Path,
    log_file: str | Path,
    build_candidate_urls: Callable[[str, str], list[str]],
    request_headers: dict,
    name_prefix: str = "img",
    max_workers: int = 4,
    retries: int = 3,
    timeout: int = 90,
    cool_every: int = 100,
    cool_seconds: float = 60.0,
    long_cool_every: int = 1500,
    long_cool_seconds: float = 300.0,
    xml_file_col: str = "xml_file",
    verbose: bool = True,
) -> pd.DataFrame:
    """Download all images referenced in *df*, with resume support.

    Parameters
    ----------
    df:
        Figure DataFrame with at least an ``image_url`` column.
        Modified in-place with ``download_status``,
        ``downloaded_image_name``, ``downloaded_image_path`` columns.
    output_dir:
        Directory where image files are saved.
    output_csv:
        CSV written after each batch (safe to resume from).
    log_file:
        Per-URL download log CSV.
    build_candidate_urls:
        ``(image_url, xml_file) -> [url, ...]``
        Publisher-specific URL builder.
    request_headers:
        HTTP headers for every request (auth keys, User-Agent, etc.).
    name_prefix:
        Image filename prefix (e.g. ``"alloy_img"`` or ``"img"``).
    max_workers:
        Thread pool size.
    retries:
        Attempts per candidate URL inside :func:`download_one`.
    timeout:
        Per-request timeout seconds.
    cool_every / cool_seconds:
        Pause after this many *real* downloads.
    long_cool_every / long_cool_seconds:
        Longer pause after this many real downloads.
    xml_file_col:
        Column in *df* holding the source XML filename (used by
        ``build_candidate_urls`` for relative-URL resolution).
    verbose:
        Print progress lines.

    Returns
    -------
    pd.DataFrame
        Updated *df* (also written to *output_csv*).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in ("download_status", "downloaded_image_name", "downloaded_image_path"):
        if col not in df.columns:
            df[col] = ""

    already_done = load_download_log(log_file)
    allocate_name = next_name_allocator(output_dir, df, name_prefix)

    # Build task list
    tasks: list[tuple] = []
    for i, row in df.iterrows():
        image_url = str(row.get("image_url", "")).strip()

        if not image_url:
            df.at[i, "download_status"] = "empty_url"
            continue

        # Resume: already succeeded in a previous run
        if image_url in already_done:
            prev = already_done[image_url]
            df.at[i, "download_status"] = prev["download_status"]
            df.at[i, "downloaded_image_name"] = prev["downloaded_image_name"]
            df.at[i, "downloaded_image_path"] = prev["downloaded_image_path"]
            continue

        # Resume: success recorded in the df itself and file exists
        existing_status = str(row.get("download_status", "")).strip()
        existing_path = str(row.get("downloaded_image_path", "")).strip()
        if existing_status == "success" and existing_path and Path(existing_path).exists():
            if Path(existing_path).parent.resolve() == output_dir.resolve():
                continue
            # Stale path pointing to a different directory — retry
            df.at[i, "downloaded_image_path"] = ""
            df.at[i, "download_status"] = ""

        candidates = build_candidate_urls(image_url, str(row.get(xml_file_col, "")))
        ext = extension_from_url(candidates[0] if candidates else image_url)
        local_name = allocate_name(ext)
        local_path = output_dir / local_name
        tasks.append((i, image_url, candidates, local_path, local_name))

    total = len(tasks)
    if verbose:
        print(f"\nPending downloads: {total}  |  output: {output_dir}  |  workers: {max_workers}\n")

    if not tasks:
        df.to_csv(output_csv, index=False)
        return df

    completed = 0
    downloaded_count = 0
    batch_log: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_one,
                idx, url, candidates, lpath, lname,
                request_headers, retries, timeout,
            ): idx
            for idx, url, candidates, lpath, lname in tasks
        }

        for future in as_completed(futures):
            result: DownloadResult = future.result()
            i = result.row_index

            df.at[i, "download_status"] = result.status
            df.at[i, "downloaded_image_name"] = result.downloaded_image_name
            df.at[i, "downloaded_image_path"] = result.local_path

            batch_log.append({
                "image_url": result.image_url,
                "status": result.status,
                "downloaded_image_name": result.downloaded_image_name,
                "local_path": result.local_path,
                "attempted_url": result.attempted_url,
                "error": result.error,
            })

            if result.status == "success":
                downloaded_count += 1

            completed += 1
            if completed % 25 == 0 or completed == total:
                append_log(log_file, batch_log)
                batch_log = []
                df.to_csv(output_csv, index=False)
                if verbose:
                    print(f"  {completed}/{total}  ✓ {downloaded_count} downloaded")

            # Cooling pauses (only triggered by real downloads)
            if downloaded_count and downloaded_count % long_cool_every == 0:
                if verbose:
                    print(f"\nLong cool-down after {long_cool_every} downloads → {long_cool_seconds}s\n")
                time.sleep(long_cool_seconds)
            elif downloaded_count and downloaded_count % cool_every == 0:
                if verbose:
                    print(f"\nCool-down after {cool_every} downloads → {cool_seconds}s\n")
                time.sleep(cool_seconds)

    # Final flush
    if batch_log:
        append_log(log_file, batch_log)
    df.to_csv(output_csv, index=False)

    if verbose:
        success = int((df["download_status"] == "success").sum())
        failed = int((df["download_status"] == "failed").sum())
        print(f"\nDone.  Success: {success}  |  Failed: {failed}  |  CSV: {output_csv}")

    return df