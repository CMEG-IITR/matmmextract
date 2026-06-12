from __future__ import annotations

import argparse
from urllib.parse import quote, urlparse

import pandas as pd

from ..shared.doi_utils import filename_to_doi
from ..shared.downloader import extension_from_url, run_downloads



SPRINGER_STATIC_BASE = "https://media.springernature.com/full/springer-static/image"

DEFAULT_CSV = "springer_figure_details.csv"
DEFAULT_OUTPUT_DIR = "springer_images_flat"
DEFAULT_OUTPUT_CSV = "springer_figure_details_with_images.csv"
DEFAULT_LOG_FILE = "download_log_springer.csv"
DEFAULT_NAME_PREFIX = "alloy_img"
DEFAULT_MAX_WORKERS = 6


def _is_absolute(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def build_candidate_urls(image_url: str, xml_file: str) -> list[str]:
    """Build ordered candidate URLs for a Springer image.

    If *image_url* is already absolute, return it directly.
    Otherwise, build CDN candidates using the DOI reconstructed from
    *xml_file*.

    Parameters
    ----------
    image_url:
        Raw URL string from the figure CSV.
    xml_file:
        Source XML filename (used to reconstruct the DOI for relative URLs).

    Returns
    -------
    list[str]
        Ordered candidate URLs to try.
    """
    url = str(image_url).strip()
    if not url:
        return []
    if _is_absolute(url):
        return [url]

    candidates: list[str] = []
    doi = filename_to_doi(xml_file)
    rel = url.lstrip("/")

    if doi:
        article_key = "art%3A" + quote(doi, safe="")
        candidates.append(f"{SPRINGER_STATIC_BASE}/{article_key}/{rel}")
        if not rel.startswith("MediaObjects/"):
            candidates.append(
                f"{SPRINGER_STATIC_BASE}/{article_key}/MediaObjects/{rel}"
            )

    candidates.append(f"https://media.springernature.com/{rel}")
    return list(dict.fromkeys(candidates))  # deduplicate preserving order


def download_all(
    csv_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    log_file: str = DEFAULT_LOG_FILE,
    name_prefix: str = DEFAULT_NAME_PREFIX,
    max_workers: int = DEFAULT_MAX_WORKERS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download all Springer figure images referenced in *csv_path*.

    Parameters
    ----------
    csv_path:
        Figure CSV (output of
        :func:`~multimat.springer.extractor.extract_all`).
    output_dir:
        Directory to save downloaded images.
    output_csv:
        Updated CSV written after each batch.
    log_file:
        Per-URL download log for resume support.
    name_prefix:
        Image filename prefix.
    max_workers:
        Thread pool size.
    verbose:
        Print progress.

    Returns
    -------
    pd.DataFrame
        Updated figure DataFrame with download status columns added.
    """
    headers = {
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "User-Agent": "Mozilla/5.0 SpringerImageDownloader/1.0",
    }

    df = pd.read_csv(csv_path)

    return run_downloads(
        df=df,
        output_dir=output_dir,
        output_csv=output_csv,
        log_file=log_file,
        build_candidate_urls=build_candidate_urls,
        request_headers=headers,
        name_prefix=name_prefix,
        max_workers=max_workers,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Springer figure images.")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    p.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    p.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    download_all(
        csv_path=args.csv,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        log_file=args.log_file,
        name_prefix=args.name_prefix,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()