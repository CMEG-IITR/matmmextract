from __future__ import annotations

import argparse
import os

import pandas as pd

from paper_pipeline.shared.downloader import run_downloads


DEFAULT_CSV = "alloy_elsevier_img_details.csv"
DEFAULT_OUTPUT_DIR = "alloy_elsevier_contents"
DEFAULT_OUTPUT_CSV = "alloy_elsevier_figures_with_images.csv"
DEFAULT_LOG_FILE = "download_log_elsevier.csv"
DEFAULT_NAME_PREFIX = "img"
DEFAULT_MAX_WORKERS = 4


def build_candidate_urls(image_url: str, xml_file: str) -> list[str]:
    """Return candidate download URLs for an Elsevier image.

    Elsevier image URLs in the figure CSV are absolute CDN URLs, so this
    is a direct passthrough.  The ``xml_file`` argument is unused but kept
    for interface compatibility with the shared engine.
    """
    url = image_url.strip()
    return [url] if url else []


def download_all(
    csv_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    log_file: str = DEFAULT_LOG_FILE,
    api_key: str | None = None,
    inst_token: str | None = None,
    name_prefix: str = DEFAULT_NAME_PREFIX,
    max_workers: int = DEFAULT_MAX_WORKERS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download all Elsevier figure images referenced in *csv_path*.

    Parameters
    ----------
    csv_path:
        Figure CSV (output of :func:`~paper_pipeline.elsevier.extractor.extract_all`).
    output_dir:
        Directory to save downloaded images.
    output_csv:
        Updated CSV written after each batch.
    log_file:
        Per-URL download log for resume support.
    api_key:
        Elsevier API key (falls back to ``ELSEVIER_API_KEY`` env var).
    inst_token:
        Elsevier institutional token (falls back to ``ELSEVIER_INST_TOKEN``).
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
    api_key = api_key or os.getenv("ELSEVIER_API_KEY", "")
    inst_token = inst_token or os.getenv("ELSEVIER_INST_TOKEN", "")

    headers = {
        "X-ELS-APIKey": api_key,
        "X-ELS-Insttoken": inst_token,
        "Accept": "image/jpeg",
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Elsevier figure images.")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    p.add_argument("--api-key", default=os.getenv("ELSEVIER_API_KEY"))
    p.add_argument("--inst-token", default=os.getenv("ELSEVIER_INST_TOKEN"))
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
        api_key=args.api_key,
        inst_token=args.inst_token,
        name_prefix=args.name_prefix,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()