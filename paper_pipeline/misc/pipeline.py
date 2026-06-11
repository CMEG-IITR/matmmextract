from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from preprocessing import (
    load_csvs,
    save_csv,
    deduplicate,
    filter_by_publisher,
    filter_open_access,
    ELSEVIER_PATTERN,
    OPEN_ACCESS_ALLOWED,
    publisher_summary,
    open_access_summary,
)
from file_ops import move_xmls_by_doi, flatten_images
from xml_extractor import extract_all


@dataclass
class CSVPipelineResult:
    """Holds all intermediate and final DataFrames from :func:`run_csv_pipeline`."""

    raw: pd.DataFrame
    """Concatenated raw data from all input CSVs."""

    deduplicated: pd.DataFrame
    """After DOI deduplication."""

    publisher_filtered: pd.DataFrame
    """After publisher regex filter."""

    open_access_filtered: pd.DataFrame
    """After open-access type filter (final output)."""

    duplicate_rows: pd.DataFrame
    """Rows identified as DOI duplicates (for QA)."""

    saved_paths: dict[str, Path] = field(default_factory=dict)
    """Paths of any CSV files written to disk."""


@dataclass
class XMLPipelineResult:
    """Holds outcomes from :func:`run_xml_pipeline`."""

    figures_df: pd.DataFrame
    """One row per figure extracted from the XML corpus."""

    moved: list[str]
    """Filenames of XML files successfully moved to the working directory."""

    missing_dois: list[str]
    """DOIs whose XML file was not found."""

    extraction_errors: list[tuple[str, str]]
    """(filename, error_message) pairs for files that failed during extraction."""

    saved_paths: dict[str, Path] = field(default_factory=dict)
    """Paths of any CSV files written to disk."""



def run_csv_pipeline(
    csv_files,
    publisher_pattern: str = ELSEVIER_PATTERN,
    open_access_allowed: frozenset[str] = OPEN_ACCESS_ALLOWED,
    output_dir: str | Path | None = None,
    doi_col: str = "DOI",
    publisher_col: str = "Publisher",
    oa_col: str = "Open Access",
    verbose: bool = True,
) -> CSVPipelineResult:
    """Load CSVs, deduplicate by DOI, filter by publisher and open-access type.

    Parameters
    ----------
    csv_files:
        Single path or iterable of paths passed to :func:`~paper_pipeline.io.load_csvs`.
    publisher_pattern:
        Regex passed to :func:`~paper_pipeline.filters.filter_by_publisher`.
        Defaults to :data:`~paper_pipeline.filters.ELSEVIER_PATTERN`.
    open_access_allowed:
        Allowed OA tag values passed to
        :func:`~paper_pipeline.filters.filter_open_access`.
        Defaults to :data:`~paper_pipeline.filters.OPEN_ACCESS_ALLOWED`.
    output_dir:
        If provided, two checkpoint CSVs are written here:
        ``publisher_filtered.csv`` and ``open_access_filtered.csv``.
    doi_col / publisher_col / oa_col:
        Column name overrides.
    verbose:
        Print progress lines.

    Returns
    -------
    CSVPipelineResult

    Examples
    --------
    >>> result = run_csv_pipeline(
    ...     ["Al_alloy_10k.csv", "Ni_alloy_10k.csv"],
    ...     output_dir="output",
    ... )
    >>> result.open_access_filtered.head()
    """
    # 1. Load
    if verbose:
        print("[pipeline] Loading CSVs …")
    raw = load_csvs(csv_files)
    if verbose:
        print(f"[pipeline] Loaded {len(raw):,} rows.")
        print(publisher_summary(raw, publisher_col=publisher_col, top_n=5).to_string())

    # 2. Deduplicate
    deduped, dupes = deduplicate(raw, doi_col=doi_col, verbose=verbose)

    # 3. Publisher filter
    pub_filtered = filter_by_publisher(
        deduped, pattern=publisher_pattern, publisher_col=publisher_col
    )
    if verbose:
        print(f"[pipeline] Publisher filter: {len(deduped):,} → {len(pub_filtered):,} rows.")

    # 4. Open-access filter
    oa_filtered = filter_open_access(
        pub_filtered, allowed=open_access_allowed, oa_col=oa_col
    )
    if verbose:
        print(f"[pipeline] Open-access filter: {len(pub_filtered):,} → {len(oa_filtered):,} rows.")
        print(open_access_summary(oa_filtered, oa_col=oa_col).to_string(index=False))

    # 5. Save checkpoints
    saved: dict[str, Path] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        saved["publisher_filtered"] = save_csv(pub_filtered, output_dir / "publisher_filtered.csv")
        saved["open_access_filtered"] = save_csv(oa_filtered, output_dir / "open_access_filtered.csv")
        if verbose:
            for label, path in saved.items():
                print(f"[pipeline] Saved {label} → {path}")

    return CSVPipelineResult(
        raw=raw,
        deduplicated=deduped,
        publisher_filtered=pub_filtered,
        open_access_filtered=oa_filtered,
        duplicate_rows=dupes,
        saved_paths=saved,
    )



def run_xml_pipeline(
    green_df: pd.DataFrame,
    source_xml_dir: str | Path,
    working_xml_dir: str | Path,
    output_dir: str | Path | None = None,
    doi_col: str = "DOI",
    verbose: bool = True,
) -> XMLPipelineResult:
    """Move qualifying XML files and extract figure metadata.

    Parameters
    ----------
    green_df:
        DataFrame containing the DOIs to process (output of
        :func:`run_csv_pipeline`, ``open_access_filtered`` field, or similar).
    source_xml_dir:
        Directory currently holding *all* XML files.
    working_xml_dir:
        Destination directory for the qualifying subset of XML files.
    output_dir:
        If provided, ``figures.csv`` is written here.
    doi_col:
        Name of the DOI column in *green_df*.
    verbose:
        Print progress lines.

    Returns
    -------
    XMLPipelineResult

    Examples
    --------
    >>> xml_result = run_xml_pipeline(
    ...     green_df=csv_result.open_access_filtered,
    ...     source_xml_dir="alloys_elsevier",
    ...     working_xml_dir="Alloy_green_open_access",
    ...     output_dir="output",
    ... )
    """
    # 1. Move XMLs
    moved, missing = move_xmls_by_doi(
        green_df[doi_col],
        source_dir=source_xml_dir,
        target_dir=working_xml_dir,
        verbose=verbose,
    )

    # 2. Extract figures
    figures_csv = None
    if output_dir is not None:
        figures_csv = Path(output_dir) / "figures.csv"

    figures_df, errors = extract_all(
        xml_dir=working_xml_dir,
        output_csv=figures_csv,
        verbose=verbose,
    )

    saved: dict[str, Path] = {}
    if figures_csv is not None:
        saved["figures"] = Path(figures_csv).resolve()

    return XMLPipelineResult(
        figures_df=figures_df,
        moved=moved,
        missing_dois=missing,
        extraction_errors=errors,
        saved_paths=saved,
    )


def run(
    csv_files,
    source_xml_dir: str | Path,
    working_xml_dir: str | Path,
    publisher_pattern: str = ELSEVIER_PATTERN,
    open_access_allowed: frozenset[str] = OPEN_ACCESS_ALLOWED,
    output_dir: str | Path = Path("output"),
    verbose: bool = True,
) -> tuple[CSVPipelineResult, XMLPipelineResult]:
    """Run the complete preprocessing pipeline end-to-end.

    Combines :func:`run_csv_pipeline` and :func:`run_xml_pipeline`.

    Parameters
    ----------
    csv_files:
        Input CSV paths.
    source_xml_dir:
        Directory holding all source XML files.
    working_xml_dir:
        Directory for the filtered XML subset.
    publisher_pattern:
        Publisher regex filter (default: Elsevier).
    open_access_allowed:
        Allowed OA tag set.
    output_dir:
        Directory for all output CSVs (default: ``"output"``).
    verbose:
        Print progress.

    Returns
    -------
    csv_result : CSVPipelineResult
    xml_result : XMLPipelineResult

    Examples
    --------
    >>> from paper_pipeline.pipeline import run
    >>> csv_result, xml_result = run(
    ...     csv_files=["Al_alloy_10k.csv", "Ni_alloy_10k.csv"],
    ...     source_xml_dir="alloys_elsevier",
    ...     working_xml_dir="Alloy_green_open_access",
    ...     output_dir="output",
    ... )
    >>> print(xml_result.figures_df.head())
    """
    csv_result = run_csv_pipeline(
        csv_files=csv_files,
        publisher_pattern=publisher_pattern,
        open_access_allowed=open_access_allowed,
        output_dir=output_dir,
        verbose=verbose,
    )

    xml_result = run_xml_pipeline(
        green_df=csv_result.open_access_filtered,
        source_xml_dir=source_xml_dir,
        working_xml_dir=working_xml_dir,
        output_dir=output_dir,
        verbose=verbose,
    )

    return csv_result, xml_result
