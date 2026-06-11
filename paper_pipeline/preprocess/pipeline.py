"""
paper_pipeline.preprocess.pipeline
====================================
Load, deduplicate, and filter Scopus export CSVs down to the set of
papers that will be fetched from publisher APIs.

This is the notebook's first half — everything before the XML download
steps — expressed as a callable API.

Notebook cells covered
-----------------------
1. Load multiple CSVs and concatenate
2. Inspect publisher counts (bar chart)
3. Identify and drop duplicate DOIs
4. Filter by publisher regex (default: Elsevier)
5. Filter by Open Access type
6. Save filtered CSVs
7. Move qualifying XMLs into a working directory (green open access)
8. Copy CC BY XMLs into a separate directory (post-license-check)
9. Flatten downloaded image sub-folders into a single flat folder
10. Prune images that have no matching .txt caption file

Cells NOT included (publisher-specific, live in elsevier/ subpackage):
- The figure extractor cell (→ elsevier/extractor.py)
- The CC license diagnostics cell (→ preprocess/cc_license.py)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TypeAlias
from collections.abc import Iterable

import pandas as pd

FileArg: TypeAlias = str | Path | Iterable[str | Path]

# ---------------------------------------------------------------------------
# Constants  (match notebook defaults)
# ---------------------------------------------------------------------------

ELSEVIER_PATTERN: str = r"\bElsevier\b"

OPEN_ACCESS_ALLOWED: frozenset[str] = frozenset({
    "All Open Access",
    "All Open Access; Green Open Access",
    "All Open Access; Bronze Open Access",
    "All Open Access; Bronze Open Access; Green Open Access",
})

IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def _clean_doi(doi: str) -> str:
    """Strip https://doi.org/ prefix if present."""
    import re
    return re.sub(r"^https?://doi\.org/", "", str(doi).strip())


def load_csvs(filenames: FileArg, **read_csv_kwargs) -> pd.DataFrame:
    """Load one or more Scopus export CSVs and concatenate them.

    Parameters
    ----------
    filenames:
        A single path or any iterable of paths.
    **read_csv_kwargs:
        Passed to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        Concatenated data, index reset to 0 … N-1.

    Examples
    --------
    >>> df = load_csvs(["Al_alloy_10k.csv", "Ni_alloy_10k.csv"])
    >>> df = load_csvs(Path("data").glob("*.csv"))
    """
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    frames = [pd.read_csv(f, **read_csv_kwargs) for f in filenames]
    if not frames:
        raise ValueError("No files provided to load_csvs().")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Inspect
# ---------------------------------------------------------------------------

def publisher_counts(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Return a value-counts Series for the top-N publishers.

    Mirrors the ``merged_df["Publisher"].value_counts().head(10)`` cell.
    """
    return df["Publisher"].value_counts().head(top_n)


# ---------------------------------------------------------------------------
# 3. Deduplication
# ---------------------------------------------------------------------------

def find_duplicate_dois(df: pd.DataFrame, doi_col: str = "DOI") -> pd.DataFrame:
    """Return all rows whose DOI appears more than once.

    Mirrors:
        ``merged_df[merged_df["DOI"].notna() & merged_df["DOI"].duplicated(keep=False)]``
    """
    mask = df[doi_col].notna() & df[doi_col].duplicated(keep=False)
    return df[mask].sort_values(doi_col)


def drop_duplicate_dois(
    df: pd.DataFrame,
    doi_col: str = "DOI",
    keep: str = "first",
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove duplicated DOIs, keeping one row per DOI.

    Mirrors:
        ``merged_df.drop_duplicates(subset=["DOI"], keep="first").reset_index(drop=True)``
    """
    # Normalize DOIs: strip https://doi.org/ prefix if present
    if doi_col in df.columns:
        df = df.copy()
        df[doi_col] = df[doi_col].apply(lambda x: _clean_doi(x) if pd.notna(x) else x)

    dupes = find_duplicate_dois(df, doi_col)
    clean = df.drop_duplicates(subset=[doi_col], keep=keep).reset_index(drop=True)
    if verbose:
        print(f"Duplicate DOI rows: {len(dupes)} | {len(df)} → {len(clean)} rows after dedup")
    return clean


# ---------------------------------------------------------------------------
# 4. Publisher filter
# ---------------------------------------------------------------------------

def filter_by_publisher(
    df: pd.DataFrame,
    pattern: str = ELSEVIER_PATTERN,
    publisher_col: str = "Publisher",
    case: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Keep rows whose publisher matches *pattern*.

    Mirrors:
        ``merged_df[merged_df["Publisher"].fillna("").str.contains(elsevier_pattern, ...)]``
    """
    mask = df[publisher_col].fillna("").str.contains(pattern, case=case, regex=True)
    result = df[mask].reset_index(drop=True)
    if verbose:
        print(f"Publisher filter '{pattern}': {len(df)} → {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# 5. Open-access filter
# ---------------------------------------------------------------------------

def filter_open_access(
    df: pd.DataFrame,
    allowed: frozenset[str] | set[str] | list[str] = OPEN_ACCESS_ALLOWED,
    oa_col: str = "Open Access",
    verbose: bool = True,
) -> pd.DataFrame:
    """Keep rows whose Open Access type is in *allowed*.

    Mirrors:
        ``green_df = df[df["Open Access"].isin(allowed_open_access)]``
    """
    result = df[df[oa_col].isin(allowed)].reset_index(drop=True)
    if verbose:
        print(f"Open-access filter: {len(df)} → {len(result)} rows")
    return result


# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: str | Path, verbose: bool = True) -> Path:
    """Save *df* to CSV (no index). Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        print(f"Saved {len(df)} rows → {path}")
    return path.resolve()


# ---------------------------------------------------------------------------
# 7. Move qualifying XMLs (green open-access subset)
# ---------------------------------------------------------------------------

def move_xmls_by_doi(
    dois: pd.Series | list[str],
    source_dir: str | Path,
    target_dir: str | Path,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """Move XML files from *source_dir* to *target_dir* for the given DOIs.

    Mirrors the notebook cell:
        ``for doi in dois: shutil.move(source_dir / doi.replace("/","_")+".xml", ...)``

    Parameters
    ----------
    dois:
        Iterable of DOI strings (from ``green_df["DOI"]``).
    source_dir:
        Directory currently holding all XML files.
    target_dir:
        Destination directory (created if it doesn't exist).

    Returns
    -------
    moved : list[str]   — filenames successfully moved
    missing : list[str] — DOIs whose XML was not found
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    clean_dois = (
        pd.Series(dois).dropna().astype(str).str.strip()
        .pipe(lambda s: s[s != ""])
        .drop_duplicates().tolist()
    )

    moved, missing = [], []
    for doi in clean_dois:
        xml_name = doi.replace("/", "_") + ".xml"
        src = source_dir / xml_name
        if src.exists():
            shutil.move(str(src), target_dir / xml_name)
            moved.append(xml_name)
        else:
            missing.append(doi)

    if verbose:
        print(f"move_xmls_by_doi: {len(clean_dois)} DOIs | {len(moved)} moved | {len(missing)} missing")
    return moved, missing


# ---------------------------------------------------------------------------
# 8. Copy CC BY XMLs
# ---------------------------------------------------------------------------

def copy_xmls_by_filename(
    filenames: Iterable[str],
    source_dir: str | Path,
    target_dir: str | Path,
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """Copy specific XML files (by filename) to *target_dir*.

    Mirrors:
        ``for fname in cc_by_files: shutil.copy2(source_dir/fname, cc_by_xml_dir/fname)``

    Returns
    -------
    copied : int
    missing : list[str]   — filenames not found in source_dir
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied, missing = 0, []
    for fname in filenames:
        src = source_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)
            copied += 1
        else:
            missing.append(fname)

    if verbose:
        print(f"copy_xmls_by_filename: {copied} copied | {len(missing)} missing")
    return copied, missing


# ---------------------------------------------------------------------------
# 9. Flatten image sub-folders
# ---------------------------------------------------------------------------

def flatten_images(
    source_root: str | Path,
    target_dir: str | Path,
    limit: int | None = None,
    image_exts: frozenset[str] = IMAGE_EXTS,
    verbose: bool = True,
) -> dict[str, int]:
    """Copy images from nested per-paper folders into one flat directory.

    Mirrors the notebook cell that scanned ``alloy_elsevier_contents``
    sub-folders and copied everything into ``alloy_images_flat``.

    Parameters
    ----------
    source_root:
        Root whose immediate children are per-paper folders.
    target_dir:
        Flat output directory.
    limit:
        If set, process only the first *limit* sub-folders (sorted).

    Returns
    -------
    dict with keys ``"copied"`` and ``"skipped"``.
    """
    source_root = Path(source_root)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    subfolders = sorted(p for p in source_root.iterdir() if p.is_dir())
    if limit is not None:
        subfolders = subfolders[:limit]

    copied = skipped = 0
    for folder in subfolders:
        image_dir = folder / "images" if (folder / "images").exists() else folder
        for img_path in image_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in image_exts:
                dst = target_dir / f"{folder.name}_{img_path.name}"
                if dst.exists():
                    skipped += 1
                else:
                    shutil.copy2(img_path, dst)
                    copied += 1

    if verbose:
        print(f"flatten_images: {len(subfolders)} folders | {copied} copied | {skipped} skipped")
    return {"copied": copied, "skipped": skipped}


# ---------------------------------------------------------------------------
# 10. Prune images without matching captions
# ---------------------------------------------------------------------------

def prune_images_without_captions(
    image_dir: str | Path,
    caption_ext: str = ".txt",
    image_exts: frozenset[str] = IMAGE_EXTS,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, int | list[str]]:
    """Delete image files that have no matching caption file.

    Mirrors:
        ``for img in images: if not img.with_suffix(".txt").exists(): img.unlink()``

    Parameters
    ----------
    dry_run:
        Report what *would* be deleted without actually deleting.
    """
    image_dir = Path(image_dir)
    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]

    kept = removed = 0
    removed_files: list[str] = []

    for img in images:
        if img.with_suffix(caption_ext).exists():
            kept += 1
        else:
            removed_files.append(img.name)
            removed += 1
            if not dry_run:
                img.unlink()

    if verbose:
        action = "would remove" if dry_run else "removed"
        print(f"prune_images: {len(images)} total | {kept} kept | {removed} {action}")
    return {"kept": kept, "removed": removed, "removed_files": removed_files}


# ---------------------------------------------------------------------------
# Convenience: full CSV preprocessing in one call
# ---------------------------------------------------------------------------

def run(
    csv_files: FileArg,
    publisher_pattern: str = ELSEVIER_PATTERN,
    open_access_allowed: frozenset[str] = OPEN_ACCESS_ALLOWED,
    output_dir: str | Path = Path("output"),
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full CSV preprocessing pipeline.

    Load → dedup → filter publisher → filter open-access → save.

    Parameters
    ----------
    csv_files:
        Input CSV path(s).
    publisher_pattern:
        Regex for publisher filter.
    open_access_allowed:
        Allowed Open Access tag values.
    output_dir:
        Directory for output CSVs.
    verbose:
        Print progress.

    Returns
    -------
    dict with keys:
        ``"raw"``              — concatenated raw data
        ``"deduped"``          — after DOI dedup
        ``"publisher"``        — after publisher filter  (saved as publisher_filtered.csv)
        ``"open_access"``      — after OA filter         (saved as open_access_filtered.csv)

    Examples
    --------
    >>> result = run(["Al_alloy_10k.csv", "Ni_alloy_10k.csv"])
    >>> green_df = result["open_access"]
    """
    raw = load_csvs(csv_files)
    if verbose:
        print(f"Loaded {len(raw):,} rows from {len(list(csv_files) if not isinstance(csv_files, (str, Path)) else [csv_files])} file(s)")
        print(publisher_counts(raw).to_string())

    deduped = drop_duplicate_dois(raw, verbose=verbose)
    pub_filtered = filter_by_publisher(deduped, pattern=publisher_pattern, verbose=verbose)
    oa_filtered = filter_open_access(pub_filtered, allowed=open_access_allowed, verbose=verbose)

    output_dir = Path(output_dir)
    save_csv(pub_filtered, output_dir / "publisher_filtered.csv", verbose=verbose)
    save_csv(oa_filtered, output_dir / "open_access_filtered.csv", verbose=verbose)

    return {
        "raw": raw,
        "deduped": deduped,
        "publisher": pub_filtered,
        "open_access": oa_filtered,
    }