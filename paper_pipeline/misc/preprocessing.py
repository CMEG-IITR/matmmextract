from __future__ import annotations

from pathlib import Path
from typing import TypeAlias
from collections.abc import Iterable

import pandas as pd

FileArg: TypeAlias = str | Path | Iterable[str | Path]


def load_csvs(filenames: FileArg, **read_csv_kwargs) -> pd.DataFrame:
    """Load one or more CSV files and concatenate them into a single DataFrame.

    Parameters
    ----------
    filenames:
        A single file path or an iterable of file paths.
    **read_csv_kwargs:
        Passed verbatim to :func:`pandas.read_csv` (e.g. ``sep``, ``encoding``).

    Returns
    -------
    pd.DataFrame
        Concatenated data, index reset to 0 … N-1.

    Examples
    --------
    >>> df = load_csvs("data.csv")
    >>> df = load_csvs(["Al_alloy_10k.csv", "Ni_alloy_10k.csv"])
    >>> df = load_csvs(Path("data_dir").glob("*.csv"))
    """
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]

    frames = [pd.read_csv(f, **read_csv_kwargs) for f in filenames]
    if not frames:
        raise ValueError("No files were provided to load_csvs().")

    return pd.concat(frames, ignore_index=True)


def save_csv(df: pd.DataFrame, path: str | Path, **to_csv_kwargs) -> Path:
    """Save a DataFrame to CSV.

    Parameters
    ----------
    df:
        The DataFrame to save.
    path:
        Destination file path.  Parent directories are created automatically.
    **to_csv_kwargs:
        Passed verbatim to :meth:`DataFrame.to_csv`
        (``index=False`` is set by default but can be overridden).

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    to_csv_kwargs.setdefault("index", False)
    df.to_csv(path, **to_csv_kwargs)
    return path.resolve()



def get_duplicate_doi_rows(df: pd.DataFrame, doi_col: str = "DOI") -> pd.DataFrame:
    """Return all rows whose DOI appears more than once.

    Rows with a missing DOI are excluded from the result (they cannot be
    meaningfully compared).

    Parameters
    ----------
    df:
        Input DataFrame.
    doi_col:
        Name of the DOI column (default ``"DOI"``).

    Returns
    -------
    pd.DataFrame
        Subset of ``df`` containing only the duplicated-DOI rows,
        sorted by DOI for easy visual inspection.

    Examples
    --------
    >>> dupes = get_duplicate_doi_rows(df)
    >>> print(f"Duplicated DOI rows: {len(dupes)}")
    """
    mask = df[doi_col].notna() & df[doi_col].duplicated(keep=False)
    return df[mask].sort_values(doi_col)


def drop_duplicate_dois(
    df: pd.DataFrame,
    doi_col: str = "DOI",
    keep: str = "first",
) -> pd.DataFrame:
    """Remove rows with a duplicated DOI, keeping one representative row.

    Parameters
    ----------
    df:
        Input DataFrame.
    doi_col:
        Name of the DOI column (default ``"DOI"``).
    keep:
        Which occurrence to keep: ``"first"`` (default), ``"last"``, or
        ``False`` (drop *all* duplicates).

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with a fresh 0-based integer index.

    Examples
    --------
    >>> clean = drop_duplicate_dois(df)
    >>> clean = drop_duplicate_dois(df, keep="last")
    """
    return (
        df
        .drop_duplicates(subset=[doi_col], keep=keep)
        .reset_index(drop=True)
    )


def deduplicate(
    df: pd.DataFrame,
    doi_col: str = "DOI",
    keep: str = "first",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report duplicate DOI rows, then return the deduplicated DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame.
    doi_col:
        Name of the DOI column.
    keep:
        Passed to :func:`drop_duplicate_dois`.
    verbose:
        If ``True``, print a summary line to stdout.

    Returns
    -------
    clean : pd.DataFrame
        Deduplicated DataFrame.
    dupes : pd.DataFrame
        The rows that were identified as duplicates (for logging / QA).

    Examples
    --------
    >>> clean, dupes = deduplicate(df)
    >>> clean, _ = deduplicate(df, verbose=False)
    """
    dupes = get_duplicate_doi_rows(df, doi_col=doi_col)
    clean = drop_duplicate_dois(df, doi_col=doi_col, keep=keep)

    if verbose:
        print(
            f"[dedup] {len(dupes)} duplicate-DOI rows found. "
            f"{len(df)} → {len(clean)} rows after deduplication."
        )

    return clean, dupes


#: Regex that matches any Elsevier publisher variant
#: (e.g. "Elsevier Ltd", "Elsevier Editora Ltda", "Elsevier B.V.").
ELSEVIER_PATTERN: str = r"\bElsevier\b"

#: Open-Access tag values that are considered "green / acceptable".
OPEN_ACCESS_ALLOWED: frozenset[str] = frozenset(
    [
        "All Open Access",
        "All Open Access; Green Open Access",
        "All Open Access; Bronze Open Access",
        "All Open Access; Bronze Open Access; Green Open Access",
    ]
)


def filter_by_publisher(
    df: pd.DataFrame,
    pattern: str = ELSEVIER_PATTERN,
    publisher_col: str = "Publisher",
    case: bool = False,
) -> pd.DataFrame:
    """Keep only rows whose publisher matches *pattern*.

    Parameters
    ----------
    df:
        Input DataFrame.
    pattern:
        Regular-expression pattern (default: :data:`ELSEVIER_PATTERN`).
    publisher_col:
        Name of the publisher column (default ``"Publisher"``).
    case:
        Case-sensitive matching (default ``False``).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with a fresh 0-based integer index.

    Examples
    --------
    >>> elsevier_df = filter_by_publisher(df)
    >>> springer_df = filter_by_publisher(df, pattern=r"\\bSpringer\\b")
    """
    mask = (
        df[publisher_col]
        .fillna("")
        .str.contains(pattern, case=case, regex=True)
    )
    return df[mask].reset_index(drop=True)


def publisher_summary(
    df: pd.DataFrame,
    publisher_col: str = "Publisher",
    top_n: int = 10,
) -> pd.Series:
    """Return a value-counts Series for the top-N publishers.

    Useful for a quick inspection before deciding which pattern to use.

    Parameters
    ----------
    df:
        Input DataFrame.
    publisher_col:
        Name of the publisher column.
    top_n:
        How many publishers to return.

    Returns
    -------
    pd.Series
        Counts, descending.
    """
    return df[publisher_col].value_counts().head(top_n)


def filter_open_access(
    df: pd.DataFrame,
    allowed: frozenset[str] | set[str] | list[str] = OPEN_ACCESS_ALLOWED,
    oa_col: str = "Open Access",
) -> pd.DataFrame:
    """Keep only rows whose Open Access tag is in the *allowed* set.

    Parameters
    ----------
    df:
        Input DataFrame.
    allowed:
        Collection of allowed Open Access tag strings
        (default: :data:`OPEN_ACCESS_ALLOWED`).
    oa_col:
        Name of the Open Access column (default ``"Open Access"``).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with a fresh 0-based integer index.

    Examples
    --------
    >>> green_df = filter_open_access(df)
    >>> only_gold = filter_open_access(df, allowed={"All Open Access"})
    """
    return df[df[oa_col].isin(allowed)].reset_index(drop=True)


def open_access_summary(
    df: pd.DataFrame,
    oa_col: str = "Open Access",
) -> pd.DataFrame:
    """Return a count + percentage breakdown of all Open Access types.

    Parameters
    ----------
    df:
        Input DataFrame.
    oa_col:
        Name of the Open Access column.

    Returns
    -------
    pd.DataFrame
        Columns: ``type``, ``count``, ``pct``.
    """
    counts = df[oa_col].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"type": counts.index, "count": counts.values, "pct": pct.values})
