from __future__ import annotations

import re
from pathlib import Path


def doi_to_filename(doi: str, suffix: str = ".xml") -> str:
    """Convert a DOI to a safe filename.

    Replaces every character that is not word-safe, a hyphen, underscore,
    or dot with ``_``.

    Parameters
    ----------
    doi:
        Raw DOI string, e.g. ``"10.1016/j.actamat.2020.01.001"``.
    suffix:
        File extension (default ``".xml"``).

    Returns
    -------
    str
        e.g. ``"10.1016_j.actamat.2020.01.001.xml"``
    """
    return re.sub(r"[^\w\-_.]", "_", doi) + suffix


def filename_to_doi(filename: str | Path) -> str:
    """Reverse ``doi_to_filename``: convert a stem back to a DOI.

    Handles the Springer convention where the first ``_`` after the
    registrant prefix (``10.XXXX``) maps back to ``/``.

    Parameters
    ----------
    filename:
        File path or bare filename, e.g.
        ``"10.1007_s42114-026-01633-w.xml"``.

    Returns
    -------
    str
        DOI string, e.g. ``"10.1007/s42114-026-01633-w"``, or ``""``
        if the stem does not look like a DOI.
    """
    stem = Path(str(filename)).stem
    match = re.match(r"^(10\.\d{4,9})_(.+)$", stem)
    if not match:
        return ""
    return f"{match.group(1)}/{match.group(2)}"


def load_set(path: str | Path) -> set[str]:
    """Load a newline-delimited text file into a set of strings.

    Returns an empty set if the file does not exist.

    Parameters
    ----------
    path:
        Path to the text file.
    """
    p = Path(path)
    if p.exists():
        return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()


def append_line(path: str | Path, line: str) -> None:
    """Append a single line (with newline) to a text file.

    Creates the file if it does not exist.

    Parameters
    ----------
    path:
        Destination file path.
    line:
        Text to append (newline is added automatically).
    """
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")