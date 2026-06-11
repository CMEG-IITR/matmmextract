from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


#: Image file extensions recognised by :func:`flatten_images`.
IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
)


def doi_to_filename(doi: str, suffix: str = ".xml") -> str:
    """Convert a DOI string to a safe filename.

    Replaces ``/`` with ``_`` (the convention used in the notebook).

    Parameters
    ----------
    doi:
        Raw DOI string, e.g. ``"10.1016/j.actamat.2020.01.001"``.
    suffix:
        File extension to append (default ``".xml"``).

    Returns
    -------
    str
        Filename, e.g. ``"10.1016_j.actamat.2020.01.001.xml"``.
    """
    return doi.replace("/", "_") + suffix


def move_xmls_by_doi(
    dois: pd.Series | list[str],
    source_dir: str | Path,
    target_dir: str | Path,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """Move XML files from *source_dir* to *target_dir* for the given DOIs.

    The expected filename for each DOI is ``<doi_with_slashes_replaced>.xml``.

    Parameters
    ----------
    dois:
        Iterable of DOI strings.
    source_dir:
        Directory that currently contains the XML files.
    target_dir:
        Destination directory (created if it does not exist).
    verbose:
        Print a summary line when done.

    Returns
    -------
    moved : list[str]
        Filenames successfully moved.
    missing : list[str]
        DOIs whose XML file was not found in *source_dir*.

    Examples
    --------
    >>> moved, missing = move_xmls_by_doi(green_df["DOI"], "alloys_elsevier", "Alloy_green_open_access")
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Normalise + deduplicate
    clean_dois = (
        pd.Series(dois)
        .dropna()
        .astype(str)
        .str.strip()
        .pipe(lambda s: s[s != ""])
        .drop_duplicates()
        .tolist()
    )

    moved: list[str] = []
    missing: list[str] = []

    for doi in clean_dois:
        fname = doi_to_filename(doi)
        src = source_dir / fname
        if src.exists():
            shutil.move(str(src), target_dir / fname)
            moved.append(fname)
        else:
            missing.append(doi)

    if verbose:
        print(
            f"[file_ops] move_xmls_by_doi: "
            f"{len(clean_dois)} DOIs checked | "
            f"{len(moved)} moved | "
            f"{len(missing)} missing"
        )

    return moved, missing


def copy_xmls_by_filename(
    filenames: list[str],
    source_dir: str | Path,
    target_dir: str | Path,
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """Copy specific XML files (by filename) from *source_dir* to *target_dir*.

    Parameters
    ----------
    filenames:
        List of bare filenames (e.g. ``["10.1016_foo.xml", …]``).
    source_dir:
        Directory containing the source XML files.
    target_dir:
        Destination directory (created if it does not exist).
    verbose:
        Print a summary line when done.

    Returns
    -------
    copied : int
        Number of files successfully copied.
    missing : list[str]
        Filenames that were not found in *source_dir*.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing: list[str] = []

    for fname in filenames:
        src = source_dir / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)
            copied += 1
        else:
            missing.append(fname)

    if verbose:
        print(
            f"[file_ops] copy_xmls_by_filename: "
            f"{len(filenames)} requested | "
            f"{copied} copied | "
            f"{len(missing)} missing"
        )

    return copied, missing


def flatten_images(
    source_root: str | Path,
    target_dir: str | Path,
    limit: int | None = None,
    image_exts: frozenset[str] = IMAGE_EXTS,
    verbose: bool = True,
) -> dict[str, int]:
    """Flatten a nested image directory into a single output folder.

    Scans first-level sub-folders inside *source_root*.  For each sub-folder
    it looks for an ``images/`` sub-directory; if absent it scans the
    sub-folder itself.  Every image file is copied to *target_dir* with a
    name ``<folder_name>_<original_filename>``.

    Parameters
    ----------
    source_root:
        Root directory whose immediate children are per-paper folders.
    target_dir:
        Flat output directory.
    limit:
        If set, process only the first *limit* sub-folders (sorted by name).
    image_exts:
        Set of lowercase extensions to treat as images.
    verbose:
        Print a summary line when done.

    Returns
    -------
    dict
        ``{"copied": int, "skipped": int}`` — skipped means the destination
        file already existed.

    Examples
    --------
    >>> stats = flatten_images("alloy_elsevier_contents", "alloy_images_flat", limit=50)
    >>> print(stats)
    {'copied': 312, 'skipped': 0}
    """
    source_root = Path(source_root)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    subfolders = sorted(p for p in source_root.iterdir() if p.is_dir())
    if limit is not None:
        subfolders = subfolders[:limit]

    copied = 0
    skipped = 0

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
        print(
            f"[file_ops] flatten_images: "
            f"{len(subfolders)} folders scanned | "
            f"{copied} copied | "
            f"{skipped} skipped (already existed)"
        )

    return {"copied": copied, "skipped": skipped}


def prune_images_without_captions(
    image_dir: str | Path,
    caption_ext: str = ".txt",
    image_exts: frozenset[str] = IMAGE_EXTS,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, int | list[str]]:
    """Delete image files that have no matching caption file.

    For each image ``foo.png`` this function checks whether ``foo.txt``
    (or whatever *caption_ext* is) exists in the same directory.  If not,
    the image is removed (unless *dry_run* is ``True``).

    Parameters
    ----------
    image_dir:
        Directory to scan.
    caption_ext:
        Extension of the expected companion caption file (default ``".txt"``).
    image_exts:
        Set of lowercase extensions to treat as images.
    dry_run:
        If ``True``, report what *would* be deleted without actually deleting.
    verbose:
        Print a summary line when done.

    Returns
    -------
    dict
        Keys: ``"kept"`` (int), ``"removed"`` (int), ``"removed_files"`` (list[str]).

    Examples
    --------
    >>> stats = prune_images_without_captions("alloy_images_flat")
    >>> stats = prune_images_without_captions("alloy_images_flat", dry_run=True)
    """
    image_dir = Path(image_dir)
    images = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    kept = 0
    removed = 0
    removed_files: list[str] = []

    for img in images:
        caption_file = img.with_suffix(caption_ext)
        if caption_file.exists():
            kept += 1
        else:
            removed_files.append(img.name)
            removed += 1
            if not dry_run:
                img.unlink()

    if verbose:
        action = "would remove" if dry_run else "removed"
        print(
            f"[file_ops] prune_images_without_captions: "
            f"{len(images)} images | "
            f"{kept} kept | "
            f"{removed} {action}"
        )

    return {"kept": kept, "removed": removed, "removed_files": removed_files}
