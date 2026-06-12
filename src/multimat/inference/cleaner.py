"""
multimat.inference.cleaner
==================================
Delete intermediate files and directories after the full pipeline
has completed successfully, keeping only what matters:

KEPT
----
- crops/                    cropped panel images
- images/                   original downloaded images
- linked_dataset.csv        final output

DELETED
-------
- inference_results/        per-image detection JSONs + _summary.json
- subcaptions/              per-crop captioning JSONs
- output/                   all intermediate CSVs
- alloys_elsevier/          fetched Elsevier XMLs
- alloys_springer/          fetched Springer XMLs
- ``*.txt``                     processed_dois / failed_dois resume files
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CleanResult:
    deleted_dirs: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    skipped_missing: list[str] = field(default_factory=list)


def clean(
    base_dir: str | Path = ".",
    delete_dirs: list[str] | None = None,
    delete_files: list[str] | None = None,
    delete_glob_patterns: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> CleanResult:
    """Delete intermediate pipeline artifacts.

    Parameters
    ----------
    base_dir:
        Root directory to resolve paths from (default: current directory).
    delete_dirs:
        List of directory names/paths to delete recursively.
        Defaults to the standard pipeline intermediates.
    delete_files:
        List of specific file paths to delete.
    delete_glob_patterns:
        Glob patterns relative to base_dir, e.g. ``["*.txt", "output/*.csv"]``.
    dry_run:
        Print what would be deleted without actually deleting.
    verbose:
        Print each deleted item.

    Returns
    -------
    CleanResult

    Examples
    --------
    >>> from multimat.inference.cleaner import clean
    >>> clean()                          # delete all standard intermediates
    >>> clean(dry_run=True)              # preview without deleting
    >>> clean(delete_dirs=["output"])    # delete only output/
    """
    base_dir = Path(base_dir).resolve()

    # Defaults
    if delete_dirs is None:
        delete_dirs = [
            "inference_results",
            "subcaptions",
            "output",
            "alloys_elsevier",
            "alloys_springer",
        ]

    if delete_files is None:
        delete_files = [
            "elsevier_with_xml_paths.csv",
            "springer_with_xml_paths.csv",
        ]

    if delete_glob_patterns is None:
        delete_glob_patterns = [
            "processed_dois_*.txt",
            "failed_dois_*.txt",
            "download_log_*.csv",
            "build_dataset.log",
        ]

    result = CleanResult()
    action = "Would delete" if dry_run else "Deleted"

    # Directories
    for d in delete_dirs:
        path = base_dir / d
        if path.exists() and path.is_dir():
            if not dry_run:
                shutil.rmtree(path)
            result.deleted_dirs.append(str(path))
            if verbose:
                print(f"[cleaner] {action} dir  : {path}")
        else:
            result.skipped_missing.append(str(path))

    # Specific files
    for f in delete_files:
        path = base_dir / f
        if path.exists() and path.is_file():
            if not dry_run:
                path.unlink()
            result.deleted_files.append(str(path))
            if verbose:
                print(f"[cleaner] {action} file : {path}")
        else:
            result.skipped_missing.append(str(path))

    # Glob patterns
    for pattern in delete_glob_patterns:
        for path in sorted(base_dir.glob(pattern)):
            if path.is_file():
                if not dry_run:
                    path.unlink()
                result.deleted_files.append(str(path))
                if verbose:
                    print(f"[cleaner] {action} file : {path}")

    if verbose:
        print(
            f"\n[cleaner] {'(dry run) ' if dry_run else ''}"
            f"dirs={len(result.deleted_dirs)}  "
            f"files={len(result.deleted_files)}  "
            f"skipped={len(result.skipped_missing)}"
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delete intermediate pipeline files, keep crops + images + linked_dataset.csv"
    )
    p.add_argument(
        "--base-dir", default=".",
        help="Root directory (default: current directory)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without deleting"
    )
    p.add_argument(
        "--dirs", nargs="+", default=None,
        help="Override default dirs to delete"
    )
    p.add_argument(
        "--files", nargs="+", default=None,
        help="Override default files to delete"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    clean(
        base_dir=args.base_dir,
        delete_dirs=args.dirs,
        delete_files=args.files,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
