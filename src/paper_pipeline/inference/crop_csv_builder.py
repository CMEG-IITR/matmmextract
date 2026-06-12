"""
paper_pipeline.inference.crop_csv_builder
==========================================
Bridge between the cropper and the captioner.

The figures CSV has xml_file + figure_id but no downloaded_image_name.
The downloaded images are named img1.jpg, img2.jpg, ... sequentially.

This module joins them by ROW ORDER — the N-th row in the figures CSV
corresponds to imgN.jpg — which is how the Elsevier downloader assigned names.

If your downloader CSV DOES have a downloaded_image_name column, pass
that CSV as figures_csv and set use_row_order=False.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

IMG_RE = re.compile(
    r"^(img\d+)_(single|common|[A-Z])(?:_(\d+))?\.jpg$",
    re.IGNORECASE,
)

# Matches the numeric part from "img14" → 14
IMG_NUM_RE = re.compile(r"^img(\d+)$", re.IGNORECASE)


def build_crop_csv(
    crops_dir: str | Path,
    figures_csv: str | Path,
    output_csv: str | Path = "crops_for_captioning.csv",
    use_row_order: bool = True,
    image_name_col: str = "downloaded_image_name",
    caption_col: str = "caption",
    reference_col: str = "reference_sentences",
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a CSV mapping each crop to its figure caption and references.

    Parameters
    ----------
    crops_dir:
        Directory of cropped panel images (output of cropper.run).
        Files named like img1_A.jpg, img1_single.jpg, img2_B.jpg ...
    figures_csv:
        Figure-level CSV from elsevier/springer extractor.
        Must have ``caption`` and ``reference_sentences`` columns.
    output_csv:
        Where to write the resulting CSV.
    use_row_order:
        If True (default): join by row position — img1 = row 0,
        img2 = row 1, etc. Use this when the figures CSV has no
        downloaded_image_name column (standard extractor output).
        If False: join by the downloaded_image_name column value.
    image_name_col:
        Only used when use_row_order=False. Column holding image stems.
    caption_col / reference_col:
        Column names for caption and references in figures_csv.
    verbose:
        Print summary.

    Returns
    -------
    pd.DataFrame
        One row per cropped image with columns:
        downloaded_image_name, caption, reference_sentences
    """
    crops_dir = Path(crops_dir)
    figures_df = pd.read_csv(figures_csv)

    if verbose:
        print(f"[crop_csv_builder] figures CSV: {len(figures_df)} rows")
        print(f"[crop_csv_builder] columns: {list(figures_df.columns)}")

    # Collect all crop filenames and their parent img_id
    crop_files = sorted(
        f for f in os.listdir(crops_dir)
        if IMG_RE.match(f)
    )

    if use_row_order:
        # img1 → row index 0, img2 → row index 1, ...
        # Build lookup: img_number (1-based) → figure row
        lookup: dict[int, dict] = {}
        for idx, row in figures_df.iterrows():
            img_num = idx + 1  # 1-based
            lookup[img_num] = {
                caption_col:   row.get(caption_col, ""),
                reference_col: row.get(reference_col, ""),
            }
    else:
        # Join by downloaded_image_name column
        if image_name_col not in figures_df.columns:
            raise KeyError(
                f"Column '{image_name_col}' not found. "
                f"Available: {list(figures_df.columns)}. "
                f"Try use_row_order=True instead."
            )
        figures_df[image_name_col] = (
            figures_df[image_name_col].astype(str).str.strip()
            .apply(lambda x: Path(x).stem)
        )
        lookup_by_name: dict[str, dict] = (
            figures_df.set_index(image_name_col)[[caption_col, reference_col]]
            .to_dict("index")
        )

    rows = []
    unmatched = []

    for fname in crop_files:
        m = IMG_RE.match(fname)
        if not m:
            continue

        img_id = m.group(1)        # e.g. "img1"
        stem   = Path(fname).stem  # e.g. "img1_A"

        if use_row_order:
            num_match = IMG_NUM_RE.match(img_id)
            if not num_match:
                unmatched.append(fname)
                rows.append({"downloaded_image_name": stem,
                             "caption": "", "reference_sentences": ""})
                continue
            img_num = int(num_match.group(1))
            fig_data = lookup.get(img_num)
        else:
            fig_data = lookup_by_name.get(img_id)

        if fig_data is None:
            unmatched.append(fname)
            rows.append({"downloaded_image_name": stem,
                         "caption": "", "reference_sentences": ""})
        else:
            rows.append({
                "downloaded_image_name": stem,
                "caption":               fig_data.get(caption_col, ""),
                "reference_sentences":   fig_data.get(reference_col, ""),
            })

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"[crop_csv_builder] {len(df)} crops → {output_csv}")
        print(f"[crop_csv_builder] matched={len(df) - len(unmatched)}  unmatched={len(unmatched)}")
        if unmatched:
            print(f"[crop_csv_builder] unmatched samples: {unmatched[:5]}")

    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build captioning CSV from crops + figures CSV")
    p.add_argument("--crops-dir",    required=True)
    p.add_argument("--figures-csv",  required=True)
    p.add_argument("--output-csv",   default="crops_for_captioning.csv")
    p.add_argument("--no-row-order", action="store_true",
                   help="Join by downloaded_image_name column instead of row order")
    p.add_argument("--image-name-col", default="downloaded_image_name")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_crop_csv(
        crops_dir=args.crops_dir,
        figures_csv=args.figures_csv,
        output_csv=args.output_csv,
        use_row_order=not args.no_row_order,
        image_name_col=args.image_name_col,
    )


if __name__ == "__main__":
    main()
