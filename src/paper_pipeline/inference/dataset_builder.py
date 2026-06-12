"""
paper_pipeline.inference.dataset_builder
==========================================
Link cropped panel images with their JSON sub-captions to produce
the final ``linked_dataset.csv``.

JSON lookup key
---------------
Each JSON file is named after the full crop stem:
    img11_A.json      ← written by captioner for crop img11_A.jpg
    img10_single.json ← written by captioner for crop img10_single.jpg

The linker looks up the JSON by full crop stem, then finds the matching
panel inside the JSON by the panel letter/key.

Filename pattern matched
------------------------
``imgXXXX_single.jpg``      →  looks for img XXXX_single.json, panel "main"
``imgXXXX_single_2.jpg``    →  looks for imgXXXX_single_2.json, panel "main"
``imgXXXX_A.jpg``           →  looks for imgXXXX_A.json, panel "a"
``imgXXXX_A_2.jpg``         →  looks for imgXXXX_A_2.json, panel "a"

Output columns
--------------
image_filename, visualization_category, visualization_subtype,
subcaption, summary
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

IMG_RE = re.compile(
    r"^(.+)_(single|[A-Z])(?:_(\d+))?\.jpg$",
    re.IGNORECASE,
)

CSV_FIELDS = [
    "image_filename",
    "image_id",
    "panel_suffix",
    "variant",
    "json_panel",
    "visualization_category",
    "visualization_subtype",
    "subcaption",
    "summary",
    "matched",
]


@dataclass
class BuildResult:
    n_images: int = 0
    n_matched: int = 0
    n_no_json: int = 0
    n_no_panel: int = 0
    n_skipped_pattern: int = 0
    n_has_caption: int = 0
    output_csv: str = ""
    log_path: str = ""


def _load_json_index(json_dir: Path) -> dict[str, dict]:
    """Load all JSON files into {stem: parsed_json} dict."""
    index: dict[str, dict] = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]  # e.g. "img11_A"
        try:
            with open(json_dir / fname, encoding="utf-8") as fh:
                index[stem] = json.load(fh)
        except json.JSONDecodeError:
            pass
    return index


def _build_panel_lookup(json_obj: dict) -> dict[str, dict]:
    """
    Return {panel_key_lower: panel_data} for simple keys only.
    Accepts: "main", single letters (a-z).
    Skips complex keys like "a1", "a1-a6", "d-e".
    """
    lookup: dict[str, dict] = {}
    for panel in json_obj.get("panels", []):
        key = panel.get("panel", "").strip()
        if key == "main" or (len(key) == 1 and key.isalpha()):
            lookup[key.lower()] = panel
    return lookup


def build(
    images_dir: str | Path,
    json_dir: str | Path,
    output_csv: str | Path = "linked_dataset.csv",
    log_path: str | Path = "build_dataset.log",
    verbose: bool = True,
) -> BuildResult:
    """Link cropped images with sub-caption JSON files.

    Parameters
    ----------
    images_dir:
        Directory of cropped panel images (output of cropper.run).
    json_dir:
        Directory of per-crop sub-caption JSON files (output of
        captioner.run or captioner_azure.run).
        Each JSON is named after the crop stem: img11_A.json.
    output_csv:
        Path for the final linked CSV.
    log_path:
        Path for the human-readable build log.
    verbose:
        Print stats to stdout.
    """
    images_dir = Path(images_dir)
    json_dir = Path(json_dir)
    output_csv = Path(output_csv)
    log_path = Path(log_path)

    # index: "img11_A" → parsed JSON
    json_index = _load_json_index(json_dir)
    image_files = sorted(os.listdir(images_dir))

    rows: list[dict] = []
    skipped_pattern: list[str] = []
    no_json: list[str] = []
    panel_not_found: list[str] = []
    matched: list[str] = []

    for fname in image_files:
        m = IMG_RE.match(fname)
        if not m:
            skipped_pattern.append(fname)
            continue

        img_id       = m.group(1)
        panel_letter = m.group(2)
        variant      = m.group(3) or ""

        if panel_letter.lower() == "single":
            panel_suffix = "single"
            lookup_key   = "main"
        else:
            panel_suffix = f"{panel_letter.upper()}_{variant}" if variant else panel_letter.upper()
            lookup_key   = panel_letter.lower()

        row: dict = {
            "image_filename":          fname,
            "image_id":                img_id,
            "panel_suffix":            panel_suffix,
            "variant":                 variant,
            "json_panel":              "",
            "visualization_category":  "",
            "visualization_subtype":   "",
            "subcaption":              "",
            "summary":                 "",
            "matched":                 False,
        }

        json_obj = json_index.get(img_id)
        if json_obj is None:
            no_json.append(fname)
        else:
            panel_data = _build_panel_lookup(json_obj).get(lookup_key)
            if panel_data:
                row["json_panel"]             = panel_data.get("panel", "")
                row["visualization_category"] = panel_data.get("visualization_category", "")
                row["visualization_subtype"]  = panel_data.get("visualization_subtype", "")
                row["subcaption"]             = panel_data.get("subcaption", "")
                row["summary"]                = panel_data.get("summary", "")
                row["matched"]                = True
                matched.append(fname)
            else:
                panel_not_found.append(fname)

        rows.append(row)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    total_rows    = len(rows)
    n_matched     = len(matched)
    n_no_json     = len(no_json)
    n_no_panel    = len(panel_not_found)
    n_skipped     = len(skipped_pattern)
    n_has_caption = sum(
        1 for r in rows if r["matched"] and r["subcaption"] and r["summary"]
    )

    def pct(n): return n / total_rows * 100 if total_rows else 0

    log_lines = [
        f"build_dataset  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Total image files scanned  : {len(image_files)}",
        f"Skipped (bad filename)     : {n_skipped}",
        f"Processed rows             : {total_rows}",
        "",
        "── Image → JSON linkage ─────────────────────────────────",
        f"  Matched (image + panel)  : {n_matched:>6}  ({pct(n_matched):.1f}%)",
        f"  No JSON file             : {n_no_json:>6}  ({pct(n_no_json):.1f}%)",
        f"  JSON found, panel missing: {n_no_panel:>6}  ({pct(n_no_panel):.1f}%)",
        "",
        "── Caption linkage (of total processed) ─────────────────",
        f"  Has subcaption + summary : {n_has_caption:>6}  ({pct(n_has_caption):.1f}%)",
        "",
        "── Output ───────────────────────────────────────────────",
        f"  CSV : {output_csv}",
        f"  Log : {log_path}",
    ]
    if skipped_pattern:
        log_lines += ["", "── Skipped filenames (bad pattern) ──────────────────────"]
        log_lines += [f"  {f}" for f in skipped_pattern[:20]]
        if len(skipped_pattern) > 20:
            log_lines.append(f"  ... and {len(skipped_pattern) - 20} more")

    log_text = "\n".join(log_lines)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text + "\n", encoding="utf-8")

    if verbose:
        print(log_text)

    return BuildResult(
        n_images=len(image_files),
        n_matched=n_matched,
        n_no_json=n_no_json,
        n_no_panel=n_no_panel,
        n_skipped_pattern=n_skipped,
        n_has_caption=n_has_caption,
        output_csv=str(output_csv),
        log_path=str(log_path),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Link crops + sub-captions → dataset CSV")
    p.add_argument("--images-dir",  required=True)
    p.add_argument("--json-dir",    required=True)
    p.add_argument("--output-csv",  default="linked_dataset.csv")
    p.add_argument("--log",         default="build_dataset.log")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build(
        images_dir=args.images_dir,
        json_dir=args.json_dir,
        output_csv=args.output_csv,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
