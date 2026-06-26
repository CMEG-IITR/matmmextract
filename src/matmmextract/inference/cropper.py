"""
matmmextract.inference.cropper
==================================
Crop detected panels from images using the JSON files produced by
:func:`~matmmextract.inference.detector.run`.

Output naming convention
------------------------
``<stem>_<label>.jpg``   — highest-confidence detection for that label.
When multiple detections share a label, only the one with the highest
score is saved.  These names are matched by dataset_builder's IMG_RE pattern.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
)


@dataclass
class CropResult:
    n_crops: int = 0
    n_skipped: int = 0
    n_no_detections: int = 0
    output_dir: str = ""
    failed: list[str] = field(default_factory=list)


def crop(
    image_dir: str | Path,
    json_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    verbose: bool = True,
) -> CropResult:
    """Crop all detected panels and save them as individual JPEG files.

    Parameters
    ----------
    image_dir:
        Directory containing the original flat images.
    json_dir:
        Directory containing per-image ``.json`` files from
        :func:`~matmmextract.inference.detector.run`.
        ``_summary.json`` is automatically skipped.
    output_dir:
        Directory where cropped panel images are saved.
    limit:
        Process at most this many JSON files (useful for testing).
    verbose:
        Print progress.
    """
    image_dir = Path(image_dir)
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(
        f for f in os.listdir(json_dir)
        if f.endswith(".json") and f != "_summary.json"
    )
    if not json_files:
        raise RuntimeError(f"No JSON files found in: {json_dir}")
    if limit is not None:
        json_files = json_files[:limit]

    if verbose:
        print(f"[cropper] {len(json_files)} JSON files  →  {output_dir}")

    result = CropResult(output_dir=str(output_dir))

    for jname in tqdm(json_files, desc="Cropping", disable=not verbose):
        jpath = json_dir / jname
        with open(jpath) as fh:
            data = json.load(fh)

        stem = Path(data["file"]).stem

        # Find the matching source image
        img_path: Path | None = None
        for ext in IMAGE_EXTS:
            candidate = image_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            result.n_skipped += 1
            if verbose:
                print(f"[cropper] skip — image not found for {jname}")
            continue

        if not data["detections"]:
            result.n_no_detections += 1
            continue

        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as exc:
            result.failed.append(str(img_path))
            if verbose:
                print(f"[cropper] ERROR opening {img_path}: {exc}")
            continue

        # Keep only the highest-confidence detection per label
        best: dict[str, dict] = {}
        for det in data["detections"]:
            lbl = det["label_name"]
            if lbl not in best or det["score"] > best[lbl]["score"]:
                best[lbl] = det

        for det in best.values():
            label = det["label_name"]
            out_name = f"{stem}_{label}.jpg"
            if label == "single":
                pil.save(os.path.join(args.output_dir, out_name))
            else:
                x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
                crop = pil.crop((x1, y1, x2, y2))
                crop.save(output_dir / out_name)

            result.n_crops += 1

    if verbose:
        print(
            f"[cropper] done — crops={result.n_crops}  "
            f"skipped={result.n_skipped}  no_detections={result.n_no_detections}"
        )

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Crop detected panels from images")
    p.add_argument("--image-dir",  required=True)
    p.add_argument("--json-dir",   required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    crop(
        image_dir=args.image_dir,
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
