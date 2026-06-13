"""
multimat.inference.detector
===================================
Batch panel detection using a YOLO fine-tuned checkpoint (Ultralytics).

Ported from ``infer.py`` (YOLO version) — globals → parameters, callable as API.

Flow
----
images → ``run()`` → one JSON file per image + ``_summary.json`` in *output_dir*

Each per-image JSON::

    {
        "meta":         { weights, conf, iou, imgsz, timestamp, id2label },
        "file":         "img123.jpg",
        "path":         "/abs/path/img123.jpg",
        "width":        W,
        "height":       H,
        "n_detections": N,
        "detections":   [
            { "bbox": [x1,y1,x2,y2], "score": 0.91,
              "label_id": 0, "label_name": "A" },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

ID2LABEL: dict[int, str] = {i: chr(ord("A") + i) for i in range(20)}
ID2LABEL[20] = "single"
ID2LABEL[21] = "common"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    n_images: int = 0
    n_saved: int = 0
    output_dir: str = ""
    failed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_images(folder: str | Path) -> list[str]:
    folder = Path(folder)
    return [
        str(folder / f)
        for f in sorted(os.listdir(folder))
        if Path(f).suffix.lower() in IMAGE_EXTS
    ]


def _is_gdrive_url(s: str) -> bool:
    return "drive.google.com" in s or "docs.google.com" in s


def _is_url(s: str) -> bool:
    return str(s).startswith(("http://", "https://"))


def _download_gdrive(url: str, cache_dir: str | Path = ".weights_cache") -> str:
    """Download a Google Drive file and return the local path.

    Accepts any of:
    - https://drive.google.com/file/d/<ID>/view?usp=sharing
    - https://drive.google.com/open?id=<ID>
    - https://drive.google.com/uc?id=<ID>
    """
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required for Google Drive downloads: pip install gdown")

    import re
    # Extract file ID from any Drive URL format
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"open\?id=([a-zA-Z0-9_-]+)",
    ]
    file_id = None
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            file_id = m.group(1)
            break

    if not file_id:
        raise ValueError(f"Could not extract Google Drive file ID from URL: {url}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(cache_dir.glob(f"{file_id}*.pt"))
    if existing:
        print(f"[detector] using cached weights: {existing[0]}")
        return str(existing[0])

    dest = cache_dir / f"{file_id}.pt"
    print(f"[detector] downloading from Google Drive (id={file_id}) → {dest}")
    gdown.download(id=file_id, output=str(dest), quiet=False)

    if not dest.exists():
        raise RuntimeError(f"Download failed — file not found at {dest}")

    return str(dest)


def resolve_weights(checkpoint: str | Path, cache_dir: str | Path = ".weights_cache") -> str:
    """Resolve weights to a local .pt path.

    Accepts:
    - A direct local .pt file path
    - A local directory containing best.pt / last.pt
    - A Google Drive share URL (downloaded and cached in *cache_dir*)
    """
    s = str(checkpoint)

    # Google Drive URL
    if _is_gdrive_url(s):
        return _download_gdrive(s, cache_dir=cache_dir)

    # Other URL — not supported yet
    if _is_url(s):
        raise ValueError(
            f"URL '{s}' is not a Google Drive link. "
            f"Only Google Drive URLs are currently supported."
        )

    # Local path
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        return str(p)
    for name in ("best.pt", "last.pt"):
        candidate = p / name
        if candidate.exists():
            return str(candidate)
    pts = list(p.glob("*.pt"))
    if pts:
        return str(sorted(pts)[0])
    raise FileNotFoundError(
        f"No .pt weights found in '{checkpoint}'. "
        f"Expected best.pt / last.pt inside the folder, or a direct .pt path."
    )


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def detect(
    image_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path = "best.pt",
    conf: float = 0.6,
    iou: float = 0.4,
    imgsz: int = 1024,
    device: str = "",
    id2label: dict[int, str] | None = None,
    write_summary: bool = True,
    weights_cache_dir: str | Path = ".weights_cache",
    verbose: bool = True,
) -> DetectionResult:
    """Run YOLO panel detection on every image in *image_dir*.

    Parameters
    ----------
    image_dir:
        Directory of input images.
    output_dir:
        Directory where one ``.json`` per image is written, plus
        ``_summary.json`` containing all records combined.
    checkpoint:
        Path to a YOLO ``.pt`` file, a directory containing
        ``best.pt`` / ``last.pt``, or a Google Drive share URL
        (e.g. ``https://drive.google.com/file/d/<ID>/view``).
    weights_cache_dir:
        Local directory to cache downloaded weights (default ``.weights_cache``).
    conf:
        Confidence threshold (default 0.6).
    iou:
        NMS IoU threshold (default 0.4).
    imgsz:
        Inference image size (default 1024).
    device:
        Device string: ``""`` for auto, ``"cpu"``, ``"0"``, ``"0,1"``, etc.
    id2label:
        Override the default label map. If ``None``, uses the built-in map.
    write_summary:
        Write ``_summary.json`` combining all records (default True).
    verbose:
        Print progress.

    Returns
    -------
    DetectionResult
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics is required: pip install ultralytics")

    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = resolve_weights(checkpoint, cache_dir=weights_cache_dir)
    label_map = id2label if id2label is not None else ID2LABEL

    if verbose:
        print(f"[detector] weights={weights}")
        print(f"[detector] conf={conf}  iou={iou}  imgsz={imgsz}  device={device!r}")

    model = YOLO(weights)

    image_paths = find_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {image_dir}")

    if verbose:
        print(f"[detector] {len(image_paths)} images\n")

    result = DetectionResult(n_images=len(image_paths), output_dir=str(output_dir))
    all_records: list[dict] = []

    for img_path in tqdm(image_paths, desc="Detecting", disable=not verbose):
        try:
            from PIL import Image
            pil = Image.open(img_path).convert("RGB")
            W, H = pil.size

            results = model.predict(
                source=img_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            r = results[0]

            detections = []
            if r.boxes is not None and len(r.boxes):
                for box, score, cls in zip(
                    r.boxes.xyxy.tolist(),
                    r.boxes.conf.tolist(),
                    r.boxes.cls.int().tolist(),
                ):
                    detections.append({
                        "bbox":       [round(v, 2) for v in box],
                        "score":      round(score, 6),
                        "label_id":   cls,
                        "label_name": label_map.get(cls, str(cls)),
                    })

            stem = Path(img_path).stem
            record = {
                "meta": {
                    "weights":   weights,
                    "conf":      conf,
                    "iou":       iou,
                    "imgsz":     imgsz,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "id2label":  {str(k): v for k, v in label_map.items()},
                },
                "file":         os.path.basename(img_path),
                "path":         img_path,
                "width":        W,
                "height":       H,
                "n_detections": len(detections),
                "detections":   detections,
            }

            out_path = output_dir / f"{stem}.json"
            out_path.write_text(json.dumps(record, indent=2))
            all_records.append(record)
            result.n_saved += 1

        except Exception as exc:
            result.failed.append(img_path)
            if verbose:
                print(f"\n[detector] ERROR {img_path}: {exc}")

    if write_summary:
        summary_path = output_dir / "_summary.json"
        summary_path.write_text(json.dumps(all_records, indent=2))
        if verbose:
            print(f"[detector] summary → {summary_path}")

    if verbose:
        print(f"[detector] saved {result.n_saved} JSON files → {output_dir}")
        if result.failed:
            print(f"[detector] {len(result.failed)} failed: {result.failed}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MatDetect — YOLO inference → JSON")
    p.add_argument("--image-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--weights",    default="best.pt",
                   help="Path to YOLO .pt weights or folder containing best.pt")
    p.add_argument("--conf",       type=float, default=0.6)
    p.add_argument("--iou",        type=float, default=0.4)
    p.add_argument("--imgsz",      type=int,   default=1024)
    p.add_argument("--device",     default="",
                   help="Device: '' auto, 'cpu', '0', '0,1', etc.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    detect(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        checkpoint=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
