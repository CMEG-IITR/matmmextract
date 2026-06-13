"""
multimat.inference.captioner_gemini
====================================
Generate per-panel sub-captions using Gemini, from figure captions and
reference sentences extracted during the XML extraction step.


Input
-----
A CSV with columns: ``downloaded_image_name``, ``caption``,
``reference_sentences``, ``download_status``.

Output
------
One JSON file per row in *output_dir*, named
``<downloaded_image_name>.json``, with schema::

    {
        "panels": [
            {
                "panel":                    "a",
                "visualization_category":  "Microscopy",
                "visualization_subtype":   "SEM",
                "subcaption":              "...",
                "summary":                 "..."
            },
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

VISUALIZATION_CATEGORIES: list[str] = [
    "Microscopy", "Diffraction", "Spectroscopy", "Thermal Analysis",
    "Phase/Equilibrium Diagram", "Mechanical Test", "Electrochemistry",
    "Magnetic/Electronic", "Optical/Photonic", "Tomography/3D",
    "Compositional Map", "Simulation", "Machine Learning",
    "Crystal Structure", "Generic Plot", "Schematic/Diagram",
    "Photograph", "Table", "other",
]

SUBTYPES_BY_CATEGORY: dict[str, list[str]] = {
    "Microscopy": [
        "SEM", "TEM", "STEM", "HAADF-STEM", "BF-TEM", "DF-TEM",
        "Optical Micrograph", "Confocal Microscopy", "AFM",
        "Fluorescence Microscopy", "Live/Dead Staining",
    ],
    "Diffraction": [
        "XRD Pattern", "SAED", "EBSD Map", "Pole Figure",
        "Inverse Pole Figure", "Neutron Diffraction", "Synchrotron Diffraction",
    ],
    "Spectroscopy": [
        "XPS Spectrum", "Raman Spectrum", "FTIR Spectrum", "EDX Spectrum",
        "EELS Spectrum", "NMR Spectrum", "Mass Spectrum", "UV-Vis Spectrum",
        "Photoluminescence Spectrum", "XANES Spectrum", "EXAFS Spectrum",
        "Mössbauer Spectrum",
    ],
    "Thermal Analysis": ["DSC Curve", "TGA Curve", "DMA Curve", "TMA Curve"],
    "Phase/Equilibrium Diagram": [
        "Binary Phase Diagram", "Ternary Phase Diagram", "TTT Diagram",
        "CCT Diagram", "CALPHAD Diagram", "Pourbaix Diagram",
    ],
    "Mechanical Test": [
        "Stress-Strain Curve", "Load-Displacement Curve", "Nanoindentation Curve",
        "Hardness Map", "Fatigue/S-N Curve", "Creep Curve",
        "Fracture Toughness Plot", "DIC Strain Map", "Wear/Tribology Plot",
    ],
    "Electrochemistry": [
        "Cyclic Voltammogram", "Charge-Discharge Curve", "Capacity Retention Plot",
        "Coulombic Efficiency Plot", "Nyquist Plot", "Bode Plot", "Tafel Plot",
        "Polarization Curve", "GITT/PITT Curve", "Rate Capability Plot",
    ],
    "Magnetic/Electronic": [
        "M-H Hysteresis Loop", "M-T Curve", "ZFC/FC Curve", "I-V Curve",
        "C-V Curve", "Band Structure", "Density of States", "Hall Effect Plot",
    ],
    "Optical/Photonic": [
        "Absorbance Spectrum", "Transmittance Spectrum", "Reflectance Spectrum",
        "EQE/IQE Plot", "J-V Curve", "Ellipsometry Plot", "Refractive Index Plot",
    ],
    "Tomography/3D": [
        "APT Reconstruction", "Micro-CT", "FIB-SEM Tomography", "3D Reconstruction",
    ],
    "Compositional Map": [
        "EDS Map", "WDS Map", "EBSD IPF Map", "Elemental Distribution Map",
    ],
    "Simulation": [
        "DFT Result", "MD Snapshot", "MD Trajectory", "Phase-Field Simulation",
        "FEA/FEM Result", "Monte Carlo Result",
    ],
    "Machine Learning": [
        "Parity Plot", "Confusion Matrix", "ROC Curve", "Learning Curve",
        "Feature Importance Plot", "SHAP Plot", "t-SNE/UMAP/PCA Plot",
    ],
    "Crystal Structure": ["Unit Cell", "Atomic Model", "Supercell"],
    "Generic Plot": [
        "Bar Chart", "Scatter Plot", "Line Graph", "Box Plot", "Contour Plot",
        "Heatmap", "Radar Chart", "Ashby Plot", "Arrhenius Plot", "Histogram",
    ],
    "Schematic/Diagram": [
        "Process Schematic", "Flowchart", "Mechanism Diagram", "Experimental Setup",
    ],
    "Photograph": ["Sample Photo", "Equipment Photo", "In-situ Photo"],
    "Table": ["Data Table"],
    "other": ["other"],
}

VISUALIZATION_SUBTYPES: list[str] = list(dict.fromkeys(
    sub for subs in SUBTYPES_BY_CATEGORY.values() for sub in subs
))

_TAXONOMY_BLOCK: str = "\n".join(
    f"- {cat}: {', '.join(subs)}"
    for cat, subs in SUBTYPES_BY_CATEGORY.items()
)

_PROMPT_TEMPLATE = """\
You are a scientific figure analysis expert covering all of materials science \
(metals/alloys, ceramics, polymers, composites, semiconductors, batteries, \
catalysts, biomaterials, 2D materials, etc.).

Using only the figure caption and the reference sentences from the paper, \
generate detailed sub-captions for each panel of the figure.

Figure caption:
{caption}

Reference sentences from the paper:
{reference_sentences}

INSTRUCTIONS:
- Generate a sub-caption for each panel (a, b, c, …); use panel id "main" for single-panel figures.
- Classify each panel with TWO fields, chosen ONLY from the allowed taxonomy below:
    * visualization_category: the broad category.
    * visualization_subtype: the specific technique/plot type, which MUST belong to the chosen category.
- Both fields are strict enums. Use "other" only if absolutely nothing matches.
- For plots/graphs: state axes, specimen/condition, test method, and any key trend or value from the caption or reference.
- For images/maps: state material, technique, orientation/view, processing condition, and key observation or structural feature.
- The caption is the ground truth — never contradict it.
- Use a reference sentence only if it clearly supports the caption content.
- Group-level references apply to all panels; references naming a specific panel letter apply only to that panel.
- Do not start every sub-caption identically — vary sentence openings.
- Expand the caption into a precise scientific description — do not copy it verbatim.
- For each panel, write a summary of 30-50 words using only that panel's subcaption and directly related references.

ALLOWED TAXONOMY (category → valid subtypes):
{taxonomy}
"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CaptionResult:
    n_total: int = 0
    n_success: int = 0
    n_error: int = 0
    n_skipped: int = 0
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def captioner(
    csv_path: str | Path,
    output_dir: str | Path,
    api_key: str | None = None,
    model_name: str = "gemini-3.1-flash-lite",
    max_tokens: int = 4096,
    max_retries: int = 4,
    overwrite: bool = False,
    verbose: bool = True,
) -> CaptionResult:
    """Generate sub-captions for every successfully downloaded figure.

    Parameters
    ----------
    csv_path:
        Figure CSV with columns ``downloaded_image_name``, ``caption``,
        ``reference_sentences``, ``download_status``.
    output_dir:
        Directory where one JSON per figure is written.
    api_key:
        Google Gemini API key. Falls back to ``GOOGLE_API_KEY`` env var.
    model_name:
        Gemini model string.
    max_tokens:
        Max output tokens per request.
    max_retries:
        Retry attempts on API error.
    overwrite:
        Re-generate even if the output JSON already exists.
    verbose:
        Print progress.

    Returns
    -------
    CaptionResult
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai is required: pip install google-genai"
        )

    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Gemini API key required. Set GOOGLE_API_KEY or pass api_key=."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)

    response_schema = types.Schema(
        type=types.Type.OBJECT,
        required=["panels"],
        properties={
            "panels": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    required=["panel", "visualization_category",
                               "visualization_subtype", "subcaption", "summary"],
                    properties={
                        "panel": types.Schema(type=types.Type.STRING),
                        "visualization_category": types.Schema(
                            type=types.Type.STRING,
                            enum=VISUALIZATION_CATEGORIES,
                        ),
                        "visualization_subtype": types.Schema(
                            type=types.Type.STRING,
                        ),
                        "subcaption": types.Schema(type=types.Type.STRING),
                        "summary": types.Schema(type=types.Type.STRING),
                    },
                ),
            ),
        },
    )

    generate_config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=response_schema,
    )

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    pending = [
        r for r in rows
        if (
            # Accept rows that explicitly succeeded, OR rows with no
            # download_status column at all (e.g. crops_for_captioning.csv)
            r.get("download_status", "success") in ("success", "")
            and str(r.get("downloaded_image_name", "")).strip()
        )
        and (overwrite or not (output_dir / f"{r['downloaded_image_name']}.json").exists())
    ]

    result = CaptionResult(
        n_total=len(pending),
        output_dir=str(output_dir),
    )

    if verbose:
        print(f"[captioner] {len(rows)} total rows | {len(pending)} pending | model={model_name}")

    if not pending:
        if verbose:
            print("[captioner] Nothing to do.")
        result.n_skipped = len(rows)
        return result

    for i, row in enumerate(pending, start=1):
        image_name = row["downloaded_image_name"]
        prompt = _PROMPT_TEMPLATE.format(
            caption=row.get("caption", ""),
            reference_sentences=row.get("reference_sentences", ""),
            taxonomy=_TAXONOMY_BLOCK,
        )

        last_error = None
        parsed = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=generate_config,
                )
                parsed = json.loads(response.text)
                break
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        if parsed is None:
            parsed = {"error": str(last_error)}
            result.n_error += 1
            if verbose:
                print(f"  [{i}/{len(pending)}] ERROR {image_name}: {last_error}")
        else:
            result.n_success += 1
            if verbose and (i % 10 == 0 or i == len(pending)):
                pct = i / len(pending) * 100
                print(
                    f"  [{pct:5.1f}%] {i}/{len(pending)} "
                    f"ok={result.n_success} err={result.n_error}"
                )

        out_path = output_dir / f"{image_name}.json"
        out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))

    if verbose:
        print(f"\n[captioner] done — success={result.n_success}  errors={result.n_error}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate sub-captions with Gemini")
    p.add_argument("--csv",        required=True, help="Figure CSV path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--api-key",    default=os.environ.get("GOOGLE_API_KEY"))
    p.add_argument("--model",      default="gemini-3.1-flash-lite")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--overwrite",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    captioner(
        csv_path=args.csv,
        output_dir=args.output_dir,
        api_key=args.api_key,
        model_name=args.model,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
