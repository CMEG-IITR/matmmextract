"""
matmmextract.inference.captioner_azure
==========================================
Generate per-panel sub-captions using Azure-hosted models via the
OpenAI-compatible API (Mistral, Llama, GPT, etc.).


Differences from :mod:`~matmmextract.inference.captioner_gemini` (Gemini):
- Uses ``openai.OpenAI`` with a custom ``base_url`` pointing at Azure
- Response format is a JSON schema dict (OpenAI structured outputs)
- Column names differ: ``image_name`` and ``reference`` instead of
``downloaded_image_name`` and ``reference_sentences``
- Retries on ``"error"`` keys in already-written JSON files


Input CSV columns required
--------------------------
``image_name``, ``caption``, ``reference``

Output
------
One JSON file per row in *output_dir*, named ``<image_name>.json``::

    {
        "panels": [
            {
                "panel":                   "a",
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
# Taxonomy  (shared with captioner_gemini.py — kept here to avoid circular imports)
# ---------------------------------------------------------------------------

from matmmextract.inference.captioner_gemini import (
    VISUALIZATION_CATEGORIES,
    SUBTYPES_BY_CATEGORY,
    _TAXONOMY_BLOCK,
)

# ---------------------------------------------------------------------------
# Azure defaults
# ---------------------------------------------------------------------------

DEFAULT_AZURE_ENDPOINT = ""
DEFAULT_MODEL = "Mistral-Large-3"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_RETRIES = 4

# ---------------------------------------------------------------------------
# Prompt  (slightly different wording from the Gemini version)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a scientific figure analysis expert specializing in materials science.

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
- For each panel, write a summary of 40-60 words using only that panel's subcaption and directly related references. Do not include "panel a" or "the figure" in the summary.

ALLOWED TAXONOMY (category -> valid subtypes):
{taxonomy}
"""

# ---------------------------------------------------------------------------
# JSON schema for structured output (OpenAI / Azure format)
# ---------------------------------------------------------------------------

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "figure_panels",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["panels"],
            "additionalProperties": False,
            "properties": {
                "panels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "panel", "visualization_category",
                            "visualization_subtype", "subcaption", "summary",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "panel":                  {"type": "string"},
                            "visualization_category": {
                                "type": "string",
                                "enum": VISUALIZATION_CATEGORIES,
                            },
                            "visualization_subtype":  {"type": "string"},
                            "subcaption":             {"type": "string"},
                            "summary":                {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


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
    azure_endpoint: str = DEFAULT_AZURE_ENDPOINT,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    overwrite: bool = False,
    image_name_col: str = "downloaded_image_name",
    caption_col: str = "caption",
    reference_col: str = "reference_sentences",
    requests_per_minute: int | None = None,
    verbose: bool = True,
) -> CaptionResult:
    """Generate sub-captions for every row in *csv_path* using an Azure model.

    Parameters
    ----------
    csv_path:
        CSV with columns ``image_name``, ``caption``, ``reference``
        (column names overridable via the ``*_col`` parameters).
    output_dir:
        Directory where one JSON per figure is written.
    api_key:
        Azure API key. Falls back to ``AZURE_API_KEY`` env var.
    azure_endpoint:
        Azure OpenAI-compatible endpoint URL.
    model_name:
        Model deployment name (e.g. ``"Mistral-Large-3"``, ``"gpt-4o"``).
    max_tokens:
        Max output tokens.  Uses ``max_completion_tokens`` for OpenAI models,
        ``max_tokens`` for Mistral/Llama.
    max_retries:
        Retry attempts on API error or existing ``"error"`` JSON.
    overwrite:
        Re-generate even if output JSON already exists and has no error.
    image_name_col / caption_col / reference_col:
        Column name overrides for non-standard CSVs.
    requests_per_minute:
        If set, throttle API calls to this rate by sleeping between requests.
        If ``None`` (default), no extra throttling beyond retry back-off.
    verbose:
        Print progress.

    Returns
    -------
    CaptionResult
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    api_key = api_key or os.environ.get("AZURE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Azure API key required. Set AZURE_API_KEY or pass api_key=."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _min_interval = 60.0 / requests_per_minute if requests_per_minute else None
    _last_request_time = [0.0]

    client = OpenAI(base_url=azure_endpoint, api_key=api_key)

    # Mistral/Llama use max_tokens; OpenAI models use max_completion_tokens
    _is_openai_model = model_name.lower().startswith(("gpt", "o1", "o3"))
    token_kwarg = (
        {"max_completion_tokens": max_tokens}
        if _is_openai_model
        else {"max_tokens": max_tokens}
    )

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # NEW: Deduplicate by figure base name (extract before underscore)
    def get_figure_base(image_name: str) -> str:
        """Extract figure ID: 'img8_A' -> 'img8', 'img10_single' -> 'img10'"""
        if "_" in image_name:
            return image_name.split("_")[0]
        return image_name

    # Group by figure base name + caption + reference (defensive dedup)
    figure_groups: dict[tuple, dict] = {}
    for row in rows:
        image_name = row.get(image_name_col, "").strip()
        if not image_name:
            continue

        fig_base = get_figure_base(image_name)
        caption = row.get(caption_col, "")
        ref = row.get(reference_col, "")

        # Create a composite key (figure base + caption + reference)
        # This ensures we don't accidentally merge figures with different content
        key = (fig_base, caption, ref)

        if key not in figure_groups:
            figure_groups[key] = {
                "row": row,
                "figure_id": fig_base,
                "crop_names": []
            }

        # Store all crop names that belong to this figure
        figure_groups[key]["crop_names"].append(image_name)

    # Use deduplicated rows for generation
    unique_rows = [g["row"] for g in figure_groups.values()]

    if verbose:
        print(f"[captioner_azure] {len(rows)} crops → {len(unique_rows)} unique figures")
        # Print which figures were deduplicated
        for g in figure_groups.values():
            if len(g["crop_names"]) > 1:
                print(f"  → {g['figure_id']}: {len(g['crop_names'])} crops ({', '.join(g['crop_names'])})")

    def _needs_run(row: dict) -> bool:
        image_name = row.get(image_name_col, "").strip()
        fig_base = get_figure_base(image_name)
        if not fig_base:
            return False
        status = row.get("download_status", "success")
        if status not in ("success", ""):
            return False
        if overwrite:
            return True
        out = output_dir / f"{fig_base}.json"
        if not out.exists():
            return True
        try:
            return "error" in json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            return True

    pending = [r for r in unique_rows if _needs_run(r)]
    result = CaptionResult(n_total=len(pending), output_dir=str(output_dir))

    if not pending:
        if verbose:
            print("[captioner_azure] Nothing to do.")
        result.n_skipped = len(rows)
        return result

    for i, row in enumerate(pending, start=1):
        image_name = row[image_name_col]
        fig_base = get_figure_base(image_name)

        if verbose:
            print(f"  [{i}/{len(pending)}] {fig_base} ...", end=" ", flush=True)

        prompt = _PROMPT_TEMPLATE.format(
            caption=row.get(caption_col, ""),
            reference_sentences=row.get(reference_col, ""),
            taxonomy=_TAXONOMY_BLOCK,
        )

        last_error = None
        parsed = None
        for attempt in range(max_retries):
            try:
                if _min_interval is not None:
                    import time as _time
                    elapsed = _time.monotonic() - _last_request_time[0]
                    if elapsed < _min_interval:
                        _time.sleep(_min_interval - elapsed)
                _last_request_time[0] = _time.monotonic() if _min_interval else 0.0
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format=_RESPONSE_FORMAT,
                    **token_kwarg,
                )
                parsed = json.loads(response.choices[0].message.content)
                break
            except Exception as exc:
                last_error = exc
                if verbose:
                    print(f"\n    attempt {attempt + 1}/{max_retries} failed: {exc}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        if parsed is None:
            parsed = {"error": str(last_error)}
            result.n_error += 1
            if verbose:
                print(f"ERROR: {last_error}", flush=True)
        else:
            result.n_success += 1
            if verbose:
                print(f"ok  (panels={len(parsed.get('panels', []))})", flush=True)

        out_path = output_dir / f"{fig_base}.json"
        out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))

    if verbose:
        print(
            f"\n[captioner_azure] done — "
            f"success={result.n_success}  errors={result.n_error}"
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate sub-captions via Azure OpenAI")
    p.add_argument("--csv",            required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--api-key",        default=os.environ.get("AZURE_API_KEY"))
    p.add_argument("--endpoint",       default=DEFAULT_AZURE_ENDPOINT)
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--max-tokens",     type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--overwrite",      action="store_true")
    p.add_argument("--image-name-col", default="image_name")
    p.add_argument("--caption-col",    default="caption")
    p.add_argument("--reference-col",  default="reference")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    captioner(
        csv_path=args.csv,
        output_dir=args.output_dir,
        api_key=args.api_key,
        azure_endpoint=args.endpoint,
        model_name=args.model,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
        image_name_col=args.image_name_col,
        caption_col=args.caption_col,
        reference_col=args.reference_col,
    )


if __name__ == "__main__":
    main()
