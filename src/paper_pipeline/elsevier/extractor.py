from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ..shared.sentence_utils import split_sentences

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


FIGURE_REFID_PREFIXES: frozenset[str] = frozenset({"fig", "f", "appsec"})
PARA_TAGS: list[str] = ["ce:para", "ce:simple-para"]
XREF_TAGS: list[str] = ["ce:cross-ref", "ce:cross-refs"]


def build_image_map(soup: BeautifulSoup) -> dict[str, str]:
    """Build ``{ref_key: url}`` from the ``<objects>`` section.

    Prefers ``category='high'``; falls back to any other category.
    """
    high: dict[str, str] = {}
    fallback: dict[str, str] = {}
    for obj in soup.find_all("object"):
        ref = obj.get("ref", "").strip()
        if not ref:
            continue
        url = obj.text.strip()
        if obj.get("category") == "high":
            high[ref] = url
        elif ref not in fallback:
            fallback[ref] = url
    return high if high else fallback



def resolve_image_url(fig_tag, image_map: dict[str, str]) -> str | None:
    """Resolve the image URL for a ``<ce:figure>`` element (3 strategies)."""
    # S1: ce:graphic href (legacy)
    graphic = fig_tag.find(["ce:graphic", "graphic"])
    if graphic:
        for attr, val in graphic.attrs.items():
            if "href" in attr:
                url = image_map.get(val.strip())
                if url:
                    return url

    # S2: ce:link locator (primary path for all current XMLs)
    link = fig_tag.find("ce:link")
    if link:
        locator = link.get("locator", "").strip()
        if locator:
            url = image_map.get(locator)
            if url:
                return url

    # S3: numeric fallback
    fig_id = fig_tag.get("id", "")
    nums = re.findall(r"\d+", fig_id)
    if nums:
        url = image_map.get(f"gr{int(nums[0])}")
        if url:
            return url

    return None


def extract_figures(soup: BeautifulSoup, image_map: dict[str, str]) -> dict[str, dict]:
    """Return ``{fig_id: {caption, image_url, fig_num}}`` for all figures."""
    figures: dict[str, dict] = {}

    for fig in soup.find_all("ce:figure"):
        fig_id = fig.get("id", "").strip()
        if not fig_id:
            continue

        caption = ""
        cap = fig.find("ce:caption", recursive=False) or fig.find("ce:caption")
        if cap:
            caption = cap.get_text(" ", strip=True)

        img_url = resolve_image_url(fig, image_map)

        nums = re.findall(r"\d+", fig_id)
        fig_num = int(nums[0]) if nums else None

        figures[fig_id] = {
            "caption": caption,
            "image_url": img_url,
            "fig_num": fig_num,
        }

    return figures


def get_merged_paragraphs(soup: BeautifulSoup) -> list[tuple]:
    """Merge sibling paragraphs split by ``<ce:float-anchor>``."""
    body = soup.find(["ce:body", "body", "ce:sections"])
    if not body:
        return [(p, p.get_text(" ", strip=True)) for p in soup.find_all(PARA_TAGS)]

    all_paras = body.find_all(PARA_TAGS)
    merged: list[tuple] = []
    skip_next = False

    for i, para in enumerate(all_paras):
        if skip_next:
            skip_next = False
            continue

        text = para.get_text(" ", strip=True)
        if para.find("ce:float-anchor") and i + 1 < len(all_paras):
            text = text + " " + all_paras[i + 1].get_text(" ", strip=True)
            skip_next = True

        merged.append((para, text))

    return merged


def extract_reference_sentences(
    soup: BeautifulSoup,
    figures: dict[str, dict],
) -> dict[str, list[str]]:
    """Find body sentences that cite each figure (structured + regex modes)."""
    ref_map: dict[str, list[str]] = {fid: [] for fid in figures}

    num_to_fids: dict[int, list[str]] = {}
    for fid, data in figures.items():
        n = data["fig_num"]
        if n is not None:
            num_to_fids.setdefault(n, []).append(fid)

    for para_tag, text in get_merged_paragraphs(soup):
        if not text:
            continue

        sentences = split_sentences(text)

        # Mode A: structured xrefs
        fids_structured: set[str] = set()
        for xr in para_tag.find_all(XREF_TAGS):
            refid_val = xr.get("refid", "")
            for rid in refid_val.split():
                prefix = re.match(r"^([A-Za-z]+)", rid)
                if prefix and prefix.group(1).lower() in FIGURE_REFID_PREFIXES and rid in ref_map:
                    fids_structured.add(rid)

        if fids_structured:
            nums_needed = {
                figures[fid]["fig_num"]
                for fid in fids_structured
                if figures[fid]["fig_num"] is not None
            }
            for sentence in sentences:
                for n in nums_needed:
                    if re.search(rf"\b(Fig\.?\s*{n}|Figs\.?\s*{n}|Figure\s*{n})\b", sentence, re.IGNORECASE):
                        for fid in num_to_fids.get(n, []):
                            ref_map[fid].append(sentence)
            continue

        # Mode B: regex fallback
        for n, fids in num_to_fids.items():
            pattern = rf"\b(Fig\.?\s*{n}|Figs\.?\s*{n}|Figure\s*{n})\b"
            if re.search(pattern, text, re.IGNORECASE):
                for sentence in sentences:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        for fid in fids:
                            ref_map[fid].append(sentence)

    return ref_map


def process_file(xml_path: str | Path) -> list[dict]:
    """Extract all figure rows from a single Elsevier XML file."""
    with open(xml_path, "rb") as fh:
        soup = BeautifulSoup(fh, "lxml")

    image_map = build_image_map(soup)
    figures = extract_figures(soup, image_map)
    ref_map = extract_reference_sentences(soup, figures)

    rows: list[dict] = []
    fname = os.path.basename(xml_path)

    for fid, data in figures.items():
        seen: set[str] = set()
        unique: list[str] = []
        for s in ref_map.get(fid, []):
            if s not in seen:
                seen.add(s)
                unique.append(s)

        rows.append({
            "xml_file": fname,
            "figure_id": fid,
            "caption": data["caption"],
            "image_url": data["image_url"] or "",
            "num_references": len(unique),
            "reference_sentences": " || ".join(unique[:5]),
        })

    return rows


def extract_all(
    xml_dir: str | Path,
    output_csv: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Process every XML file in *xml_dir* and return a figures DataFrame.

    Parameters
    ----------
    xml_dir:
        Directory containing Elsevier ``.xml`` files.
    output_csv:
        If provided, write the DataFrame here.
    verbose:
        Print per-file progress.

    Returns
    -------
    df : pd.DataFrame
        One row per figure.
    errors : list of (filename, error_message)
    """
    xml_dir = Path(xml_dir)
    xml_files = sorted(f for f in xml_dir.iterdir() if f.suffix == ".xml")

    if verbose:
        print(f"Found {len(xml_files)} XML files in '{xml_dir}'")

    all_rows: list[dict] = []
    errors: list[tuple[str, str]] = []

    for path in xml_files:
        try:
            rows = process_file(path)
            all_rows.extend(rows)
            if verbose:
                print(f"  ✓  {path.name}  ({len(rows)} figures)")
        except Exception as exc:
            errors.append((path.name, str(exc)))
            if verbose:
                print(f"  ✗  {path.name}  ERROR: {exc}")

    df = pd.DataFrame(all_rows)

    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\n{len(all_rows)} rows → {output_csv}")

    return df, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract figures from Elsevier XML files.")
    p.add_argument("--xml-dir", default="alloys_elsevier")
    p.add_argument("--output-csv", default="alloy_elsevier_img_details.csv")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    extract_all(xml_dir=args.xml_dir, output_csv=args.output_csv)


if __name__ == "__main__":
    main()