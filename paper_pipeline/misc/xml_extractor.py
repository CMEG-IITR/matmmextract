from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from preprocessing import save_csv

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


#: refid prefixes that identify a cross-reference as pointing to a figure.
FIGURE_REFID_PREFIXES: frozenset[str] = frozenset({"fig", "f", "appsec"})

#: Paragraph tag names to scan for body text.
PARA_TAGS: list[str] = ["ce:para", "ce:simple-para"]

#: Cross-reference tag names.
XREF_TAGS: list[str] = ["ce:cross-ref", "ce:cross-refs"]

#: Abbreviations that must NOT trigger a sentence boundary.
_ABBREV_PATTERN = re.compile(
    r"\b(Fig|Figs|fig|figs|e\.g|i\.e|et al|vs|approx|Dr|Prof|cf|Eq|Eqs|No|Vol|pp)\."
)


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, protecting scientific abbreviations.

    Uses a simple regex split on ``[.!?]`` followed by whitespace + capital,
    after temporarily masking known abbreviation dots.

    Parameters
    ----------
    text:
        Plain text to split.

    Returns
    -------
    list[str]
        Non-empty sentences.
    """
    protected = _ABBREV_PATTERN.sub(
        lambda m: m.group(0).replace(".", "<DOT>"), text
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\(])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def build_image_map(soup: BeautifulSoup) -> dict[str, str]:
    """Build a ``{ref_key: url}`` mapping from the ``<objects>`` section.

    Prefers ``category='high'``; falls back to any category if no
    high-resolution object is found.

    Parameters
    ----------
    soup:
        Parsed XML document.

    Returns
    -------
    dict[str, str]
        Maps locator strings (e.g. ``"gr1"``) to URLs.
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


def resolve_image_url(
    fig_tag: Any,
    image_map: dict[str, str],
) -> str | None:
    """Resolve the image URL for a ``<ce:figure>`` element.

    Tries three strategies in order:

    1. ``<ce:graphic *href="grN">``  (legacy, uncommon)
    2. ``<ce:link locator="grN">``   (primary path for current XMLs)
    3. Numeric fallback: strip leading zeros from figure ID → ``grN``

    Parameters
    ----------
    fig_tag:
        BeautifulSoup tag for a ``<ce:figure>`` element.
    image_map:
        Mapping produced by :func:`build_image_map`.

    Returns
    -------
    str or None
        Resolved URL, or ``None`` if not found.
    """
    # Strategy 1 — ce:graphic href
    graphic = fig_tag.find(["ce:graphic", "graphic"])
    if graphic:
        for attr, val in graphic.attrs.items():
            if "href" in attr:
                url = image_map.get(val.strip())
                if url:
                    return url

    # Strategy 2 — ce:link locator  (primary)
    link = fig_tag.find("ce:link")
    if link:
        locator = link.get("locator", "").strip()
        if locator:
            url = image_map.get(locator)
            if url:
                return url

    # Strategy 3 — numeric fallback
    fig_id = fig_tag.get("id", "")
    nums = re.findall(r"\d+", fig_id)
    if nums:
        url = image_map.get(f"gr{int(nums[0])}")
        if url:
            return url

    return None


def extract_figures(
    soup: BeautifulSoup,
    image_map: dict[str, str],
) -> dict[str, dict]:
    """Extract all figures from the document.

    Handles:

    * ``fig####`` and ``f####`` ID patterns.
    * Nested ``<ce:figure>`` inside ``<ce:figure>`` (sub-figures).
    * Caption from ``<ce:caption>``.
    * ``fig_num``: integer used later for regex-based sentence matching.

    Parameters
    ----------
    soup:
        Parsed XML document.
    image_map:
        Mapping produced by :func:`build_image_map`.

    Returns
    -------
    dict
        ``{fig_id: {"caption": str, "image_url": str|None, "fig_num": int|None}}``
    """
    figures: dict[str, dict] = {}

    for fig in soup.find_all("ce:figure"):
        fig_id = fig.get("id", "").strip()
        if not fig_id:
            continue

        # Caption
        caption = ""
        cap = fig.find("ce:caption", recursive=False) or fig.find("ce:caption")
        if cap:
            caption = cap.get_text(" ", strip=True)

        # Image URL
        img_url = resolve_image_url(fig, image_map)

        # Numeric part for sentence matching (fig0001→1, f0005→5, fig2a→2)
        nums = re.findall(r"\d+", fig_id)
        fig_num = int(nums[0]) if nums else None

        figures[fig_id] = {
            "caption": caption,
            "image_url": img_url,
            "fig_num": fig_num,
        }

    return figures

def get_merged_paragraphs(soup: BeautifulSoup) -> list[tuple[Any, str]]:
    """Return ``(tag, merged_text)`` pairs for all logical paragraphs.

    Many Elsevier XMLs split a single logical sentence across two sibling
    ``<ce:para>`` nodes using ``<ce:float-anchor>``.  This function detects
    that pattern and stitches the two halves together so that the sentence
    splitter sees the complete text.

    Parameters
    ----------
    soup:
        Parsed XML document.

    Returns
    -------
    list of (tag, str)
        Each entry is the original paragraph tag and its (merged) text.
    """
    body = soup.find(["ce:body", "body", "ce:sections"])
    if not body:
        return [
            (p, p.get_text(" ", strip=True))
            for p in soup.find_all(PARA_TAGS)
        ]

    all_paras = body.find_all(PARA_TAGS)
    merged: list[tuple[Any, str]] = []
    skip_next = False

    for i, para in enumerate(all_paras):
        if skip_next:
            skip_next = False
            continue

        text = para.get_text(" ", strip=True)

        if para.find("ce:float-anchor") and i + 1 < len(all_paras):
            next_text = all_paras[i + 1].get_text(" ", strip=True)
            text = text + " " + next_text
            skip_next = True

        merged.append((para, text))

    return merged


def extract_reference_sentences(
    soup: BeautifulSoup,
    figures: dict[str, dict],
) -> dict[str, list[str]]:
    """Find body sentences that cite each figure.

    Two detection modes per paragraph:

    A) **Structured** — ``<ce:cross-ref refid="fig0003">`` inside the paragraph.
       Reliable; used when available.
    B) **Regex fallback** — ``"Fig. 3"``, ``"Figs. 3"``, ``"Figure 3"``
       pattern scan.  Used when no structured cross-ref tags are present.

    Parameters
    ----------
    soup:
        Parsed XML document.
    figures:
        Dict produced by :func:`extract_figures`.

    Returns
    -------
    dict
        ``{fig_id: [sentence, …]}`` — each value is a list of sentences
        (may contain duplicates; de-duplication happens in
        :func:`process_file`).
    """
    ref_map: dict[str, list[str]] = {fid: [] for fid in figures}

    # num → list of fig_ids sharing that number (handles sub-figures)
    num_to_fids: dict[int, list[str]] = {}
    for fid, data in figures.items():
        n = data["fig_num"]
        if n is not None:
            num_to_fids.setdefault(n, []).append(fid)

    para_blocks = get_merged_paragraphs(soup)

    for para_tag, text in para_blocks:
        if not text:
            continue

        sentences = split_sentences(text)

        # ── Mode A: structured xrefs ─────────────────────────────────────
        fids_structured: set[str] = set()
        for xr in para_tag.find_all(XREF_TAGS):
            refid_val = xr.get("refid", "")
            for rid in refid_val.split():
                prefix = re.match(r"^([A-Za-z]+)", rid)
                if (
                    prefix
                    and prefix.group(1).lower() in FIGURE_REFID_PREFIXES
                    and rid in ref_map
                ):
                    fids_structured.add(rid)

        if fids_structured:
            nums_needed = {
                figures[fid]["fig_num"]
                for fid in fids_structured
                if figures[fid]["fig_num"] is not None
            }
            for sentence in sentences:
                for n in nums_needed:
                    pattern = rf"\b(Fig\.?\s*{n}|Figs\.?\s*{n}|Figure\s*{n})\b"
                    if re.search(pattern, sentence, re.IGNORECASE):
                        for fid in num_to_fids.get(n, []):
                            ref_map[fid].append(sentence)
            continue  # structured is reliable; skip regex mode

        # ── Mode B: regex fallback ───────────────────────────────────────
        for n, fids in num_to_fids.items():
            pattern = rf"\b(Fig\.?\s*{n}|Figs\.?\s*{n}|Figure\s*{n})\b"
            if re.search(pattern, text, re.IGNORECASE):
                for sentence in sentences:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        for fid in fids:
                            ref_map[fid].append(sentence)

    return ref_map


def process_file(xml_path: str | Path) -> list[dict]:
    """Extract all figure rows from a single XML file.

    Parameters
    ----------
    xml_path:
        Path to the Elsevier XML file.

    Returns
    -------
    list[dict]
        One dict per figure, with keys:
        ``xml_file``, ``figure_id``, ``caption``, ``image_url``,
        ``num_references``, ``reference_sentences``.
    """
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

        rows.append(
            {
                "xml_file": fname,
                "figure_id": fid,
                "caption": data["caption"],
                "image_url": data["image_url"] or "",
                "num_references": len(unique),
                "reference_sentences": " || ".join(unique[:5]),
            }
        )

    return rows


def extract_all(
    xml_dir: str | Path,
    output_csv: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Process every XML file in *xml_dir* and return the results.

    Parameters
    ----------
    xml_dir:
        Directory containing ``.xml`` files.
    output_csv:
        Optional path to write the resulting DataFrame as CSV.
        If ``None``, no file is written.
    verbose:
        Print a progress line for each file.

    Returns
    -------
    df : pd.DataFrame
        One row per figure across all files.
    errors : list of (filename, error_message)
        Files that raised an exception during processing.

    Examples
    --------
    >>> df, errors = extract_all("alloys_elsevier")
    >>> df, errors = extract_all("alloys_elsevier", output_csv="figures.csv")
    """
    xml_dir = Path(xml_dir)
    xml_files = sorted(f for f in xml_dir.iterdir() if f.suffix == ".xml")

    if verbose:
        print(f"[xml_extractor] Found {len(xml_files)} XML files in '{xml_dir}'")

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
        save_csv(df, output_csv)
        if verbose:
            print(f"\n[xml_extractor] {len(all_rows)} rows → {output_csv}")

    if errors and verbose:
        print(f"[xml_extractor] {len(errors)} file(s) had errors.")

    return df, errors
