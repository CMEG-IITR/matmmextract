from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

from ..shared.sentence_utils import split_sentences

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)



PARA_TAGS: frozenset[str] = frozenset({"p", "para", "simple-para"})
FIG_TAGS: frozenset[str] = frozenset({"fig"})
XREF_TAGS: frozenset[str] = frozenset({"xref"})

SKIP_TEXT_TAGS: frozenset[str] = frozenset({
    "math", "mml:math", "mml:semantics",
    "disp-formula", "inline-formula",
    "tex-math", "alternatives",
})

GA_RE = re.compile(r"graphical\s+abstract", re.IGNORECASE)


def _tag_name(node) -> str:
    if not isinstance(node, Tag) or not node.name:
        return ""
    return node.name.lower()


def _local_name(node) -> str:
    return _tag_name(node).split(":", 1)[-1]


def _attr_value(tag, wanted: str) -> str:
    for key, value in tag.attrs.items():
        if key == wanted or key.split(":", 1)[-1] == wanted:
            if isinstance(value, list):
                return " ".join(str(v) for v in value)
            return str(value)
    return ""


def _clean_markers(text: str) -> str:
    text = re.sub(r"__XREF_.*?__\s*", "", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def _normalize_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(Fig(?:ure)?\.?\s*\d+[A-Za-z]?\s*){2,}", r"\1", text)


def _figure_number_from_id(fig_id: str | None) -> int | None:
    m = re.search(r"(\d+)", fig_id or "")
    return int(m.group(1)) if m else None


def _is_figure_xref(xref) -> bool:
    ref_type = _attr_value(xref, "ref-type").lower()
    rid = _attr_value(xref, "rid")
    if ref_type in {"fig", "figure"}:
        return bool(rid)
    return bool(re.search(r"\b(fig|f)\d+[a-z]?\b", rid, re.IGNORECASE))


def _extract_fallback_figure_numbers(sentence: str) -> set[int]:
    """Extract figure numbers from untagged mentions (e.g. 'Figs. 2-5')."""
    numbers: set[int] = set()
    for match in re.finditer(
        r"\bFig(?:ure)?s?\.?\s+([0-9A-Za-z\(\)\[\],\s\-–—]+)",
        sentence, re.IGNORECASE,
    ):
        phrase = match.group(1)
        phrase = re.split(
            r"\b(?:show|shows|present|presents|illustrate|illustrates|"
            r"display|displays|compare|compares|summarize|summarizes)\b",
            phrase, 1, flags=re.IGNORECASE,
        )[0]
        for start, end in re.findall(r"(\d+)\s*[-–—]\s*(\d+)", phrase):
            lo, hi = int(start), int(end)
            if lo <= hi and hi - lo <= 50:
                numbers.update(range(lo, hi + 1))
        for number in re.findall(r"\d+", phrase):
            numbers.add(int(number))
    return numbers


def resolve_image_url(fig_tag) -> str:
    """Resolve image location from JATS ``<graphic xlink:href="...">``, etc."""
    for child in fig_tag.find_all(True):
        if _local_name(child) not in {"graphic", "inline-graphic", "media"}:
            continue
        href = _attr_value(child, "href")
        if href:
            return href.strip()
    return ""


def _is_graphical_abstract(fig_tag) -> bool:
    if GA_RE.search(_attr_value(fig_tag, "id")):
        return True
    label = fig_tag.find(lambda t: _local_name(t) == "label")
    if label and GA_RE.search(label.get_text(" ", strip=True)):
        return True
    caption = fig_tag.find(lambda t: _local_name(t) == "caption")
    if caption and GA_RE.search(caption.get_text(" ", strip=True)):
        return True
    return False


def extract_figures(root) -> dict[str, dict]:
    """Return ``{fig_id: {caption, image_url, fig_num, is_graphical_abstract}}``."""
    figures: dict[str, dict] = {}

    for fig in root.find_all(lambda t: _local_name(t) in FIG_TAGS):
        fig_id = _attr_value(fig, "id").strip()
        if not fig_id:
            label_tag = fig.find(lambda t: _local_name(t) == "label")
            if label_tag:
                fig_id = re.sub(r"\W+", "", label_tag.get_text(" ", strip=True))
        if not fig_id:
            continue

        label_text = ""
        label = fig.find(lambda t: _local_name(t) == "label", recursive=False)
        if label:
            label_text = label.get_text(" ", strip=True)

        caption_text = ""
        caption = fig.find(lambda t: _local_name(t) == "caption", recursive=False)
        if caption:
            caption_text = caption.get_text(" ", strip=True)

        caption_full = _normalize_caption(
            " ".join(p for p in [label_text, caption_text] if p)
        )

        figures[fig_id] = {
            "caption": caption_full,
            "image_url": resolve_image_url(fig),
            "fig_num": _figure_number_from_id(fig_id),
            "is_graphical_abstract": _is_graphical_abstract(fig),
        }

    return figures



def _get_marked_text(para_tag) -> tuple[str, dict[str, str]]:
    """Reconstruct paragraph text with ``__XREF_<rid>__`` sentinel markers."""
    parts: list[str] = []
    marker_map: dict[str, str] = {}

    def walk(node):
        if isinstance(node, NavigableString):
            parts.append(str(node))
            return
        if not isinstance(node, Tag):
            return

        name = _local_name(node)
        full = _tag_name(node)
        if full in SKIP_TEXT_TAGS or name in SKIP_TEXT_TAGS:
            return

        if name in XREF_TAGS:
            rid = _attr_value(node, "rid")
            if rid and _is_figure_xref(node):
                marker = f"__XREF_{rid}__"
                marker_map[marker] = rid
                parts.append(marker)
            parts.append(node.get_text(" ", strip=True))
            return

        for child in node.children:
            walk(child)

    for child in para_tag.children:
        walk(child)

    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return text, marker_map


def extract_reference_sentences(root, figures: dict[str, dict]) -> dict[str, list[str]]:
    """Find body sentences that cite each figure."""
    ref_map: dict[str, list[str]] = {fid: [] for fid in figures}

    num_to_fids: dict[int, list[str]] = {}
    for fid, data in figures.items():
        n = data["fig_num"]
        if n is not None:
            num_to_fids.setdefault(n, []).append(fid)

    body = root.find(lambda t: _local_name(t) == "body")
    search_root = body or root

    for para in search_root.find_all(lambda t: _local_name(t) in PARA_TAGS):
        text_marked, marker_map = _get_marked_text(para)
        if not text_marked:
            continue

        has_any_fig_marker = bool(marker_map)

        for sentence in split_sentences(text_marked):
            found = re.findall(r"__XREF_(.*?)__", sentence)
            clean = _clean_markers(sentence)

            if found:
                for marker_content in found:
                    for rid in marker_content.split():
                        if rid in ref_map and clean not in ref_map[rid]:
                            ref_map[rid].append(clean)
            elif not has_any_fig_marker:
                for number in _extract_fallback_figure_numbers(clean):
                    for fid in num_to_fids.get(number, []):
                        if clean not in ref_map[fid]:
                            ref_map[fid].append(clean)

    return ref_map


def process_file(xml_path: str | Path) -> list[dict]:
    """Extract all figure rows from a single Springer JATS XML file."""
    with open(xml_path, "rb") as fh:
        soup = BeautifulSoup(fh, "lxml")

    figures = extract_figures(soup)
    ref_map = extract_reference_sentences(soup, figures)

    fname = os.path.basename(xml_path)
    rows: list[dict] = []

    for fid, data in figures.items():
        sentences = ref_map.get(fid, [])
        rows.append({
            "xml_file": fname,
            "figure_id": fid,
            "caption": data["caption"],
            "image_url": data["image_url"],
            "is_graphical_abstract": data["is_graphical_abstract"],
            "num_references": len(sentences),
            "reference_sentences": " || ".join(sentences[:5]),
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
        Directory containing Springer ``.xml`` files.
    output_csv:
        If provided, write the DataFrame here.
    verbose:
        Print per-file progress and summary.

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
                n_refs = sum(1 for r in rows if r["num_references"] > 0)
                n_ga = sum(1 for r in rows if r["is_graphical_abstract"])
                print(f"  OK  {path.name}  figs={len(rows)}  with_refs={n_refs}  ga={n_ga}")
        except Exception as exc:
            errors.append((path.name, str(exc)))
            if verbose:
                print(f"  ERR {path.name}  ERROR: {exc}")

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
    p = argparse.ArgumentParser(description="Extract figures from Springer JATS XML files.")
    p.add_argument("--xml-dir", default="alloys_springer")
    p.add_argument("--output-csv", default="springer_figure_details.csv")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    extract_all(xml_dir=args.xml_dir, output_csv=args.output_csv)


if __name__ == "__main__":
    main()