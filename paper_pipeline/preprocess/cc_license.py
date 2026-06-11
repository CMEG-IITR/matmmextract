"""
paper_pipeline.preprocess.cc_license
======================================
Scan Elsevier XML files to identify their CC license type, then filter
figure rows down to CC BY only (with at least one reference sentence).

This is the notebook's CC diagnostics cell — expressed as a callable API.

Notebook cells covered
-----------------------
- ``analyse_file()``        — detect CC license from a single XML
- The main scan loop        → ``scan_directory()``
- CC BY filter on figures   → ``filter_figures_cc_by()``
- Copy CC BY XMLs           → lives in preprocess.pipeline.copy_xmls_by_filename

Detection strategies (in order):
  1. <oa:userLicense> / <oa:openAccessInformation>  (most reliable)
  2. <prism:copyright> / <dc:rights>
  3. <ce:copyright> and any tag whose name contains "copyright"
  4. <license> / <ali:license_ref> (JATS style)
  5. Free-text scan for creativecommons.org URLs
  5b. Free-text scan for plain "CC BY" text
  6. © Elsevier without any CC marker → Subscription/Copyright
"""

from __future__ import annotations

import csv
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ---------------------------------------------------------------------------
# CC pattern → canonical label
# ---------------------------------------------------------------------------

_CC_PATTERNS: list[tuple[str, str]] = [
    # Most restrictive first so we match correctly
    (r"by-nc-nd",                                              "CC BY-NC-ND"),
    (r"by-nc-sa",                                              "CC BY-NC-SA"),
    (r"by-nc",                                                 "CC BY-NC"),
    (r"by-nd",                                                 "CC BY-ND"),
    (r"by-sa",                                                 "CC BY-SA"),
    (r"by/4\.0|by/3\.0|by/2\.0|by/1\.0|creativecommons\.org/licenses/by/", "CC BY"),
    # Bare text forms
    (r"\bcc[\s\-]by[\s\-]nc[\s\-]nd\b",                       "CC BY-NC-ND"),
    (r"\bcc[\s\-]by[\s\-]nc[\s\-]sa\b",                       "CC BY-NC-SA"),
    (r"\bcc[\s\-]by[\s\-]nc\b",                               "CC BY-NC"),
    (r"\bcc[\s\-]by[\s\-]nd\b",                               "CC BY-ND"),
    (r"\bcc[\s\-]by[\s\-]sa\b",                               "CC BY-SA"),
    (r"\bcc[\s\-]by\b",                                        "CC BY"),
    # Elsevier-specific
    (r"elsevier user licen",                                   "Elsevier User License"),
    (r"open archive",                                          "Elsevier Open Archive"),
    (r"creativecommons\.org",                                  "CC (version unknown)"),
]

#: Human-readable redistribution rights for each license label.
REDISTRIB_RIGHTS: dict[str, str] = {
    "CC BY":                  "✅  Full reuse incl. commercial — safe for dataset",
    "CC BY-SA":               "✅  Reuse allowed (share-alike) — safe for dataset",
    "CC BY-NC":               "⚠️  Non-commercial only — dataset must be NC too",
    "CC BY-NC-SA":            "⚠️  Non-commercial + share-alike",
    "CC BY-ND":               "❌  No derivatives — cannot redistribute image dataset",
    "CC BY-NC-ND":            "❌  No derivatives + non-commercial",
    "Elsevier User License":  "❌  Restricted reuse — not suitable for redistribution",
    "Elsevier Open Archive":  "❌  Read-only — not suitable for redistribution",
    "CC (version unknown)":   "⚠️  CC present but type unclear — check manually",
    "Subscription/Copyright": "❌  No open license — cannot reuse",
    "UNKNOWN":                "❓  Could not detect license — check manually",
}


def _normalise_license(text: str) -> str | None:
    """Return canonical CC label from any license string/URL, or ``None``."""
    if not text:
        return None
    t = text.lower().strip()
    for pattern, label in _CC_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE):
            return label
    return None


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyse_file(xml_path: str | Path) -> dict:
    """Detect the CC license of a single Elsevier XML file.

    Parameters
    ----------
    xml_path:
        Path to the XML file.

    Returns
    -------
    dict with keys:
        ``file``, ``license``, ``license_raw``, ``detection_path``,
        ``doi``, ``title``, ``notes``
    """
    fname = os.path.basename(xml_path)
    result: dict = {
        "file": fname,
        "license": "UNKNOWN",
        "license_raw": "",
        "detection_path": "",
        "doi": "",
        "title": "",
        "notes": [],
    }

    with open(xml_path, "rb") as fh:
        soup = BeautifulSoup(fh, "lxml")

    # DOI
    doi_tag = (
        soup.find("ce:doi")
        or soup.find("prism:doi")
        or soup.find("dc:identifier")
        or soup.find(attrs={"name": "citation_doi"})
    )
    if doi_tag:
        result["doi"] = doi_tag.get_text(strip=True)

    # Title
    title_tag = (
        soup.find("ce:title")
        or soup.find("dc:title")
        or soup.find("article-title")
    )
    if title_tag:
        result["title"] = title_tag.get_text(separator=" ", strip=True)[:120]

    # Strategy 1: <oa:userLicense> / <oa:openAccessInformation>
    for tag_name in ["oa:useraccessright", "oa:userlicense", "oa:openaccess",
                     "userAccessRight", "userLicense"]:
        tag = soup.find(tag_name)
        if tag:
            raw = tag.get_text(strip=True)
            lic = _normalise_license(raw)
            if lic:
                result.update(license=lic, license_raw=raw,
                               detection_path="Strategy1: <oa:userLicense>")
                return result

    oa_block = soup.find(re.compile(r"openaccess", re.I))
    if oa_block:
        for attr_val in oa_block.attrs.values():
            lic = _normalise_license(str(attr_val))
            if lic:
                result.update(license=lic, license_raw=str(attr_val),
                               detection_path="Strategy1b: <oa:*> attribute")
                return result

    # Strategy 2: <prism:copyright> / <dc:rights>
    for tag_name in ["prism:copyright", "dc:rights", "rights"]:
        tag = soup.find(tag_name)
        if tag:
            raw = tag.get_text(strip=True)
            lic = _normalise_license(raw)
            if lic:
                result.update(license=lic, license_raw=raw,
                               detection_path=f"Strategy2: <{tag_name}>")
                return result
            if raw:
                result["license_raw"] = raw
                result["notes"].append(f"<{tag_name}> found but no CC: {raw[:80]}")

    # Strategy 3: <ce:copyright> and any copyright-named tag
    for tag in soup.find_all(re.compile(r"copyright", re.I)):
        raw = tag.get_text(strip=True)
        lic = _normalise_license(raw)
        if lic:
            result.update(license=lic, license_raw=raw,
                           detection_path=f"Strategy3: <{tag.name}>")
            return result
        for attr_val in tag.attrs.values():
            lic = _normalise_license(str(attr_val))
            if lic:
                result.update(license=lic, license_raw=str(attr_val),
                               detection_path=f"Strategy3b: <{tag.name}> attr")
                return result

    # Strategy 4: <license> / <ali:license_ref> (JATS)
    for tag_name in ["license", "ali:license_ref", "license-p"]:
        for tag in soup.find_all(tag_name):
            href = tag.get("href", tag.get("xlink:href", ""))
            raw = href or tag.get_text(strip=True)
            lic = _normalise_license(raw)
            if lic:
                result.update(license=lic, license_raw=raw,
                               detection_path=f"Strategy4: <{tag_name}>")
                return result

    # Strategy 5: free-text scan of raw XML
    with open(xml_path, "r", encoding="utf-8", errors="replace") as fh:
        raw_text = fh.read()

    cc_urls = re.findall(
        r"(https?://creativecommons\.org/licenses/[^\s\"\'<>]+)", raw_text, re.IGNORECASE
    )
    if cc_urls:
        raw = cc_urls[0]
        lic = _normalise_license(raw)
        if lic:
            result.update(license=lic, license_raw=raw,
                           detection_path="Strategy5: CC URL in raw text")
            return result

    cc_texts = re.findall(
        r"(CC[\s\-]BY[\s\-]?(?:NC[\s\-]?)?(?:ND|SA)?)", raw_text, re.IGNORECASE
    )
    if cc_texts:
        raw = cc_texts[0]
        lic = _normalise_license(raw)
        if lic:
            result.update(license=lic, license_raw=raw,
                           detection_path="Strategy5b: CC text in raw XML")
            return result

    # Strategy 6: © Elsevier with no CC → subscription
    if re.search(r"©\s*(20\d\d)?\s*elsevier", raw_text, re.IGNORECASE):
        if not any(kw in raw_text.lower() for kw in ["creativecommons", "cc by", "cc-by"]):
            result.update(license="Subscription/Copyright",
                           detection_path="Strategy6: © Elsevier, no CC found")
            return result

    return result


# ---------------------------------------------------------------------------
# Batch scan
# ---------------------------------------------------------------------------

def scan_directory(
    xml_dir: str | Path,
    output_txt: str | Path | None = None,
    output_csv: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Scan every XML file in *xml_dir* and return license information.

    Parameters
    ----------
    xml_dir:
        Directory containing Elsevier XML files.
    output_txt:
        Optional path for the full human-readable report.
    output_csv:
        Optional path for the machine-readable per-file CSV.
    verbose:
        Print per-file progress.

    Returns
    -------
    df : pd.DataFrame
        One row per file with columns:
        ``file``, ``license``, ``license_raw``, ``detection_path``,
        ``doi``, ``title``, ``reuse_rights``
    tally : dict[str, int]
        ``{license_label: count}``
    """
    xml_dir = Path(xml_dir)
    xml_files = sorted(f for f in xml_dir.iterdir() if f.suffix == ".xml")

    if verbose:
        print(f"Found {len(xml_files)} XML files in '{xml_dir}'")

    results: list[dict] = []
    tally: dict[str, int] = defaultdict(int)

    for path in xml_files:
        if verbose:
            print(f"  Analysing {path.name} ...", end=" ", flush=True)
        try:
            r = analyse_file(path)
            results.append(r)
            tally[r["license"]] += 1
            if verbose:
                print(r["license"])
        except Exception as exc:
            entry = {
                "file": path.name, "license": "ERROR",
                "license_raw": str(exc), "detection_path": "",
                "doi": "", "title": "", "notes": [],
            }
            results.append(entry)
            tally["ERROR"] += 1
            if verbose:
                print(f"ERROR: {exc}")

    # Build DataFrame
    rows = []
    for r in results:
        rows.append({
            "file": r["file"],
            "license": r["license"],
            "license_raw": r["license_raw"],
            "detection_path": r["detection_path"],
            "doi": r["doi"],
            "title": r["title"],
            "reuse_rights": REDISTRIB_RIGHTS.get(r["license"], ""),
        })
    df = pd.DataFrame(rows)

    # Summary print
    if verbose:
        print(f"\n{'='*60}")
        print("LICENSE SUMMARY")
        for lic, cnt in sorted(tally.items(), key=lambda x: -x[1]):
            pct = cnt / len(results) * 100 if results else 0
            print(f"  {lic:<30} {cnt:4d} ({pct:5.1f}%)  {REDISTRIB_RIGHTS.get(lic,'')}")
        print(f"{'='*60}")

    # Optional file outputs
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"CSV saved → {output_csv}")

    if output_txt is not None:
        output_txt = Path(output_txt)
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        _write_text_report(results, tally, output_txt)
        if verbose:
            print(f"Report saved → {output_txt}")

    return df, dict(tally)


# ---------------------------------------------------------------------------
# Figure filtering (post-scan)
# ---------------------------------------------------------------------------

def filter_figures_cc_by(
    figures_df: pd.DataFrame,
    cc_summary_df: pd.DataFrame,
    require_references: bool = True,
    output_csv: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Keep only figures from CC BY papers that have reference sentences.

    Mirrors the notebook cells:
        ``cc_by_df = cc_df[cc_df["license"] == "CC BY"]``
        ``filtered = figures_df[figures_df["xml_file"].isin(cc_by_files) & has_ref]``

    Parameters
    ----------
    figures_df:
        Output of :func:`~paper_pipeline.elsevier.extractor.extract_all`.
    cc_summary_df:
        Output of :func:`scan_directory` (the ``df`` return value).
    require_references:
        If ``True`` (default), also require ``num_references > 0`` OR
        a non-empty ``reference_sentences`` value.
    output_csv:
        If provided, save the filtered DataFrame here.
    verbose:
        Print counts.

    Returns
    -------
    pd.DataFrame
        Filtered figure rows.
    """
    cc_by_files = set(
        cc_summary_df.loc[cc_summary_df["license"] == "CC BY", "file"]
        .dropna().astype(str).str.strip()
    )

    mask = figures_df["xml_file"].astype(str).isin(cc_by_files)

    if require_references:
        num_refs = pd.to_numeric(figures_df["num_references"], errors="coerce").fillna(0)
        has_ref_text = figures_df["reference_sentences"].fillna("").str.strip().ne("")
        mask = mask & ((num_refs > 0) | has_ref_text)

    result = figures_df[mask].reset_index(drop=True)

    if verbose:
        print(f"filter_figures_cc_by: {len(figures_df)} → {len(result)} rows "
              f"({len(cc_by_files)} CC BY files)")

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        if verbose:
            print(f"Saved → {output_csv}")

    return result


# ---------------------------------------------------------------------------
# Text report writer (internal)
# ---------------------------------------------------------------------------

def _write_text_report(results: list[dict], tally: dict[str, int], path: Path) -> None:
    W = 70
    lines: list[str] = []

    def h(t):
        lines.append("\n" + "═" * W + f"\n  {t}\n" + "═" * W)

    def kv(k, v, indent=4):
        lines.append(f"{' ' * indent}{k}: {v}")

    h("CC LICENSE DIAGNOSTICS — ELSEVIER XML")
    kv("Total files", len(results))

    h("SUMMARY — LICENSE DISTRIBUTION")
    for lic, cnt in sorted(tally.items(), key=lambda x: -x[1]):
        pct = cnt / len(results) * 100 if results else 0
        right = REDISTRIB_RIGHTS.get(lic, "")
        lines.append(f"    {lic:<30} {cnt:4d} ({pct:5.1f}%)   {right}")

    groups = {
        "✅  FREELY REDISTRIBUTABLE (CC BY / CC BY-SA)": ["CC BY", "CC BY-SA"],
        "⚠️  NON-COMMERCIAL (CC BY-NC / BY-NC-SA)": ["CC BY-NC", "CC BY-NC-SA"],
        "❌  RESTRICTED": ["CC BY-ND", "CC BY-NC-ND", "Elsevier User License",
                           "Elsevier Open Archive", "Subscription/Copyright"],
        "❓  UNKNOWN": ["UNKNOWN", "CC (version unknown)", "ERROR"],
    }
    for heading, licenses in groups.items():
        subset = [r for r in results if r["license"] in licenses]
        if not subset:
            continue
        h(heading)
        for r in subset:
            lines.append(f"    {r['file']}")
            if r["doi"]:
                kv("DOI", r["doi"])
            kv("License", r["license"])
            kv("Raw", r["license_raw"][:100])
            kv("Found via", r["detection_path"])
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
