from __future__ import annotations

import re

# Abbreviations that end with a period but must NOT trigger a sentence split.
ABBREV_RE = re.compile(
    r"\b(Fig|Figs|fig|figs|e\.g|i\.e|et al|vs|approx|Dr|Prof|cf|Eq|Eqs|No|Vol|pp)\."
)


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, protecting scientific abbreviations.

    Strategy: temporarily replace abbreviation dots with a placeholder,
    split on ``[.!?]`` followed by whitespace + uppercase/bracket, then
    restore the placeholder.

    Parameters
    ----------
    text:
        Plain text to split.

    Returns
    -------
    list[str]
        Non-empty sentence strings.

    Examples
    --------
    >>> split_sentences("As shown in Fig. 3, the yield is high. See also Fig. 4.")
    ['As shown in Fig. 3, the yield is high.', 'See also Fig. 4.']
    """
    protected = ABBREV_RE.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\(\[])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]