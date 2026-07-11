"""
Shared word-boundary phrase matching — used everywhere a keyword/cue list is
checked against free-text user input (planner routing, Section-12 eligibility
detection).

Bug this fixes (verified live): naive `substring in text` checks match INSIDE
unrelated words — the eligibility cue "her " matched inside "weather", and the
wages keyword "site" matched inside "opposite". Both caused wrong routing /
false eligibility claims on completely unrelated queries. `contains_phrase()`
requires the phrase to appear as whole word(s), not as a fragment of a longer
word, while still matching multi-word phrases like "domestic worker" as a
contiguous span.
"""

import re
from functools import lru_cache


@lru_cache(maxsize=None)
def _compiled(phrase: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(phrase.strip()) + r"\b", re.IGNORECASE)


def contains_phrase(text: str, phrase: str) -> bool:
    """True if `phrase` appears in `text` as whole word(s) — not as a
    substring fragment of a longer word."""
    phrase = phrase.strip()
    if not phrase:
        return False
    return _compiled(phrase).search(text) is not None


def any_phrase(text: str, phrases) -> bool:
    return any(contains_phrase(text, p) for p in phrases)


def count_phrases(text: str, phrases) -> int:
    return sum(1 for p in phrases if contains_phrase(text, p))
