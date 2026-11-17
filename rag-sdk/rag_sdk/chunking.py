"""Token-capped, sentence-aware text chunking.

A pure module: no provider or tokenizer import. The caller passes ``count_tokens``, which is what
makes the cap real (a heuristic default could not guarantee one). Use the counter that matches
your embedding model, e.g. ``rag_sdk.tokenizer.openai_token_counter``.

Method
------
1. Split into paragraphs (blank-line separated), keeping a paragraph whole if it fits the target.
2. Otherwise split it into sentences (a break needs ``.``, ``!`` or ``?`` followed by whitespace
   and then an upper-case letter, digit, quote or bracket, so ``0.75 percent``, ``BMS-3.2.1`` and
   ``e.g. this`` are not split).
3. A sentence over the cap is cut into the longest pieces that fit, preferring whitespace.
4. Pieces are packed greedily, in order, up to ``target_tokens``. Consecutive chunks may overlap
   by whole units (sentences or paragraphs) totalling at most ``overlap_tokens``.

Guarantees (tested)
-------------------
- Every chunk is an exact, contiguous substring of the input, so ``text[start:end]`` for
  ``chunk_spans`` equals the chunk. Containment labels and content IDs rely on this.
- Every chunk is within ``max_tokens`` by ``count_tokens`` and is never empty.
- Every non-whitespace character of the input is in at least one chunk.
- Each chunk contains at least one unit its predecessor did not, so no chunk sits inside another.
- Deterministic; input within the target is returned as a single chunk.

Not handled: headings are not treated specially, and two identical passages in one input can
produce identical chunks (which would share a content ID).
"""

import re
from collections.abc import Callable
from functools import lru_cache

TokenCounter = Callable[[str], int]
Span = tuple[int, int]

_PARAGRAPH_BREAK = re.compile(r"\n[ \t\r]*\n\s*")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _trim(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _split(text: str, pattern: re.Pattern[str], outer: Span) -> list[Span]:
    spans: list[Span] = []
    cursor = outer[0]
    for match in pattern.finditer(text, outer[0], outer[1]):
        if (span := _trim(text, cursor, match.start())) is not None:
            spans.append(span)
        cursor = match.end()
    if (span := _trim(text, cursor, outer[1])) is not None:
        spans.append(span)
    return spans


def _longest_fitting_end(text: str, start: int, end: int, fits: Callable[[int, int], bool]) -> int:
    """Largest ``e`` in ``(start, end]`` with ``text[start:e]`` within the cap (>= start + 1)."""
    lo, hi = start + 1, end  # binary search for the longest prefix that fits
    if not fits(start, lo):
        raise ValueError("max_tokens is smaller than a single character's token count")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(start, mid):
            lo = mid
        else:
            hi = mid - 1
    while lo > start + 1 and not fits(start, lo):  # token counts are not strictly monotonic
        lo -= 1
    return lo


def chunk_spans(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int = 256,
    overlap_tokens: int = 0,
    target_tokens: int | None = None,
) -> list[Span]:
    """Return ``(start, end)`` character spans of the chunks; see the module docstring."""
    target = max_tokens if target_tokens is None else target_tokens
    if max_tokens < 1 or target < 1 or target > max_tokens:
        raise ValueError("need 1 <= target_tokens <= max_tokens")
    if overlap_tokens < 0 or overlap_tokens >= target:
        raise ValueError("overlap_tokens must be >= 0 and smaller than the target size")

    @lru_cache(maxsize=None)
    def tokens(start: int, end: int) -> int:
        return count_tokens(text[start:end])

    def fits_cap(start: int, end: int) -> bool:
        return tokens(start, end) <= max_tokens

    units: list[Span] = []
    for paragraph in _split(text, _PARAGRAPH_BREAK, (0, len(text))):
        if tokens(*paragraph) <= target:
            units.append(paragraph)
            continue
        for sentence in _split(text, _SENTENCE_BREAK, paragraph):
            if fits_cap(*sentence):
                units.append(sentence)
                continue
            start, end = sentence  # oversized: cut into the longest pieces that fit the cap
            while start < end:
                while start < end and text[start].isspace():
                    start += 1  # measure what will be emitted: leading whitespace is dropped
                if start >= end:
                    break
                cut = _longest_fitting_end(text, start, end, fits_cap)
                if cut < end:  # prefer to break after whitespace inside the piece
                    space = max(text.rfind(" ", start + 1, cut), text.rfind("\n", start + 1, cut))
                    if space > start:
                        cut = space
                # Trimming and the whitespace back-off can change how BPE tokenizes the piece
                # (a shorter string is not always fewer tokens), so re-measure the final piece
                # and shrink it until it really fits.
                piece = _trim(text, start, cut)
                while piece is not None and not fits_cap(*piece):
                    if piece[1] - piece[0] <= 1:
                        raise ValueError("max_tokens is smaller than a single character's tokens")
                    cut = piece[1] - 1
                    piece = _trim(text, start, cut)
                if piece is not None:
                    units.append(piece)
                start = cut

    spans: list[Span] = []
    n = len(units)
    i = 0  # first unit of the next chunk, possibly an overlap from the previous one
    new = 0  # first unit not yet covered by any chunk
    while new < n:
        first = i
        while first < new and not fits_cap(units[first][0], units[new][1]):
            first += 1  # drop overlap until at least one new unit fits beside it
        last = new
        while last + 1 < n and tokens(units[first][0], units[last + 1][1]) <= target:
            last += 1
        spans.append((units[first][0], units[last][1]))
        new = last + 1
        i = new  # choose the overlap: as many trailing whole units as the budget allows
        while i - 1 > first and tokens(units[i - 1][0], units[last][1]) <= overlap_tokens:
            i -= 1
    return spans


def chunk_text(
    text: str,
    count_tokens: TokenCounter,
    max_tokens: int = 256,
    overlap_tokens: int = 0,
    target_tokens: int | None = None,
) -> list[str]:
    """Chunk ``text`` into exact substrings; see the module docstring for the guarantees."""
    return [
        text[start:end]
        for start, end in chunk_spans(text, count_tokens, max_tokens, overlap_tokens, target_tokens)
    ]
