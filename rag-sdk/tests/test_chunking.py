import random

import pytest
from rag_sdk.chunking import chunk_spans, chunk_text


def words(text: str) -> int:
    return len(text.split())


def chars4(text: str) -> int:
    return -(-len(text) // 4)  # ~ BPE on English; also bounds unbroken strings


def check_invariants(text, counter, max_tokens, overlap_tokens=0, target_tokens=None):
    spans = chunk_spans(text, counter, max_tokens, overlap_tokens, target_tokens)
    chunks = [text[a:b] for a, b in spans]
    assert chunks == chunk_text(text, counter, max_tokens, overlap_tokens, target_tokens)
    assert spans == chunk_spans(text, counter, max_tokens, overlap_tokens, target_tokens)  # stable
    covered = [False] * len(text)
    for (a, b), chunk in zip(spans, chunks):
        assert chunk and chunk == chunk.strip()  # non-empty, trimmed only at its edges
        assert counter(chunk) <= max_tokens, (counter(chunk), chunk[:60])
        for i in range(a, b):
            covered[i] = True
    for i, ch in enumerate(text):
        assert covered[i] or ch.isspace(), f"character {ch!r} at {i} is in no chunk"
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert b2 > b1 and a2 > a1, "a chunk must add new text and start later"
    assert len(set(chunks)) == len(chunks), "duplicate chunks would collide on content ID"
    return chunks


SAMPLE = (
    "Battery management system firmware BMS-3.2.1 added cell balancing that trims voltage "
    "differences above fifteen millivolts. Replace a chain stretched past 0.75 percent wear "
    "promptly, or it will chew through the cassette. Error code E-4471 means the pump pressure "
    "sensor reads below 6 bar. Use e.g. a citric acid solution. Bake at 250 degrees.\n\n"
    "Second paragraph here. It has two sentences.\n\n\n"
    "Third paragraph is short."
)


def test_text_within_target_is_one_exact_chunk():
    assert chunk_text("  Short text.  ", words, max_tokens=50) == ["Short text."]
    assert chunk_text(SAMPLE, words, max_tokens=500) == [SAMPLE.strip()]


def test_empty_and_whitespace_input_gives_no_chunks():
    assert chunk_text("", words, max_tokens=5) == []
    assert chunk_text(" \n\n \t ", words, max_tokens=5) == []


@pytest.mark.parametrize("max_tokens", [5, 8, 12, 20, 40])
@pytest.mark.parametrize("overlap", [0, 2, 4])
def test_invariants_on_realistic_text(max_tokens, overlap):
    check_invariants(SAMPLE, words, max_tokens, overlap if overlap < max_tokens else 0)


def test_identifiers_and_decimals_are_not_split_by_sentence_breaking():
    chunks = check_invariants(SAMPLE, words, max_tokens=14)
    joined = "\n".join(chunks)
    for token in ("BMS-3.2.1", "0.75 percent", "E-4471", "e.g. a citric"):
        assert any(token in c for c in chunks), f"{token!r} was cut by a chunk boundary"
    assert joined  # chunks exist


def test_paragraphs_stay_whole_when_they_fit():
    text = "Alpha one. Alpha two.\n\nBeta one. Beta two.\n\nGamma one. Gamma two."
    assert chunk_text(text, words, max_tokens=4) == [
        "Alpha one. Alpha two.",
        "Beta one. Beta two.",
        "Gamma one. Gamma two.",
    ]


def test_windows_line_endings_split_paragraphs():
    text = "First paragraph here.\r\n\r\nSecond paragraph here."
    assert chunk_text(text, words, max_tokens=3) == [
        "First paragraph here.",
        "Second paragraph here.",
    ]


def test_unbroken_50k_character_string_is_cut_to_the_cap():
    rng = random.Random(7)  # unique content: identical passages would give identical chunks
    text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(50_000))
    chunks = check_invariants(text, chars4, max_tokens=256)
    assert len(chunks) == pytest.approx(50_000 / 1024, abs=1)
    assert "".join(chunks) == text


def test_long_sentence_is_cut_at_whitespace_when_possible():
    sentence = " ".join(f"word{i}" for i in range(200)) + "."
    chunks = check_invariants(sentence, words, max_tokens=30)
    assert all(not c.startswith(("ord", "rd")) for c in chunks)  # never starts mid-word
    assert " ".join(chunks).split() == sentence.split()


def test_overlap_repeats_whole_trailing_sentences_within_budget():
    text = " ".join(f"Sentence number {i} is here." for i in range(12))  # 5 words each
    plain = check_invariants(text, words, max_tokens=15, overlap_tokens=0)
    overlapped = check_invariants(text, words, max_tokens=15, overlap_tokens=5)
    assert len(overlapped) > len(plain)
    for before, after in zip(overlapped, overlapped[1:]):
        shared = before.split(". ")[-1]  # last whole sentence of the earlier chunk
        assert after.startswith(shared.rstrip("."))  # repeated at the start of the next chunk


def test_overlap_counts_inside_the_cap_and_never_stalls():
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    # Overlap budget nearly as large as the target: still within the cap, still makes progress.
    chunks = check_invariants(text, words, max_tokens=10, overlap_tokens=9)
    assert all(words(c) <= 10 for c in chunks)


def test_target_below_cap_packs_to_target_but_never_exceeds_cap():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    chunks = check_invariants(text, words, max_tokens=20, target_tokens=10)
    assert max(words(c) for c in chunks) <= 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_tokens": 5, "overlap_tokens": 5},
        {"max_tokens": 5, "overlap_tokens": 9},
        {"max_tokens": 5, "overlap_tokens": -1},
        {"max_tokens": 5, "target_tokens": 6},
        {"max_tokens": 5, "target_tokens": 0},
    ],
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        chunk_text("some text here", words, **kwargs)


def test_cap_smaller_than_one_character_raises_instead_of_looping():
    with pytest.raises(ValueError, match="single character"):
        chunk_text("abcdef", lambda t: 5 * len(t), max_tokens=3)


def test_deterministic_fuzz_over_random_text_and_settings():
    rng = random.Random(1234)
    vocab = ["alpha", "beta", "gamma", "delta", "E-4471", "0.75", "BMS-3.2.1", "x" * 30, "ok"]
    serial = iter(range(10**9))  # makes every word unique so no two passages repeat
    for _ in range(150):
        paragraphs = []
        for _ in range(rng.randint(1, 6)):
            sentences = []
            for _ in range(rng.randint(1, 6)):
                body = " ".join(
                    f"{rng.choice(vocab)}_{next(serial)}" for _ in range(rng.randint(1, 15))
                )
                sentences.append(body.capitalize() + rng.choice([".", "!", "?", ""]))
            paragraphs.append(" ".join(sentences))
        text = rng.choice(["\n\n", "\n \n", "\r\n\r\n", "\n\n\n"]).join(paragraphs)
        counter = rng.choice([words, chars4])
        cap = rng.randint(8, 60) if counter is chars4 else rng.randint(3, 25)
        overlap = rng.randint(0, max(0, cap // 2 - 1))
        check_invariants(text, counter, cap, overlap)
