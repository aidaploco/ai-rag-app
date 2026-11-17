import pytest

tiktoken = pytest.importorskip("tiktoken")

from rag_sdk.chunking import chunk_text  # noqa: E402
from rag_sdk.tokenizer import openai_token_counter  # noqa: E402

MODEL = "text-embedding-3-small"


def test_counts_match_tiktoken_for_the_models_encoding():
    count = openai_token_counter(MODEL)
    enc = tiktoken.encoding_for_model(MODEL)
    assert enc.name == "cl100k_base"
    for text in ["hello world", "E-4471 and BMS-3.2.1", "naïve café 日本語", "", " \n\n "]:
        assert count(text) == len(enc.encode(text))
    assert count("hello world") == 2


def test_special_token_lookalikes_are_counted_as_text_not_rejected():
    assert openai_token_counter(MODEL)("a <|endoftext|> b") > 3


def test_unknown_model_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="no known tokenizer"):
        openai_token_counter("definitely-not-a-model")


def test_chunk_cap_holds_with_the_real_tokenizer():
    count = openai_token_counter(MODEL)
    text = " ".join(
        f"Sentence {i} mentions E-{4000 + i} and BMS-3.2.{i} at 0.75 percent." for i in range(300)
    )
    text += "\n\n" + "Ünïcödé 日本語テキスト " * 400  # dense multi-byte text: many tokens per char
    chunks = chunk_text(text, count, max_tokens=64, overlap_tokens=8)
    assert chunks and all(count(c) <= 64 for c in chunks)


def test_regression_trimmed_piece_is_remeasured_under_bpe():
    # Found by fuzzing with the real tokenizer: the piece after a whitespace cut was measured with
    # its leading space but emitted without it, and tokenized to one token over the cap.
    count = openai_token_counter(MODEL)
    text = "本ücé語 😀ébü日語b"
    chunks = chunk_text(text, count, max_tokens=8)
    assert chunks and all(count(c) <= 8 for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks) == text.replace(" ", "")


def test_cap_holds_over_seeded_fuzz_with_the_real_tokenizer():
    import random
    import string

    from rag_sdk.chunking import chunk_spans

    count = openai_token_counter(MODEL)
    rng = random.Random(2026)
    alphabet = string.ascii_letters + string.digits + "     .,;:!?-_/\n\té ü日本語😀"
    for _ in range(400):
        kind = rng.choice(["unbroken", "mixed", "unicode", "spaces"])
        if kind == "unbroken":
            text = "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(20, 400)))
        elif kind == "mixed":
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(30, 400)))
        elif kind == "unicode":
            text = "".join(rng.choice("éü日本語дa😀 abc") for _ in range(rng.randint(10, 250)))
        else:
            text = " ".join(
                " " * rng.randint(0, 4)
                + "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(1, 15)))
                for _ in range(rng.randint(5, 50))
            )
        cap = rng.randint(8, 16)
        spans = chunk_spans(
            text, count, max_tokens=cap, overlap_tokens=rng.randint(0, cap // 2 - 1)
        )
        covered = [False] * len(text)
        for a, b in spans:
            chunk = text[a:b]
            assert chunk and chunk == chunk.strip() and count(chunk) <= cap, (cap, repr(chunk))
            for i in range(a, b):
                covered[i] = True
        assert all(covered[i] or ch.isspace() for i, ch in enumerate(text))
