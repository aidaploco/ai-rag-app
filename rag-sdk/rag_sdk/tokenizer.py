"""Token counting that matches an OpenAI model, for use as ``count_tokens`` in ``rag_sdk.chunking``.

Needs the ``[tokenizer]`` extra (tiktoken). tiktoken downloads an encoding's vocabulary on first
use and caches it (set ``TIKTOKEN_CACHE_DIR`` to control where), so an offline deployment must
warm that cache at build time.
"""

from collections.abc import Callable


def openai_token_counter(model: str) -> Callable[[str], int]:
    """Return a function counting tokens exactly as ``model``'s tokenizer does.

    Raises ``ValueError`` for a model tiktoken does not know, rather than guessing an encoding:
    a wrong guess would make a "hard" token cap silently wrong.
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        raise ValueError(f"no known tokenizer for model {model!r}") from None

    def count(text: str) -> int:
        # Count special-token lookalikes ("<|endoftext|>") as ordinary text, as the API does.
        return len(encoding.encode(text, disallowed_special=()))

    return count
