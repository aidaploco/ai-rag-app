import logging

import pytest
from rag_sdk.providers.base import Chatter, GeneratedResponse, Prompt, SearchResult, TokenUsage
from rag_sdk.providers.llm_reranker import LLMReranker


class ReplyChatter(Chatter):
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[Prompt] = []

    async def generate(self, prompt: Prompt) -> GeneratedResponse:
        self.prompts.append(prompt)
        return GeneratedResponse(self._reply, TokenUsage(1, 1, 2))


def _results(*contents: str) -> list[SearchResult]:
    return [
        SearchResult(content=c, score=1.0 - i / 10, metadata={}) for i, c in enumerate(contents)
    ]


async def _rerank(reply: str, n: int = 3, top_n: int = 3) -> list[str]:
    contents = [f"p{i}" for i in range(1, n + 1)]
    out = await LLMReranker(ReplyChatter(reply)).rerank("q", _results(*contents), top_n)
    return [r.content for r in out]


async def test_reorders_by_model_ranking():
    assert await _rerank("[3, 1, 2]") == ["p3", "p1", "p2"]


async def test_accepts_array_inside_prose_and_code_fence():
    assert await _rerank("Here you go:\n```json\n[2, 3, 1]\n```\nHope it helps") == [
        "p2",
        "p3",
        "p1",
    ]


async def test_omitted_passages_follow_in_original_order():
    assert await _rerank("[3]") == ["p3", "p1", "p2"]


async def test_duplicates_and_out_of_range_are_dropped():
    assert await _rerank("[2, 2, 9, 0, -1, true, 1.5, 3]") == ["p2", "p3", "p1"]


async def test_top_n_cuts_after_ordering():
    assert await _rerank("[3, 2, 1]", top_n=2) == ["p3", "p2"]


async def test_unparseable_reply_keeps_original_order_and_warns(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="rag_sdk.providers.llm_reranker"):
        assert await _rerank("I cannot rank these.") == ["p1", "p2", "p3"]
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


@pytest.mark.parametrize("reply", ["[]", "```json\n[]\n```", "[9, 10]", "[0, -1]"])
async def test_empty_ranking_keeps_original_order_and_logs_info_not_a_warning(
    reply: str, caplog: pytest.LogCaptureFixture
):
    # The model returned a well-formed array with nothing usable: "no passage is relevant".
    with caplog.at_level(logging.INFO, logger="rag_sdk.providers.llm_reranker"):
        assert await _rerank(reply) == ["p1", "p2", "p3"]
    assert [r.levelno for r in caplog.records] == [logging.INFO]


async def test_returns_only_the_given_result_objects():
    results = _results("a", "b", "c")
    out = await LLMReranker(ReplyChatter("[2, 3, 1]")).rerank("q", results, 3)
    assert {id(r) for r in out} == {id(r) for r in results}


async def test_zero_or_one_result_makes_no_chat_call():
    chatter = ReplyChatter("[1]")
    reranker = LLMReranker(chatter)
    assert await reranker.rerank("q", [], 3) == []
    only = _results("a")
    assert await reranker.rerank("q", only, 3) == only
    assert chatter.prompts == []


async def test_prompt_numbers_passages_and_truncates_only_the_prompt():
    chatter = ReplyChatter("[1, 2]")
    results = _results("x" * 50, "short")
    out = await LLMReranker(chatter, max_passage_chars=10).rerank("my query", results, 2)
    user = chatter.prompts[0].user
    assert "Query: my query" in user and "[1] " + "x" * 10 + "\n" in user and "[2] short" in user
    assert out[0].content == "x" * 50  # result itself is not truncated


async def test_rejects_non_positive_top_n():
    with pytest.raises(ValueError):
        await LLMReranker(ReplyChatter("[1]")).rerank("q", _results("a", "b"), 0)
