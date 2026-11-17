from typing import Any

from rag_sdk.providers.base import (
    Chatter,
    GeneratedResponse,
    Prompt,
    Retriever,
    SearchResult,
    SearchResults,
    TokenUsage,
)
from rag_sdk.workflow.rag import RagWorkflow


class StubRetriever(Retriever):
    def __init__(self, results: SearchResults) -> None:
        self._results = results
        self.last_filters: dict[str, Any] | None = None

    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchResults:
        self.last_filters = filters
        return self._results


class CapturingChatter(Chatter):
    def __init__(self, response: str = "answer") -> None:
        self._response = response
        self.last_prompt: Prompt | None = None

    async def generate(self, prompt: Prompt) -> GeneratedResponse:
        self.last_prompt = prompt
        return GeneratedResponse(
            content=self._response,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def _result(content: str) -> SearchResult:
    return SearchResult(content=content, score=0.9, metadata={})


async def test_rag_returns_chatter_response():
    workflow = RagWorkflow(
        retriever=StubRetriever([_result("Paris is the capital of France")]),
        chatter=CapturingChatter("The capital is Paris."),
    )
    result = await workflow.rag(
        query="What is the capital of France?",
        user_template="{question}",
        system_template="{context}",
    )
    assert result.response == "The capital is Paris."


async def test_rag_includes_retrieved_context_in_result():
    results = [_result("fact one"), _result("fact two")]
    workflow = RagWorkflow(
        retriever=StubRetriever(results),
        chatter=CapturingChatter(),
    )
    result = await workflow.rag(query="q", user_template="{question}", system_template="{context}")
    assert len(result.context) == 2


async def test_rag_substitutes_question_in_user_template():
    chatter = CapturingChatter()
    workflow = RagWorkflow(retriever=StubRetriever([]), chatter=chatter)
    await workflow.rag(
        query="What is 2+2?",
        user_template="Please answer: {question}",
        system_template="{context}",
    )
    assert chatter.last_prompt is not None
    assert "What is 2+2?" in chatter.last_prompt.user


async def test_rag_substitutes_context_in_system_template():
    chatter = CapturingChatter()
    workflow = RagWorkflow(
        retriever=StubRetriever([_result("chunk A"), _result("chunk B")]),
        chatter=chatter,
    )
    await workflow.rag(
        query="q",
        user_template="{question}",
        system_template="Use this: {context}",
    )
    assert "chunk A" in chatter.last_prompt.system
    assert "chunk B" in chatter.last_prompt.system


async def test_rag_joins_multiple_results_with_newlines():
    chatter = CapturingChatter()
    workflow = RagWorkflow(
        retriever=StubRetriever([_result("first"), _result("second")]),
        chatter=chatter,
    )
    await workflow.rag(query="q", user_template="{question}", system_template="{context}")
    assert "first\n\nsecond" in chatter.last_prompt.system


async def test_rag_passes_filters_to_retriever():
    retriever = StubRetriever([])
    workflow = RagWorkflow(retriever=retriever, chatter=CapturingChatter())
    await workflow.rag(
        query="q",
        user_template="{question}",
        system_template="{context}",
        filters={"category": "science"},
    )
    assert retriever.last_filters == {"category": "science"}


async def test_rag_includes_token_usage_in_result():
    workflow = RagWorkflow(
        retriever=StubRetriever([]),
        chatter=CapturingChatter(),
    )
    result = await workflow.rag(query="q", user_template="{question}", system_template="{context}")
    assert result.token_usage.total_tokens == 15


async def test_rag_with_reranking_retriever_reaches_chatter_in_reranked_order():
    from rag_sdk.providers.base import Reranker
    from rag_sdk.workflow.reranking import RerankingRetriever

    class ReverseReranker(Reranker):
        async def rerank(self, query: str, results: SearchResults, top_n: int) -> SearchResults:
            return list(reversed(results))[:top_n]

    chatter = CapturingChatter()
    retriever = RerankingRetriever(
        StubRetriever([_result("c1"), _result("c2"), _result("c3"), _result("c4")]),
        ReverseReranker(),
        retrieve_k=4,
        top_n=2,
    )
    result = await RagWorkflow(retriever=retriever, chatter=chatter).rag(
        query="q", user_template="{question}", system_template="{context}"
    )
    assert chatter.last_prompt is not None
    assert chatter.last_prompt.system == "c4\n\nc3"
    assert [r.content for r in result.context] == ["c4", "c3"]
