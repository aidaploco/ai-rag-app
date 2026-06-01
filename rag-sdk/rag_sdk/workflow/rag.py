from dataclasses import dataclass
from typing import Any, LiteralString

from rag_sdk.core.tracing import trace
from rag_sdk.providers.base import Chatter, Prompt, Retriever, SearchResults, TokenUsage


@dataclass(frozen=True)
class RagResult:
    response: str
    context: SearchResults
    token_usage: TokenUsage


class RagWorkflow:
    def __init__(self, retriever: Retriever, chatter: Chatter) -> None:
        self._retriever = retriever
        self._chatter = chatter

    @trace()
    async def rag(
        self,
        query: str,
        user_template: LiteralString = "Question: {question}",
        system_template: LiteralString = (
            "You are a helpful assistant. "
            "Answer questions based only on the provided context.\n\n"
            "Context:\n{context}"
        ),
        filters: dict[str, Any] | None = None,
    ) -> RagResult:
        results = await self._retriever.retrieve(query, filters)
        context_text = "\n\n".join(r.content for r in results)
        prompt = Prompt(
            system=system_template.replace("{context}", context_text),
            user=user_template.replace("{question}", query),
        )
        response = await self._chatter.generate(prompt)
        return RagResult(
            response=response.content,
            context=results,
            token_usage=response.token_usage,
        )
