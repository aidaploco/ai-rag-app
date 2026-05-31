from typing import Any

from rag_sdk.workflow.rag import RagResult, RagWorkflow


class RagService:
    def __init__(self, workflow: RagWorkflow) -> None:
        self._workflow = workflow

    async def answer(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> RagResult:
        return await self._workflow.rag(
            query=query,
            user_template="Question: {question}",
            system_template=(
                "You are a helpful assistant. "
                "Answer questions based only on the provided context.\n\n"
                "Context:\n{context}"
            ),
            filters=filters,
        )
