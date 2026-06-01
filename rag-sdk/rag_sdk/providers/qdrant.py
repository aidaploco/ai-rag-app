from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag_sdk.core.tracing import trace
from rag_sdk.providers.base import (
    Documents,
    Embedder,
    Indexer,
    Retriever,
    Schema,
    SearchResult,
    SearchResults,
)


def _build_filter(filters: dict[str, Any]) -> Filter:
    return Filter(
        must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
    )


class QdrantRetriever(Retriever):
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        embedder: Embedder,
        top_k: int = 5,
        call_options: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._top_k = top_k
        self._call_options = call_options or {}

    @trace()
    async def retrieve(self, query: str, filters: dict[str, Any] | None = None) -> SearchResults:
        query_vector = await self._embedder.embed(query)
        qdrant_filter = _build_filter(filters) if filters else None
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=self._top_k,
            query_filter=qdrant_filter,
            **self._call_options,
        )
        return [
            SearchResult(
                content=hit.payload["content"],
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "content"},
            )
            for hit in hits
        ]


class QdrantIndexer(Indexer):
    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: Embedder,
        collection: str,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection

    @trace()
    async def create_or_update_index(self, name: str, schema: Schema) -> None:
        if not await self._client.collection_exists(name):
            await self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._embedder.dimensions,
                    distance=Distance.COSINE,
                ),
            )

    @trace()
    async def upload_documents(self, docs: Documents) -> None:
        points = [
            PointStruct(
                id=doc.content_id,
                vector=doc.content_vector,
                payload={"content": doc.content, **doc.metadata},
            )
            for doc in docs
        ]
        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    async def has_document(self, filter: dict[str, Any]) -> bool:
        if "content_id" in filter:
            results = await self._client.retrieve(
                collection_name=self._collection,
                ids=[filter["content_id"]],
            )
            return len(results) > 0
        return await self.count_documents(filter) > 0

    async def count_documents(self, filter: dict[str, Any]) -> int:
        if "content_id" in filter:
            results = await self._client.retrieve(
                collection_name=self._collection,
                ids=[filter["content_id"]],
            )
            return len(results)
        result = await self._client.count(
            collection_name=self._collection,
            count_filter=_build_filter(filter),
        )
        return result.count

    async def existing_content_ids(self, content_ids: list[str]) -> set[str]:
        results = await self._client.retrieve(
            collection_name=self._collection,
            ids=content_ids,
        )
        return {str(r.id) for r in results}

    def create_searcher(
        self,
        index: str,
        top_k: int = 5,
        call_options: dict[str, Any] | None = None,
    ) -> Retriever:
        return QdrantRetriever(
            client=self._client,
            collection=index,
            embedder=self._embedder,
            top_k=top_k,
            call_options=call_options,
        )
