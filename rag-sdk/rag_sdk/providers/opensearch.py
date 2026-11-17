"""OpenSearch-backed ``Indexer`` / ``Retriever`` (dense kNN, Lucene HNSW engine).

Behaves like the Qdrant and pgvector providers so the three are interchangeable: scores are cosine
*similarity* (higher is better), filters are AND-equality on metadata keys, and ``content_id`` is
the document ``_id``. The caller owns the ``AsyncOpenSearch`` client.

Differences worth knowing:

- Metadata strings are mapped as ``keyword`` (exact match). A metadata key whose type changes
  between documents (a string in one, a number in another) is rejected by OpenSearch; Qdrant and
  pgvector accept that.
- Search is approximate (HNSW); the filter is applied inside the kNN search, not after it.
- Writes use ``refresh=True`` so documents are searchable as soon as ``upload_documents`` returns.
"""

import re
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

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

_INDEX_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _index_name(name: str) -> str:
    if not _INDEX_NAME.match(name):
        raise ValueError(
            f"invalid index name {name!r}: lowercase letters, digits, '_' and '-', max 64 chars"
        )
    return name


def _filter_clauses(filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    """AND-equality on metadata keys."""
    return [{"term": {f"metadata.{key}": value}} for key, value in (filters or {}).items()]


def _to_result(hit: dict[str, Any], score: float) -> SearchResult:
    source = hit["_source"]
    return SearchResult(
        content=source["content"],
        score=score,
        metadata=source.get("metadata", {}),
        content_id=str(hit["_id"]),
    )


class OpenSearchRetriever(Retriever):
    def __init__(
        self,
        client: AsyncOpenSearch,
        index: str,
        embedder: Embedder,
        top_k: int = 5,
    ) -> None:
        self._client = client
        self._index = _index_name(index)
        self._embedder = embedder
        self._top_k = top_k

    @trace()
    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchResults:
        k = top_k if top_k is not None else self._top_k
        knn: dict[str, Any] = {"vector": await self._embedder.embed(query), "k": k}
        clauses = _filter_clauses(filters)
        if clauses:
            knn["filter"] = {"bool": {"filter": clauses}}  # inside kNN: filters during the search
        response = await self._client.search(
            index=self._index,
            body={"size": k, "query": {"knn": {"embedding": knn}}},
        )
        # With space_type cosinesimil the engine reports (1 + cosine) / 2; convert back so scores
        # are cosine similarity like the other stores.
        return [_to_result(h, 2.0 * float(h["_score"]) - 1.0) for h in response["hits"]["hits"]]


class OpenSearchLexicalRetriever(Retriever):
    """BM25 keyword search over ``content`` (default analyzer, default BM25 parameters).

    The score is OpenSearch's raw BM25 score: unbounded, and not comparable with cosine
    similarity. Combine it with a dense retriever through ``HybridRetriever``.
    """

    def __init__(self, client: AsyncOpenSearch, index: str, top_k: int = 5) -> None:
        self._client = client
        self._index = _index_name(index)
        self._top_k = top_k

    @trace()
    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchResults:
        k = top_k if top_k is not None else self._top_k
        response = await self._client.search(
            index=self._index,
            body={
                "size": k,
                "query": {
                    "bool": {
                        "must": {"match": {"content": {"query": query}}},
                        "filter": _filter_clauses(filters),
                    }
                },
            },
        )
        return [_to_result(h, float(h["_score"])) for h in response["hits"]["hits"]]


class OpenSearchIndexer(Indexer):
    def __init__(self, client: AsyncOpenSearch, embedder: Embedder, index: str) -> None:
        self._client = client
        self._embedder = embedder
        self._index = _index_name(index)

    @trace()
    async def create_or_update_index(self, name: str, schema: Schema) -> None:
        index = _index_name(name)
        dims = self._embedder.dimensions
        if not await self._client.indices.exists(index=index):
            await self._client.indices.create(
                index=index,
                body={
                    # One shard, no replicas: stable BM25 statistics and a green single node.
                    "settings": {
                        "index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}
                    },
                    "mappings": {
                        "dynamic_templates": [
                            {
                                "metadata_strings_as_keywords": {
                                    "path_match": "metadata.*",
                                    "match_mapping_type": "string",
                                    "mapping": {"type": "keyword"},
                                }
                            }
                        ],
                        "properties": {
                            "content": {"type": "text"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": dims,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "lucene",
                                },
                            },
                            "metadata": {"type": "object", "dynamic": True},
                        },
                    },
                },
            )
            return
        mapping = await self._client.indices.get_mapping(index=index)
        existing = mapping[index]["mappings"]["properties"]["embedding"]["dimension"]
        if existing != dims:
            raise ValueError(
                f"index {name!r} has dimension {existing} but the embedder produces {dims}"
            )

    @trace()
    async def upload_documents(self, docs: Documents) -> None:
        if not docs:
            return
        # Last write wins for a repeated ID within the batch, as in the other stores.
        latest = {doc.content_id: doc for doc in docs}
        actions = (
            {
                "_op_type": "index",
                "_index": self._index,
                "_id": doc.content_id,
                "_source": {
                    "content": doc.content,
                    "embedding": doc.content_vector,
                    "metadata": doc.metadata,
                },
            }
            for doc in latest.values()
        )
        # Bulk reports per-item failures inside a 200 response; raise_on_error turns them into
        # an exception instead of silently dropping documents.
        await async_bulk(self._client, actions, raise_on_error=True, refresh=True)

    async def has_document(self, filter: dict[str, Any]) -> bool:
        return await self.count_documents(filter) > 0

    async def count_documents(self, filter: dict[str, Any]) -> int:
        if "content_id" in filter:
            query: dict[str, Any] = {"ids": {"values": [str(filter["content_id"])]}}
        else:
            query = {"bool": {"filter": _filter_clauses(filter)}}
        response = await self._client.count(index=self._index, body={"query": query})
        return int(response["count"])

    async def existing_content_ids(self, content_ids: list[str]) -> set[str]:
        if not content_ids:
            return set()
        response = await self._client.mget(
            index=self._index, body={"ids": content_ids}, _source=False
        )
        return {str(doc["_id"]) for doc in response["docs"] if doc.get("found")}

    def create_searcher(self, index: str, top_k: int = 5) -> Retriever:
        return OpenSearchRetriever(self._client, index, self._embedder, top_k)

    def create_lexical_searcher(self, index: str, top_k: int = 5) -> Retriever:
        return OpenSearchLexicalRetriever(self._client, index, top_k)
