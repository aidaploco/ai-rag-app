"""pgvector-backed ``Indexer`` / ``Retriever``.

Behaves like the Qdrant provider so the two are interchangeable: scores are cosine *similarity*
(1 - cosine distance, higher is better), filters are AND-equality on metadata keys, and
``Document.content_id`` must be a UUID string.

The caller owns the ``asyncpg`` pool. Vectors and metadata are sent as text and cast in SQL, so
no per-connection codec registration is needed. Search is an exact scan by default; the HNSW
index is opt-in (``create_hnsw_index``) because it applies filters *after* the index search, so
a selective filter can return fewer than ``top_k`` results unless iterative scans are enabled
(pgvector >= 0.8). With ``create_hnsw_index=True`` the searchers enable ``strict_order``
iterative scans, which keep results in distance order and fill ``top_k`` (HNSW itself
remains approximate).
"""

import json
import re
from typing import Any

import asyncpg  # type: ignore[import-untyped]

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

# Max 40 chars: Postgres truncates identifiers at 63, and index names add a suffix of up to 19.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,39}$")
_HNSW_MAX_DIMENSIONS = 2000  # limit of pgvector's `vector` type for HNSW indexes
_ITERATIVE_SCAN_MODES = {"strict_order", "relaxed_order"}


def _table(name: str) -> str:
    """Quote a table name after validating it; identifiers cannot be bound parameters."""
    if not _IDENTIFIER.match(name):
        raise ValueError(
            f"invalid table name {name!r}: letters, digits and underscores, max 40 characters"
        )
    return f'"{name}"'


def _vector(values: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _filter_clause(filters: dict[str, Any] | None, arg: int) -> tuple[str, list[Any]]:
    """AND-equality on metadata keys, as a jsonb containment test."""
    if not filters:
        return "", []
    return f"WHERE metadata @> ${arg}::jsonb", [json.dumps(filters)]


class PgVectorRetriever(Retriever):
    def __init__(
        self,
        pool: asyncpg.Pool,
        table: str,
        embedder: Embedder,
        top_k: int = 5,
        hnsw_iterative_scan: str | None = None,
    ) -> None:
        if hnsw_iterative_scan is not None and hnsw_iterative_scan not in _ITERATIVE_SCAN_MODES:
            raise ValueError(f"hnsw_iterative_scan must be one of {sorted(_ITERATIVE_SCAN_MODES)}")
        self._pool = pool
        self._table = _table(table)
        self._embedder = embedder
        self._top_k = top_k
        self._hnsw_iterative_scan = hnsw_iterative_scan

    @trace()
    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchResults:
        vector = _vector(await self._embedder.embed(query))
        where, params = _filter_clause(filters, arg=2)
        limit_arg = 2 + len(params)
        sql = (
            f"SELECT id, content, metadata, 1 - (embedding <=> $1::vector) AS score "
            f"FROM {self._table} {where} "
            f"ORDER BY embedding <=> $1::vector LIMIT ${limit_arg}"
        )
        limit = top_k if top_k is not None else self._top_k
        async with self._pool.acquire() as conn:
            if self._hnsw_iterative_scan:
                # SET LOCAL lasts for this transaction only; the mode is validated above.
                async with conn.transaction():
                    await conn.execute(
                        f"SET LOCAL hnsw.iterative_scan = {self._hnsw_iterative_scan}"
                    )
                    rows = await conn.fetch(sql, vector, *params, limit)
            else:
                rows = await conn.fetch(sql, vector, *params, limit)
        return [
            SearchResult(
                content=row["content"],
                score=float(row["score"]),
                metadata=json.loads(row["metadata"]),
                content_id=str(row["id"]),
            )
            for row in rows
        ]


class PgVectorIndexer(Indexer):
    def __init__(
        self,
        pool: asyncpg.Pool,
        embedder: Embedder,
        table: str,
        create_hnsw_index: bool = False,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._table_name = table
        self._table = _table(table)
        self._create_hnsw_index = create_hnsw_index

    @trace()
    async def create_or_update_index(self, name: str, schema: Schema) -> None:
        table = _table(name)
        dims = self._embedder.dimensions
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                f"id uuid PRIMARY KEY, content text NOT NULL, "
                f"embedding vector({dims}) NOT NULL, metadata jsonb NOT NULL DEFAULT '{{}}')"
            )
            # `atttypmod` holds the declared dimension; an existing table must agree.
            existing = await conn.fetchval(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = $1::regclass AND attname = 'embedding'",
                table,
            )
            if existing != dims:
                raise ValueError(
                    f"table {name!r} has vector({existing}) but the embedder produces {dims}"
                )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{name}_metadata_idx" '
                f"ON {table} USING gin (metadata jsonb_path_ops)"
            )
            if self._create_hnsw_index:
                if dims > _HNSW_MAX_DIMENSIONS:
                    raise ValueError(f"HNSW supports at most {_HNSW_MAX_DIMENSIONS} dimensions")
                await conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "{name}_embedding_hnsw_idx" '
                    f"ON {table} USING hnsw (embedding vector_cosine_ops)"
                )

    @trace()
    async def upload_documents(self, docs: Documents) -> None:
        if not docs:
            return
        # Last write wins for a repeated ID within the batch (as upsert does in Qdrant); a single
        # multi-row ON CONFLICT would reject it.
        latest = {doc.content_id: doc for doc in docs}
        rows = [
            (d.content_id, d.content, _vector(d.content_vector), json.dumps(d.metadata))
            for d in latest.values()
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"INSERT INTO {self._table} (id, content, embedding, metadata) "
                f"VALUES ($1::uuid, $2, $3::vector, $4::jsonb) "
                f"ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, "
                f"embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata",
                rows,
            )

    async def has_document(self, filter: dict[str, Any]) -> bool:
        return await self.count_documents(filter) > 0

    async def count_documents(self, filter: dict[str, Any]) -> int:
        async with self._pool.acquire() as conn:
            if "content_id" in filter:
                return int(
                    await conn.fetchval(
                        f"SELECT count(*) FROM {self._table} WHERE id = $1::uuid",
                        filter["content_id"],
                    )
                )
            where, params = _filter_clause(filter, arg=1)
            return int(await conn.fetchval(f"SELECT count(*) FROM {self._table} {where}", *params))

    async def existing_content_ids(self, content_ids: list[str]) -> set[str]:
        if not content_ids:
            return set()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT id FROM {self._table} WHERE id = ANY($1::uuid[])", content_ids
            )
        return {str(row["id"]) for row in rows}

    def create_searcher(self, index: str, top_k: int = 5) -> Retriever:
        return PgVectorRetriever(
            pool=self._pool,
            table=index,
            embedder=self._embedder,
            top_k=top_k,
            hnsw_iterative_scan="strict_order" if self._create_hnsw_index else None,
        )
