"""FAISS-backed ``Indexer`` / ``Retriever``: an in-process, exact (flat) vector index.

Behaves like the other providers so they are interchangeable: scores are cosine *similarity*
(vectors are L2-normalised and searched with an inner-product flat index, so the search is exact),
filters are AND-equality on metadata keys and are applied *before* ranking (FAISS id selector),
``content_id`` is any string, and uploading an existing ID overwrites it.

Things to know:

- It is a library, not a server: one process, no concurrent writers, nothing shared. Persistence
  is explicit (``save`` / ``load``); there is no automatic durability.
- Writes mark the index dirty and it is rebuilt on the next search (O(N) per rebuild), so it
  suits corpora that fit in memory and are written in batches, not one-by-one streaming.
- Metadata filtering is done in Python over the stored metadata (FAISS stores only vectors), so a
  filtered search costs O(N) before ranking.
- Search is exhaustive, so it is exact, not approximate; there is no ANN index here.
"""

import json
import math
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np

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


def _same(a: Any, b: Any) -> bool:
    """Equality that does not treat ``True`` as ``1`` (JSON stores them as different values)."""
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def _matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(k in metadata and _same(metadata[k], v) for k, v in filters.items())


class FaissStore:
    """The vectors, contents and metadata of one index. Not thread-safe."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self._ids: list[str] = []
        self._contents: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._vectors = np.empty((0, dimensions), dtype=np.float32)
        self._row: dict[str, int] = {}
        self._index: Any = None  # built lazily; None means dirty

    def __len__(self) -> int:
        return len(self._ids)

    def _normalise(self, vector: list[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (self.dimensions,):
            raise ValueError(f"vector has {arr.size} dimensions, expected {self.dimensions}")
        norm = float(np.linalg.norm(arr))
        if norm == 0 or math.isnan(norm):
            raise ValueError("cannot index a zero or NaN vector (cosine is undefined)")
        return arr / norm

    def upsert(self, docs: Documents) -> None:
        latest = {doc.content_id: doc for doc in docs}  # last write wins within a batch
        rows = {
            cid: self._normalise(d.content_vector) for cid, d in latest.items()
        }  # validate first
        new_vectors = []
        for cid, doc in latest.items():
            if cid in self._row:  # overwrite in place
                row = self._row[cid]
                self._contents[row] = doc.content
                self._metadata[row] = dict(doc.metadata)
                self._vectors[row] = rows[cid]
            else:
                self._row[cid] = len(self._ids)
                self._ids.append(cid)
                self._contents.append(doc.content)
                self._metadata.append(dict(doc.metadata))
                new_vectors.append(rows[cid])
        if new_vectors:
            self._vectors = np.vstack([self._vectors, np.asarray(new_vectors, dtype=np.float32)])
        self._index = None

    def _built(self) -> Any:
        if self._index is None:
            index = faiss.IndexFlatIP(self.dimensions)
            if len(self._ids):
                index.add(self._vectors)
            self._index = index
        return self._index

    def search(
        self, query: list[float], top_k: int, filters: dict[str, Any] | None
    ) -> list[tuple[int, float]]:
        """``(row, cosine)`` pairs, best first, restricted to rows matching ``filters``."""
        if not len(self) or top_k < 1:
            return []
        q = self._normalise(query).reshape(1, -1)
        params = None
        selector = None  # FAISS does not own the selector: keep it alive until search returns
        candidates = len(self)
        if filters:
            allowed = np.asarray(
                [i for i, m in enumerate(self._metadata) if _matches(m, filters)], dtype=np.int64
            )
            if not len(allowed):
                return []
            candidates = len(allowed)
            selector = faiss.IDSelectorBatch(allowed)  # filter inside the search, pre-ranking
            params = faiss.SearchParameters()
            params.sel = selector
        scores, rows = self._built().search(q, min(top_k, candidates), params=params)
        del selector  # only now may it be freed
        return [(int(r), float(s)) for s, r in zip(scores[0], rows[0]) if r != -1]

    def result(self, row: int, score: float) -> SearchResult:
        return SearchResult(
            content=self._contents[row],
            score=score,
            metadata=dict(self._metadata[row]),
            content_id=self._ids[row],
        )

    def count(self, filters: dict[str, Any]) -> int:
        if "content_id" in filters:
            return int(str(filters["content_id"]) in self._row)
        return sum(_matches(m, filters) for m in self._metadata)

    def existing(self, content_ids: list[str]) -> set[str]:
        return {c for c in content_ids if c in self._row}

    def save(self, directory: Path) -> None:
        """Write the index (``index.faiss``) and its records (``records.json``) to ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._built(), str(directory / "index.faiss"))
        records = {
            "dimensions": self.dimensions,
            "ids": self._ids,
            "contents": self._contents,
            "metadata": self._metadata,
        }
        (directory / "records.json").write_text(json.dumps(records), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "FaissStore":
        records = json.loads((directory / "records.json").read_text(encoding="utf-8"))
        store = cls(records["dimensions"])
        index = faiss.read_index(str(directory / "index.faiss"))
        if index.ntotal != len(records["ids"]) or index.d != store.dimensions:
            raise ValueError("index.faiss and records.json do not match")
        store._ids = list(records["ids"])
        store._contents = list(records["contents"])
        store._metadata = list(records["metadata"])
        store._row = {cid: i for i, cid in enumerate(store._ids)}
        store._vectors = (
            index.reconstruct_n(0, index.ntotal)
            if index.ntotal
            else np.empty((0, store.dimensions), dtype=np.float32)
        )
        store._index = index
        return store


class FaissRetriever(Retriever):
    def __init__(self, store: FaissStore, embedder: Embedder, top_k: int = 5) -> None:
        self._store = store
        self._embedder = embedder
        self._top_k = top_k

    @trace()
    async def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> SearchResults:
        vector = await self._embedder.embed(query)
        k = top_k if top_k is not None else self._top_k
        return [
            self._store.result(row, score) for row, score in self._store.search(vector, k, filters)
        ]


class FaissIndexer(Indexer):
    def __init__(self, embedder: Embedder, index: str = "default") -> None:
        self._embedder = embedder
        self._index = index
        self._stores: dict[str, FaissStore] = {}

    @property
    def store(self) -> FaissStore:
        return self._stores[self._index]

    def attach(self, name: str, store: FaissStore) -> None:
        """Use an existing (e.g. loaded) store for ``name``."""
        if store.dimensions != self._embedder.dimensions:
            raise ValueError(
                f"store has {store.dimensions} dimensions, the embedder produces "
                f"{self._embedder.dimensions}"
            )
        self._stores[name] = store

    @trace()
    async def create_or_update_index(self, name: str, schema: Schema) -> None:
        if name not in self._stores:
            self._stores[name] = FaissStore(self._embedder.dimensions)

    @trace()
    async def upload_documents(self, docs: Documents) -> None:
        if docs:
            self.store.upsert(docs)

    async def has_document(self, filter: dict[str, Any]) -> bool:
        return self.store.count(filter) > 0

    async def count_documents(self, filter: dict[str, Any]) -> int:
        return self.store.count(filter)

    async def existing_content_ids(self, content_ids: list[str]) -> set[str]:
        return self.store.existing(content_ids)

    def create_searcher(self, index: str, top_k: int = 5) -> Retriever:
        return FaissRetriever(self._stores[index], self._embedder, top_k)
