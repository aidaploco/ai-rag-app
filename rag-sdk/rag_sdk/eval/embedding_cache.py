"""A disk cache for embeddings, so an expensive embedding pass is paid for once.

``CachedEmbedder`` wraps any ``Embedder``. Vectors are keyed by (model key, dimensions, SHA-256 of
the text) in a SQLite file, so changing the model or dimensions never reuses a stale vector.
Misses are sent to the wrapped embedder *sequentially* in batches bounded by both an input count
and a token count (providers cap both per request), instead of one huge concurrent burst.
"""

import hashlib
import sqlite3
import struct
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from rag_sdk.providers.base import Embedder


def batch_by_tokens(
    texts: Sequence[str],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    max_inputs: int,
) -> Iterator[list[int]]:
    """Yield lists of indexes into ``texts``, in order, each within both limits.

    A single text over ``max_tokens`` gets a batch of its own (the provider will reject it, which
    is the right place to find out) rather than being silently dropped.
    """
    if max_tokens < 1 or max_inputs < 1:
        raise ValueError("max_tokens and max_inputs must be >= 1")
    batch: list[int] = []
    used = 0
    for i, text in enumerate(texts):
        n = count_tokens(text)
        if batch and (used + n > max_tokens or len(batch) >= max_inputs):
            yield batch
            batch, used = [], 0
        batch.append(i)
        used += n
    if batch:
        yield batch


class EmbeddingCache:
    def __init__(self, path: Path, model_key: str, dimensions: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (model TEXT, dims INTEGER, sha TEXT, vec BLOB, "
            "PRIMARY KEY (model, dims, sha))"
        )
        self._model, self._dims = model_key, dimensions

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        row = self._db.execute(
            "SELECT vec FROM vectors WHERE model=? AND dims=? AND sha=?",
            (self._model, self._dims, self._sha(text)),
        ).fetchone()
        return list(struct.unpack(f"{self._dims}f", row[0])) if row else None

    def put_many(self, items: Sequence[tuple[str, list[float]]]) -> None:
        for _, vec in items:
            if len(vec) != self._dims:
                raise ValueError(f"vector has {len(vec)} dimensions, expected {self._dims}")
        self._db.executemany(
            "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)",
            [
                (self._model, self._dims, self._sha(t), struct.pack(f"{self._dims}f", *v))
                for t, v in items
            ],
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class CachedEmbedder(Embedder):
    """Embeds each distinct text once, ever. ``texts_embedded`` / ``tokens_embedded`` count what
    was actually sent to the wrapped embedder (cache hits are free)."""

    def __init__(
        self,
        inner: Embedder,
        cache: EmbeddingCache,
        count_tokens: Callable[[str], int],
        max_batch_tokens: int = 100_000,
        max_batch_inputs: int = 256,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._count = count_tokens
        self._max_tokens = max_batch_tokens
        self._max_inputs = max_batch_inputs
        self.texts_embedded = 0
        self.tokens_embedded = 0
        self.requests = 0

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_chunks([text]))[0]

    async def embed_chunks(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        found: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):
            cached = self._cache.get(text)
            if cached is not None:
                found[text] = cached
            else:
                missing.append(text)
        for indexes in batch_by_tokens(missing, self._count, self._max_tokens, self._max_inputs):
            batch = [missing[i] for i in indexes]
            vectors = await self._inner.embed_chunks(batch, batch_size=len(batch))
            if len(vectors) != len(batch):
                raise ValueError("embedder returned a different number of vectors than inputs")
            self._cache.put_many(list(zip(batch, vectors)))
            found.update(zip(batch, vectors))
            self.requests += 1
            self.texts_embedded += len(batch)
            self.tokens_embedded += sum(self._count(t) for t in batch)
        return [found[t] for t in texts]
