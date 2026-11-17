import pytest
from rag_sdk.eval.embedding_cache import CachedEmbedder, EmbeddingCache, batch_by_tokens
from rag_sdk.providers.base import Embedder

DIMS = 4


def words(text: str) -> int:
    return len(text.split())


class CountingEmbedder(Embedder):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return DIMS

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_chunks([text]))[0]

    async def embed_chunks(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 2.0, 3.0] for t in texts]


def test_batches_respect_both_limits_and_keep_order():
    texts = ["a b", "c d e", "f", "g h i j", "k"]  # tokens 2,3,1,4,1
    assert list(batch_by_tokens(texts, words, max_tokens=5, max_inputs=10)) == [[0, 1], [2, 3], [4]]
    assert list(batch_by_tokens(texts, words, max_tokens=100, max_inputs=2)) == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_an_oversized_text_gets_its_own_batch_instead_of_vanishing():
    assert list(batch_by_tokens(["a b c d e f", "x"], words, max_tokens=3, max_inputs=9)) == [
        [0],
        [1],
    ]


def test_batching_rejects_bad_limits():
    with pytest.raises(ValueError):
        list(batch_by_tokens(["a"], words, 0, 1))


async def test_each_distinct_text_is_embedded_once_and_order_is_preserved(tmp_path):
    inner = CountingEmbedder()
    cache = EmbeddingCache(tmp_path / "c.db", "model-x", DIMS)
    embedder = CachedEmbedder(inner, cache, words, max_batch_tokens=100, max_batch_inputs=10)
    out = await embedder.embed_chunks(["aa", "b", "aa", "ccc"])
    assert [v[0] for v in out] == [2.0, 1.0, 2.0, 3.0]
    assert sorted(t for call in inner.calls for t in call) == ["aa", "b", "ccc"]  # "aa" once
    again = await embedder.embed_chunks(["ccc", "aa"])
    assert [v[0] for v in again] == [3.0, 2.0]
    assert len(inner.calls) == 1  # all hits: nothing more sent
    assert embedder.texts_embedded == 3 and embedder.requests == 1


async def test_cache_persists_across_instances_and_is_keyed_by_model_and_dims(tmp_path):
    path = tmp_path / "c.db"
    first = CachedEmbedder(CountingEmbedder(), EmbeddingCache(path, "m1", DIMS), words)
    await first.embed("hello")
    inner = CountingEmbedder()
    await CachedEmbedder(inner, EmbeddingCache(path, "m1", DIMS), words).embed("hello")
    assert inner.calls == []  # same model + dims: served from disk
    await CachedEmbedder(inner, EmbeddingCache(path, "m2", DIMS), words).embed("hello")
    assert inner.calls == [["hello"]]  # different model: not reused


async def test_misses_are_sent_in_sequential_bounded_batches(tmp_path):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(
        inner,
        EmbeddingCache(tmp_path / "c.db", "m", DIMS),
        words,
        max_batch_tokens=4,
        max_batch_inputs=3,
    )
    await embedder.embed_chunks([f"w{i} x" for i in range(5)])  # 2 tokens each
    assert [len(c) for c in inner.calls] == [2, 2, 1]


async def test_wrong_dimension_vectors_are_refused_not_cached(tmp_path):
    class BadEmbedder(CountingEmbedder):
        async def embed_chunks(self, texts, batch_size=2048):
            return [[1.0, 2.0] for _ in texts]

    embedder = CachedEmbedder(BadEmbedder(), EmbeddingCache(tmp_path / "c.db", "m", DIMS), words)
    with pytest.raises(ValueError, match="dimensions"):
        await embedder.embed("x")
