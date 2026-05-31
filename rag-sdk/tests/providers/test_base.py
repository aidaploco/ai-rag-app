from typing import Any

import pytest

from rag_sdk.providers.base import (
    Chatter,
    Document,
    Documents,
    Embedder,
    GeneratedResponse,
    Indexer,
    Prompt,
    Retriever,
    Schema,
    SearchResult,
    SearchResults,
    TokenUsage,
)


# --- ABC instantiation guards ---

def test_embedder_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Embedder()  # type: ignore[abstract]


def test_chatter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Chatter()  # type: ignore[abstract]


def test_retriever_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]


def test_indexer_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Indexer()  # type: ignore[abstract]


# --- Stub implementations (also serve as usage examples) ---

class StubEmbedder(Embedder):
    @property
    def dimensions(self) -> int:
        return 1536

    async def embed(self, text: str) -> list[float]:
        return [0.1] * self.dimensions

    async def embed_chunks(
        self, texts: list[str], batch_size: int = 2048
    ) -> list[list[float]]:
        return [[0.1] * self.dimensions for _ in texts]


class StubChatter(Chatter):
    async def generate(self, prompt: Prompt) -> GeneratedResponse:
        return GeneratedResponse(
            content="stub response",
            token_usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )


class StubRetriever(Retriever):
    async def retrieve(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> SearchResults:
        return [SearchResult(content="stub result", score=0.9, metadata={})]


class StubIndexer(Indexer):
    async def upload_documents(self, docs: Documents) -> None:
        pass

    async def create_or_update_index(self, name: str, schema: Schema) -> None:
        pass

    async def has_document(self, filter: dict[str, Any]) -> bool:
        return False

    async def count_documents(self, filter: dict[str, Any]) -> int:
        return 0

    def create_searcher(self, index: str) -> Retriever:
        return StubRetriever()


# --- Stub behaviour tests ---

async def test_embedder_returns_vector_of_correct_dimension():
    emb = StubEmbedder()
    assert emb.dimensions == 1536
    assert len(await emb.embed("hello")) == 1536


async def test_embed_chunks_returns_one_vector_per_text():
    emb = StubEmbedder()
    texts = ["one", "two", "three"]
    results = await emb.embed_chunks(texts)
    assert len(results) == len(texts)
    assert all(len(v) == emb.dimensions for v in results)


async def test_chatter_returns_generated_response():
    chatter = StubChatter()
    prompt = Prompt(system="you are helpful", user="hello")
    response = await chatter.generate(prompt)
    assert response.content == "stub response"
    assert response.token_usage.total_tokens == 15


async def test_retriever_returns_search_results():
    retriever = StubRetriever()
    results = await retriever.retrieve("my query")
    assert len(results) == 1
    assert results[0].score == 0.9


async def test_indexer_create_searcher_returns_retriever():
    indexer = StubIndexer()
    searcher = indexer.create_searcher("my-index")
    assert isinstance(searcher, Retriever)


def test_document_is_mutable():
    doc = Document(
        content_id="abc",
        content="hello",
        content_vector=[0.1, 0.2],
        metadata={"page_nr": 1},
    )
    doc.metadata["source"] = "test"
    assert doc.metadata["source"] == "test"


def test_prompt_is_immutable():
    prompt = Prompt(system="sys", user="usr")
    with pytest.raises(Exception):
        prompt.user = "changed"  # type: ignore[misc]
