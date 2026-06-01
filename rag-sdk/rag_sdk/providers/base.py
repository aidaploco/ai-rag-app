from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class GeneratedResponse:
    content: str
    token_usage: TokenUsage


@dataclass(frozen=True)
class SearchResult:
    content: str
    score: float
    metadata: dict[str, Any]


@dataclass
class Document:
    content_id: str  # UUID5 of content, as string
    content: str
    content_vector: list[float]
    metadata: dict[str, Any]


SearchResults: TypeAlias = list[SearchResult]
Documents: TypeAlias = list[Document]
Schema: TypeAlias = dict[str, Any]


class Embedder(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_chunks(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...


class Chatter(ABC):
    @abstractmethod
    async def generate(self, prompt: Prompt) -> GeneratedResponse: ...


class Retriever(ABC):
    @abstractmethod
    async def retrieve(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> SearchResults: ...


class Indexer(ABC):
    @abstractmethod
    async def upload_documents(self, docs: Documents) -> None: ...

    @abstractmethod
    async def create_or_update_index(self, name: str, schema: Schema) -> None: ...

    @abstractmethod
    async def has_document(self, filter: dict[str, Any]) -> bool: ...

    @abstractmethod
    async def count_documents(self, filter: dict[str, Any]) -> int: ...

    async def existing_content_ids(self, content_ids: list[str]) -> set[str]:
        result = set()
        for cid in content_ids:
            if await self.has_document({"content_id": cid}):
                result.add(cid)
        return result

    @abstractmethod
    def create_searcher(self, index: str) -> Retriever: ...
