from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"
    vector_store: Literal["qdrant", "pgvector"] = "qdrant"
    postgres_dsn: str | None = None
    pgvector_table: str = "documents"
    # "paragraph" (default) keeps the blank-line split, so content IDs are unchanged.
    # "recursive" uses rag_sdk.chunking with the limits below, counted with the embedding
    # model's tokenizer.
    chunking: Literal["paragraph", "recursive"] = "paragraph"
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 32
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chat_model: str = "gpt-4o-mini"

    @field_validator("qdrant_api_key", mode="before")
    @classmethod
    def _blank_api_key_is_none(cls, value: object) -> object:
        # `.env.example` ships `QDRANT_API_KEY=` (blank); treat that as "no key", not "".
        return value or None

    @model_validator(mode="after")
    def _pgvector_needs_a_dsn(self) -> "Config":
        if self.vector_store == "pgvector" and not self.postgres_dsn:
            raise ValueError("POSTGRES_DSN is required when VECTOR_STORE=pgvector")
        if self.chunking == "recursive" and not (
            0 <= self.chunk_overlap_tokens < self.chunk_max_tokens
        ):
            raise ValueError("CHUNK_OVERLAP_TOKENS must be >= 0 and smaller than CHUNK_MAX_TOKENS")
        return self
