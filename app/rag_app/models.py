from typing import Any

from pydantic import BaseModel


class IngestResponse(BaseModel):
    pages_ingested: int
    pages_skipped: int


class GenerateRequest(BaseModel):
    query: str
    filters: dict[str, Any] | None = None


class SearchResultModel(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


class TokenUsageModel(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    response: str
    context: list[SearchResultModel]
    token_usage: TokenUsageModel


class E2EResponse(BaseModel):
    response: str
    context: list[SearchResultModel]
    pages_ingested: int
