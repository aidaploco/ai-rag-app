from fastapi import APIRouter, Request

from rag_app.models import GenerateRequest, GenerateResponse, SearchResultModel, TokenUsageModel
from rag_app.services.rag_service import RagService

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    service: RagService = request.app.state.rag_service
    result = await service.answer(body.query, body.filters)
    return GenerateResponse(
        response=result.response,
        context=[
            SearchResultModel(
                content=r.content, score=r.score, metadata=r.metadata
            )
            for r in result.context
        ],
        token_usage=TokenUsageModel(
            prompt_tokens=result.token_usage.prompt_tokens,
            completion_tokens=result.token_usage.completion_tokens,
            total_tokens=result.token_usage.total_tokens,
        ),
    )
