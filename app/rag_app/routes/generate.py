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
        context=[SearchResultModel.model_validate(r) for r in result.context],
        token_usage=TokenUsageModel.model_validate(result.token_usage),
    )
