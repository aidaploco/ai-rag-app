from fastapi import APIRouter, File, Form, Request, UploadFile

from rag_app.models import E2EResponse, SearchResultModel
from rag_app.schema import DocumentContext
from rag_app.services.ingestion_service import IngestionService
from rag_app.services.rag_service import RagService

router = APIRouter()


@router.post("/", response_model=E2EResponse)
async def e2e(
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(...),
) -> E2EResponse:
    content = (await file.read()).decode("utf-8")
    pages = [p.strip() for p in content.split("\n\n") if p.strip()]
    context = DocumentContext(filename=file.filename or "unknown")

    ingestion_service: IngestionService = request.app.state.ingestion_service
    rag_service: RagService = request.app.state.rag_service

    ingested, _ = await ingestion_service.ingest_with_dedup(context, pages)
    result = await rag_service.answer(query)

    return E2EResponse(
        response=result.response,
        context=[
            SearchResultModel(
                content=r.content, score=r.score, metadata=r.metadata
            )
            for r in result.context
        ],
        pages_ingested=ingested,
    )
