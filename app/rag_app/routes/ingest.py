from fastapi import APIRouter, File, Form, Request, UploadFile

from rag_app.models import IngestResponse
from rag_app.schema import DocumentContext
from rag_app.services.ingestion_service import IngestionService

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    filename: str = Form(default=""),
) -> IngestResponse:
    content = (await file.read()).decode("utf-8")
    pages = [p.strip() for p in content.split("\n\n") if p.strip()]
    context = DocumentContext(filename=filename or file.filename or "unknown")
    service: IngestionService = request.app.state.ingestion_service
    ingested, skipped = await service.ingest_with_dedup(context, pages)
    return IngestResponse(pages_ingested=ingested, pages_skipped=skipped)
