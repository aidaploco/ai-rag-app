from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import asyncpg
from fastapi import FastAPI
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from rag_sdk.providers.base import Embedder, Indexer
from rag_sdk.providers.openai import OpenAIChatter, OpenAIEmbedder
from rag_sdk.providers.pgvector import PgVectorIndexer
from rag_sdk.providers.qdrant import QdrantIndexer
from rag_sdk.workflow.ingestion import IngestionWorkflow
from rag_sdk.workflow.rag import RagWorkflow

from rag_app.config import Config
from rag_app.routes.e2e import router as e2e_router
from rag_app.routes.generate import router as generate_router
from rag_app.routes.ingest import router as ingest_router
from rag_app.schema import AppSchemaExtension
from rag_app.services.ingestion_service import IngestionService
from rag_app.services.rag_service import RagService


async def _build_indexer(
    config: Config, embedder: Embedder, stack: AsyncExitStack
) -> tuple[Indexer, str]:
    """Create the configured vector store; its client/pool is closed when ``stack`` exits."""
    if config.vector_store == "pgvector":
        assert config.postgres_dsn is not None  # enforced by Config
        pool = await asyncpg.create_pool(config.postgres_dsn)
        stack.push_async_callback(pool.close)
        return PgVectorIndexer(pool, embedder, config.pgvector_table), config.pgvector_table

    qdrant_client = AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    stack.push_async_callback(qdrant_client.close)
    return (
        QdrantIndexer(client=qdrant_client, embedder=embedder, collection=config.qdrant_collection),
        config.qdrant_collection,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = Config()

    async with AsyncExitStack() as stack:
        openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        stack.push_async_callback(openai_client.close)
        embedder = OpenAIEmbedder(
            client=openai_client,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        chatter = OpenAIChatter(client=openai_client, model=config.chat_model)
        indexer, index = await _build_indexer(config, embedder, stack)
        ingestion_workflow = IngestionWorkflow(
            embedder=embedder,
            indexer=indexer,
            index=index,
            schema_extension=AppSchemaExtension(),
        )
        rag_workflow = RagWorkflow(retriever=indexer.create_searcher(index), chatter=chatter)

        app.state.ingestion_service = IngestionService(workflow=ingestion_workflow, indexer=indexer)
        app.state.rag_service = RagService(workflow=rag_workflow)

        await ingestion_workflow.ensure_index_exists()
        yield


def create_app(lifespan=lifespan) -> FastAPI:
    application = FastAPI(title="RAG App", lifespan=lifespan)
    application.include_router(ingest_router)
    application.include_router(generate_router)
    application.include_router(e2e_router)
    return application


app = create_app()
