import uuid
from dataclasses import dataclass
from typing import Any

from rag_sdk.providers.base import Documents, Embedder, Indexer, Retriever, Schema
from rag_sdk.workflow.ingestion import IngestionWorkflow
from rag_sdk.workflow.schema_extension import RagSchemaExtension


@dataclass
class DocContext:
    source: str


class SimpleSchemaExtension(RagSchemaExtension[DocContext]):
    def get_schema_extension(self) -> Schema:
        return {"extra_field": "string"}

    def get_document_extension(
        self, content: str, context: DocContext
    ) -> dict[str, Any]:
        return {"metadata": {"source": context.source}}


class StubEmbedder(Embedder):
    @property
    def dimensions(self) -> int:
        return 3

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_chunks(
        self, texts: list[str], batch_size: int = 2048
    ) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class CapturingIndexer(Indexer):
    def __init__(self) -> None:
        self.uploaded: Documents = []
        self.created_index: str | None = None

    async def upload_documents(self, docs: Documents) -> None:
        self.uploaded.extend(docs)

    async def create_or_update_index(self, name: str, schema: Schema) -> None:
        self.created_index = name

    async def has_document(self, filter: dict[str, Any]) -> bool:
        return False

    async def count_documents(self, filter: dict[str, Any]) -> int:
        return 0

    def create_searcher(self, index: str) -> Retriever:
        raise NotImplementedError


def _make_workflow(indexer: CapturingIndexer) -> IngestionWorkflow[DocContext]:
    return IngestionWorkflow(
        embedder=StubEmbedder(),
        indexer=indexer,
        index="test-index",
        schema_extension=SimpleSchemaExtension(),
    )


async def test_ingest_returns_one_doc_per_page():
    indexer = CapturingIndexer()
    docs = await _make_workflow(indexer).ingest(DocContext("a.pdf"), ["p1", "p2", "p3"])
    assert len(docs) == 3


async def test_ingest_uploads_to_indexer():
    indexer = CapturingIndexer()
    await _make_workflow(indexer).ingest(DocContext("a.pdf"), ["hello", "world"])
    assert len(indexer.uploaded) == 2


async def test_content_id_is_deterministic_uuid5():
    indexer = CapturingIndexer()
    docs = await _make_workflow(indexer).ingest(DocContext("a.pdf"), ["hello world"])
    expected = str(uuid.uuid5(uuid.NAMESPACE_OID, "hello world"))
    assert docs[0].content_id == expected


async def test_same_content_produces_same_id():
    indexer = CapturingIndexer()
    wf = _make_workflow(indexer)
    docs1 = await wf.ingest(DocContext("a.pdf"), ["same content"])
    docs2 = await wf.ingest(DocContext("b.pdf"), ["same content"])
    assert docs1[0].content_id == docs2[0].content_id


async def test_extension_metadata_merged_with_base_metadata():
    indexer = CapturingIndexer()
    docs = await _make_workflow(indexer).ingest(DocContext("file.pdf"), ["content"])
    assert docs[0].metadata["source"] == "file.pdf"
    assert docs[0].metadata["page_nr"] == 0


async def test_base_fields_not_overwritten_by_extension():
    class OverwritingExtension(RagSchemaExtension[DocContext]):
        def get_schema_extension(self) -> Schema:
            return {}

        def get_document_extension(
            self, content: str, context: DocContext
        ) -> dict[str, Any]:
            return {"content_id": "hijacked", "metadata": {}}

    indexer = CapturingIndexer()
    wf = IngestionWorkflow(
        embedder=StubEmbedder(),
        indexer=indexer,
        index="test-index",
        schema_extension=OverwritingExtension(),
    )
    docs = await wf.ingest(DocContext("x.pdf"), ["hello"])
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_OID, "hello"))
    assert docs[0].content_id == expected_id


async def test_ensure_index_exists_calls_indexer():
    indexer = CapturingIndexer()
    await _make_workflow(indexer).ensure_index_exists()
    assert indexer.created_index == "test-index"
