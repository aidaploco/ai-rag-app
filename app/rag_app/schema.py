from dataclasses import dataclass
from typing import Any

from rag_sdk.providers.base import Schema
from rag_sdk.workflow.schema_extension import RagSchemaExtension


@dataclass
class DocumentContext:
    filename: str


class AppSchemaExtension(RagSchemaExtension[DocumentContext]):
    def get_schema_extension(self) -> Schema:
        return {"filename": "keyword"}

    def get_document_extension(self, content: str, context: DocumentContext) -> dict[str, Any]:
        return {"metadata": {"filename": context.filename}}
