import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from deepmerge import Merger

from rag_sdk.providers.base import Document, Schema

TContext = TypeVar("TContext")


def content_id_for(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, text))

# Keep base fields on scalar conflict — extensions can add but not overwrite
# content_id, content, or content_vector.
_document_merger = Merger(
    [(list, ["append"]), (dict, ["merge"]), (set, ["union"])],
    ["use_existing"],
    ["use_existing"],
)


class RagSchemaExtension(Generic[TContext], ABC):
    @abstractmethod
    def get_schema_extension(self) -> Schema: ...

    @abstractmethod
    def get_document_extension(
        self, content: str, context: TContext
    ) -> dict[str, Any]: ...


class DocumentProcessor(Generic[TContext]):
    def __init__(self, schema_extension: RagSchemaExtension[TContext]) -> None:
        self._extension = schema_extension

    def process(
        self,
        content: str,
        content_vector: list[float],
        context: TContext,
        page_nr: int,
    ) -> Document:
        content_id = content_id_for(content)
        base: dict[str, Any] = {
            "content_id": content_id,
            "content": content,
            "content_vector": content_vector,
            "metadata": {"page_nr": page_nr},
        }
        extension = self._extension.get_document_extension(content, context)
        merged = _document_merger.merge(base, extension)
        return Document(
            content_id=merged["content_id"],
            content=merged["content"],
            content_vector=merged["content_vector"],
            metadata=merged["metadata"],
        )
