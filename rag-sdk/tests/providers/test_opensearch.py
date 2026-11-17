"""OpenSearch-specific behaviour. The shared contract lives in test_indexer_contract.py."""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import BulkIndexError
from rag_sdk.providers.base import Document
from rag_sdk.providers.opensearch import OpenSearchIndexer, OpenSearchRetriever
from rag_sdk.workflow.schema_extension import content_id_for

from tests.providers.test_indexer_contract import DIMS, HashEmbedder, make_docs


class OtherDimsEmbedder(HashEmbedder):
    @property
    def dimensions(self) -> int:
        return DIMS // 2


@pytest.fixture
async def client() -> AsyncIterator[AsyncOpenSearch]:
    url = os.environ.get("OPENSEARCH_TEST_URL")
    if not url:
        if os.environ.get("CI"):
            pytest.fail("OPENSEARCH_TEST_URL is not set in CI; opensearch tests would not run")
        pytest.skip("OPENSEARCH_TEST_URL not set")
    created = AsyncOpenSearch(hosts=[url])
    yield created
    await created.close()


@pytest.fixture
async def index(client: AsyncOpenSearch) -> AsyncIterator[str]:
    name = f"os_{uuid.uuid4().hex[:12]}"
    yield name
    await client.indices.delete(index=name, ignore_unavailable=True)


async def test_existing_index_with_other_dimensions_is_rejected(client, index):
    await OpenSearchIndexer(client, HashEmbedder(), index).create_or_update_index(index, {})
    with pytest.raises(ValueError, match="dimension 16.*produces 8"):
        await OpenSearchIndexer(client, OtherDimsEmbedder(), index).create_or_update_index(
            index, {}
        )


async def test_wrong_dimension_vector_raises_instead_of_being_dropped(client, index):
    indexer = OpenSearchIndexer(client, HashEmbedder(), index)
    await indexer.create_or_update_index(index, {})
    good = (await make_docs(("fine", {})))[0]
    bad = Document(content_id_for("bad"), "bad", [0.1, 0.2, 0.3], {})  # 3 dims, index wants 16
    with pytest.raises(BulkIndexError):
        await indexer.upload_documents([good, bad])


async def test_metadata_strings_are_exact_keywords(client, index):
    indexer = OpenSearchIndexer(client, HashEmbedder(), index)
    await indexer.create_or_update_index(index, {})
    await indexer.upload_documents(
        await make_docs(("a", {"filename": "My Report v2.pdf"}), ("b", {"filename": "My Report"}))
    )
    # An analysed text field would match "My Report" against both; a keyword matches exactly.
    assert await indexer.count_documents({"filename": "My Report"}) == 1
    assert await indexer.count_documents({"filename": "My Report v2.pdf"}) == 1


async def test_pinned_limit_a_metadata_key_cannot_change_type(client, index):
    # Qdrant and pgvector accept this; OpenSearch rejects the second document. Pinned, not endorsed.
    indexer = OpenSearchIndexer(client, HashEmbedder(), index)
    await indexer.create_or_update_index(index, {})
    await indexer.upload_documents(await make_docs(("first", {"page_nr": 1})))
    with pytest.raises(BulkIndexError):
        await indexer.upload_documents(await make_docs(("second", {"page_nr": "one"})))


@pytest.mark.parametrize("bad", ["", "Upper", "has space", "-lead", "a/b", "a" * 65, "x.y"])
def test_invalid_index_names_are_rejected(bad):
    with pytest.raises(ValueError, match="invalid index name"):
        OpenSearchRetriever(None, bad, HashEmbedder())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid index name"):
        OpenSearchIndexer(None, HashEmbedder(), bad)  # type: ignore[arg-type]
