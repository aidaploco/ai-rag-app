"""FAISS-specific behaviour. The shared contract lives in test_indexer_contract.py."""

import numpy as np
import pytest
from rag_sdk.providers.base import Document
from rag_sdk.providers.faiss import FaissIndexer, FaissStore
from rag_sdk.workflow.schema_extension import content_id_for

from tests.providers.test_indexer_contract import DIMS, HashEmbedder, make_docs


async def _loaded(items) -> FaissIndexer:
    indexer = FaissIndexer(HashEmbedder(), "idx")
    await indexer.create_or_update_index("idx", {})
    await indexer.upload_documents(await make_docs(*items))
    return indexer


async def test_save_and_load_round_trip_preserves_search_results_and_allows_more_writes(tmp_path):
    items = [(f"doc {i}", {"n": i, "group": "a" if i % 2 else "b"}) for i in range(20)]
    original = await _loaded(items)
    before = await original.create_searcher("idx", top_k=5).retrieve(
        "doc 3", filters={"group": "a"}
    )
    original.store.save(tmp_path / "saved")

    restored = FaissIndexer(HashEmbedder(), "idx")
    restored.attach("idx", FaissStore.load(tmp_path / "saved"))
    after = await restored.create_searcher("idx", top_k=5).retrieve("doc 3", filters={"group": "a"})
    assert [(r.content_id, r.content, r.metadata) for r in after] == [
        (r.content_id, r.content, r.metadata) for r in before
    ]
    assert [r.score for r in after] == pytest.approx([r.score for r in before], abs=1e-6)
    assert await restored.count_documents({"group": "b"}) == 10

    await restored.upload_documents(
        await make_docs(("a new doc", {"group": "c"}))
    )  # still writable
    assert await restored.count_documents({"group": "c"}) == 1
    assert (await restored.create_searcher("idx").retrieve("a new doc"))[0].content == "a new doc"


async def test_load_rejects_mismatched_files(tmp_path):
    (await _loaded([("a", {}), ("b", {})])).store.save(tmp_path / "x")
    other = await _loaded([("only one", {})])
    other.store.save(tmp_path / "y")
    (tmp_path / "x" / "records.json").write_text((tmp_path / "y" / "records.json").read_text())
    with pytest.raises(ValueError, match="do not match"):
        FaissStore.load(tmp_path / "x")


async def test_empty_store_saves_and_loads(tmp_path):
    empty = FaissIndexer(HashEmbedder(), "idx")
    await empty.create_or_update_index("idx", {})
    empty.store.save(tmp_path / "e")
    store = FaissStore.load(tmp_path / "e")
    assert len(store) == 0 and store.search([1.0] * DIMS, 3, None) == []


async def test_overwrite_replaces_the_vector_as_well_as_the_payload():
    indexer = await _loaded([("same text", {"v": 1})])
    cid = content_id_for("same text")
    other_vec = await HashEmbedder().embed("a completely different text")
    await indexer.upload_documents([Document(cid, "replaced", other_vec, {"v": 2})])
    top = (await indexer.create_searcher("idx").retrieve("a completely different text"))[0]
    assert (top.content, top.metadata, top.content_id) == ("replaced", {"v": 2}, cid)
    assert top.score == pytest.approx(1.0, abs=1e-5)  # the stored vector really changed
    assert await indexer.count_documents({"v": 1}) == 0 and len(indexer.store) == 1


async def test_top_k_larger_than_the_index_and_empty_index_are_fine():
    indexer = await _loaded([("a", {}), ("b", {}), ("c", {})])
    assert len(await indexer.create_searcher("idx", top_k=50).retrieve("a")) == 3
    empty = FaissIndexer(HashEmbedder(), "e")
    await empty.create_or_update_index("e", {})
    assert await empty.create_searcher("e").retrieve("anything") == []


async def test_filter_matching_nothing_returns_empty_not_an_error():
    indexer = await _loaded([("a", {"k": 1})])
    assert await indexer.create_searcher("idx").retrieve("a", filters={"k": 2}) == []
    assert await indexer.create_searcher("idx").retrieve("a", filters={"missing": 1}) == []


async def test_filters_do_not_treat_true_as_one():
    indexer = await _loaded([("flag", {"on": True}), ("num", {"on": 1})])
    assert await indexer.count_documents({"on": True}) == 1
    assert await indexer.count_documents({"on": 1}) == 1
    hit = await indexer.create_searcher("idx").retrieve("flag", filters={"on": True})
    assert [h.content for h in hit] == ["flag"]


async def test_bad_vectors_are_rejected_before_anything_is_written():
    indexer = await _loaded([("good", {})])
    good = Document(content_id_for("x"), "x", [0.1] * DIMS, {})
    for bad in ([0.0] * DIMS, [float("nan")] * DIMS, [0.1] * (DIMS - 1)):
        with pytest.raises(ValueError):
            await indexer.upload_documents([good, Document(content_id_for("y"), "y", bad, {})])
    assert len(indexer.store) == 1  # the batch containing a bad vector was not partly applied


async def test_attaching_a_store_with_other_dimensions_is_rejected():
    indexer = FaissIndexer(HashEmbedder(), "idx")
    with pytest.raises(ValueError, match="dimensions"):
        indexer.attach("other", FaissStore(DIMS // 2))
    indexer.attach("fine", FaissStore(DIMS))  # the matching size is accepted


async def test_scores_are_cosine_not_raw_inner_product():
    # Vectors with very different lengths: an un-normalised inner product would exceed 1.
    indexer = FaissIndexer(HashEmbedder(), "idx")
    await indexer.create_or_update_index("idx", {})
    big = Document(content_id_for("big"), "big", [10.0] + [0.0] * (DIMS - 1), {})
    small = Document(content_id_for("small"), "small", [0.001] + [0.0] * (DIMS - 1), {})
    await indexer.upload_documents([big, small])
    scores = [s for _, s in indexer.store.search([5.0] + [0.0] * (DIMS - 1), 2, None)]
    assert scores == pytest.approx([1.0, 1.0], abs=1e-5) and np.all(np.asarray(scores) <= 1.00001)


async def test_filtered_search_matches_numpy_brute_force_under_memory_churn():
    # FAISS does not own the id selector it is handed, so the provider keeps a reference until
    # the search returns. (With faiss-cpu 1.16 the wrapper also retains one, so dropping it did
    # not fail in testing; this guards against a version where that changes.) Hammer filtered
    # searches with allocation churn and compare each against an independent brute-force cosine
    # over the allowed rows.
    import gc

    rng = np.random.default_rng(0)
    indexer = FaissIndexer(HashEmbedder(), "idx")
    await indexer.create_or_update_index("idx", {})
    n = 300
    vectors = rng.normal(size=(n, DIMS)).astype(np.float32)
    docs = [
        Document(content_id_for(f"d{i}"), f"d{i}", vectors[i].tolist(), {"g": i % 3})
        for i in range(n)
    ]
    await indexer.upload_documents(docs)
    store = indexer.store
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    for trial in range(300):
        group = trial % 3
        query = rng.normal(size=DIMS).astype(np.float32)
        junk = [bytearray(4096) for _ in range(50)]  # churn memory a freed selector would reuse
        gc.collect(0)  # young generation only: a full collect costs ~0.5 s in a large process
        got = store.search(query.tolist(), 10, {"g": group})
        del junk
        allowed = np.flatnonzero(np.arange(n) % 3 == group)
        scores = unit[allowed] @ (query / np.linalg.norm(query))
        expected = allowed[np.argsort(-scores, kind="stable")[:10]]
        assert [row for row, _ in got] == expected.tolist()
        assert [s for _, s in got] == pytest.approx(np.sort(scores)[::-1][:10].tolist(), abs=1e-5)
