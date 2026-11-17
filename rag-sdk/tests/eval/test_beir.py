import json
from pathlib import Path

import pytest
from rag_sdk.eval.beir import doc_text, load_beir


def _write(tmp_path: Path, corpus, queries, qrels, split="test") -> Path:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(c) for c in corpus) + "\n")
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    lines = ["query-id\tcorpus-id\tscore"] + [f"{q}\t{d}\t{s}" for q, d, s in qrels]
    (tmp_path / "qrels" / f"{split}.tsv").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_doc_text_joins_title_and_body_and_tolerates_no_title():
    assert doc_text("A title", " body text ") == "A title. body text"
    assert doc_text("", "body only") == "body only"


def test_loads_corpus_and_only_the_judged_queries(tmp_path):
    root = _write(
        tmp_path,
        corpus=[{"_id": "1", "title": "T1", "text": "x"}, {"_id": "2", "title": "", "text": "y"}],
        queries=[{"_id": "q1", "text": "first"}, {"_id": "q2", "text": "unjudged"}],
        qrels=[("q1", "1", 1), ("q1", "2", 0)],
    )
    data = load_beir(root)
    assert data.corpus == {"1": "T1. x", "2": "y"}
    assert data.queries == {"q1": "first"}  # q2 has no judgements in this split
    assert data.qrels == {"q1": {"1": 1}}  # score 0 is not relevant


def test_graded_relevance_is_kept(tmp_path):
    root = _write(
        tmp_path,
        [{"_id": "1", "text": "x"}, {"_id": "2", "text": "y"}],
        [{"_id": "q", "text": "z"}],
        [("q", "1", 2), ("q", "2", 1)],
    )
    assert load_beir(root).qrels == {"q": {"1": 2, "2": 1}}


def test_missing_judged_document_raises(tmp_path):
    root = _write(
        tmp_path, [{"_id": "1", "text": "x"}], [{"_id": "q", "text": "z"}], [("q", "9", 1)]
    )
    with pytest.raises(ValueError, match="not in the corpus"):
        load_beir(root)


def test_missing_judged_query_raises(tmp_path):
    root = _write(
        tmp_path, [{"_id": "1", "text": "x"}], [{"_id": "other", "text": "z"}], [("q", "1", 1)]
    )
    with pytest.raises(ValueError, match="not in queries"):
        load_beir(root)


def test_split_selects_the_qrels_file(tmp_path):
    root = _write(
        tmp_path, [{"_id": "1", "text": "x"}], [{"_id": "q", "text": "z"}], [("q", "1", 1)], "train"
    )
    assert load_beir(root, "train").qrels == {"q": {"1": 1}}
    with pytest.raises(FileNotFoundError):
        load_beir(root, "test")
