"""Loader for datasets in the BEIR layout (https://github.com/beir-cellar/beir).

A dataset directory holds ``corpus.jsonl`` (``_id``, ``title``, ``text``), ``queries.jsonl``
(``_id``, ``text``) and ``qrels/<split>.tsv`` (``query-id``, ``corpus-id``, ``score``). Only the
queries that have judgements in the chosen split are returned.

The data is never bundled with this repo: check each dataset's own licence before using it.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeirDataset:
    corpus: dict[str, str]  # doc id -> text to index (see ``doc_text``)
    queries: dict[str, str]  # query id -> query text, only those with judgements
    qrels: dict[str, dict[str, int]]  # query id -> {doc id: graded relevance > 0}


def doc_text(title: str, text: str) -> str:
    """The text indexed for a document: ``"<title>. <text>"``, or just the text without a title."""
    title, text = title.strip(), text.strip()
    return f"{title}. {text}" if title else text


def load_beir(directory: Path, split: str = "test") -> BeirDataset:
    corpus: dict[str, str] = {}
    with (directory / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                corpus[str(row["_id"])] = doc_text(row.get("title", ""), row.get("text", ""))

    all_queries: dict[str, str] = {}
    with (directory / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                all_queries[str(row["_id"])] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with (directory / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if not row or row[0] == "query-id":
                continue
            query_id, doc_id, score = row[0], row[1], int(row[2])
            if score > 0:
                qrels.setdefault(query_id, {})[doc_id] = score

    missing_docs = {d for rels in qrels.values() for d in rels if d not in corpus}
    if missing_docs:
        raise ValueError(f"{len(missing_docs)} judged documents are not in the corpus")
    missing_queries = [q for q in qrels if q not in all_queries]
    if missing_queries:
        raise ValueError(f"{len(missing_queries)} judged queries are not in queries.jsonl")
    return BeirDataset(corpus, {q: all_queries[q] for q in qrels}, qrels)
