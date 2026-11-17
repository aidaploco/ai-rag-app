"""Paired-bootstrap comparison of the arms in results/beir_scifact.json (per-query nDCG@10).

    uv run python app/eval/compare_arms.py

Reads only the committed record (per-query rows are keyed by query ID, so no dataset is needed)
and writes results/beir_scifact_comparisons.json. The record's rerank arm is built on top of
dense_opensearch, so it is compared with that arm; the others are compared with dense_exact.
"""

import json
from pathlib import Path

from rag_sdk.eval.stats import compare_paired

HERE = Path(__file__).parent
RECORD = HERE / "results" / "beir_scifact.json"
OUT = HERE / "results" / "beir_scifact_comparisons.json"


def main() -> None:
    record = json.loads(RECORD.read_text())
    rows = record["per_query"]
    qids = list(rows["dense_exact"])

    def ndcg(arm: str) -> list[float]:
        return [rows[arm][q]["ndcg"]["10"] for q in qids]

    pairs = [("dense_exact", "bm25_opensearch"), ("dense_exact", "hybrid_rrf")]
    pairs += [("dense_opensearch", arm) for arm in rows if arm.startswith("dense_rerank_run")]
    result = {}
    print(f"paired bootstrap on per-query nDCG@10, {len(qids)} queries, 10,000 resamples, seed 0")
    for a, b in pairs:
        c = compare_paired(ndcg(a), ndcg(b))
        result[f"{b}_minus_{a}"] = {
            "mean_difference": round(c.mean_difference, 4),
            "ci95": [round(c.ci_low, 4), round(c.ci_high, 4)],
            "significant": c.significant,
            "queries_better": c.b_better,
            "queries_worse": c.a_better,
            "ties": c.ties,
        }
        print(
            f"  {b} - {a}: {c.mean_difference:+.4f}  95% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}]  "
            f"better {c.b_better} / worse {c.a_better} / tied {c.ties}  "
            f"{'significant' if c.significant else 'not significant'}"
        )
    OUT.write_text(
        json.dumps({"metric": "ndcg@10", "queries": len(qids), **result}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
