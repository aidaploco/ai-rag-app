"""Compare a fresh dense-baseline run with the recorded one and fail on a real regression.

    uv run python app/eval/check_baseline.py --new /tmp/new.json \\
        --recorded app/eval/results/dense_baseline.json

Exit codes: 0 no regression, 1 an overall metric dropped by more than TOLERANCE, 2 the two records
are not comparable (different corpus, queries, k values or embedding model), so no verdict is given.

The tolerance is declared here, before any run, and is not meant to be tuned to make a run pass.
With 27 queries one query changing rank moves an overall metric by about 0.03 or more, so 0.02
flags any single-query regression. Per-kind rows have only 6 to 12 queries (one query is 0.08 to
0.17), so they are reported for information and never fail the check.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOLERANCE = 0.02
METRICS = ("recall", "mrr", "ndcg")
MUST_MATCH = ("ks", "queries", "corpus_paragraphs", "embedding_model", "embedding_dimensions")


@dataclass(frozen=True)
class Difference:
    scope: str  # "overall" or a query kind
    metric: str
    k: str
    recorded: float
    new: float

    @property
    def delta(self) -> float:
        return self.new - self.recorded


@dataclass
class Comparison:
    regressions: list[Difference] = field(default_factory=list)
    improvements: list[Difference] = field(default_factory=list)
    informational: list[Difference] = field(default_factory=list)  # per-kind moves past tolerance
    incomparable: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if self.incomparable:
            return 2
        return 1 if self.regressions else 0


def _blocks(record: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    blocks = {"overall": record["overall"]}
    blocks.update({kind: data for kind, data in record.get("by_kind", {}).items()})
    return blocks


def compare(
    new: dict[str, Any], recorded: dict[str, Any], tolerance: float = TOLERANCE
) -> Comparison:
    result = Comparison()
    for key in MUST_MATCH:
        if new.get(key) != recorded.get(key):
            result.incomparable.append(
                f"{key}: recorded {recorded.get(key)!r}, new {new.get(key)!r}"
            )
    if result.incomparable:
        return result

    new_blocks, recorded_blocks = _blocks(new), _blocks(recorded)
    for scope, recorded_block in recorded_blocks.items():
        if scope not in new_blocks:
            result.incomparable.append(f"query kind {scope!r} is missing from the new run")
            continue
        for metric in METRICS:
            for k, recorded_value in recorded_block[metric].items():
                try:
                    new_value = new_blocks[scope][metric][k]
                except KeyError:
                    result.incomparable.append(f"{scope} {metric}@{k} is missing from the new run")
                    continue
                diff = Difference(scope, metric, k, recorded_value, new_value)
                if abs(diff.delta) <= tolerance:
                    continue
                if scope != "overall":
                    result.informational.append(diff)
                elif diff.delta < 0:
                    result.regressions.append(diff)
                else:
                    result.improvements.append(diff)
    return result


def _rows(title: str, diffs: list[Difference]) -> list[str]:
    if not diffs:
        return []
    lines = [f"{title}:", "| scope | metric | recorded | new | change |", "|---|---|---|---|---|"]
    lines += [
        f"| {d.scope} | {d.metric}@{d.k} | {d.recorded} | {d.new} | {d.delta:+.4f} |" for d in diffs
    ]
    return lines + [""]


def render(result: Comparison, tolerance: float = TOLERANCE) -> str:
    if result.incomparable:
        lines = ["NOT COMPARABLE, no verdict given:"] + [f"- {m}" for m in result.incomparable]
        return "\n".join(lines)
    lines = _rows(
        "REGRESSIONS (overall metric dropped by more than the tolerance)", result.regressions
    )
    lines += _rows("Improvements (overall)", result.improvements)
    lines += _rows(
        "Per-kind changes past the tolerance (informational, never failing)", result.informational
    )
    verdict = "FAIL" if result.regressions else "OK"
    lines.append(
        f"{verdict}: tolerance {tolerance}; {len(result.regressions)} overall regression(s)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--recorded", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(json.loads(args.new.read_text()), json.loads(args.recorded.read_text()))
    print(render(result))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
