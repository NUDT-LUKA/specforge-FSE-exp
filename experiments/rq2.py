"""RQ2: classify controlled specification errors using the bounded oracle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from experiments.rq_common import fraction, read_jsonl, require_complete_oracle, write_json


def summarize(rows: list[dict]) -> dict:
    labels = Counter(row["gold"] for row in rows if row.get("family") != "REF")
    equivalent = labels["equivalent"]
    silent = labels["weaker"]
    loud = labels["stronger"] + labels["incomparable"]
    wrong = silent + loud
    return {
        "candidates": sum(labels.values()),
        "equivalent": equivalent, "silent": silent, "loud": loud,
        "oracle_unknown": labels["unknown"], "non_equivalent": wrong,
        "silent_error_share": fraction(silent, wrong),
        "loud_error_share": fraction(loud, wrong),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("outputs/rq1/controlled_xlang.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/rq2/summary.json"))
    args = parser.parse_args(argv)
    rows = list(read_jsonl(args.records))
    require_complete_oracle(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["lang"]].append(row)
    report = {
        "rq": "RQ2", "population": "oracle-labelled non-equivalent temporal candidates",
        "per_language": {lang: summarize(group) for lang, group in sorted(groups.items())},
        "pooled": summarize(rows),
    }
    write_json(args.out, report)
    print(f"RQ2: {report['pooled']['silent']}/{report['pooled']['non_equivalent']} "
          f"silent errors -> {args.out}")


if __name__ == "__main__":
    main()
