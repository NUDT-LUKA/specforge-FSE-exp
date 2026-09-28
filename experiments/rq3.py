"""RQ3: witness-verdict recall, false alarms and witness-budget sensitivity."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

from experiments.rq_common import LANG_FILES, fraction, read_jsonl, require_complete_oracle, write_json
from specforge.xlang.monitor import Trace
from specforge.xlang.tfvr_x import judge
from specforge.xlang.tlast import from_json


def exact_mcnemar_log10p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 0.0
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + n * math.log(0.5) for i in range(min(b, c) + 1)]
    peak = max(terms)
    logp = min(0.0, math.log(2.0) + peak + math.log(sum(math.exp(x - peak) for x in terms)))
    return logp / math.log(10.0)


def summarize(rows: list[dict]) -> dict:
    labelled = [row for row in rows if row.get("family") != "REF"
                and row.get("gold") not in (None, "unknown")
                and row.get("faithful") is not None]
    equivalent = [row for row in labelled if row["gold"] == "equivalent"]
    wrong = [row for row in labelled if row["gold"] != "equivalent"]
    silent = [row for row in wrong if row["gold"] == "weaker"]
    caught = lambda row: row["faithful"] is False
    check_caught = lambda row: row["gold"] != "weaker"
    b = sum(caught(row) and not check_caught(row) for row in wrong)
    c = sum(check_caught(row) and not caught(row) for row in wrong)
    return {
        "oracle_labelled": len(labelled), "non_equivalent": len(wrong),
        "silent_errors": len(silent),
        "witness_recall_all": fraction(sum(map(caught, wrong)), len(wrong)),
        "symbolic_check_recall_all": fraction(sum(map(check_caught, wrong)), len(wrong)),
        "witness_recall_silent": fraction(sum(map(caught, silent)), len(silent)),
        "false_alarms": sum(caught(row) for row in equivalent),
        "equivalent_candidates": len(equivalent),
        "mcnemar_witness_only": b, "mcnemar_check_only": c,
        "mcnemar_log10_p": exact_mcnemar_log10p(b, c),
    }


def witness_budget(rows: list[dict], data_dir: Path, caps=(1, 2, 3, 4, 6)) -> dict:
    tasks = {}
    for filename in LANG_FILES.values():
        for task in read_jsonl(data_dir / filename):
            tasks[task["task_id"]] = task
    silent = [row for row in rows if row.get("family") != "REF"
              and row.get("gold") == "weaker" and row.get("faithful") is not None]
    detected = {cap: 0 for cap in caps}
    for row in silent:
        task = tasks[row["task_id"]]
        candidate = from_json(row["spec_cand_json"])
        for cap in caps:
            positives = [Trace.from_json(item) for item in task["pos"][:cap]]
            negatives = [Trace.from_json(item) for item in task["neg"][:cap]]
            if judge(candidate, positives, negatives).faithful is False:
                detected[cap] += 1
    return {str(cap): {"detected": detected[cap], "silent_errors": len(silent),
                       "recall": fraction(detected[cap], len(silent))} for cap in caps}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("outputs/rq1/controlled_xlang.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/xlang"))
    parser.add_argument("--out", type=Path, default=Path("outputs/rq3/summary.json"))
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    rows = list(read_jsonl(args.records))
    require_complete_oracle(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["lang"]].append(row)
    report = {
        "rq": "RQ3", "population": "conclusive witness and symbolic verdicts",
        "per_language": {lang: summarize(group) for lang, group in sorted(groups.items())},
        "pooled": summarize(rows),
    }
    if not args.skip_sensitivity:
        report["witness_budget"] = witness_budget(rows, args.data_dir)
    write_json(args.out, report)
    print(f"RQ3: {report['pooled']['non_equivalent']} labelled errors -> {args.out}")


if __name__ == "__main__":
    main()
