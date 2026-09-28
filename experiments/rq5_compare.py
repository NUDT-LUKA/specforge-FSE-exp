"""Pair RQ5 outcomes with independently generated method outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path

from experiments.rq3 import exact_mcnemar_log10p
from experiments.rq_common import fraction, read_jsonl, write_json


def load(path: Path, forced_method: str | None = None) -> list[dict]:
    rows = list(read_jsonl(path))
    for row in rows:
        for field in ("arm", "run", "task", "tfvr"):
            if field not in row:
                raise ValueError(f"{path}: missing {field}")
        if forced_method:
            row["method"] = forced_method
        elif "method" not in row:
            raise ValueError(f"{path}: missing method")
        if row["tfvr"] is not None and not isinstance(row["tfvr"], bool):
            raise ValueError(f"{path}: tfvr must be true, false, or null")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", action="append", type=Path, default=[],
                        help="SpecForge/GivenSpec JSONL; repeat for separate arms")
    parser.add_argument("--external", action="append", default=[], metavar="METHOD=FILE",
                        help="JSONL from a separately installed comparison system")
    parser.add_argument("--weak-only", action="store_true",
                        help="controlled-arm sensitivity with one weak variant per reference task")
    parser.add_argument("--out", type=Path, default=Path("outputs/rq5/comparison.json"))
    args = parser.parse_args(argv)
    rows = []
    for path in args.records or [Path("outputs/rq5/records.jsonl")]:
        rows.extend(load(path))
    for item in args.external:
        if "=" not in item:
            parser.error("--external must be METHOD=FILE")
        label, filename = item.split("=", 1)
        rows.extend(load(Path(filename), label))
    if args.weak_only:
        rows = [row for row in rows if row["arm"] != "controlled"
                or row["task"].endswith("#weak")]
    indexed = {}
    for row in rows:
        key = (row["arm"], row["run"], row["method"], row["task"])
        if key in indexed:
            raise ValueError(f"duplicate method-task outcome: {key}")
        indexed[key] = row
    groups = defaultdict(dict)
    for (arm, run, method, task), row in indexed.items():
        groups[(arm, run)][(method, task)] = row
    report = {}
    for (arm, run), group in sorted(groups.items()):
        methods = sorted({method for method, _ in group})
        section = {"methods": {}, "paired": {}}
        for method in methods:
            scored = [row for (name, _), row in group.items()
                      if name == method and row["tfvr"] is not None]
            section["methods"][method] = {
                "scored": len(scored), "tfvr_count": sum(row["tfvr"] is True for row in scored),
                "tfvr_rate": fraction(sum(row["tfvr"] is True for row in scored), len(scored)),
            }
        if "SpecForge" in methods:
            for method in methods:
                if method == "SpecForge":
                    continue
                pairs = [(a, row) for (name, task), row in group.items()
                         if name == "SpecForge" and (a := group.get((method, task))) is not None
                         and row["tfvr"] is not None and a["tfvr"] is not None]
                for other, sf in pairs:
                    other_hash, sf_hash = other.get("dataset_sha256"), sf.get("dataset_sha256")
                    if other_hash and sf_hash and other_hash != sf_hash:
                        raise ValueError(f"dataset mismatch for {arm}/run-{run}/{method}")
                wins = sum(sf["tfvr"] is True and other["tfvr"] is False
                           for other, sf in pairs)
                losses = sum(sf["tfvr"] is False and other["tfvr"] is True
                             for other, sf in pairs)
                section["paired"][method] = {
                    "paired": len(pairs), "wins": wins, "losses": losses,
                    "mcnemar_log10_p": exact_mcnemar_log10p(wins, losses),
                }
            prior = [(method, item["mcnemar_log10_p"])
                     for method, item in section["paired"].items() if method != "GivenSpec"]
            previous = float("-inf")
            for index, (method, logp) in enumerate(sorted(prior, key=lambda pair: pair[1])):
                adjusted = min(0.0, max(previous, logp + math.log10(len(prior) - index)))
                section["paired"][method]["holm_adjusted_log10_p"] = adjusted
                previous = adjusted
        report[f"{arm}/run-{run}"] = section
    write_json(args.out, {"rq": "RQ5", "groups": report})
    print(f"RQ5 comparison: {args.out}")


if __name__ == "__main__":
    main()
