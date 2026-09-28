"""RQ1: pre-repair check versus witness-faithfulness gaps in six languages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from experiments.rq_common import LANG_FILES, fraction, read_jsonl, sha256, write_json, write_record
from experiments.run_xlang import run_task
from specforge.tfvr import _verified_success, assess_faithful
from specforge.verifier.dafny import DafnyVerifier


def temporal_summary(records: list[dict]) -> dict:
    weak = [row for row in records if row["family"] == "W" and row.get("verified") is not None]
    verified = sum(row["verified"] is True for row in weak)
    faithful = sum(row["faithful"] is True for row in weak)
    symbolic = [row for row in weak if row.get("gold_ref_entails_cand") is not None]
    return {
        "weak_candidates": len(weak),
        "positive_check": fraction(verified, len(weak)),
        "tfvr": fraction(faithful, len(weak)),
        "gap_points": fraction(100 * (verified - faithful), len(weak)),
        "symbolic_check": fraction(
            sum(row["gold_ref_entails_cand"] is True for row in symbolic), len(symbolic)
        ),
        "symbolic_conclusive": len(symbolic),
    }


def run_temporal(data_dir: Path, out: Path, limit: int, k: int, seed: int, timeout: int) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with out.open("w", encoding="utf-8") as stream:
        for lang, filename in LANG_FILES.items():
            dataset = data_dir / filename
            records = []
            for index, task in enumerate(read_jsonl(dataset)):
                if limit and index >= limit:
                    break
                for row in run_task(task, k, True, timeout, seed + index):
                    write_record(stream, row)
                    records.append(row)
            summary[lang] = {"dataset_sha256": sha256(dataset), **temporal_summary(records)}
            summary[lang]["tasks"] = sum(row["family"] == "REF" for row in records)
            print(f"RQ1 {lang}: {summary[lang]['tasks']} tasks, {summary[lang]['weak_candidates']} weakenings")
    return summary


def run_dafny(data: Path, out: Path, limit: int, path: str, timeout: int) -> dict:
    verifier = DafnyVerifier(path, timeout)
    if not verifier.available:
        raise RuntimeError(f"Dafny executable not found: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w", encoding="utf-8") as stream:
        for index, task in enumerate(read_jsonl(data)):
            if limit and index >= limit:
                break
            source = task["unfaithful_src"]
            verified = _verified_success(verifier.verify(source, None))
            assessment = assess_faithful(
                source, task["positives"], task["negatives"], verifier,
                source_verified=verified,
            )
            row = {
                "task": task["name"], "verified": verified,
                "faithful": assessment.faithful,
                "assessment": asdict(assessment),
            }
            write_record(stream, row)
            rows.append(row)
            print(f"RQ1 Dafny: {index + 1} tasks", flush=True)
    checked = sum(row["verified"] for row in rows)
    credited = sum(row["verified"] and row["faithful"] is True for row in rows)
    return {
        "dataset_sha256": sha256(data), "tasks": len(rows),
        "verified": checked, "tfvr_count": credited,
        "verify_rate": fraction(checked, len(rows)), "tfvr": fraction(credited, len(rows)),
        "gap_points": fraction(100 * (checked - credited), len(rows)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/xlang"))
    parser.add_argument("--dafny-data", type=Path, default=Path("data/dafny/controlled.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rq1"))
    parser.add_argument("--limit", type=int, default=0, help="tasks per dataset; 0 means all")
    parser.add_argument("--k", type=int, default=14, help="candidate mutations per temporal task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smt-timeout", type=int, default=5000)
    parser.add_argument("--dafny-path", default="dafny")
    parser.add_argument("--dafny-timeout", type=int, default=45)
    parser.add_argument("--skip-dafny", action="store_true", help="temporal-only smoke run")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.k < 0:
        parser.error("--limit and --k must be nonnegative")
    report = {
        "rq": "RQ1", "configuration": {"limit": args.limit, "k": args.k, "seed": args.seed},
        "temporal": run_temporal(
            args.data_dir, args.out_dir / "controlled_xlang.jsonl",
            args.limit, args.k, args.seed, args.smt_timeout,
        ),
    }
    if not args.skip_dafny:
        report["dafny"] = run_dafny(
            args.dafny_data, args.out_dir / "controlled_dafny.jsonl",
            args.limit, args.dafny_path, args.dafny_timeout,
        )
    write_json(args.out_dir / "summary.json", report)
    print(f"RQ1 summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
