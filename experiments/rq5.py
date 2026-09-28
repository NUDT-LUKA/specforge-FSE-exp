"""RQ5: run GivenSpec and SpecForge on the frozen Dafny evaluation arms."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import yaml

from experiments.rq_common import fraction, read_jsonl, sha256, write_json, write_record
from specforge.adapters import ExternalProofRecoverer
from specforge.algorithm import SpecForge
from specforge.guard import guarded_success
from specforge.providers import make_provider
from specforge.tfvr import _verified_success, assess_faithful
from specforge.verifier.dafny import DafnyVerifier


DATASETS = {
    "controlled": Path("data/dafny/controlled.jsonl"),
    "nl-a": Path("data/dafny/nl_primary.jsonl"),
    "nl-b": Path("data/dafny/nl_secondary.jsonl"),
}


def variants(task: dict, arm: str):
    if arm == "controlled":
        yield task["name"] + "#faithful", task["faithful_src"]
        yield task["name"] + "#weak", task["unfaithful_src"]
    else:
        yield task["name"], task["nl_src"]


def run_given(source: str, positives: list, negatives: list, verifier: DafnyVerifier) -> dict:
    verified = _verified_success(verifier.verify(source, None))
    assessment = assess_faithful(
        source, positives, negatives, verifier, source_verified=verified,
    )
    return {
        "source": source, "verified": verified, "guarded": verified,
        "faithful": assessment.faithful,
        "tfvr": verified and assessment.faithful is True,
        "status": "success" if verified and assessment.faithful is True else
                  ("unfaithful" if assessment.faithful is False else "inconclusive"),
        "assessment": asdict(assessment),
    }


def run_specforge(source: str, positives: list, negatives: list,
                  method: SpecForge, verifier: DafnyVerifier) -> dict:
    result = method.run(source, positives, negatives)
    guarded, _ = guarded_success(source, result.source, result.verified, allow_code_edit=False)
    provider_failed = any(event.startswith("provider_error:") for event in result.events)
    return {
        "source": result.source, "verified": result.verified,
        "guarded": guarded, "faithful": result.faithful.faithful,
        "tfvr": None if provider_failed else guarded and result.faithful.faithful is True,
        "status": "provider_error" if provider_failed else result.status,
        "assessment": asdict(result.faithful),
        "repair_attempts": result.repair_attempts,
        "retained_edits": result.retained_edits, "events": result.events,
    }


def summary(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        key = (row["arm"], row["run"], row["method"])
        groups.setdefault(key, []).append(row)
    result = {}
    for (arm, run, method), group in sorted(groups.items()):
        scored = [row for row in group if row["tfvr"] is not None]
        result[f"{arm}/run-{run}/{method}"] = {
            "attempted": len(group), "scored": len(scored),
            "verified": sum(row["verified"] is True for row in scored),
            "guarded": sum(row["guarded"] is True for row in scored),
            "tfvr_count": sum(row["tfvr"] is True for row in scored),
            "guarded_verify_rate": fraction(
                sum(row["guarded"] is True for row in scored), len(scored)
            ),
            "tfvr_rate": fraction(sum(row["tfvr"] is True for row in scored), len(scored)),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["all", *DATASETS], default="all")
    parser.add_argument("--method", choices=["both", "given", "specforge"], default="both")
    parser.add_argument("--config", type=Path, default=Path("config/anthropic.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/rq5/records.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="reference tasks per arm; 0 means all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dafny-path", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-no-proof-recovery", action="store_true",
                        help="development run for NL arms without Phase 1")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.repeats < 1:
        parser.error("--limit must be nonnegative and --repeats must be positive")
    if args.out.exists() and not args.resume:
        parser.error(f"output exists: {args.out}; choose another path or --resume")
    arms = list(DATASETS) if args.arm == "all" else [args.arm]
    methods = ["GivenSpec", "SpecForge"] if args.method == "both" else [
        "GivenSpec" if args.method == "given" else "SpecForge"
    ]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_hash = sha256(args.config)
    verifier_config = config.get("verifier") or {}
    verifier = DafnyVerifier(
        args.dafny_path or verifier_config.get("path", "dafny"),
        int(verifier_config.get("timeout", 45)),
    )
    if not verifier.available:
        parser.error(f"Dafny executable not found: {verifier.dafny_path}")
    method = None
    if "SpecForge" in methods:
        proof_config = config.get("proof_recovery") or {}
        command = proof_config.get("command")
        if any(arm != "controlled" for arm in arms) and not command \
                and not args.allow_no_proof_recovery:
            parser.error("NL arms require proof_recovery.command for the paper's Phase 1 "
                         "protocol; use --allow-no-proof-recovery only for a development run")
        recoverer = ExternalProofRecoverer(
            list(command), int(proof_config.get("timeout", 900))
        ) if command else None
        method = SpecForge(
            make_provider(config["llm"]), verifier,
            repair_rounds=int(config.get("repair_rounds", 3)),
            proof_recoverer=recoverer,
        )
    hashes = {arm: sha256(DATASETS[arm]) for arm in arms}
    previous = list(read_jsonl(args.out)) if args.out.exists() else []
    if any(row.get("dataset_sha256") != hashes.get(row.get("arm")) for row in previous):
        parser.error("existing output has a different dataset or arm")
    if any(row.get("config_sha256") != config_hash for row in previous):
        parser.error("existing output has a different experiment config")
    done = {(row["run"], row["arm"], row["method"], row["task"]) for row in previous}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a" if args.resume else "w", encoding="utf-8") as output:
        for run in range(1, args.repeats + 1):
            for arm in arms:
                for index, task in enumerate(read_jsonl(DATASETS[arm])):
                    if args.limit and index >= args.limit:
                        break
                    for task_name, source in variants(task, arm):
                        for label in methods:
                            key = (run, arm, label, task_name)
                            if key in done:
                                continue
                            before = verifier.calls
                            started = time.perf_counter()
                            if label == "GivenSpec":
                                result = run_given(source, task["positives"], task["negatives"], verifier)
                            else:
                                result = run_specforge(
                                    source, task["positives"], task["negatives"], method, verifier
                                )
                            row = {
                                "run": run, "arm": arm, "method": label, "task": task_name,
                                "dataset_sha256": hashes[arm], "config_sha256": config_hash,
                                "verifier_calls": verifier.calls - before,
                                "elapsed_seconds": time.perf_counter() - started,
                                **result,
                            }
                            write_record(output, row)
                            done.add(key)
                            print(f"RQ5 run {run} {arm} {label} {task_name}: {row['status']}", flush=True)
    rows = list(read_jsonl(args.out))
    report = {"rq": "RQ5", "configuration": {
        "arm": args.arm, "method": args.method, "limit": args.limit,
        "repeats": args.repeats,
    }, "groups": summary(rows)}
    report_path = args.out.with_suffix(".summary.json")
    write_json(report_path, report)
    print(f"RQ5 summary: {report_path}")


if __name__ == "__main__":
    main()
