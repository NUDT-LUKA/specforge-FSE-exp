"""Run SpecForge on a released Dafny input dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import yaml

from specforge.adapters import ExternalProofRecoverer
from specforge.algorithm import SpecForge
from specforge.providers import make_provider
from specforge.verifier.dafny import DafnyVerifier


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def variants(record: dict, arm: str):
    if arm == "controlled":
        yield record["name"] + "#faithful", record["faithful_src"]
        yield record["name"] + "#weak", record["unfaithful_src"]
    else:
        yield record["name"], record["nl_src"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dafny/controlled.jsonl"))
    parser.add_argument("--arm", choices=["controlled", "nl"], default="controlled")
    parser.add_argument("--config", type=Path, default=Path("config/anthropic.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/specforge.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    provider = make_provider(config["llm"])
    verifier_cfg = config.get("verifier", {})
    verifier = DafnyVerifier(
        verifier_cfg.get("path", "dafny"), int(verifier_cfg.get("timeout", 45))
    )
    proof_cfg = config.get("proof_recovery") or {}
    recoverer = None
    if proof_cfg.get("command"):
        recoverer = ExternalProofRecoverer(
            list(proof_cfg["command"]), int(proof_cfg.get("timeout", 900))
        )
    method = SpecForge(
        provider,
        verifier,
        repair_rounds=int(config.get("repair_rounds", 3)),
        proof_recoverer=recoverer,
    )

    dataset_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as output:
        for record in rows(args.data):
            for task_name, source in variants(record, args.arm):
                if args.limit and count >= args.limit:
                    return
                started = time.perf_counter()
                result = method.run(source, record["positives"], record["negatives"])
                payload = asdict(result)
                payload.update(
                    task=task_name,
                    dataset_sha256=dataset_hash,
                    elapsed_seconds=time.perf_counter() - started,
                )
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output.flush()
                count += 1
                print(f"[{count}] {task_name}: {result.status}", flush=True)


if __name__ == "__main__":
    main()

