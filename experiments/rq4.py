"""RQ4: translate untouched requirements and score the resulting specifications."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import yaml

from experiments.rq_common import LANG_FILES, fraction, read_jsonl, sha256, write_json, write_record
from specforge.providers import make_provider
from specforge.xlang import parser as temporal_parser, smt
from specforge.xlang.domain import infer_enums
from specforge.xlang.monitor import Trace
from specforge.xlang.parse_sva import parse_sva
from specforge.xlang.tfvr_x import judge
from specforge.xlang.tlast import atoms, from_json, to_json, to_str


SYNTAX = {
    "LTL": ("linear temporal logic (LTL)",
            "G (always), F (eventually), X (next), U (until), ! & | -> <->",
            "G(request -> F grant)"),
    "MTL": ("metric temporal logic (MTL)",
            "G (always), F (eventually), U (until), X (next), ! & | -> <->, "
            "and time-bounded G[a,b], F[a,b], U[a,b]",
            "G(prop_1 -> F[0,5] prop_2)"),
    "STL": ("signal temporal logic (STL)",
            "always[a,b], eventually[a,b], until[a,b], since[a,b], not, and, or, ->, "
            "signal predicates such as (speed > 120), rise(p), fall(p)",
            "always ( (speed > 120) -> eventually [0:3] (brake == 1) )"),
    "ptLTL": ("past-time linear temporal logic in NASA FRET's expanded form",
              "H (historically), O (once), Y (previous), Z (weak previous), S (since), "
              "bounded H[a,b] / O[a,b], ! & | ->, arithmetic comparisons with a single =",
              "(H ((Y (mode = takeoff)) -> (thrust > 0.5)))"),
    "SVA": ("a SystemVerilog assertion",
            "assert property (@(posedge clk) ...) with |->, |=>, ##k, ##[a:b], "
            "$past, $rose, $fell, $stable, &&, ||, !, and comparisons",
            "assert property (@(posedge clk) req |-> ##2 gnt );"),
}


def build_prompt(task: dict) -> str:
    full, ops, example = SYNTAX[task["lang"]]
    signature = []
    for name, domain in task["vars"].items():
        kind = domain["kind"]
        if kind == "bool":
            detail = "boolean"
        elif kind == "enum":
            values = [value for value in domain["values"] if value != "__other__"]
            detail = "enumerated, values {%s}" % ", ".join(values)
        else:
            detail = "numeric signal"
        signature.append(f"  - {name} : {detail}")
    return (
        f"You are formalising a requirement as {full}.\n\n"
        f"Requirement (natural language):\n  {task['nl'].strip()}\n\n"
        "You must use exactly these variables, spelled exactly as given:\n"
        + "\n".join(signature) + "\n\n"
        f"Allowed syntax: {ops}\nExample of the expected output format: {example}\n\n"
        "Write the single formula that captures the requirement. Output ONLY the formula "
        "on one line, with no explanation, no code fence, and no surrounding prose."
    )


def clean_response(raw: str) -> str:
    text = (raw or "").strip()
    fenced = re.search(r"```[a-zA-Z]*\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    lines = [line.strip().strip("`").strip() for line in text.splitlines() if line.strip()]
    return max(lines, key=lambda line: (sum(ch in "()[]&|!->" for ch in line), len(line))) if lines else ""


def parse_candidate(task: dict, text: str):
    if task["lang"] == "SVA":
        return parse_sva(text)
    names = set(task["vars"])
    return infer_enums(temporal_parser.parse(
        text, bool_atoms=names, eq_is_cmp=(task["lang"] == "ptLTL")
    ))


def score(task: dict, candidate_text: str, *, model: str, timeout: int, raw: str = "") -> dict:
    record = {
        "task_id": task["task_id"], "lang": task["lang"], "source": task["source"],
        "model": model, "spec_cand_src": candidate_text, "raw": raw,
    }
    if not candidate_text:
        record["parse_error"] = "empty candidate"
        return record
    try:
        candidate = parse_candidate(task, candidate_text)
    except Exception as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return record
    record["spec_cand"] = to_str(candidate)
    record["spec_cand_json"] = to_json(candidate)
    record["unknown_vars"] = sorted(set(atoms(candidate)) - set(task["vars"]))
    try:
        positives = [Trace.from_json(item) for item in task["pos"]]
        negatives = [Trace.from_json(item) for item in task["neg"]]
        record.update(judge(candidate, positives, negatives).to_json())
    except Exception as exc:
        record["witness_error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        relation = smt.relation(
            from_json(task["spec_json"]), candidate, task["vars"],
            task["horizon"], task["dt"], timeout_ms=timeout,
        )
        record["gold"] = relation["verdict"]
    except Exception as exc:
        record["oracle_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return record


def summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if row.get("verified") is not None
              and row.get("faithful") is not None]
    checked = sum(row["verified"] is True for row in scored)
    faithful = sum(row["faithful"] is True for row in scored)
    oracle = [row for row in rows if row.get("gold") not in (None, "unknown")]
    return {
        "attempted": len(rows),
        "parse_fail": sum("parse_error" in row for row in rows),
        "provider_fail": sum("provider_error" in row for row in rows),
        "witness_scored": len(scored),
        "positive_check": fraction(checked, len(scored)),
        "tfvr": fraction(faithful, len(scored)),
        "unfaithful_share_among_accepted": fraction(checked - faithful, checked),
        "oracle_conclusive": len(oracle),
        "equivalence_accuracy": fraction(
            sum(row["gold"] == "equivalent" for row in oracle), len(oracle)
        ),
    }


def report_rows(rows: list[dict], out: Path) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["lang"])].append(row)
    report = {
        "rq": "RQ4", "population": "untouched translated requirements",
        "groups": {
            f"{model}/{lang}": summarize(group)
            for (model, lang), group in sorted(groups.items())
        },
    }
    write_json(out, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/xlang"))
    parser.add_argument("--config", type=Path, default=Path("config/rq4_anthropic.yaml"))
    parser.add_argument("--candidates", type=Path,
                        help="JSONL with task_id and spec_cand_src from an external translator")
    parser.add_argument("--label", default="external", help="model label for candidate input")
    parser.add_argument("--out", type=Path, default=Path("outputs/rq4/translations.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="tasks per language; 0 means all")
    parser.add_argument("--smt-timeout", type=int, default=5000)
    parser.add_argument("--resume", action="store_true", help="append missing model tasks")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    tasks = {}
    hashes = {}
    for filename in LANG_FILES.values():
        dataset = args.data_dir / filename
        hashes[filename] = sha256(dataset)
        for index, task in enumerate(read_jsonl(dataset)):
            if args.limit and index >= args.limit:
                break
            if task["task_id"] in tasks:
                raise ValueError(f"duplicate task_id: {task['task_id']}")
            tasks[task["task_id"]] = task
    if args.out.exists() and not args.resume:
        parser.error(f"output exists: {args.out}; choose another path or --resume")
    previous = list(read_jsonl(args.out)) if args.out.exists() else []
    done = {(row["model"], row["task_id"]) for row in previous
            if "provider_error" not in row}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.out.open(mode, encoding="utf-8") as output:
        if args.candidates:
            for candidate in read_jsonl(args.candidates):
                task = tasks.get(candidate.get("task_id"))
                if task is None:
                    continue
                model = str(candidate.get("model") or args.label)
                key = (model, task["task_id"])
                if key in done:
                    continue
                row = score(task, candidate.get("spec_cand_src", ""), model=model,
                            timeout=args.smt_timeout)
                write_record(output, row)
                done.add(key)
        else:
            config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            model = str(config["llm"]["model"])
            provider = make_provider(config["llm"])
            for task in tasks.values():
                key = (model, task["task_id"])
                if key in done:
                    continue
                try:
                    raw = provider.complete(
                        "Translate natural-language requirements into one formal specification.",
                        build_prompt(task),
                    )
                except Exception as exc:
                    row = {"task_id": task["task_id"], "lang": task["lang"],
                           "source": task["source"], "model": model,
                           "provider_error": f"{type(exc).__name__}: {exc}"[:300]}
                else:
                    row = score(task, clean_response(raw), model=model,
                                timeout=args.smt_timeout, raw=raw)
                    done.add(key)
                write_record(output, row)
                print(f"RQ4 {model}: {task['task_id']}", flush=True)
    latest = {}
    for row in read_jsonl(args.out):
        latest[(row["model"], row["task_id"])] = row
    summary_path = args.out.with_suffix(".summary.json")
    report_rows(list(latest.values()), summary_path)
    print(f"RQ4: {len(latest)} model-task outputs -> {summary_path}")


if __name__ == "__main__":
    main()
