import json
from pathlib import Path

from experiments import rq2, rq3, rq4, rq5_compare
from experiments.rq_common import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_denominators_and_paired_recall():
    rows = [
        {"family": "W", "gold": "weaker", "faithful": False},
        {"family": "S", "gold": "stronger", "faithful": False},
        {"family": "S", "gold": "stronger", "faithful": True},
        {"family": "N", "gold": "equivalent", "faithful": True},
    ]
    rq2_stats = rq2.summarize(rows)
    rq3_stats = rq3.summarize(rows)
    assert rq2_stats["silent_error_share"] == 1 / 3
    assert rq2_stats["non_equivalent"] == 3
    assert rq3_stats["witness_recall_all"] == 2 / 3
    assert rq3_stats["symbolic_check_recall_all"] == 2 / 3
    assert rq3_stats["mcnemar_witness_only"] == 1
    assert rq3_stats["mcnemar_check_only"] == 1
    assert rq3_stats["false_alarms"] == 0


def test_rq4_reference_and_candidate_file(tmp_path):
    task = next(read_jsonl(ROOT / "data/xlang/ltl.jsonl"))
    assert task["spec_ref"] not in rq4.build_prompt(task)
    scored = rq4.score(task, task["spec_ref"], model="fixture", timeout=5000)
    assert scored["gold"] == "equivalent"
    assert scored["verified"] is True
    assert scored["faithful"] is True
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({
        "task_id": task["task_id"], "spec_cand_src": task["spec_ref"]
    }) + "\n", encoding="utf-8")
    output = tmp_path / "translations.jsonl"
    rq4.main(["--data-dir", str(ROOT / "data/xlang"), "--limit", "1",
              "--candidates", str(candidates), "--label", "fixture", "--out", str(output)])
    assert len(list(read_jsonl(output))) == 1
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["groups"]["fixture/LTL"]["equivalence_accuracy"] == 1.0


def test_rq4_provider_entry_point_without_network(tmp_path, monkeypatch):
    first_tasks = [next(read_jsonl(ROOT / "data/xlang" / filename))
                   for filename in ("ltl.jsonl", "mtl.jsonl", "stl.jsonl", "ptltl.jsonl", "sva.jsonl")]
    answers = {rq4.build_prompt(task): task["spec_ref"] for task in first_tasks}

    class OfflineProvider:
        def complete(self, system, prompt):
            return answers[prompt]

    monkeypatch.setattr(rq4, "make_provider", lambda config: OfflineProvider())
    output = tmp_path / "provider.jsonl"
    rq4.main(["--data-dir", str(ROOT / "data/xlang"), "--limit", "1",
              "--config", str(ROOT / "config/anthropic.yaml"), "--out", str(output)])
    rows = list(read_jsonl(output))
    assert len(rows) == 5
    assert all(row.get("gold") == "equivalent" and row.get("faithful") is True
               for row in rows)


def test_rq5_compare_pairs_exact_task(tmp_path):
    rows = [
        {"arm": "controlled", "run": 1, "method": "SpecForge", "task": "a#weak", "tfvr": True},
        {"arm": "controlled", "run": 1, "method": "GivenSpec", "task": "a#weak", "tfvr": False},
        {"arm": "controlled", "run": 1, "method": "SpecForge", "task": "b#weak", "tfvr": False},
        {"arm": "controlled", "run": 1, "method": "GivenSpec", "task": "b#weak", "tfvr": True},
    ]
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "comparison.json"
    rq5_compare.main(["--records", str(records), "--out", str(output)])
    paired = json.loads(output.read_text(encoding="utf-8"))["groups"]["controlled/run-1"]["paired"]["GivenSpec"]
    assert (paired["paired"], paired["wins"], paired["losses"]) == (2, 1, 1)
