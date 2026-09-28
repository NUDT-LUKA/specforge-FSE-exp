"""Shared input and summary helpers for the five RQ entry points."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


LANG_FILES = {
    "LTL": "ltl.jsonl",
    "MTL": "mtl.jsonl",
    "STL": "stl.jsonl",
    "ptLTL": "ptltl.jsonl",
    "SVA": "sva.jsonl",
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_record(stream, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def require_complete_oracle(records: list[dict]) -> None:
    if not records or any("gold" not in row for row in records):
        raise ValueError("Controlled records with symbolic verdicts are required; run RQ1 first.")
