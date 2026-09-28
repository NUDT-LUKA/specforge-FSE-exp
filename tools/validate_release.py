"""Validate the repository payload before upload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "data/dafny/controlled.jsonl": (191, "29ab973359e8038da37ac4c6c49efaa7e00cff497967f17ada25123a96b2b8bf"),
    "data/dafny/nl_primary.jsonl": (191, "adff804d927c40817d4ae1aa3982146498e23d93872fb8a5d302abaf71d0fa3a"),
    "data/dafny/nl_secondary.jsonl": (191, "4b66bb9c15a908ed416a39a209eee0744dc24a5f02b256f4f02114950f94837b"),
    "data/xlang/ltl.jsonl": (200, "016d9ed5e5d92054cb17e389081eafb586eb0bba731d5a7d340acd51a1106e5e"),
    "data/xlang/mtl.jsonl": (200, "9f5927efc86cded2a7f583a0cca41e7cd39f6693a48d641a5e1839275143f1b0"),
    "data/xlang/ptltl.jsonl": (151, "09187d11d3a97bc077a9b46553ec58b39dd2ebd1ef810a516e67502aee6ee038"),
    "data/xlang/stl.jsonl": (200, "3ba11e2a4d89bf27d1b11af406f8e70c9ef02ce7c0723691bdffceb44018b8ad"),
    "data/xlang/sva.jsonl": (200, "65638ee11f55851fe567b38a2c66d179459e5d0db4c4b795882a0c6e6632f665"),
}


def fail(message: str):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main():
    for relative, (expected_lines, expected_hash) in DATASETS.items():
        path = ROOT / relative
        if not path.is_file():
            fail(f"missing dataset: {relative}")
        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        actual_lines = sum(1 for line in payload.splitlines() if line.strip())
        if (actual_lines, actual_hash) != (expected_lines, expected_hash):
            fail(f"dataset mismatch: {relative}")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL at {relative}:{number}: {exc}")

    forbidden_names = re.compile(
        r"(^|/)(results?|outputs?|logs?)(/|$)|\.partial\.jsonl$|\.log$",
        re.IGNORECASE,
    )
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False
        )
    except OSError:
        tracked = None
    if tracked is not None and tracked.returncode == 0:
        files = [ROOT / name.decode("utf-8") for name in tracked.stdout.split(b"\0") if name]
    else:
        files = [path for path in ROOT.rglob("*") if path.is_file()
                 and ".git" not in path.relative_to(ROOT).parts]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if forbidden_names.search(relative):
            fail(f"generated output file present: {relative}")

    secret_pattern = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
    for path in files:
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if secret_pattern.search(text):
            fail(f"credential-like token present: {path.relative_to(ROOT)}")

    adapter_files = [path for path in (ROOT / "experiments/adapters").rglob("*.py")
                     if path.name != "__init__.py"]
    if adapter_files:
        fail("comparison implementation found under experiments/adapters")

    controlled = [
        json.loads(line)
        for line in (ROOT / "data/dafny/controlled.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if sum(len(row["positives"]) for row in controlled) != 576:
        fail("controlled positive-witness count changed")
    if sum(len(row["negatives"]) for row in controlled) != 1017:
        fail("controlled negative-witness count changed")

    print(f"release validation passed: {len(files)} files, {len(DATASETS)} datasets")


if __name__ == "__main__":
    main()
