import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED = {
    "data/dafny/controlled.jsonl": (191, "29ab973359e8038da37ac4c6c49efaa7e00cff497967f17ada25123a96b2b8bf"),
    "data/dafny/nl_primary.jsonl": (191, "adff804d927c40817d4ae1aa3982146498e23d93872fb8a5d302abaf71d0fa3a"),
    "data/dafny/nl_secondary.jsonl": (191, "4b66bb9c15a908ed416a39a209eee0744dc24a5f02b256f4f02114950f94837b"),
    "data/xlang/ltl.jsonl": (200, "016d9ed5e5d92054cb17e389081eafb586eb0bba731d5a7d340acd51a1106e5e"),
    "data/xlang/mtl.jsonl": (200, "9f5927efc86cded2a7f583a0cca41e7cd39f6693a48d641a5e1839275143f1b0"),
    "data/xlang/ptltl.jsonl": (151, "09187d11d3a97bc077a9b46553ec58b39dd2ebd1ef810a516e67502aee6ee038"),
    "data/xlang/stl.jsonl": (200, "3ba11e2a4d89bf27d1b11af406f8e70c9ef02ce7c0723691bdffceb44018b8ad"),
    "data/xlang/sva.jsonl": (200, "65638ee11f55851fe567b38a2c66d179459e5d0db4c4b795882a0c6e6632f665"),
}


def test_dataset_counts_hashes_and_json():
    for relative, (count, digest) in EXPECTED.items():
        payload = (ROOT / relative).read_bytes()
        lines = [line for line in payload.splitlines() if line.strip()]
        assert len(lines) == count
        assert hashlib.sha256(payload).hexdigest() == digest
        for line in lines:
            json.loads(line)


def test_paper_witness_totals():
    path = ROOT / "data/dafny/controlled.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(len(row["positives"]) for row in rows) == 576
    assert sum(len(row["negatives"]) for row in rows) == 1017

