from types import SimpleNamespace

import specforge.algorithm as algorithm
from specforge.tfvr import Faithful


SOURCE = """method Id(x: int) returns (y: int)
  ensures y >= 0
{
  y := x;
}
"""


class FakeProvider:
    def complete(self, system, prompt):
        assert "append" not in system.lower() or "ensures" in system
        assert "wrong return" in prompt
        return "ensures y == x"


class FakeVerifier:
    def verify(self, source, spec=None):
        return SimpleNamespace(success=True, failures=[], raw="verified")


def test_append_only_repair(monkeypatch):
    def assess(source, positives, negatives, verifier, source_verified=None):
        if "ensures y == x" in source:
            return Faithful(True, True, True, 1, 1, "faithful")
        return Faithful(
            False,
            True,
            False,
            1,
            1,
            "accepted negative",
            {"kind": "negative_acceptance", "inputs": [1], "wrong": [0]},
        )

    monkeypatch.setattr(algorithm, "assess_faithful", assess)
    result = algorithm.SpecForge(FakeProvider(), FakeVerifier()).run(
        SOURCE,
        [{"inputs": [1], "out_lits": ["1"]}],
        [{"inputs": [1], "out_lits": ["0"]}],
    )
    assert result.status == "success"
    assert result.retained_edits == 1
    assert "ensures y >= 0" in result.source
    assert "ensures y == x" in result.source
    assert "y := x;" in result.source


def test_response_parser_ignores_prose():
    assert algorithm.parse_ensures("try this\nensures y == x\n```") == ["ensures y == x"]

