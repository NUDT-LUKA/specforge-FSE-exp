from specforge.guard import check_integrity


ORIGINAL = """method M(x: int) returns (y: int)
  requires x >= 0
  ensures y >= 0
{
  y := x;
}
"""


def test_guard_accepts_appended_postcondition():
    produced = ORIGINAL.replace("  ensures y >= 0", "  ensures y >= 0\n  ensures y == x")
    assert check_integrity(ORIGINAL, produced).ok


def test_guard_rejects_dropped_clause_and_code_edit():
    produced = ORIGINAL.replace("  ensures y >= 0\n", "").replace("y := x", "y := x + 1")
    verdict = check_integrity(ORIGINAL, produced)
    assert not verdict.ok
    assert verdict.dropped_spec
    assert verdict.code_changed

