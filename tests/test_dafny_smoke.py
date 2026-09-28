import os
import shutil

import pytest

from specforge.verifier.dafny import DafnyVerifier


DAFNY = os.environ.get("DAFNY_PATH") or shutil.which("dafny")


@pytest.mark.skipif(DAFNY is None, reason="Dafny is not installed")
def test_real_dafny_verifier():
    source = """method Id(x: int) returns (y: int)
  ensures y == x
{
  y := x;
}
"""
    result = DafnyVerifier(DAFNY, timeout=30).verify(source)
    assert result.success, result.raw
