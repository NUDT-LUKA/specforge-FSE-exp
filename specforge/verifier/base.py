"""验证器抽象接口与结果数据类(设计文档 8.3 ``verifier/base.py``)。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..ir import SrcLoc


# 失败类别:与 feedback.aligner 的映射表对应(设计文档 5.5)。
FAILURE_KINDS = {
    "post_unproved",  # postcondition could not be proved
    "inv_entry",  # invariant could not be proved on entry
    "inv_maintain",  # invariant might not be maintained
    "assert",  # assertion violation
    "decreases",  # decreases might not decrease
    "spec_range",  # index out of range / null in spec
    "other",
}


@dataclass
class Failure:
    kind: str
    location: SrcLoc
    message: str
    counterexample: dict | None = None  # 变量赋值模型
    # 反例/报错中涉及的变量,供 aligner 决定连接哪些变量。
    related_vars: list[str] = field(default_factory=list)


@dataclass
class VerifyResult:
    success: bool
    obligations_total: int = 0
    obligations_discharged: int = 0
    failures: list[Failure] = field(default_factory=list)
    raw: str = ""

    @property
    def discharge_rate(self) -> float:
        if self.obligations_total == 0:
            return 1.0 if self.success else 0.0
        return self.obligations_discharged / self.obligations_total


@runtime_checkable
class Verifier(Protocol):
    """验证器协议。``spec`` 为目标规约(模式 A 给定 S;模式 B 用 ir.contract)。"""

    def verify(self, program_src: str, spec=None) -> VerifyResult: ...
