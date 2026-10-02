"""完整性守卫:确保修复过程没有 **偷改另一边**(设计文档 v2 §6)。

动机
----
验证通过本身不能证明修复保留了给定规约和可执行代码。删除原有后置条件或
改动方法体可能改变任务,却仍使验证通过。本模块单独检查这些保留条件:

* **G1**:产出必须保留全部给定的 ``requires`` / ``ensures`` / ``modifies``;
* **G2**:未被裁决为 CODE 时,可执行代码必须逐字符不变;
* **G3**:裁决为 CODE 时可改代码,但不得改动 ``S``。

违反守卫的产出不计为守卫通过。具体评测率由调用方按相应协议计算。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .util import normalize_clause

# 契约子句(给定规约 S 的载体)
_SPEC_RE = re.compile(r"^[ \t]*(requires|ensures|modifies)[ \t]+(.*\S)[ \t]*$", re.M)
# 全部标注行:允许被增删的部分(证明提示,不属于 S)
_ANNOT_RE = re.compile(
    r"^[ \t]*(?:requires|ensures|modifies|decreases|invariant|assert|reads)\b.*$", re.M)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)


def _canon(expr: str) -> str:
    """规约子句的规范形:**去掉全部空白**。

    仅折叠空白(``normalize_clause``)是不够的:模型常把 ``x==191/7`` 重排版成
    ``x == 191 / 7``,二者语义完全相同,却可能被误判为"删了一条规约",
    导致守卫拒绝有效候选。
    这里对两侧施加同一规范化,只比较去空白后的字面。
    """
    return re.sub(r"\s+", "", expr)


def spec_clauses(src: str) -> set[str]:
    """提取给定规约的规范化子句集合(``关键字:表达式``,空白无关)。"""
    return {f"{kw}:{_canon(body)}" for kw, body in _SPEC_RE.findall(src)}


def executable_skeleton(src: str) -> str:
    """去掉注释与全部标注行后的可执行骨架(空白归一),用于判断代码是否被改。

    ``decreases`` / ``invariant`` / ``assert`` 属于证明提示,**允许**增删;
    ``requires`` / ``ensures`` / ``modifies`` 由 G1 单独把关。
    """
    s = _BLOCK_COMMENT_RE.sub("", src)
    s = _LINE_COMMENT_RE.sub("", s)
    s = _ANNOT_RE.sub("", s)
    return re.sub(r"\s+", "", s)


@dataclass
class GuardVerdict:
    ok: bool
    dropped_spec: list[str] = field(default_factory=list)   # 被删/改写的 S 子句
    code_changed: bool = False
    code_edit_authorized: bool = False                      # 归责为 CODE 时才为真
    reasons: list[str] = field(default_factory=list)

    @property
    def cheated(self) -> bool:
        """是否"靠削弱制品换取通过"。

        经裁决授权的代码修改属于正当修复,不算作弊;删改给定规约永远算。
        """
        return bool(self.dropped_spec) or (self.code_changed
                                           and not self.code_edit_authorized)


def check_integrity(original: str, produced: str,
                    allow_code_edit: bool = False) -> GuardVerdict:
    """比对产出与原始程序,返回守卫裁定。

    :param allow_code_edit: 仅当归责裁决为 CODE 时置 True(G3)。
    """
    v = GuardVerdict(ok=True, code_edit_authorized=allow_code_edit)

    # ---- G1:给定规约不得丢失或被改写 ----
    lost = spec_clauses(original) - spec_clauses(produced)
    if lost:
        v.ok = False
        v.dropped_spec = sorted(lost)
        v.reasons.append(f"G1 违反:丢失/改写 {len(lost)} 条给定规约子句")

    # ---- G2 / G3:可执行代码 ----
    if executable_skeleton(original) != executable_skeleton(produced):
        v.code_changed = True
        if not allow_code_edit:
            v.ok = False
            v.reasons.append("G2 违反:未获授权却改动了可执行代码")

    return v


def guarded_success(original: str, produced: str, verified: bool,
                    allow_code_edit: bool = False) -> tuple[bool, GuardVerdict]:
    """把"验证通过"收紧为"验证通过 **且** 未破坏制品完整性"。

    返回 ``(守卫通过, 裁定)``。上层应同时记录原始 ``verified`` 与本结果,
    以便区分验证成功和守卫通过。
    """
    v = check_integrity(original, produced, allow_code_edit=allow_code_edit)
    return (bool(verified) and v.ok), v
