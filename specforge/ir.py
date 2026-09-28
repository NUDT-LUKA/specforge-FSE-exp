"""核心数据结构 ``AnnotIR``(类型化标注中间表示)。

对应设计文档第 4 节。设计原则:
  * 强类型、可序列化为 JSON;
  * 每个叶子可独立定位修复(配合 :mod:`specforge.feedback`);
  * 保留一个 "开放原子" 逃生口 :class:`RawPred`,容纳类型节点表达不了的工业谓词。

为了让整个项目仅依赖标准库即可运行,这里用 ``dataclasses`` 实现结构,并在
:func:`AnnotIR.validate` 中做轻量校验(等价于 pydantic 的"严格解析/拒绝"职责)。
若需要更严格的 schema 校验,可在 :func:`validate` 内接入 pydantic。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from typing import Any, Union

# Expr 在 IR 中以字符串表示目标语言表达式片段(如 "a.Length", "k", "a.Length - k")。
Expr = str


class InvKind(str, Enum):
    """不变量种类(影响提示分槽与 Critic 规则)。"""

    numeric = "numeric"
    relational = "relational"
    shape = "shape"
    quantified = "quantified"
    bound = "bound"


# --------------------------------------------------------------------------- #
# 谓词节点 Pred(类型化,带逃生口)—— 设计文档 4.2
# --------------------------------------------------------------------------- #
@dataclass
class Relational:
    """关系谓词,如 ``x <= y``、``a[i] == m``。"""

    op: str  # "<" | "<=" | "==" | "!=" | ">" | ">="
    lhs: Expr
    rhs: Expr


@dataclass
class Quantified:
    """有界量化谓词:``forall j :: lo <= j < hi ==> body``。"""

    quant: str  # "forall" | "exists"
    var: str
    lo: Expr
    hi: Expr
    body: "Pred"


@dataclass
class Logical:
    """逻辑组合:``and`` | ``or`` | ``implies`` | ``not``。"""

    op: str
    children: list["Pred"]


@dataclass
class HeapPred:
    """分离逻辑/形状谓词,如 ``sorted(a)``、``list(p)``。"""

    kind: str
    args: list[Expr]


@dataclass
class RawPred:
    """开放原子:类型节点表达不了的整段塞进来(对应 SignalL 的 RawSTLAtom)。"""

    raw: str


Pred = Union[Relational, Quantified, Logical, HeapPred, RawPred]


# --------------------------------------------------------------------------- #
# 定位信息
# --------------------------------------------------------------------------- #
@dataclass
class SrcLoc:
    """源码位置(行号 / AST 路径 / 唯一 id)。"""

    line: int = -1
    ast_path: str | None = None
    id: str | None = None


@dataclass
class LoopLoc(SrcLoc):
    """循环定位,语义上同 :class:`SrcLoc`,单独命名以提升可读性。"""


# --------------------------------------------------------------------------- #
# 子句与顶层结构 —— 设计文档 4.1
# --------------------------------------------------------------------------- #
@dataclass
class InvClause:
    pred: Pred
    kind: InvKind = InvKind.relational
    strength_tier: int = 1  # 强度档,调质环可调
    provenance: list[str] = field(default_factory=list)  # 相关变量(静态分析提示)


@dataclass
class Assertion:
    loc: SrcLoc
    pred: Pred


@dataclass
class Lemma:
    name: str
    body: str  # 引理/ghost 以原始片段保留


@dataclass
class LoopAnnot:
    loc: LoopLoc
    invariants: list[InvClause] = field(default_factory=list)
    decreases: list[Expr] = field(default_factory=list)


@dataclass
class Contract:
    requires: list[Pred] = field(default_factory=list)
    ensures: list[Pred] = field(default_factory=list)
    modifies: list[str] = field(default_factory=list)
    decreases: list[Expr] = field(default_factory=list)

    def contract_only(self) -> "Contract":
        """返回仅含本契约的拷贝(调质环用契约去"验收"变异体时使用)。"""
        return Contract(
            requires=list(self.requires),
            ensures=list(self.ensures),
            modifies=list(self.modifies),
            decreases=list(self.decreases),
        )


@dataclass
class AnnotIR:
    function: str
    contract: Contract = field(default_factory=Contract)
    loops: list[LoopAnnot] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    lemmas: list[Lemma] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ----------------------- 序列化 ----------------------- #
    def to_dict(self) -> dict:
        return _to_jsonable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def from_dict(d: dict) -> "AnnotIR":
        return AnnotIR(
            function=d.get("function", ""),
            contract=_contract_from_dict(d.get("contract", {})),
            loops=[_loop_from_dict(x) for x in _as_list(d.get("loops")) if isinstance(x, dict)],
            assertions=[_assertion_from_dict(x) for x in _as_list(d.get("assertions"))],
            lemmas=[Lemma(**x) if isinstance(x, dict) else Lemma(name="", body=str(x))
                    for x in _as_list(d.get("lemmas"))],
            meta=dict(d.get("meta", {})) if isinstance(d.get("meta"), dict) else {},
        )

    @staticmethod
    def from_json(s: str) -> "AnnotIR":
        return AnnotIR.from_dict(json.loads(s))

    # ----------------------- 工具方法 ----------------------- #
    def clone(self) -> "AnnotIR":
        """深拷贝(经 JSON 往返),供调质环做 trial/回滚。"""
        return AnnotIR.from_dict(self.to_dict())

    def without_invariant(self, loop_idx: int, clause_idx: int) -> "AnnotIR":
        """返回去掉指定不变量合取项的新 IR(Houdini 式极小化用)。"""
        c = self.clone()
        del c.loops[loop_idx].invariants[clause_idx]
        return c

    def validate(self) -> list[str]:
        """轻量结构校验,返回问题列表(空列表表示 OK)。

        这是 pydantic 职责的最小实现:保证 LLM 输出可被严格解析/拒绝。
        """
        problems: list[str] = []
        if not self.function:
            problems.append("AnnotIR.function 为空")
        for li, loop in enumerate(self.loops):
            for ci, clause in enumerate(loop.invariants):
                problems.extend(
                    f"loops[{li}].invariants[{ci}]: {m}"
                    for m in _validate_pred(clause.pred)
                )
        for pi, p in enumerate(self.contract.ensures):
            problems.extend(f"contract.ensures[{pi}]: {m}" for m in _validate_pred(p))
        for pi, p in enumerate(self.contract.requires):
            problems.extend(f"contract.requires[{pi}]: {m}" for m in _validate_pred(p))
        return problems


# --------------------------------------------------------------------------- #
# (反)序列化辅助
# --------------------------------------------------------------------------- #
def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        # 过滤掉默认空值以贴近设计文档里的精简 JSON 风格
        out = {}
        for k, v in asdict(obj).items():
            if isinstance(v, (list, dict)) and len(v) == 0 and k in _SKIP_IF_EMPTY:
                continue
            out[k] = _to_jsonable(v)
        return out
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


_SKIP_IF_EMPTY = {"modifies", "decreases", "provenance", "assertions", "lemmas", "meta"}


def pred_from_dict(d: Any) -> Pred:
    """从 dict / str 还原 :data:`Pred`,按字段形状判别节点类型。

    对真实 LLM 输出做健壮化容错(离线 Mock 不触发,但真实实验高频出现):
      * 整段字符串 -> RawPred;
      * InvClause/Assertion 形状(带 ``pred`` 包一层)-> 递归解包内层 pred;
      * ``expr``/``text``/``dafny`` 等别名承载原始表达式 -> RawPred;
      * 结构无法判别时 **降级为 RawPred**(交给验证器判定)而非抛错,
        避免单个畸形谓词中断整轮实验(§7.8 构念效度:字段漂移兜底)。
    """
    if isinstance(d, str):
        return RawPred(raw=d)
    if isinstance(d, (int, float, bool)):
        return RawPred(raw=str(d))
    if isinstance(d, list):
        # LLM 偶尔把合取写成裸列表 -> and 组合
        return Logical(op="and", children=[pred_from_dict(x) for x in d])
    if not isinstance(d, dict):
        raise ValueError(f"无法解析 Pred: {d!r}")
    node = d.get("node") or d.get("type")
    # 量词优先判别:LLM 可能用 "quant" 或 "op" 承载 "forall"/"exists"。必须在下方
    # "pred 解包" 与 "children->Logical" 之前处理,否则:
    #   * {"op":"exists","children":[...]} 会被误建成 Logical(op="exists"),
    #     渲染时 pred_to_dafny 做 {...}["exists"] -> KeyError,中断整轮;
    #   * {"quant":"exists","pred":body} 会被当 InvClause 误解包,丢掉量词。
    _q = d.get("quant") or d.get("op")
    if _q in ("forall", "exists") or node == "quantified":
        quant = _q if _q in ("forall", "exists") else (d.get("quant") or "forall")
        var = d.get("var") or d.get("bound") or d.get("v")
        body = d.get("body")
        if body is None and isinstance(d.get("pred"), (dict, str)):
            body = d["pred"]
        hi = d.get("hi", d.get("high"))
        lo = d.get("lo", d.get("low", 0))
        if var is not None and body is not None and hi is not None:
            return Quantified(quant=quant, var=str(var), lo=str(lo), hi=str(hi),
                              body=pred_from_dict(body))
        # 量词字段不全 -> 降级 RawPred(交验证器判定),绝不误建成 Logical(exists)
        return RawPred(raw=json.dumps(d, ensure_ascii=False))
    # InvClause / Assertion 包了一层 pred:解包内层
    if "pred" in d and isinstance(d["pred"], (dict, str)):
        return pred_from_dict(d["pred"])
    # 显式 tag 优先
    if "raw" in d or node == "raw":
        return RawPred(raw=str(d["raw"]))
    # 常见别名:整段表达式字符串
    for alias in ("expr", "expression", "text", "dafny", "formula", "predicate"):
        if alias in d and isinstance(d[alias], str):
            return RawPred(raw=d[alias])
    if "children" in d or node == "logical":
        op = d.get("op", "and")
        kids = d.get("children") or d.get("operands") or []
        if kids:
            return Logical(op=op, children=[pred_from_dict(c) for c in kids])
    if ("lhs" in d and "rhs" in d) or node == "relational":
        return Relational(op=d.get("op", "=="), lhs=str(d["lhs"]), rhs=str(d["rhs"]))
    if ("kind" in d and "args" in d) or node == "heap":
        return HeapPred(kind=d["kind"], args=list(d["args"]))
    # 兜底:无法判别则降级为 RawPred(不中断实验)
    return RawPred(raw=json.dumps(d, ensure_ascii=False))


def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _contract_from_dict(d: dict) -> Contract:
    if not isinstance(d, dict):
        return Contract()
    return Contract(
        requires=[pred_from_dict(x) for x in _as_list(d.get("requires"))],
        ensures=[pred_from_dict(x) for x in _as_list(d.get("ensures"))],
        modifies=[str(x) for x in _as_list(d.get("modifies"))],
        decreases=[str(x) for x in _as_list(d.get("decreases"))],
    )


def _loc_from_dict(d: dict, cls=LoopLoc) -> SrcLoc:
    return cls(line=d.get("line", -1), ast_path=d.get("ast_path"), id=d.get("id"))


def _loop_from_dict(d: dict) -> LoopAnnot:
    return LoopAnnot(
        loc=_loc_from_dict(d.get("loc", {}), LoopLoc),
        invariants=[_inv_from_dict(x) for x in d.get("invariants", [])],
        decreases=list(d.get("decreases", [])),
    )


def _safe_inv_kind(v) -> InvKind:
    try:
        return InvKind(v)
    except (ValueError, KeyError):
        return InvKind.relational  # LLM 给了未知 kind -> 兜底


def _inv_from_dict(d: dict) -> InvClause:
    if not isinstance(d, dict):  # LLM 直接给了裸谓词/字符串
        return InvClause(pred=pred_from_dict(d))
    return InvClause(
        pred=pred_from_dict(d.get("pred", d)),
        kind=_safe_inv_kind(d.get("kind", "relational")),
        strength_tier=d.get("strength_tier", 1),
        provenance=list(d.get("provenance", []) or []),
    )


def _assertion_from_dict(d: dict) -> Assertion:
    if not isinstance(d, dict):
        return Assertion(loc=SrcLoc(), pred=pred_from_dict(d))
    return Assertion(loc=_loc_from_dict(d.get("loc", {}), SrcLoc), pred=pred_from_dict(d.get("pred", d)))


# --------------------------------------------------------------------------- #
# 谓词校验(供 validate 与 Critic 复用的基础检查)
# --------------------------------------------------------------------------- #
def _validate_pred(p: Pred) -> list[str]:
    problems: list[str] = []
    if isinstance(p, Relational):
        if p.op not in {"<", "<=", "==", "!=", ">", ">="}:
            problems.append(f"非法关系运算符 {p.op!r}")
        if p.lhs == "" or p.rhs == "":
            problems.append("关系谓词左/右操作数为空")
    elif isinstance(p, Quantified):
        if p.quant not in {"forall", "exists"}:
            problems.append(f"非法量词 {p.quant!r}")
        if p.lo == "" or p.hi == "":
            problems.append("量化变量缺少上/下界 (C5)")
        problems.extend(_validate_pred(p.body))
    elif isinstance(p, Logical):
        if p.op not in {"and", "or", "implies", "not"}:
            problems.append(f"非法逻辑运算符 {p.op!r}")
        if not p.children:
            problems.append("逻辑节点没有子节点")
        for c in p.children:
            problems.extend(_validate_pred(c))
    elif isinstance(p, HeapPred):
        if not p.kind:
            problems.append("堆谓词缺少 kind")
    elif isinstance(p, RawPred):
        if not p.raw.strip():
            problems.append("RawPred 为空")
    else:
        problems.append(f"未知 Pred 节点: {type(p)}")
    return problems
