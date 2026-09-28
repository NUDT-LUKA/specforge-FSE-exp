"""统一时序规约 IR。

五种表层语言(LTL / STL / MTL / ptLTL / SVA)全部编译到这一棵树上,语义核只写一遍。
设计取舍:

* **离散时间**。区间 ``[lo, hi]`` 记的是**时间单位**,由 ``dt`` 换算成采样步数;
  LTL/MTL/SVA 的 ``dt=1``,STL 按数据集给的采样率。这是 RTAMT/Breach 做采样信号
  监控时的标准口径,也让 SMT 编码保持可判定。
* **有界迹**。所有语义都在长度 H 的有限迹上定义(LTLf 口径):``X`` 分强/弱,
  ``G`` 在空区间上真空为真。FRET 的 ``LAST`` 作为"只在末点为真"的原子处理,
  于是 ``LAST V phi`` 恰好退化成 ``G phi``,与 FRET 自己的 ft 语义一致。
* **算术原子**。STL 与 ptLTL 的原子是实值比较(``speed > 120``),所以布尔原子
  ``Var`` 与比较原子 ``Cmp`` 并存;``Cmp`` 两侧的表达式树支持四则、取负与
  **未解释函数**(``sin``/``cos``/``abs`` 等)—— 后者在 SMT 侧映射成 Z3 的
  uninterpreted function,等价性判定因此仍然可靠(只是更保守)。
"""

from __future__ import annotations

from dataclasses import dataclass

INF = float("inf")


# --------------------------------------------------------------------------- #
# 算术表达式(比较原子的两侧)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Sig:
    """信号 / 变量引用。布尔变量出现在这里时按 0/1 数值参与比较。"""
    name: str


@dataclass(frozen=True)
class EnumLit:
    """枚举字面量(FRET 里 lift_mode = semi_thrust_borne 的右侧)。"""
    name: str


@dataclass(frozen=True)
class BinArith:
    op: str            # + - * /
    lhs: object
    rhs: object


@dataclass(frozen=True)
class NegArith:
    x: object


@dataclass(frozen=True)
class Func:
    """未解释函数应用(sin / cos / sqrt / abs / ...)。"""
    name: str
    args: tuple


# --------------------------------------------------------------------------- #
# 布尔 / 时序节点
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Const:
    value: bool


@dataclass(frozen=True)
class Var:
    """布尔命题原子。"""
    name: str


@dataclass(frozen=True)
class Last:
    """只在迹末点为真的原子(FRET LAST)。"""


@dataclass(frozen=True)
class Cmp:
    op: str            # <  <=  >  >=  ==  !=
    lhs: object
    rhs: object


@dataclass(frozen=True)
class Not:
    x: object


@dataclass(frozen=True)
class And:
    xs: tuple


@dataclass(frozen=True)
class Or:
    xs: tuple


@dataclass(frozen=True)
class Implies:
    a: object
    b: object


@dataclass(frozen=True)
class Iff:
    a: object
    b: object


@dataclass(frozen=True)
class Next:
    x: object
    weak: bool = False        # weak=True 即 LTLf 的弱 next


@dataclass(frozen=True)
class Prev:
    x: object
    weak: bool = False        # Y = strong(weak=False), Z = weak(weak=True)


@dataclass(frozen=True)
class Eventually:
    x: object
    lo: float = 0.0
    hi: float = INF


@dataclass(frozen=True)
class Always:
    x: object
    lo: float = 0.0
    hi: float = INF


@dataclass(frozen=True)
class Until:
    a: object
    b: object
    lo: float = 0.0
    hi: float = INF


@dataclass(frozen=True)
class Release:
    a: object
    b: object
    lo: float = 0.0
    hi: float = INF


@dataclass(frozen=True)
class Once:
    x: object
    lo: float = 0.0
    hi: float = INF


@dataclass(frozen=True)
class Historically:
    x: object
    lo: float = 0.0
    hi: float = INF


@dataclass(frozen=True)
class Since:
    a: object
    b: object
    lo: float = 0.0
    hi: float = INF


UNARY_TEMPORAL = (Next, Prev, Eventually, Always, Once, Historically)
BINARY_TEMPORAL = (Until, Release, Since)
TEMPORAL = UNARY_TEMPORAL + BINARY_TEMPORAL
TIMED = (Eventually, Always, Once, Historically, Until, Release, Since)


# --------------------------------------------------------------------------- #
# 结构工具
# --------------------------------------------------------------------------- #
def children(f) -> tuple:
    """时序/布尔节点的子公式(不含算术表达式)。"""
    if isinstance(f, Not):
        return (f.x,)
    if isinstance(f, (And, Or)):
        return tuple(f.xs)
    if isinstance(f, (Implies, Iff)):
        return (f.a, f.b)
    if isinstance(f, UNARY_TEMPORAL):
        return (f.x,)
    if isinstance(f, BINARY_TEMPORAL):
        return (f.a, f.b)
    return ()


def rebuild(f, kids: tuple):
    """按同一节点类型用新的子公式重建(供变异算子用)。"""
    if isinstance(f, Not):
        return Not(kids[0])
    if isinstance(f, And):
        return And(tuple(kids))
    if isinstance(f, Or):
        return Or(tuple(kids))
    if isinstance(f, Implies):
        return Implies(kids[0], kids[1])
    if isinstance(f, Iff):
        return Iff(kids[0], kids[1])
    if isinstance(f, Next):
        return Next(kids[0], f.weak)
    if isinstance(f, Prev):
        return Prev(kids[0], f.weak)
    if isinstance(f, Eventually):
        return Eventually(kids[0], f.lo, f.hi)
    if isinstance(f, Always):
        return Always(kids[0], f.lo, f.hi)
    if isinstance(f, Once):
        return Once(kids[0], f.lo, f.hi)
    if isinstance(f, Historically):
        return Historically(kids[0], f.lo, f.hi)
    if isinstance(f, Until):
        return Until(kids[0], kids[1], f.lo, f.hi)
    if isinstance(f, Release):
        return Release(kids[0], kids[1], f.lo, f.hi)
    if isinstance(f, Since):
        return Since(kids[0], kids[1], f.lo, f.hi)
    return f


def walk(f):
    """先序遍历所有布尔/时序子节点。"""
    yield f
    for k in children(f):
        yield from walk(k)


def size(f) -> int:
    return 1 + sum(size(k) for k in children(f))


def has_past(f) -> bool:
    return any(isinstance(n, (Prev, Once, Historically, Since)) for n in walk(f))


def arith_sigs(e, out: set) -> None:
    if isinstance(e, Sig):
        out.add(e.name)
    elif isinstance(e, BinArith):
        arith_sigs(e.lhs, out)
        arith_sigs(e.rhs, out)
    elif isinstance(e, NegArith):
        arith_sigs(e.x, out)
    elif isinstance(e, Func):
        for a in e.args:
            arith_sigs(a, out)


def atoms(f) -> set:
    """公式里出现的变量名(布尔原子 + 比较里引用的信号)。"""
    out: set = set()
    for n in walk(f):
        if isinstance(n, Var):
            out.add(n.name)
        elif isinstance(n, Cmp):
            arith_sigs(n.lhs, out)
            arith_sigs(n.rhs, out)
    return out


def enum_values(f) -> dict:
    """收集 ``x == SomeEnum`` 形式里每个变量用到的枚举字面量。"""
    out: dict = {}
    for n in walk(f):
        if not isinstance(n, Cmp):
            continue
        if isinstance(n.lhs, Sig) and isinstance(n.rhs, EnumLit):
            out.setdefault(n.lhs.name, set()).add(n.rhs.name)
        elif isinstance(n.rhs, Sig) and isinstance(n.lhs, EnumLit):
            out.setdefault(n.rhs.name, set()).add(n.lhs.name)
    return out


def thresholds(f) -> list:
    """公式里出现的数值常量(见证采样时用来把随机值钉在阈值附近)。"""
    out = []
    for n in walk(f):
        if isinstance(n, Cmp):
            for side in (n.lhs, n.rhs):
                for e in _arith_walk(side):
                    if isinstance(e, Num):
                        out.append(float(e.value))
    return out


def _arith_walk(e):
    yield e
    if isinstance(e, BinArith):
        yield from _arith_walk(e.lhs)
        yield from _arith_walk(e.rhs)
    elif isinstance(e, NegArith):
        yield from _arith_walk(e.x)
    elif isinstance(e, Func):
        for a in e.args:
            yield from _arith_walk(a)


# --------------------------------------------------------------------------- #
# 打印(回到一种可读的通用语法,供论文示例与 LLM prompt 用)
# --------------------------------------------------------------------------- #
def _num(v) -> str:
    fv = float(v)
    return str(int(fv)) if fv == int(fv) else str(fv)


def _iv(lo, hi) -> str:
    if lo == 0 and hi == INF:
        return ""
    h = "inf" if hi == INF else _num(hi)
    return "[" + _num(lo) + "," + h + "]"


def arith_str(e) -> str:
    if isinstance(e, Num):
        return _num(e.value)
    if isinstance(e, (Sig, EnumLit)):
        return e.name
    if isinstance(e, BinArith):
        return "(" + arith_str(e.lhs) + " " + e.op + " " + arith_str(e.rhs) + ")"
    if isinstance(e, NegArith):
        return "(- " + arith_str(e.x) + ")"
    if isinstance(e, Func):
        return e.name + "(" + ", ".join(arith_str(a) for a in e.args) + ")"
    return str(e)


def to_str(f) -> str:
    if isinstance(f, Const):
        return "true" if f.value else "false"
    if isinstance(f, Var):
        return f.name
    if isinstance(f, Last):
        return "LAST"
    if isinstance(f, Cmp):
        return "(" + arith_str(f.lhs) + " " + f.op + " " + arith_str(f.rhs) + ")"
    if isinstance(f, Not):
        return "!" + to_str(f.x)
    if isinstance(f, And):
        return "(" + " & ".join(to_str(x) for x in f.xs) + ")"
    if isinstance(f, Or):
        return "(" + " | ".join(to_str(x) for x in f.xs) + ")"
    if isinstance(f, Implies):
        return "(" + to_str(f.a) + " -> " + to_str(f.b) + ")"
    if isinstance(f, Iff):
        return "(" + to_str(f.a) + " <-> " + to_str(f.b) + ")"
    if isinstance(f, Next):
        return ("wX" if f.weak else "X") + to_str(f.x)
    if isinstance(f, Prev):
        return ("Z" if f.weak else "Y") + to_str(f.x)
    if isinstance(f, Eventually):
        return "F" + _iv(f.lo, f.hi) + to_str(f.x)
    if isinstance(f, Always):
        return "G" + _iv(f.lo, f.hi) + to_str(f.x)
    if isinstance(f, Once):
        return "O" + _iv(f.lo, f.hi) + to_str(f.x)
    if isinstance(f, Historically):
        return "H" + _iv(f.lo, f.hi) + to_str(f.x)
    if isinstance(f, Until):
        return "(" + to_str(f.a) + " U" + _iv(f.lo, f.hi) + " " + to_str(f.b) + ")"
    if isinstance(f, Release):
        return "(" + to_str(f.a) + " V" + _iv(f.lo, f.hi) + " " + to_str(f.b) + ")"
    if isinstance(f, Since):
        return "(" + to_str(f.a) + " S" + _iv(f.lo, f.hi) + " " + to_str(f.b) + ")"
    return str(f)


# --------------------------------------------------------------------------- #
# JSON 序列化(建集与评测之间传 IR,不走"重新解析"那条会漂移的路)
# --------------------------------------------------------------------------- #
_ARITH_CLASSES = {"Num": Num, "Sig": Sig, "EnumLit": EnumLit,
                  "BinArith": BinArith, "NegArith": NegArith, "Func": Func}
_NODE_CLASSES = {"Const": Const, "Var": Var, "Last": Last, "Cmp": Cmp, "Not": Not,
                 "And": And, "Or": Or, "Implies": Implies, "Iff": Iff,
                 "Next": Next, "Prev": Prev, "Eventually": Eventually,
                 "Always": Always, "Once": Once, "Historically": Historically,
                 "Until": Until, "Release": Release, "Since": Since}


def _b(v):
    """inf 在 JSON 里没有字面量,统一编码成字符串 'inf'。"""
    return "inf" if v == INF else float(v)


def _ub(v):
    return INF if v == "inf" else float(v)


def to_json(f):
    t = type(f).__name__
    if t == "Num":
        return {"t": t, "v": float(f.value)}
    if t in ("Sig", "EnumLit", "Var"):
        return {"t": t, "n": f.name}
    if t == "BinArith":
        return {"t": t, "op": f.op, "l": to_json(f.lhs), "r": to_json(f.rhs)}
    if t == "NegArith":
        return {"t": t, "x": to_json(f.x)}
    if t == "Func":
        return {"t": t, "n": f.name, "a": [to_json(a) for a in f.args]}
    if t == "Const":
        return {"t": t, "v": bool(f.value)}
    if t == "Last":
        return {"t": t}
    if t == "Cmp":
        return {"t": t, "op": f.op, "l": to_json(f.lhs), "r": to_json(f.rhs)}
    if t == "Not":
        return {"t": t, "x": to_json(f.x)}
    if t in ("And", "Or"):
        return {"t": t, "xs": [to_json(x) for x in f.xs]}
    if t in ("Implies", "Iff"):
        return {"t": t, "a": to_json(f.a), "b": to_json(f.b)}
    if t in ("Next", "Prev"):
        return {"t": t, "x": to_json(f.x), "w": bool(f.weak)}
    if t in ("Eventually", "Always", "Once", "Historically"):
        return {"t": t, "x": to_json(f.x), "lo": _b(f.lo), "hi": _b(f.hi)}
    if t in ("Until", "Release", "Since"):
        return {"t": t, "a": to_json(f.a), "b": to_json(f.b),
                "lo": _b(f.lo), "hi": _b(f.hi)}
    raise ValueError("cannot serialise " + t)


def from_json(d):
    t = d["t"]
    if t == "Num":
        return Num(float(d["v"]))
    if t == "Sig":
        return Sig(d["n"])
    if t == "EnumLit":
        return EnumLit(d["n"])
    if t == "Var":
        return Var(d["n"])
    if t == "BinArith":
        return BinArith(d["op"], from_json(d["l"]), from_json(d["r"]))
    if t == "NegArith":
        return NegArith(from_json(d["x"]))
    if t == "Func":
        return Func(d["n"], tuple(from_json(a) for a in d["a"]))
    if t == "Const":
        return Const(bool(d["v"]))
    if t == "Last":
        return Last()
    if t == "Cmp":
        return Cmp(d["op"], from_json(d["l"]), from_json(d["r"]))
    if t == "Not":
        return Not(from_json(d["x"]))
    if t == "And":
        return And(tuple(from_json(x) for x in d["xs"]))
    if t == "Or":
        return Or(tuple(from_json(x) for x in d["xs"]))
    if t == "Implies":
        return Implies(from_json(d["a"]), from_json(d["b"]))
    if t == "Iff":
        return Iff(from_json(d["a"]), from_json(d["b"]))
    if t == "Next":
        return Next(from_json(d["x"]), bool(d.get("w", False)))
    if t == "Prev":
        return Prev(from_json(d["x"]), bool(d.get("w", False)))
    if t == "Eventually":
        return Eventually(from_json(d["x"]), _ub(d["lo"]), _ub(d["hi"]))
    if t == "Always":
        return Always(from_json(d["x"]), _ub(d["lo"]), _ub(d["hi"]))
    if t == "Once":
        return Once(from_json(d["x"]), _ub(d["lo"]), _ub(d["hi"]))
    if t == "Historically":
        return Historically(from_json(d["x"]), _ub(d["lo"]), _ub(d["hi"]))
    if t == "Until":
        return Until(from_json(d["a"]), from_json(d["b"]), _ub(d["lo"]), _ub(d["hi"]))
    if t == "Release":
        return Release(from_json(d["a"]), from_json(d["b"]), _ub(d["lo"]), _ub(d["hi"]))
    if t == "Since":
        return Since(from_json(d["a"]), from_json(d["b"]), _ub(d["lo"]), _ub(d["hi"]))
    raise ValueError("cannot deserialise " + t)
