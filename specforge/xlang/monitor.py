"""离散时间轨迹语义(布尔判定 + STL 鲁棒度)。

这是整个跨语言 TFVR 的**唯一语义核**:五种表层语言解析成同一棵 IR 之后,忠实性
判据 E-spec+/E-spec- 全都落到这里的 ``sat`` 上。

口径(全部按有限迹 LTLf,并在 docstring 里写死,免得日后漂移):

* 区间 ``[lo,hi]`` 是**时间单位**,换算成采样步 ``a=ceil(lo/dt)``、``b=floor(hi/dt)``;
  ``hi=inf`` 时 ``b`` 取到迹尾。
* ``X phi`` (strong) 在末点为假;``wX phi`` (weak) 在末点为真。
* ``G[a,b]`` 在窗口落到迹外时**真空为真**,``F[a,b]`` 则为假 —— 与 RTAMT 的
  有限迹口径一致。
* ``Y``(strong prev)在 t=0 为假,``Z``(weak prev)在 t=0 为真。
* 求值点:未来时态公式看 t=0;过去时态公式(ptLTL)看 t=H-1。FRET 的 pt 公式
  全部以 ``H(...)`` 为根,"在末点成立"与"在每点成立"等价,所以这一约定不损失语义。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .tlast import (
    INF,
    Always,
    And,
    BinArith,
    Cmp,
    Const,
    EnumLit,
    Eventually,
    Func,
    Historically,
    Iff,
    Implies,
    Last,
    NegArith,
    Next,
    Not,
    Num,
    Once,
    Or,
    Prev,
    Release,
    Sig,
    Since,
    Until,
    Var,
    has_past,
)


class Undecidable(Exception):
    """轨迹上判不出来(引用了迹里没有的变量等),按惯例保守丢弃。"""


@dataclass
class Trace:
    """一条采样轨迹。``data`` 是 变量名 -> 长度 H 的取值表(bool / float / str)。"""
    data: dict
    dt: float = 1.0

    @property
    def H(self) -> int:
        for v in self.data.values():
            return len(v)
        return 0

    def get(self, name: str, t: int):
        try:
            col = self.data[name]
        except KeyError:
            raise Undecidable("trace has no variable " + name)
        if t < 0 or t >= len(col):
            raise Undecidable("index out of trace")
        return col[t]

    def to_json(self) -> dict:
        return {"dt": self.dt, "data": {k: list(v) for k, v in self.data.items()}}

    @staticmethod
    def from_json(d: dict) -> "Trace":
        return Trace(data={k: list(v) for k, v in d["data"].items()},
                     dt=float(d.get("dt", 1.0)))


# --------------------------------------------------------------------------- #
# 算术求值
# --------------------------------------------------------------------------- #
_FUNCS = {
    "abs": abs, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": lambda x: math.sqrt(x) if x >= 0 else float("nan"),
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
    "atan": math.atan, "asin": lambda x: math.asin(max(-1.0, min(1.0, x))),
    "acos": lambda x: math.acos(max(-1.0, min(1.0, x))),
    # SVA 位运算:信号按整数建模,这里给出真实的按位语义。SMT 侧同名函数是
    # uninterpreted,两者不一致的方向是 SMT 更保守(多判不等价),不会夸大结论。
    "b2i": lambda x: 1.0 if x else 0.0,
    "bitand": lambda a, b: float(int(round(a)) & int(round(b))),
    "bitor": lambda a, b: float(int(round(a)) | int(round(b))),
    "bitxor": lambda a, b: float(int(round(a)) ^ int(round(b))),
    "countones": lambda x: float(bin(int(round(x)) & 0xFFFF).count("1")),
    "concat": lambda *xs: float(sum(int(round(v)) << (4 * i) for i, v in enumerate(reversed(xs)))),
}


def _arith(e, tr: Trace, t: int):
    """返回 float,或返回 str(枚举/字符串取值)。"""
    if isinstance(e, Num):
        return float(e.value)
    if isinstance(e, EnumLit):
        return e.name
    if isinstance(e, Sig):
        v = tr.get(e.name, t)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, str):
            return v
        return float(v)
    if isinstance(e, NegArith):
        return -_num(_arith(e.x, tr, t))
    if isinstance(e, BinArith):
        a = _num(_arith(e.lhs, tr, t))
        b = _num(_arith(e.rhs, tr, t))
        if e.op == "+":
            return a + b
        if e.op == "-":
            return a - b
        if e.op == "*":
            return a * b
        if e.op == "/":
            return a / b if b != 0 else float("inf") * (1 if a >= 0 else -1)
        raise Undecidable("bad arith op " + e.op)
    if isinstance(e, Func):
        fn = _FUNCS.get(e.name)
        if fn is None:
            raise Undecidable("uninterpreted function " + e.name)
        return float(fn(*[_num(_arith(a, tr, t)) for a in e.args]))
    raise Undecidable("bad arith node " + type(e).__name__)


def _num(v) -> float:
    if isinstance(v, str):
        raise Undecidable("string used in arithmetic")
    return float(v)


def _cmp(op: str, a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        if op == "==":
            return str(a) == str(b)
        if op == "!=":
            return str(a) != str(b)
        raise Undecidable("ordering on non-numeric values")
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    raise Undecidable("bad cmp op " + op)


# --------------------------------------------------------------------------- #
# 布尔语义
# --------------------------------------------------------------------------- #
def _window(lo, hi, dt, H, t, past=False):
    """把时间区间换算成采样下标闭区间 [i0, i1](越界后已裁剪);空窗返回 None。"""
    a = int(math.ceil(lo / dt - 1e-9))
    b = H if hi == INF else int(math.floor(hi / dt + 1e-9))
    if past:
        i0 = max(0, t - b)
        i1 = t - a
        if i1 < i0 or i1 < 0:
            return None
        return i0, min(i1, H - 1)
    i0 = t + a
    i1 = t + b
    if i0 > H - 1:
        return None
    return i0, min(i1, H - 1)


def sat(f, tr: Trace, t: int = 0) -> bool:
    """公式 f 在轨迹 tr 的时刻 t 是否成立。"""
    if isinstance(f, Const):
        return f.value
    if isinstance(f, Last):
        return t == tr.H - 1
    if isinstance(f, Var):
        v = tr.get(f.name, t)
        if isinstance(v, str):
            raise Undecidable("boolean atom bound to string")
        return bool(v)
    if isinstance(f, Cmp):
        return _cmp(f.op, _arith(f.lhs, tr, t), _arith(f.rhs, tr, t))
    if isinstance(f, Not):
        return not sat(f.x, tr, t)
    if isinstance(f, And):
        return all(sat(x, tr, t) for x in f.xs)
    if isinstance(f, Or):
        return any(sat(x, tr, t) for x in f.xs)
    if isinstance(f, Implies):
        return (not sat(f.a, tr, t)) or sat(f.b, tr, t)
    if isinstance(f, Iff):
        return sat(f.a, tr, t) == sat(f.b, tr, t)
    if isinstance(f, Next):
        if t + 1 >= tr.H:
            return bool(f.weak)
        return sat(f.x, tr, t + 1)
    if isinstance(f, Prev):
        if t - 1 < 0:
            return bool(f.weak)
        return sat(f.x, tr, t - 1)

    H = tr.H
    if isinstance(f, Eventually):
        w = _window(f.lo, f.hi, tr.dt, H, t)
        return False if w is None else any(sat(f.x, tr, i) for i in range(w[0], w[1] + 1))
    if isinstance(f, Always):
        w = _window(f.lo, f.hi, tr.dt, H, t)
        return True if w is None else all(sat(f.x, tr, i) for i in range(w[0], w[1] + 1))
    if isinstance(f, Once):
        w = _window(f.lo, f.hi, tr.dt, H, t, past=True)
        return False if w is None else any(sat(f.x, tr, i) for i in range(w[0], w[1] + 1))
    if isinstance(f, Historically):
        w = _window(f.lo, f.hi, tr.dt, H, t, past=True)
        return True if w is None else all(sat(f.x, tr, i) for i in range(w[0], w[1] + 1))
    if isinstance(f, Until):
        w = _window(f.lo, f.hi, tr.dt, H, t)
        if w is None:
            return False
        for i in range(w[0], w[1] + 1):
            if sat(f.b, tr, i) and all(sat(f.a, tr, j) for j in range(t, i)):
                return True
        return False
    if isinstance(f, Release):
        w = _window(f.lo, f.hi, tr.dt, H, t)
        if w is None:
            return True
        for i in range(w[0], w[1] + 1):
            if not sat(f.b, tr, i):
                if not any(sat(f.a, tr, j) for j in range(w[0], i)):
                    return False
        return True
    if isinstance(f, Since):
        w = _window(f.lo, f.hi, tr.dt, H, t, past=True)
        if w is None:
            return False
        for i in range(w[1], w[0] - 1, -1):
            if sat(f.b, tr, i) and all(sat(f.a, tr, j) for j in range(i + 1, t + 1)):
                return True
        return False
    raise Undecidable("bad node " + type(f).__name__)


def eval_formula(f, tr: Trace) -> bool:
    """整条轨迹的判决。过去时态在末点求值,其余在起点求值(见模块 docstring)。"""
    if tr.H == 0:
        raise Undecidable("empty trace")
    return sat(f, tr, tr.H - 1 if has_past(f) else 0)


# --------------------------------------------------------------------------- #
# STL 鲁棒度(定量语义)—— 只用于诊断与和 RTAMT 的交叉校验
# --------------------------------------------------------------------------- #
def rob(f, tr: Trace, t: int = 0) -> float:
    if isinstance(f, Const):
        return INF if f.value else -INF
    if isinstance(f, Last):
        return INF if t == tr.H - 1 else -INF
    if isinstance(f, Var):
        return INF if bool(tr.get(f.name, t)) else -INF
    if isinstance(f, Cmp):
        a, b = _arith(f.lhs, tr, t), _arith(f.rhs, tr, t)
        if isinstance(a, str) or isinstance(b, str):
            return INF if _cmp(f.op, a, b) else -INF
        d = {"<": b - a, "<=": b - a, ">": a - b, ">=": a - b,
             "==": -abs(a - b), "!=": abs(a - b)}[f.op]
        return float(d)
    if isinstance(f, Not):
        return -rob(f.x, tr, t)
    if isinstance(f, And):
        return min(rob(x, tr, t) for x in f.xs)
    if isinstance(f, Or):
        return max(rob(x, tr, t) for x in f.xs)
    if isinstance(f, Implies):
        return max(-rob(f.a, tr, t), rob(f.b, tr, t))
    if isinstance(f, Iff):
        return min(max(-rob(f.a, tr, t), rob(f.b, tr, t)),
                   max(rob(f.a, tr, t), -rob(f.b, tr, t)))
    if isinstance(f, Next):
        if t + 1 >= tr.H:
            return INF if f.weak else -INF
        return rob(f.x, tr, t + 1)
    if isinstance(f, Prev):
        if t - 1 < 0:
            return INF if f.weak else -INF
        return rob(f.x, tr, t - 1)

    H = tr.H
    if isinstance(f, (Eventually, Always, Once, Historically)):
        past = isinstance(f, (Once, Historically))
        w = _window(f.lo, f.hi, tr.dt, H, t, past=past)
        if w is None:
            return -INF if isinstance(f, (Eventually, Once)) else INF
        vals = [rob(f.x, tr, i) for i in range(w[0], w[1] + 1)]
        return max(vals) if isinstance(f, (Eventually, Once)) else min(vals)
    if isinstance(f, Until):
        w = _window(f.lo, f.hi, tr.dt, H, t)
        if w is None:
            return -INF
        best = -INF
        for i in range(w[0], w[1] + 1):
            pre = min([rob(f.a, tr, j) for j in range(t, i)] or [INF])
            best = max(best, min(rob(f.b, tr, i), pre))
        return best
    if isinstance(f, Release):
        return -rob(Until(Not(f.a), Not(f.b), f.lo, f.hi), tr, t)
    if isinstance(f, Since):
        w = _window(f.lo, f.hi, tr.dt, H, t, past=True)
        if w is None:
            return -INF
        best = -INF
        for i in range(w[0], w[1] + 1):
            pre = min([rob(f.a, tr, j) for j in range(i + 1, t + 1)] or [INF])
            best = max(best, min(rob(f.b, tr, i), pre))
        return best
    raise Undecidable("bad node " + type(f).__name__)
