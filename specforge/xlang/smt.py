"""Z3 有界编码:蕴含 / 等价 / 区分轨迹合成。

作用有二:

1. **给 TFVR 做效度检验的 gold standard**。轨迹见证给出的是"这几条轨迹上没抓到
   区别",符号判定给出的是"长度 H 的论域里根本不存在区别"。RQ-G2 就是拿后者去
   验证前者。
2. **合成区分轨迹**。SAT 时的模型本身就是一条把 phi* 与 phi-hat 分开的轨迹 ——
   既是最有信息量的见证,也是反例制导修复的输入。

口径与 ``monitor`` 严格对齐(同样的有限迹语义、同样的求值点、同样的区间换算),
否则 gold 和 proxy 比的就不是同一件事。实数变量默认约束在 ``domain`` 给的量程内:
量程是信号的物理论域,采样器也只在这里面取值,两侧口径必须一致。未解释函数
(``sin``/``cos``/自定义谓词)映射成 Z3 的 uninterpreted function —— 判定因此偏
**保守**(可能把实际等价的判成不等价),这个方向的偏差不会夸大我们的结论。
"""

from __future__ import annotations

import math

import z3

from .monitor import Trace
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


class Unsupported(Exception):
    pass


class Encoder:
    def __init__(self, doms: dict, H: int, dt: float = 1.0, bound_reals: bool = True):
        self.doms, self.H, self.dt = doms, H, dt
        self.bound_reals = bound_reals
        self.cells: dict = {}
        self.enum_map: dict = {}
        self.funcs: dict = {}
        self.axioms: list = []
        for name, dom in doms.items():
            k = dom["kind"]
            if k == "bool":
                self.cells[name] = [z3.Bool("%s@%d" % (name, t)) for t in range(H)]
            elif k == "enum":
                vals = list(dom["values"])
                self.enum_map[name] = {v: i for i, v in enumerate(vals)}
                col = [z3.Int("%s@%d" % (name, t)) for t in range(H)]
                self.cells[name] = col
                for c in col:
                    self.axioms.append(z3.And(c >= 0, c < len(vals)))
            else:
                col = [z3.Real("%s@%d" % (name, t)) for t in range(H)]
                self.cells[name] = col
                if bound_reals:
                    lo, hi = dom.get("range", [-1e4, 1e4])
                    for c in col:
                        self.axioms.append(z3.And(c >= lo, c <= hi))
        self.memo: dict = {}

    # -- 变量取值 ------------------------------------------------------------ #
    def cell(self, name, t):
        col = self.cells.get(name)
        if col is None:
            raise Unsupported("variable not in domain: " + name)
        if t < 0 or t >= self.H:
            raise Unsupported("index out of horizon")
        return col[t]

    def func(self, name, arity):
        key = (name, arity)
        if key not in self.funcs:
            self.funcs[key] = z3.Function(name, *([z3.RealSort()] * arity), z3.RealSort())
        return self.funcs[key]

    # -- 算术 ---------------------------------------------------------------- #
    def arith(self, e, t):
        if isinstance(e, Num):
            return z3.RealVal(float(e.value))
        if isinstance(e, EnumLit):
            raise Unsupported("enum literal in arithmetic position")
        if isinstance(e, Sig):
            c = self.cell(e.name, t)
            if z3.is_bool(c):
                return z3.If(c, z3.RealVal(1), z3.RealVal(0))
            if self.doms[e.name]["kind"] == "enum":
                return c
            return c
        if isinstance(e, NegArith):
            return -self.arith(e.x, t)
        if isinstance(e, BinArith):
            a, b = self.arith(e.lhs, t), self.arith(e.rhs, t)
            if e.op == "+":
                return a + b
            if e.op == "-":
                return a - b
            if e.op == "*":
                return a * b
            if e.op == "/":
                return a / b
            raise Unsupported("arith op " + e.op)
        if isinstance(e, Func):
            args = [self.arith(a, t) for a in e.args]
            return self.func(e.name, len(args))(*[z3.ToReal(a) if a.sort() == z3.IntSort() else a for a in args])
        raise Unsupported("arith node " + type(e).__name__)

    def cmp(self, node, t):
        lhs, rhs = node.lhs, node.rhs
        # 枚举等值:两侧都换成整数编码
        for a, b in ((lhs, rhs), (rhs, lhs)):
            if isinstance(a, Sig) and isinstance(b, EnumLit):
                m = self.enum_map.get(a.name)
                if m is None or b.name not in m:
                    return z3.BoolVal(False) if node.op == "==" else z3.BoolVal(True)
                eq = self.cell(a.name, t) == m[b.name]
                return eq if node.op == "==" else z3.Not(eq)
        A, B = self.arith(lhs, t), self.arith(rhs, t)
        op = node.op
        if op == "<":
            return A < B
        if op == "<=":
            return A <= B
        if op == ">":
            return A > B
        if op == ">=":
            return A >= B
        if op == "==":
            return A == B
        if op == "!=":
            return A != B
        raise Unsupported("cmp op " + op)

    # -- 时序 ---------------------------------------------------------------- #
    def window(self, lo, hi, t, past=False):
        a = int(math.ceil(lo / self.dt - 1e-9))
        b = self.H if hi == INF else int(math.floor(hi / self.dt + 1e-9))
        if past:
            i0, i1 = max(0, t - b), t - a
            if i1 < i0 or i1 < 0:
                return None
            return i0, min(i1, self.H - 1)
        i0, i1 = t + a, t + b
        if i0 > self.H - 1:
            return None
        return i0, min(i1, self.H - 1)

    def sat(self, f, t):
        key = (id(f), t)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        v = self._sat(f, t)
        self.memo[key] = v
        return v

    def _sat(self, f, t):
        if isinstance(f, Const):
            return z3.BoolVal(f.value)
        if isinstance(f, Last):
            return z3.BoolVal(t == self.H - 1)
        if isinstance(f, Var):
            c = self.cell(f.name, t)
            if z3.is_bool(c):
                return c
            return c != 0
        if isinstance(f, Cmp):
            return self.cmp(f, t)
        if isinstance(f, Not):
            return z3.Not(self.sat(f.x, t))
        if isinstance(f, And):
            return z3.And(*[self.sat(x, t) for x in f.xs])
        if isinstance(f, Or):
            return z3.Or(*[self.sat(x, t) for x in f.xs])
        if isinstance(f, Implies):
            return z3.Implies(self.sat(f.a, t), self.sat(f.b, t))
        if isinstance(f, Iff):
            return self.sat(f.a, t) == self.sat(f.b, t)
        if isinstance(f, Next):
            if t + 1 >= self.H:
                return z3.BoolVal(bool(f.weak))
            return self.sat(f.x, t + 1)
        if isinstance(f, Prev):
            if t - 1 < 0:
                return z3.BoolVal(bool(f.weak))
            return self.sat(f.x, t - 1)
        if isinstance(f, (Eventually, Always, Once, Historically)):
            past = isinstance(f, (Once, Historically))
            w = self.window(f.lo, f.hi, t, past)
            anyk = isinstance(f, (Eventually, Once))
            if w is None:
                return z3.BoolVal(not anyk)
            parts = [self.sat(f.x, i) for i in range(w[0], w[1] + 1)]
            return z3.Or(*parts) if anyk else z3.And(*parts)
        if isinstance(f, Until):
            w = self.window(f.lo, f.hi, t)
            if w is None:
                return z3.BoolVal(False)
            outs = []
            for i in range(w[0], w[1] + 1):
                pre = [self.sat(f.a, j) for j in range(t, i)]
                outs.append(z3.And(self.sat(f.b, i), *pre) if pre else self.sat(f.b, i))
            return z3.Or(*outs)
        if isinstance(f, Release):
            w = self.window(f.lo, f.hi, t)
            if w is None:
                return z3.BoolVal(True)
            outs = []
            for i in range(w[0], w[1] + 1):
                rel = [self.sat(f.a, j) for j in range(w[0], i)]
                outs.append(z3.Or(self.sat(f.b, i), *rel) if rel else self.sat(f.b, i))
            return z3.And(*outs)
        if isinstance(f, Since):
            w = self.window(f.lo, f.hi, t, past=True)
            if w is None:
                return z3.BoolVal(False)
            outs = []
            for i in range(w[0], w[1] + 1):
                pre = [self.sat(f.a, j) for j in range(i + 1, t + 1)]
                outs.append(z3.And(self.sat(f.b, i), *pre) if pre else self.sat(f.b, i))
            return z3.Or(*outs)
        raise Unsupported("node " + type(f).__name__)

    def root(self, f):
        return self.sat(f, self.H - 1 if has_past(f) else 0)

    # -- 反解模型成一条具体轨迹 ---------------------------------------------- #
    def model_to_trace(self, m) -> Trace:
        data = {}
        for name, dom in self.doms.items():
            col = []
            for t in range(self.H):
                c = self.cells[name][t]
                v = m.eval(c, model_completion=True)
                if dom["kind"] == "bool":
                    col.append(bool(z3.is_true(v)))
                elif dom["kind"] == "enum":
                    col.append(dom["values"][int(v.as_long()) % len(dom["values"])])
                else:
                    col.append(float(v.as_fraction()) if hasattr(v, "as_fraction") else float(str(v)))
            data[name] = col
        return Trace(data=data, dt=self.dt)


# --------------------------------------------------------------------------- #
def distinguish(f1, f2, doms, H, dt=1.0, timeout_ms=8000, bound_reals=True):
    """找一条轨迹使 ``f1`` 成立而 ``f2`` 不成立。

    返回 ``("sat", trace)`` / ``("unsat", None)`` / ``("unknown", None)``。
    ``unsat`` 即有界蕴含 ``f1 |= f2`` 成立。
    """
    try:
        enc = Encoder(doms, H, dt, bound_reals)
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        for a in enc.axioms:
            s.add(a)
        s.add(enc.root(f1))
        s.add(z3.Not(enc.root(f2)))
    except (Unsupported, z3.Z3Exception) as e:
        return "unknown", None
    r = s.check()
    if r == z3.sat:
        try:
            return "sat", enc.model_to_trace(s.model())
        except Exception:
            return "sat", None
    if r == z3.unsat:
        return "unsat", None
    return "unknown", None


def entails(f1, f2, doms, H, dt=1.0, timeout_ms=8000, bound_reals=True):
    """f1 |= f2 (有界)。返回 True / False / None(判不了)。"""
    st, _ = distinguish(f1, f2, doms, H, dt, timeout_ms, bound_reals)
    if st == "unsat":
        return True
    if st == "sat":
        return False
    return None


def relation(ref, cand, doms, H, dt=1.0, timeout_ms=8000, bound_reals=True):
    """参考 vs 候选的语义关系(有界)。

    返回 dict:``ref_entails_cand``(候选不过强 -> 验证会通过)、
    ``cand_entails_ref``(候选不过弱 -> 忠实)、以及派生的 ``verdict``:

    * ``equivalent``  两向都成立
    * ``weaker``      只有 ref|=cand —— **沉默错误**(验证通过但规约过弱)
    * ``stronger``    只有 cand|=ref —— 自曝错误(验证失败)
    * ``incomparable``两向都不成立 —— 也是自曝错误
    * ``unknown``     判不了
    """
    a = entails(ref, cand, doms, H, dt, timeout_ms, bound_reals)
    b = entails(cand, ref, doms, H, dt, timeout_ms, bound_reals)
    if a is None or b is None:
        v = "unknown"
    elif a and b:
        v = "equivalent"
    elif a:
        v = "weaker"
    elif b:
        v = "stronger"
    else:
        v = "incomparable"
    return {"ref_entails_cand": a, "cand_entails_ref": b, "verdict": v}


# --------------------------------------------------------------------------- #
def agrees_with_monitor(phi, doms, H, dt, traces, bound_reals=True):
    """检查 SMT 编码与轨迹监控器在给定轨迹上是否给出同一判决。

    两套语义只要在任何一点上分岔(枚举/数值双重解释、未解释函数取值不同等),
    "gold" 与"proxy"比的就不是同一件事,忠实性判据的可靠性论证也随之失效。
    所以建集期把这条不变量当门槛:对不上就整条任务丢掉,而不是留着悄悄污染统计。

    返回 True / False / None(判不了)。
    """
    from .monitor import Undecidable, eval_formula
    try:
        enc = Encoder(doms, H, dt, bound_reals)
        root = enc.root(phi)
    except (Unsupported, z3.Z3Exception):
        return None
    for tr in traces:
        try:
            want = eval_formula(phi, tr)
        except Undecidable:
            return None
        s = z3.Solver()
        s.set("timeout", 4000)
        for name, dom in doms.items():
            for t in range(H):
                v = tr.data[name][t]
                c = enc.cells[name][t]
                if dom["kind"] == "bool":
                    s.add(c if v else z3.Not(c))
                elif dom["kind"] == "enum":
                    if v not in dom["values"]:
                        return None
                    s.add(c == dom["values"].index(v))
                else:
                    s.add(c == z3.RealVal(float(v)))
        s.add(root if want else z3.Not(root))
        r = s.check()
        if r == z3.unsat:
            return False
        if r != z3.sat:
            return None
    return True
