"""从参考公式反推变量论域,并给每个任务定采样率 dt 与视界 H。

这一步决定了见证集 R 的取值空间,所以它必须只依赖**参考公式 phi***(意图侧),
绝不能看候选公式 —— 否则忠实性判据就循环论证了。

三类变量:

* ``bool``:以布尔命题出现(``Var``),或被拿去和 0/1 比较。
* ``enum``:出现在 ``x == SomeLiteral`` 里且该字面量从不作为信号使用。
* ``real``:其余,论域由公式里出现过的阈值撑开(阈值附近是判决翻转的地方,
  随机采样必须打得到那里,否则 Sigma- 全是"离谱到毫无信息量"的反例)。
"""

from __future__ import annotations

import math

from .tlast import (
    INF,
    BinArith,
    Cmp,
    EnumLit,
    Func,
    NegArith,
    Num,
    Sig,
    TIMED,
    Var,
    walk,
)


def infer_enums(f):
    """把"只在 ==/!= 右侧出现、且从不作为信号参与运算"的标识符改判为枚举字面量。

    DeepSTL 的 ``W2 == RYoU_1`` 里 ``RYoU_1`` 是状态取值不是信号;而 ``id2 == id3``
    两侧都是信号。区分办法只能是看这个名字在整条公式里还有没有别的用法。
    """
    rhs_only, used_elsewhere, numeric = set(), set(), set()

    def arith_used(e, numeric_ctx=False):
        if isinstance(e, Sig):
            used_elsewhere.add(e.name)
            if numeric_ctx:
                numeric.add(e.name)
        elif isinstance(e, BinArith):
            arith_used(e.lhs, True); arith_used(e.rhs, True)
        elif isinstance(e, NegArith):
            arith_used(e.x, True)
        elif isinstance(e, Func):
            for a in e.args:
                arith_used(a, True)

    for n in walk(f):
        if isinstance(n, Var):
            used_elsewhere.add(n.name)
        elif isinstance(n, Cmp):
            num_ctx = n.op not in ("==", "!=")          # 序关系一定是数值语义
            for side, other in ((n.lhs, n.rhs), (n.rhs, n.lhs)):
                if isinstance(side, Sig) and isinstance(other, Num):
                    numeric.add(side.name)              # 和常数比 -> 数值语义
            if n.op in ("==", "!=") and isinstance(n.rhs, Sig) and isinstance(n.lhs, Sig):
                arith_used(n.lhs)
                rhs_only.add(n.rhs.name)
            else:
                arith_used(n.lhs, num_ctx); arith_used(n.rhs, num_ctx)

    # 只有当**左侧变量本身从不参与数值运算**时,右侧标识符才可能是枚举字面量。
    # 漏掉这一条就会出现 `sig_E == sig_F` 判成枚举、而同一条公式里 `sig_E != 0`
    # 又按数值解释的分裂语义 —— 监控器与 SMT 会给出不同判决。
    enums = set()
    for n in walk(f):
        if (isinstance(n, Cmp) and n.op in ("==", "!=")
                and isinstance(n.lhs, Sig) and isinstance(n.rhs, Sig)
                and n.rhs.name in rhs_only - used_elsewhere
                and n.lhs.name not in numeric):
            enums.add(n.rhs.name)
    if not enums:
        return f
    return _relabel(f, enums)


def _relabel(f, enums: set):
    from .tlast import rebuild, children

    def fix(node):
        if isinstance(node, Cmp) and isinstance(node.rhs, Sig) and node.rhs.name in enums:
            return Cmp(node.op, node.lhs, EnumLit(node.rhs.name))
        kids = children(node)
        if not kids:
            return node
        return rebuild(node, tuple(fix(k) for k in kids))

    return fix(f)


# --------------------------------------------------------------------------- #
def variable_domains(f, extra_enum: dict | None = None) -> dict:
    """返回 {name: {"kind": ..., ...}}。"""
    doms: dict = {}
    thresholds: dict = {}

    def note_real(name):
        doms.setdefault(name, {"kind": "real"})

    def arith_scan(e, owner=None):
        if isinstance(e, Sig):
            note_real(e.name)
        elif isinstance(e, BinArith):
            arith_scan(e.lhs); arith_scan(e.rhs)
        elif isinstance(e, NegArith):
            arith_scan(e.x)
        elif isinstance(e, Func):
            for a in e.args:
                arith_scan(a)

    for n in walk(f):
        if isinstance(n, Var):
            doms[n.name] = {"kind": "bool"}
        elif isinstance(n, Cmp):
            lhs, rhs = n.lhs, n.rhs
            if isinstance(lhs, Sig) and isinstance(rhs, EnumLit):
                d = doms.setdefault(lhs.name, {"kind": "enum", "values": []})
                d["kind"] = "enum"
                d.setdefault("values", [])
                if rhs.name not in d["values"]:
                    d["values"].append(rhs.name)
                continue
            if isinstance(rhs, Sig) and isinstance(lhs, EnumLit):
                d = doms.setdefault(rhs.name, {"kind": "enum", "values": []})
                d["kind"] = "enum"
                d.setdefault("values", [])
                if lhs.name not in d["values"]:
                    d["values"].append(lhs.name)
                continue
            arith_scan(lhs); arith_scan(rhs)
            # 记录"信号 op 常数"里的阈值
            if isinstance(lhs, Sig) and isinstance(rhs, Num):
                thresholds.setdefault(lhs.name, []).append(float(rhs.value))
            if isinstance(rhs, Sig) and isinstance(lhs, Num):
                thresholds.setdefault(rhs.name, []).append(float(lhs.value))

    for name, d in doms.items():
        if d["kind"] == "enum":
            vals = list(d.get("values") or [])
            # 补一个语料里没出现过的取值:否则"变量必取这几个值之一"会变成隐含
            # 前提,过弱的候选规约就白白蒙对了。
            d["values"] = vals + ["__other__"]
        elif d["kind"] == "real":
            ts = sorted(set(thresholds.get(name, [])))
            d["thresholds"] = ts
            if ts:
                lo, hi = min(ts), max(ts)
                pad = max(1.0, (hi - lo) * 0.6)
                d["range"] = [lo - pad, hi + pad]
            else:
                d["range"] = [-10.0, 10.0]
    if extra_enum:
        for name, vals in extra_enum.items():
            if name in doms and doms[name]["kind"] != "bool":
                doms[name] = {"kind": "enum", "values": list(vals) + ["__other__"]}
    return doms


# --------------------------------------------------------------------------- #
def time_scale(f, h_cap: int = 24, slack: int = 4):
    """给公式定 (dt, H)。

    区间界都装得下时 ``dt=1``;装不下就把采样率放粗到刚好让最大界落在
    ``h_cap - slack`` 步之内。放粗采样只改变"每步代表多少时间",不改变公式语义,
    而且候选与参考公式共用同一 (dt, H),判决因此仍然是公平的。
    """
    bounds = []
    for n in walk(f):
        if isinstance(n, TIMED):
            for b in (n.lo, n.hi):
                if b != INF and b > 0:
                    bounds.append(float(b))
    nest = _next_depth(f)
    need = max(bounds) if bounds else 0.0
    if need <= h_cap - slack:
        dt = 1.0
        H = int(max(nest + 3, math.ceil(need) + slack, 8))
        return dt, min(H, h_cap)
    dt = need / float(h_cap - slack)
    return dt, h_cap


def _next_depth(f) -> int:
    from .tlast import Next, Prev, children
    best = 0
    stack = [(f, 0)]
    while stack:
        node, d = stack.pop()
        d2 = d + 1 if isinstance(node, (Next, Prev)) else d
        best = max(best, d2)
        for k in children(node):
            stack.append((k, d2))
    return best
