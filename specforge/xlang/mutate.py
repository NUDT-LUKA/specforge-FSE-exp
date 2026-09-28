"""规约变异算子族(受控注入臂)。

三个族,分别对应候选规约相对意图的三种关系:

* **W(weakening)**:结果被 phi* 蕴含但更弱 —— 验证照样通过,错误是**沉默的**。
* **S(strengthening)**:结果比 phi* 强 —— 验证会失败,错误是**自曝的**。
* **N(neutral)**:方向不定的改写(交换算子、平移区间、取反原子),用来做
  "中性变异"臂,量化"自然发生的规约错误里有多大比例是沉默的"。

算子作用点带**极性**:把一个子公式换成 ``true`` 只有在正极性位置上才是弱化,
在否定/蕴含前件下反而是强化。极性算错会让"受控弱化"这条臂彻底失去意义,所以
这里显式维护极性,而不是靠"大多数情况下成立"。
"""

from __future__ import annotations

import random

from .tlast import (
    INF,
    Always,
    And,
    Cmp,
    Const,
    Eventually,
    Historically,
    Iff,
    Implies,
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
    children,
    rebuild,
    walk,
)

POS, NEG, BOTH = 1, -1, 0


def sites(f, pol=POS, path=()):
    """产出 (path, node, polarity) —— 公式里每个可变异点。"""
    yield path, f, pol
    if isinstance(f, Not):
        yield from sites(f.x, -pol if pol else BOTH, path + (0,))
        return
    if isinstance(f, Implies):
        yield from sites(f.a, -pol if pol else BOTH, path + (0,))
        yield from sites(f.b, pol, path + (1,))
        return
    if isinstance(f, Iff):
        yield from sites(f.a, BOTH, path + (0,))
        yield from sites(f.b, BOTH, path + (1,))
        return
    for i, k in enumerate(children(f)):
        yield from sites(k, pol, path + (i,))


def at(f, path):
    for i in path:
        f = children(f)[i]
    return f


def replace(f, path, new):
    if not path:
        return new
    kids = list(children(f))
    kids[path[0]] = replace(kids[path[0]], path[1:], new)
    return rebuild(f, tuple(kids))


def _atoms_pool(f):
    out = []
    for n in walk(f):
        if isinstance(n, (Var, Cmp)):
            out.append(n)
    return out


def _shift(v, d, floor=None):
    if v == INF:
        return INF
    v2 = v + d
    if floor is not None:
        v2 = max(floor, v2)
    return v2


# --------------------------------------------------------------------------- #
# 单点候选生成:给定一个作用点,返回 [(op_name, family, new_node), ...]
# --------------------------------------------------------------------------- #
def _candidates(node, pol, pool, rnd):
    out = []
    W, S, N = "W", "S", "N"

    if isinstance(node, And) and len(node.xs) >= 2:
        i = rnd.randrange(len(node.xs))
        rest = tuple(x for j, x in enumerate(node.xs) if j != i)
        drop = rest[0] if len(rest) == 1 else And(rest)
        out.append(("drop_conjunct", W if pol == POS else S, drop))
        if pool:
            out.append(("add_conjunct", S if pol == POS else W,
                        And(node.xs + (rnd.choice(pool),))))
        out.append(("and_to_or", N, Or(node.xs)))

    if isinstance(node, Or) and len(node.xs) >= 2:
        i = rnd.randrange(len(node.xs))
        rest = tuple(x for j, x in enumerate(node.xs) if j != i)
        keep = rest[0] if len(rest) == 1 else Or(rest)
        out.append(("drop_disjunct", S if pol == POS else W, keep))
        if pool:
            out.append(("add_disjunct", W if pol == POS else S,
                        Or(node.xs + (rnd.choice(pool),))))
        out.append(("or_to_and", N, And(node.xs)))

    if isinstance(node, Implies):
        if pool:
            out.append(("strengthen_antecedent", W if pol == POS else S,
                        Implies(And((node.a, rnd.choice(pool))), node.b)))
            out.append(("weaken_consequent", W if pol == POS else S,
                        Implies(node.a, Or((node.b, rnd.choice(pool))))))
            out.append(("weaken_antecedent", S if pol == POS else W,
                        Implies(Or((node.a, rnd.choice(pool))), node.b)))
        out.append(("implies_to_and", S if pol == POS else W, And((node.a, node.b))))
        out.append(("implies_to_iff", N, Iff(node.a, node.b)))

    if isinstance(node, Always):
        out.append(("G_to_F", W if pol == POS else S, Eventually(node.x, node.lo, node.hi)))
        out.append(("drop_G", W if pol == POS else S, node.x))
        if node.hi != INF:
            out.append(("G_shrink_window", W if pol == POS else S,
                        Always(node.x, node.lo, max(node.lo, node.hi - max(1.0, 0.3 * (node.hi - node.lo))))))
            out.append(("G_widen_window", S if pol == POS else W,
                        Always(node.x, node.lo, node.hi + max(1.0, 0.3 * (node.hi - node.lo)))))
        else:
            out.append(("G_shrink_window", W if pol == POS else S, Always(node.x, node.lo, 3.0)))
        out.append(("G_shift_window", N, Always(node.x, _shift(node.lo, 1.0, 0.0), _shift(node.hi, 1.0))))

    if isinstance(node, Eventually):
        out.append(("F_to_G", S if pol == POS else W, Always(node.x, node.lo, node.hi)))
        if node.hi != INF:
            out.append(("F_widen_window", W if pol == POS else S,
                        Eventually(node.x, node.lo, node.hi + max(1.0, 0.5 * (node.hi - node.lo)))))
            out.append(("F_shrink_window", S if pol == POS else W,
                        Eventually(node.x, node.lo, max(node.lo, node.hi - max(1.0, 0.3 * (node.hi - node.lo))))))
        out.append(("F_shift_window", N, Eventually(node.x, _shift(node.lo, 1.0, 0.0), _shift(node.hi, 1.0))))

    if isinstance(node, Until):
        out.append(("U_to_F", W if pol == POS else S, Eventually(node.b, node.lo, node.hi)))
        out.append(("U_swap_operands", N, Until(node.b, node.a, node.lo, node.hi)))
        out.append(("U_to_and", S if pol == POS else W, And((node.a, Eventually(node.b, node.lo, node.hi)))))

    if isinstance(node, Release):
        out.append(("V_to_G", N, Always(node.b, node.lo, node.hi)))

    if isinstance(node, Next):
        out.append(("X_to_F", W if pol == POS else S, Eventually(node.x)))
        out.append(("X_to_G", S if pol == POS else W, Always(node.x)))
        out.append(("X_drop", N, node.x))

    if isinstance(node, Historically):
        out.append(("H_to_O", W if pol == POS else S, Once(node.x, node.lo, node.hi)))
        out.append(("drop_H", W if pol == POS else S, node.x))
        if node.hi != INF:
            out.append(("H_shrink_window", W if pol == POS else S,
                        Historically(node.x, node.lo, max(node.lo, node.hi * 0.6))))

    if isinstance(node, Once):
        out.append(("O_to_H", S if pol == POS else W, Historically(node.x, node.lo, node.hi)))
        if node.hi != INF:
            out.append(("O_widen_window", W if pol == POS else S,
                        Once(node.x, node.lo, node.hi * 1.5)))

    if isinstance(node, Since):
        out.append(("S_to_O", W if pol == POS else S, Once(node.b, node.lo, node.hi)))
        out.append(("S_swap_operands", N, Since(node.b, node.a, node.lo, node.hi)))

    if isinstance(node, Prev):
        out.append(("Y_to_Z" if not node.weak else "Z_to_Y",
                    (W if not node.weak else S) if pol == POS else (S if not node.weak else W),
                    Prev(node.x, weak=not node.weak)))

    if isinstance(node, Cmp):
        op, lhs, rhs = node.op, node.lhs, node.rhs
        num = rhs if isinstance(rhs, Num) else (lhs if isinstance(lhs, Num) else None)
        if num is not None:
            d = max(abs(float(num.value)) * 0.15, 0.5)
            def shifted(delta):
                nn = Num(float(num.value) + delta)
                return Cmp(op, nn if lhs is num else lhs, nn if rhs is num else rhs)
            loosen = -d if op in (">", ">=") else d
            if op in ("<", "<=", ">", ">="):
                out.append(("cmp_loosen", W if pol == POS else S, shifted(loosen if rhs is num else -loosen)))
                out.append(("cmp_tighten", S if pol == POS else W, shifted(-loosen if rhs is num else loosen)))
            if op == "==":
                out.append(("eq_to_ineq", W if pol == POS else S,
                            Cmp(">=", lhs, Num(float(num.value) - d)) if rhs is num
                            else Cmp("<=", Num(float(num.value) - d), rhs)))
        if op in ("<", ">"):
            out.append(("strict_to_nonstrict", W if pol == POS else S,
                        Cmp("<=" if op == "<" else ">=", lhs, rhs)))
        if op in ("<=", ">="):
            out.append(("nonstrict_to_strict", S if pol == POS else W,
                        Cmp("<" if op == "<=" else ">", lhs, rhs)))
        if op == "==":
            out.append(("eq_to_neq", N, Cmp("!=", lhs, rhs)))
        out.append(("negate_atom", N, Not(node)))

    if isinstance(node, Var):
        out.append(("negate_atom", N, Not(node)))
        if pool:
            alt = [a for a in pool if a != node]
            if alt:
                out.append(("swap_atom", N, rnd.choice(alt)))

    # 极性明确时,"整块置真/置假"是最纯粹的弱化/强化
    if pol in (POS, NEG) and not isinstance(node, Const):
        out.append(("subformula_to_true", W if pol == POS else S, Const(True)))
        out.append(("subformula_to_false", S if pol == POS else W, Const(False)))
    return out


# --------------------------------------------------------------------------- #
def mutants(phi, family=None, limit=40, seed=0):
    """枚举 phi 的变异体。返回 [(op, family, formula), ...],已按 op 去重。

    ``family`` 传 ``"W"`` / ``"S"`` / ``"N"`` 时只出该族;传 None 出全部。
    """
    rnd = random.Random(seed)
    pool = _atoms_pool(phi)
    seen, out = set(), []
    all_sites = list(sites(phi))
    rnd.shuffle(all_sites)
    for path, node, pol in all_sites:
        for op, fam, new in _candidates(node, pol, pool, rnd):
            if family and fam != family:
                continue
            cand = replace(phi, path, new)
            key = (op, _key(cand))
            if key in seen or _key(cand) == _key(phi):
                continue
            seen.add(key)
            out.append((op, fam, cand))
            if len(out) >= limit:
                return out
    return out


def _key(f):
    from .tlast import to_str
    return to_str(f)
