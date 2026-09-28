"""SystemVerilog Assertions(SVA)子集 -> 统一 IR。

SVA 是五种语言里唯一以**序列正则**为骨架的:``a ##3 b |-> ##[1:4] c``。把它压到
同一棵时序 IR 上,靠的是"序列 = (在起点求值的公式, 到序列末端的时钟偏移)"这一对
表示:

    e1 ##k e2          ->  (e1 & X^k e2, k)
    S |-> P            ->  S.f -> X^{S.d} P
    S |=> P            ->  S.f -> X^{S.d+1} P
    assert property(P) ->  G(P)                    (每个时钟沿都要成立)
    disable iff (c)    ->  G(!c -> P)

信号建模:FVEval 的信号是多位向量,这里按**整数**建模(论域 0..15),于是位归约
算子有真实语义而不是三个互不相干的原子:

    &x -> x == 15 ,  |x -> x != 0 ,  ~|x -> x == 0 ,  ~&x -> x != 15

``^x``(奇偶归约)整数论域下表达不了,退化成一个不透明命题原子;``$stable`` 按
"布尔读数不变"近似。两处近似对参考公式与候选公式**同样施加**,所以比较仍然公平,
但会让语义关系判定偏保守 —— 这个方向不会夸大结论。
"""

from __future__ import annotations

import re

from .tlast import (
    And,
    Func,
    Always,
    Cmp,
    Const,
    Eventually,
    Iff,
    Implies,
    Next,
    Not,
    Num,
    Or,
    Prev,
    Sig,
    Until,
    Var,
)

WIDTH_MAX = 15.0          # 位向量论域上界(4 位);归约与算术都按这个宽度解释


class SVAParseError(Exception):
    pass


_TOK = re.compile(r"""
    (?P<ws>\s+|//[^\n]*)
  | (?P<sysf>\$[A-Za-z_][A-Za-z_0-9]*)
  | (?P<sized>\d*'[sS]?[bBhHdDoO][0-9a-fA-FxXzZ_]+)
  | (?P<unsized>'[01])
  | (?P<num>\d+\.\d+|\d+)
  | (?P<id>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>\#\#|\|->|\|=>|===|!==|==|!=|<=|>=|&&|\|\||~\||~&|[()\[\]{},;:@!~&|^<>+\-*/?.$])
""", re.X)


def _tokenize(s):
    toks, i = [], 0
    while i < len(s):
        m = _TOK.match(s, i)
        if not m:
            raise SVAParseError("bad char %r in %r" % (s[i], s[max(0, i - 30):i + 30]))
        i = m.end()
        if m.lastgroup == "ws":
            continue
        toks.append((m.lastgroup, m.group()))
    toks.append(("eof", ""))
    return toks


def _sized_value(txt):
    """``4'b1010`` / ``1'b1`` -> 数值;含 x/z 的按不可判处理。"""
    m = re.match(r"(\d*)'([sS]?)([bBhHdDoO])([0-9a-fA-FxXzZ_]+)", txt)
    if not m:
        raise SVAParseError("bad sized literal " + txt)
    base, digits = m.group(3).lower(), m.group(4).replace("_", "")
    if re.search(r"[xXzZ]", digits):
        raise SVAParseError("x/z literal not modelled: " + txt)
    return float(int(digits, {"b": 2, "o": 8, "d": 10, "h": 16}[base]))


class SVAParser:
    def __init__(self, src):
        self.toks = _tokenize(src)
        self.i = 0
        self.src = src

    # -- helpers ------------------------------------------------------------ #
    def k(self, o=0):
        return self.toks[min(self.i + o, len(self.toks) - 1)][0]

    def t(self, o=0):
        return self.toks[min(self.i + o, len(self.toks) - 1)][1]

    def bump(self):
        v = self.toks[self.i]
        self.i += 1
        return v

    def eat(self, txt):
        if self.t() != txt:
            raise SVAParseError("expected %r got %r" % (txt, self.t()))
        return self.bump()[1]

    def opt(self, txt):
        if self.t() == txt:
            self.bump()
            return True
        return False

    # -- 顶层 ---------------------------------------------------------------- #
    def parse_assertion(self):
        # 可能带标签 `asrt:`
        if self.k() == "id" and self.t(1) == ":":
            self.bump(); self.bump()
        if self.t() in ("assert", "assume", "cover"):
            self.bump()
        self.eat("property")
        self.eat("(")
        if self.opt("@"):
            self.eat("(")
            depth = 1
            while depth:                       # 时钟事件表达式整段跳过
                tt = self.bump()[1]
                if tt == "(":
                    depth += 1
                elif tt == ")":
                    depth -= 1
                elif tt == "":
                    raise SVAParseError("unterminated clocking event")
        guard = None
        if self.t() == "disable":
            self.bump()
            self.eat("iff")
            self.eat("(")
            guard = self.p_bool()
            self.eat(")")
        body = self.p_property()
        self.eat(")")
        self.opt(";")
        if self.k() != "eof":
            raise SVAParseError("trailing %r" % (self.t(),))
        if guard is not None:
            body = Implies(Not(guard), body)
        return Always(body)

    # -- property / sequence ------------------------------------------------- #
    def p_property(self):
        if self.t() in ("s_eventually", "eventually"):
            self.bump()
            return Eventually(self.p_property())
        if self.t() == "always":
            self.bump()
            return Always(self.p_property())
        if self.t() == "not":
            self.bump()
            return Not(self.p_property())
        if self.t() == "##":
            f, d, iv = self.p_delay_head()
            if iv is not None:
                return Eventually(self.p_property(), iv[0], iv[1])
            return _shift(self.p_property(), d)

        f, d = self.p_sequence()
        op = self.t()
        if op in ("|->", "|=>"):
            self.bump()
            extra = 0 if op == "|->" else 1
            if self.t() == "##":
                _, dd, iv = self.p_delay_head()
                cons = self.p_property()
                if iv is not None:
                    return Implies(f, _shift(Eventually(cons, iv[0], iv[1]), d + extra))
                return Implies(f, _shift(cons, d + extra + dd))
            return Implies(f, _shift(self.p_property(), d + extra))
        if self.t() in ("until", "s_until", "until_with", "s_until_with"):
            self.bump()
            return Until(f, self.p_property())
        if d:
            raise SVAParseError("bare sequence with delay is not a property")
        return f

    def p_delay_head(self):
        """吃掉一个 ``##k`` 或 ``##[a:b]``,返回 (None, k, interval_or_None)。"""
        self.eat("##")
        if self.t() == "[":
            self.bump()
            a = self.p_int()
            self.eat(":")
            if self.t() == "$":
                self.bump()
                b = float("inf")
            else:
                b = float(self.p_int())
            self.eat("]")
            return None, 0, (float(a), b)
        return None, self.p_int(), None

    def p_int(self):
        if self.k() == "num":
            return int(float(self.bump()[1]))
        if self.k() == "sized":
            return int(_sized_value(self.bump()[1]))
        if self.t() == "$":
            self.bump()
            return 8
        raise SVAParseError("expected delay count got %r" % (self.t(),))

    def p_sequence(self):
        """返回 (在起点求值的公式, 序列末端相对起点的时钟偏移)。"""
        f = self.p_bool()
        d = 0
        while self.t() == "##":
            _, k, iv = self.p_delay_head()
            if iv is not None:
                raise SVAParseError("ranged delay inside a sequence is unsupported")
            d += k
            f = And((f, _shift(self.p_bool(), d)))
        return f, d

    # -- 布尔表达式(双模:公式 / 算术项)------------------------------------- #
    def p_bool(self):
        v = self.p_or()
        return _as_formula(v)

    def p_or(self):
        a = self.p_and()
        while self.t() == "||":
            self.bump()
            a = ("f", Or((_as_formula(a), _as_formula(self.p_and()))))
        return a

    def p_and(self):
        a = self.p_bitor()
        while self.t() == "&&":
            self.bump()
            a = ("f", And((_as_formula(a), _as_formula(self.p_bitor()))))
        return a

    # Verilog 位运算优先级:& 高于 ^ 高于 |,三者都低于等值、高于 &&。
    # 位运算在整数论域上没法用四则表达,统一压成未解释函数 bitand/bitor/bitxor:
    # 监控器里按真实整数位运算求值,SMT 侧当 UF —— 后者更保守,不会夸大结论。
    def p_bitor(self):
        a = self.p_bitxor()
        while self.t() == "|" and self.t(1) != "|":
            self.bump()
            a = ("a", Func("bitor", (_as_arith_loose(a), _as_arith_loose(self.p_bitxor()))))
        return a

    def p_bitxor(self):
        a = self.p_bitand()
        while self.t() == "^":
            self.bump()
            a = ("a", Func("bitxor", (_as_arith_loose(a), _as_arith_loose(self.p_bitand()))))
        return a

    def p_bitand(self):
        a = self.p_eq()
        while self.t() == "&" and self.t(1) != "&":
            self.bump()
            a = ("a", Func("bitand", (_as_arith_loose(a), _as_arith_loose(self.p_eq()))))
        return a

    def p_eq(self):
        a = self.p_rel()
        while self.t() in ("==", "!=", "===", "!=="):
            op = self.bump()[1]
            b = self.p_rel()
            neg = op in ("!=", "!==")
            if a[0] == "f" or b[0] == "f":
                # 一侧已经是命题:等值退化成同或
                node = Iff(_as_formula(a), _as_formula(b))
            else:
                node = Cmp("==", a[1], b[1])
            a = ("f", Not(node) if neg else node)
        return a

    def p_rel(self):
        a = self.p_add()
        while self.t() in ("<", ">", "<=", ">="):
            op = self.bump()[1]
            b = self.p_add()
            a = ("f", Cmp(op, _as_arith_loose(a), _as_arith_loose(b)))
        return a

    def p_add(self):
        from .tlast import BinArith
        a = self.p_unary()
        while self.t() in ("+", "-"):
            op = self.bump()[1]
            a = ("a", BinArith(op, _as_arith_loose(a), _as_arith_loose(self.p_unary())))
        return a

    def p_unary(self):
        t = self.t()
        if t in ("!",):
            self.bump()
            return ("f", Not(_as_formula(self.p_unary())))
        if t == "~":
            self.bump()
            return ("f", Not(_as_formula(self.p_unary())))
        if t in ("&", "|", "^", "~|", "~&"):
            self.bump()
            inner = self.p_unary()
            return ("f", _reduction(t, inner))
        if t == "-":
            from .tlast import NegArith
            self.bump()
            return ("a", NegArith(_as_arith_loose(self.p_unary())))
        return self.p_primary()

    def p_primary(self):
        if self.t() == "{":                     # 位拼接:整体当一个未解释函数
            self.bump()
            args = [_as_arith_loose(self.p_or())]
            while self.t() == ",":
                self.bump()
                args.append(_as_arith_loose(self.p_or()))
            self.eat("}")
            return ("a", Func("concat", tuple(args)))
        if self.t() in ("strong", "weak") and self.t(1) == "(":
            self.bump(); self.eat("(")
            v = self.p_property()
            self.eat(")")
            return ("f", v)
        if self.t() == "(":
            # 括号里既可能是布尔表达式,也可能是完整 property(嵌套的 |-> / ##)。
            save = self.i
            try:
                self.bump()
                v = self.p_or()
                self.eat(")")
                return v
            except SVAParseError:
                self.i = save
            self.bump()
            v = self.p_property()
            self.eat(")")
            return ("f", v)
        if self.k() == "sysf":
            return self.p_sysfunc()
        if self.k() == "sized":
            return ("a", Num(_sized_value(self.bump()[1])))
        if self.k() == "unsized":
            # `'1` 是"所有位为 1",在本文的 4 位整数建模里就是 WIDTH_MAX;`'0` 是 0
            return ("a", Num(WIDTH_MAX if self.bump()[1] == "'1" else 0.0))
        if self.k() == "num":
            return ("a", Num(float(self.bump()[1])))
        if self.k() == "id":
            name = self.bump()[1]
            if name in ("1", "0"):
                return ("a", Num(float(name)))
            if self.t() == "[":                 # 位选 / 位段:整体当一个新信号
                depth, buf = 0, ""
                while True:
                    tt = self.bump()[1]
                    if tt == "[":
                        depth += 1
                    elif tt == "]":
                        depth -= 1
                        if depth == 0:
                            break
                    elif tt == "":
                        raise SVAParseError("unterminated bit-select")
                    buf += tt
                name = "%s_%s" % (name, re.sub(r"\W+", "_", buf))
            return ("a", Sig(name))
        raise SVAParseError("cannot parse primary at %r" % (self.t(),))

    def p_sysfunc(self):
        fn = self.bump()[1]
        self.eat("(")
        arg = self.p_or()
        n = 1
        if self.t() == ",":
            self.bump()
            n = self.p_int()
        self.eat(")")
        f = _as_formula(arg)
        if fn == "$past":
            out = f
            for _ in range(max(1, n)):
                out = Prev(out)
            return ("f", out)
        if fn in ("$rose",):
            return ("f", And((f, Prev(Not(f)))))
        if fn in ("$fell",):
            return ("f", And((Not(f), Prev(f))))
        if fn in ("$stable",):
            return ("f", Iff(f, Prev(f, weak=True)))
        if fn in ("$changed",):
            return ("f", Not(Iff(f, Prev(f, weak=True))))
        if fn in ("$onehot", "$onehot0"):
            # 整数论域下"至多/恰好一位为 1"就是取值落在 2 的幂(含 0)上
            val = _as_arith_loose(arg)
            pows = [0.0, 1.0, 2.0, 4.0, 8.0]
            if fn == "$onehot":
                pows = pows[1:]
            return ("f", Or(tuple(Cmp("==", val, Num(v)) for v in pows)))
        if fn == "$isunknown":
            return ("f", Const(False))          # 本建模里不引入 x/z
        if fn == "$countones":
            return ("a", Func("countones", (_as_arith_loose(arg),)))
        raise SVAParseError("unknown system function " + fn)


# --------------------------------------------------------------------------- #
def _shift(f, d):
    for _ in range(int(d)):
        f = Next(f)
    return f


def _as_formula(v):
    kind, node = v
    if kind == "f":
        return node
    if isinstance(node, Num):
        return Const(bool(node.value))
    return Cmp("!=", node, Num(0.0))          # 非零即真


def _as_arith(v):
    kind, node = v
    if kind == "a":
        return node
    raise SVAParseError("proposition used where a value is required")


def _as_arith_loose(v):
    """算术/位运算位置上的取值:命题按 0/1 提升(Verilog 就是这么做的)。"""
    kind, node = v
    if kind == "a":
        return node
    return Func("b2i", (node,))


def _reduction(op, inner):
    kind, node = inner
    if kind == "f":
        # 对已经是命题的东西做位归约:1 位向量,& 和 | 都退化成它本身
        return node if op in ("&", "|") else Not(node)
    if op == "&":
        return Cmp("==", node, Num(WIDTH_MAX))
    if op == "|":
        return Cmp("!=", node, Num(0.0))
    if op == "~|":
        return Cmp("==", node, Num(0.0))
    if op == "~&":
        return Cmp("!=", node, Num(WIDTH_MAX))
    if op == "^":
        from .tlast import arith_str
        return Var("parity__" + re.sub(r"\W+", "_", arith_str(node)))
    raise SVAParseError("bad reduction " + op)


def parse_sva(src: str):
    """解析一条 ``assert property (...)``,返回统一 IR。"""
    s = " ".join(src.split())
    return SVAParser(s).parse_assertion()
