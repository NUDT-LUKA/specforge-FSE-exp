"""表层语法 -> 统一 IR 的通用前端。

LTL / MTL / STL / ptLTL 四种语言在纸面上写法各异(``G`` vs ``always`` vs
``globally`` vs ``[]``),但文法骨架是同一套中缀时序逻辑,所以这里只写一个
可配置的递归下降解析器,用关键词表吸收差异。SVA 的序列/蕴含语法差得远,单独放
``parse_sva.py``。

另外提供 ``parse_prefix``:Efficient-Eng-2-LTL 的 clean-up / pick-and-place 子集
用的是波兰前缀式(``F & B F C``),不走中缀通道。
"""

from __future__ import annotations

import re

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
)


class ParseError(Exception):
    pass


# --------------------------------------------------------------------------- #
# 词法
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>\d+\.\d+|\.\d+|\d+)
  | (?P<id>[A-Za-z_][A-Za-z_0-9.']*)
  | (?P<inf>∞)
  | (?P<op><->|<=>|->|=>|\|\||&&|/\\|\\/|<=|>=|==|!=|<>|\[\]|<>|\.\.|[<>=!&|~^()\[\],:;+\-*/])
""", re.X)

# 上面 op 分支里 `<>` 出现两次是有意的:第一次给 "不等于"(某些语料),第二次给
# LTL 的 eventually。谁先匹配由这里的顺序决定 —— 我们把它统一当 eventually,
# 不等于一律写 `!=`,数据集里没有反例。


def tokenize(s: str):
    toks, i = [], 0
    while i < len(s):
        m = _TOKEN_RE.match(s, i)
        if not m:
            raise ParseError("bad char %r at %d in %r" % (s[i], i, s[:80]))
        i = m.end()
        if m.lastgroup == "ws":
            continue
        toks.append((m.lastgroup, m.group()))
    toks.append(("eof", ""))
    return toks


# --------------------------------------------------------------------------- #
# 关键词表(小写匹配)
# --------------------------------------------------------------------------- #
ALWAYS = {"g", "always", "globally", "alw", "[]"}
EVENTUALLY = {"f", "eventually", "finally", "ev", "<>"}
NEXT = {"x", "next"}
WEAK_NEXT = {"wx"}
HIST = {"h", "historically", "hist"}
ONCE = {"o", "once"}
PREV_STRONG = {"y", "prev", "pre", "yesterday"}
PREV_WEAK = {"z", "wprev"}
UNTIL = {"u", "until"}
RELEASE = {"v", "r", "release"}
SINCE = {"s", "si", "since"}
NOT = {"!", "~", "not", "negation", "neg"}
AND = {"&", "&&", "and", "/\\"}
OR = {"|", "||", "or", "\\/"}
IMPLIES = {"->", "=>", "imply", "implies"}
IFF = {"<->", "<=>", "equal", "iff", "equivalent"}
TRUE = {"true", "1'b1"}
FALSE = {"false", "1'b0"}
CMP_OPS = {"<", "<=", ">", ">=", "==", "!=", "===", "!=="}
# STL 语料里的边沿函数,展开成 IR 里已有的过去算子
EDGE = {"rise", "fall", "rose", "fell"}
ARITH_FUNCS = {"abs", "sin", "cos", "tan", "sqrt", "exp", "floor", "ceil",
               "atan", "asin", "acos", "min", "max", "mod", "pow"}


class Parser:
    """可配置的中缀时序逻辑解析器。

    ``bool_atoms``:声明为布尔命题的标识符集合(其余标识符在比较里当信号)。
    ``enum_of``:变量名 -> 该变量的枚举取值集合,用来把 ``mode = idle`` 的右侧
    识别成枚举字面量而不是另一个变量。``eq_is_cmp`` 决定单个 ``=`` 是不是等号
    (FRET 是,nl2spec 不是)。
    """

    def __init__(self, src: str, bool_atoms=None, enum_of=None, eq_is_cmp=True):
        self.toks = tokenize(src)
        self.i = 0
        self.src = src
        self.bool_atoms = set(bool_atoms or ())
        self.enum_of = dict(enum_of or {})
        self.eq_is_cmp = eq_is_cmp

    # -- token helpers ------------------------------------------------------ #
    def peek(self, k=0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def kind(self, k=0):
        return self.peek(k)[0]

    def text(self, k=0):
        return self.peek(k)[1]

    def low(self, k=0):
        return self.peek(k)[1].lower()

    def bump(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, txt):
        if self.text() != txt:
            raise ParseError("expected %r got %r in %r" % (txt, self.text(), self.src[:90]))
        return self.bump()

    def at_word(self, words) -> bool:
        return self.low() in words

    # -- entry -------------------------------------------------------------- #
    def parse(self):
        f = self.p_iff()
        if self.kind() != "eof":
            raise ParseError("trailing %r in %r" % (self.text(), self.src[:90]))
        return f

    # -- 区间 [a,b] / [a:b] -------------------------------------------------- #
    def maybe_interval(self):
        if self.text() != "[":
            return 0.0, INF
        # `[]` 已在词法阶段合成一个 token,所以这里的 `[` 一定是区间开头
        save = self.i
        self.bump()
        try:
            lo = self.p_number()
            if self.text() not in (",", ":"):
                self.i = save
                return 0.0, INF
            self.bump()
            hi = self.p_number()
            self.expect("]")
            return lo, hi
        except ParseError:
            self.i = save
            return 0.0, INF

    def p_number(self) -> float:
        neg = False
        if self.text() == "-":
            self.bump()
            neg = True
        if self.kind() == "num":
            v = float(self.bump()[1])
            return -v if neg else v
        if self.low() in ("inf", "infinite", "infinity") or self.text() in ("∞",):
            self.bump()
            return INF
        raise ParseError("expected number got %r" % (self.text(),))

    # -- 二元层 -------------------------------------------------------------- #
    def p_iff(self):
        a = self.p_implies()
        while self.at_word(IFF):
            self.bump()
            a = Iff(a, self.p_implies())
        return a

    def p_implies(self):
        a = self.p_or()
        if self.at_word(IMPLIES):
            self.bump()
            return Implies(a, self.p_implies())
        return a

    def p_or(self):
        xs = [self.p_and()]
        while self.at_word(OR):
            self.bump()
            xs.append(self.p_and())
        return xs[0] if len(xs) == 1 else Or(tuple(xs))

    def p_and(self):
        xs = [self.p_binary_temporal()]
        while self.at_word(AND):
            self.bump()
            xs.append(self.p_binary_temporal())
        return xs[0] if len(xs) == 1 else And(tuple(xs))

    def p_binary_temporal(self):
        a = self.p_unary()
        while True:
            w = self.low()
            if w in UNTIL:
                self.bump()
                lo, hi = self.maybe_interval()
                a = Until(a, self.p_unary(), lo, hi)
            elif w in RELEASE and self.kind() == "id":
                self.bump()
                lo, hi = self.maybe_interval()
                a = Release(a, self.p_unary(), lo, hi)
            elif w in SINCE:
                self.bump()
                lo, hi = self.maybe_interval()
                a = Since(a, self.p_unary(), lo, hi)
            else:
                return a

    # -- 一元层 -------------------------------------------------------------- #
    def p_unary(self):
        w = self.low()
        if w in NOT and not (w == "!" and self.text(1) == "="):
            self.bump()
            return Not(self.p_unary())
        split = self._split_glued_operators()
        if split is not None:
            letters, rest = split
            self.bump()
            self.toks.insert(self.i, ("id", rest))
            f = self.p_unary()
            for ch in reversed(letters):
                f = _STACKED[ch](f)
            return f
        if re.fullmatch(r"[GFXHOYZ]{2,}", self.text()) and self._is_temporal_kw():
            # 叠写的单字母时序算子(`XX !a`、`GF(d)`、`FG a`),词法上是一个标识符。
            # 不展开的话会掉进"未解释谓词"兜底,把 `GF(d)` 静默读成一个原子 —— 判决
            # 照样出得来,但语义已经完全错了,这类静默错误必须在解析期就堵死。
            letters = self.bump()[1]
            f = self.p_unary()
            for ch in reversed(letters):
                f = _STACKED[ch](f)
            return f
        if w in ALWAYS and self._is_temporal_kw():
            self.bump()
            lo, hi = self.maybe_interval()
            return Always(self.p_unary(), lo, hi)
        if w in EVENTUALLY and self._is_temporal_kw():
            self.bump()
            lo, hi = self.maybe_interval()
            return Eventually(self.p_unary(), lo, hi)
        if w in HIST and self._is_temporal_kw():
            self.bump()
            lo, hi = self.maybe_interval()
            return Historically(self.p_unary(), lo, hi)
        if w in ONCE and self._is_temporal_kw():
            self.bump()
            lo, hi = self.maybe_interval()
            return Once(self.p_unary(), lo, hi)
        if w in NEXT and self._is_temporal_kw():
            self.bump()
            return Next(self.p_unary())
        if w in WEAK_NEXT and self._is_temporal_kw():
            self.bump()
            return Next(self.p_unary(), weak=True)
        if w in PREV_STRONG and self._is_temporal_kw():
            self.bump()
            return Prev(self.p_unary())
        if w in PREV_WEAK and self._is_temporal_kw():
            self.bump()
            return Prev(self.p_unary(), weak=True)
        return self.p_atom()

    def _split_glued_operators(self):
        """``XXXb`` / ``Gfoo``:算子和变量名黏在一起时拆开。

        只有在**调用方声明了变量集**(``bool_atoms``)时才启用 —— 没有声明就无从
        判断 ``Xa`` 是"next a"还是一个叫 ``Xa`` 的变量,宁可不拆。参考公式的解析
        不传变量集,所以建集口径完全不受影响;这条规则只服务于评测 LLM 输出。
        """
        if not self.bool_atoms or self.kind() != "id":
            return None
        t = self.text()
        if t in self.bool_atoms:
            return None
        m = re.match(r"^([GFXHOYZ]+)(.+)$", t)
        if not m:
            return None
        letters, rest = m.group(1), m.group(2)
        if rest in self.bool_atoms:
            return letters, rest
        return None

    def _is_temporal_kw(self) -> bool:
        """单字母时序算子(G/F/X/H/O/Y/Z/V/S/U)与同名变量的消歧。

        判据:后面紧跟 `(`、`[`、`!`、另一个时序算子或标识符时当算子;后面跟
        比较号/二元算符/右括号/eof 时当变量。这条规则在五个数据集上都成立,
        踩不中的会在解析期直接抛错而不是静默错判。
        """
        t = self.text()
        if len(t) > 1 and t.lower() not in ("wx",):
            return True                      # always / eventually 之类的词形
        if t in self.bool_atoms:
            return False
        nk, nt = self.kind(1), self.text(1)
        if nt in ("(", "[", "!", "~", "-"):
            return True
        if nk == "id" or nk == "num":
            return True
        return False

    # -- 原子 ---------------------------------------------------------------- #
    def p_atom(self):
        if self.text() == "(":
            # 括号既可能包一个子公式,也可能包一段**算术**(FRET 里
            # `((Psi - a) / b) <= 0.1` 就是后者)。先按子公式试,失败或者
            # 括号闭合后紧跟比较号时回退走算术通道 —— 两种写法都不丢。
            save = self.i
            try:
                self.bump()
                f = self.p_iff()
                self.expect(")")
                if self._cmp_op() is None:
                    return f
            except ParseError:
                pass
            self.i = save
        w = self.low()
        if w in TRUE:
            self.bump()
            return Const(True)
        if w in FALSE:
            self.bump()
            return Const(False)
        if self.text() == "LAST":
            self.bump()
            return Last()
        if w in EDGE and self.text(1) == "(":
            self.bump()
            self.expect("(")
            inner = self.p_iff()
            self.expect(")")
            if w in ("rise", "rose"):
                return And((inner, Prev(Not(inner))))
            return And((Not(inner), Prev(inner)))

        # 走比较通道:先试着当算术表达式解析,后面跟比较号就是 Cmp,否则回退成
        # 布尔原子。
        save = self.i
        try:
            lhs = self.p_arith()
        except ParseError:
            self.i = save
            lhs = None
        if lhs is not None:
            op = self._cmp_op()
            if op:
                self.bump()
                if op == "=" and not self.eq_is_cmp:
                    raise ParseError("assignment '=' where comparison not allowed")
                rhs = self.p_arith()
                return Cmp(_norm_cmp(op), lhs, self._as_enum(lhs, rhs))
            if isinstance(lhs, Sig):
                return Var(lhs.name)
            if isinstance(lhs, Func):
                # 布尔值的未解释谓词(FRET 的 `det_3x3(...)`)。整个应用当成一个
                # 不透明命题原子,名字带上实参,不同实参就是不同原子。
                from .tlast import arith_str
                return Var(arith_str(lhs))
            raise ParseError("arithmetic term used as a proposition: %r" % (self.src[:80],))
        raise ParseError("cannot parse atom at %r" % (self.text(),))

    def _cmp_op(self):
        t = self.text()
        if t in CMP_OPS:
            return t
        if t == "=" and self.eq_is_cmp:
            return "="
        if t == "<" and self.text(1) == "=":
            return "<="
        return None

    def _as_enum(self, lhs, rhs):
        """``mode = idle`` 的右侧若是该变量声明过的枚举值,转成 EnumLit。"""
        if isinstance(lhs, Sig) and isinstance(rhs, Sig):
            vals = self.enum_of.get(lhs.name)
            if vals and rhs.name in vals:
                return EnumLit(rhs.name)
        return rhs

    # -- 算术 ---------------------------------------------------------------- #
    def p_arith(self):
        return self.p_add()

    def p_add(self):
        a = self.p_mul()
        while self.text() in ("+", "-"):
            op = self.bump()[1]
            a = BinArith(op, a, self.p_mul())
        return a

    def p_mul(self):
        a = self.p_uarith()
        while self.text() in ("*", "/"):
            op = self.bump()[1]
            a = BinArith(op, a, self.p_uarith())
        return a

    def p_uarith(self):
        if self.text() == "-":
            self.bump()
            return NegArith(self.p_uarith())
        if self.text() == "(":
            self.bump()
            e = self.p_add()
            self.expect(")")
            return e
        if self.kind() == "num":
            return Num(float(self.bump()[1]))
        if self.kind() == "id":
            name = self.bump()[1]
            if self.text() == "(":
                self.bump()
                args = [self.p_add()]
                while self.text() == ",":
                    self.bump()
                    args.append(self.p_add())
                self.expect(")")
                return Func(name, tuple(args))
            return Sig(name)
        raise ParseError("bad arithmetic token %r" % (self.text(),))


_STACKED = {
    "G": lambda f: Always(f),
    "F": lambda f: Eventually(f),
    "X": lambda f: Next(f),
    "H": lambda f: Historically(f),
    "O": lambda f: Once(f),
    "Y": lambda f: Prev(f),
    "Z": lambda f: Prev(f, weak=True),
}


def _norm_cmp(op: str) -> str:
    return {"=": "==", "===": "==", "!==": "!=", "<>": "!="}.get(op, op)


def parse(src: str, bool_atoms=None, enum_of=None, eq_is_cmp=True):
    return Parser(src, bool_atoms, enum_of, eq_is_cmp).parse()


# --------------------------------------------------------------------------- #
# 波兰前缀式(Efficient-Eng-2-LTL clean-up / pick-and-place)
# --------------------------------------------------------------------------- #
_PREFIX_UNARY = {"F": Eventually, "G": Always, "X": Next, "!": Not, "~": Not}
_PREFIX_BINARY = {"&": "and", "|": "or", "U": "until", "->": "imply", "i": "imply"}


def parse_prefix(src: str, use_next: bool = True):
    """解析 ``F & B F C`` 这样的前缀式。标识符一律当布尔命题。

    ``use_next=False``:该语料把 ``X`` 当房间名之类的原子而不是 next 算子
    (Efficient-Eng-2-LTL 的 clean-up 子集就是这样),此时不要把它读成时序算子。
    """
    toks = src.replace("(", " ").replace(")", " ").split()
    pos = [0]
    unary = dict(_PREFIX_UNARY)
    if not use_next:
        unary.pop("X", None)

    def rec():
        if pos[0] >= len(toks):
            raise ParseError("prefix formula ran out of tokens: %r" % (src,))
        t = toks[pos[0]]
        pos[0] += 1
        if t in unary:
            cls = unary[t]
            return cls(rec())
        if t in _PREFIX_BINARY:
            a, b = rec(), rec()
            kind = _PREFIX_BINARY[t]
            if kind == "and":
                return And((a, b))
            if kind == "or":
                return Or((a, b))
            if kind == "until":
                return Until(a, b)
            return Implies(a, b)
        if t.lower() in TRUE:
            return Const(True)
        if t.lower() in FALSE:
            return Const(False)
        return Var(t)

    f = rec()
    if pos[0] != len(toks):
        raise ParseError("prefix formula has %d trailing tokens: %r" % (len(toks) - pos[0], src))
    return f
