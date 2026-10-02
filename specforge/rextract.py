"""R 抽取器:执行 Dafny ground_truth 程序,产出 ``(输入, 期望输出)`` 测试用例。

这是 v3 框架的使能器(设计文档 v3 §3)。Dafny 可 translate 到 Python 并执行
(``dafny translate py --include-runtime``,**无需 .NET**),因此可以:随机生成
输入 -> 用 ground_truth(已验证正确)算出期望输出 -> 得到测试用例 T。T 是自然
语言意图 R 的可执行投影,用来打破 ``S∧¬P`` 的信息对称(v2 裁决正是缺了它)。

支持的参数/返回类型见 ``_SUPPORTED``:标量(int/nat/bool/real/char)、序列
(seq<T>/string)与数组(array<T>)。其余签名跳过(返回空用例,上层据此排除)。

**目标选择(2026-08-05 扩展)**:早期版本只看文件里的**第一个 method**,
因此会漏掉目标方法排在辅助声明之后的文件。现在 ``find_targets`` 枚举全部可编译的
method/function,按"legacy 优先"排序后逐个尝试 —— 第一个候选与旧实现完全一致
(含随机数抽取顺序),故旧评测集可逐位复现,新增候选只做增量。
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass

_SIG_RE = re.compile(
    r"method\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*"
    r"(?:returns\s*\(([^)]*)\))?", re.S)
_PARAM_RE = re.compile(r"(\w+)\s*:\s*([\w<>]+)")


def parse_signature(src: str):
    """返回 (方法名, [(参数名,类型)], [(返回名,类型)]);解析失败返回 None。

    **保持旧语义**:只认文件里的第一个 ``method``。多目标枚举请用
    ``find_targets``;需要按目标定位规约子句请用 ``spec_region``。
    """
    m = _SIG_RE.search(src)
    if not m:
        return None
    name = m.group(1)
    params = _PARAM_RE.findall(m.group(2) or "")
    rets = _PARAM_RE.findall(m.group(3) or "")
    return name, params, rets


# --------------------------------------------------------------------------- #
# 类型片段
# --------------------------------------------------------------------------- #
_SCALARS = {"int", "nat", "bool", "real", "char"}
#: 序列/数组的元素类型(能生成字面量、能打印回读)
_ELEMS = {"int", "nat", "bool", "real", "char"}

_SUPPORTED = (
    _SCALARS
    | {"string"}
    | {f"seq<{e}>" for e in _ELEMS}
    | {f"array<{e}>" for e in _ELEMS}
    | {f"array?<{e}>" for e in _ELEMS}
)

_ARRAY_RE = re.compile(r"^array\??<(\w+)>$")
_SEQ_RE = re.compile(r"^seq<(\w+)>$")


def elem_type(ty: str) -> str | None:
    """array<T>/seq<T>/string 的元素类型;标量返回 None。"""
    if ty == "string":
        return "char"
    m = _ARRAY_RE.match(ty) or _SEQ_RE.match(ty)
    return m.group(1) if m else None


def is_array(ty: str) -> bool:
    return bool(_ARRAY_RE.match(ty))


# 边界值优先:off-by-one / 比较符翻转类缺陷只在边界处显形,
# 随机大整数往往触发不了。以 _BOUNDARY_PROB 的概率注入边界值,提高 E-code 见证率。
# 这些常量被 SpecForge-tex/scripts/make_extractor_macros.py 读出为论文宏,
# 改动会直接反映到论文正文 —— 不要在 tex 里手抄。
_BOUNDARY_INT = [0, 1, -1, 2, 3, -2]
_BOUNDARY_PROB = 0.6
_BOUNDARY_REAL = [0.0, 1.0, -1.0, 0.5, 2.0, -0.5]
_ALPHABET = "abcdez"

DEFAULT_N_TESTS = 3        # 每个任务保留的测试条数上限
DEFAULT_TIMEOUT = 40       # translate + 执行的秒数上限
RESAMPLE_FACTOR = 3        # 采样尝试次数 = DEFAULT_N_TESTS * RESAMPLE_FACTOR


# --------------------------------------------------------------------------- #
# 目标(method / function)枚举与规约作用域
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    """文件里一个可执行、可打分的目标声明。"""
    kind: str                       # "method" | "function"
    name: str
    params: list                    # [(名, 类型)]
    rets: list                      # [(名, 类型)]
    sig_start: int                  # 声明起点在 src 中的偏移
    body_start: int                 # 方法体 '{' 的偏移(-1 = 未找到)

    @property
    def signature(self):
        return self.name, self.params, self.rets


_DECL_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:(ghost|twostate|least|greatest)[ \t]+)?"
    r"(method|function|predicate)[ \t]+(?:method[ \t]+)?"
    #: 签名可以跨行 —— DafnyBench 里 ``returns (...)`` 常另起一行,用 [ \t]*
    #: 会整片漏掉这些文件(实测 Tangent/max 两例)。
    r"(?:\{[^}]*\}\s*)*(\w+)\s*(<[^>]*>)?\s*\(([^)]*)\)"
    r"\s*(?:returns\s*\(([^)]*)\)|:\s*([\w<>?]+))?", re.S)
#: 允许 ``a, b: int`` 这种共享类型的参数组(旧 _PARAM_RE 会漏掉 a)
_PARAM_GROUP_RE = re.compile(
    r"([\w\s,]+?)\s*:\s*([\w<>?]+)(?:\s*,\s*(?=[\w\s]+:)|\s*$)")


def _split_params(text: str):
    out = []
    for m in _PARAM_GROUP_RE.finditer(text or ""):
        ty = m.group(2)
        out += [(n.strip(), ty) for n in m.group(1).split(",") if n.strip()]
    return out


def _body_offset(src: str, from_pos: int) -> int:
    """从签名结束处找方法体 '{'(跳过 requires/ensures/reads/... 子句)。"""
    i = src.find("{", from_pos)
    return i


def find_targets(src: str) -> list[Target]:
    """枚举文件里所有类型受支持、带返回值的 method/function。

    排序:与 ``parse_signature`` 一致的 legacy 目标排第一(保证旧评测集逐位
    复现),其余按出现顺序;ghost/泛型声明被排除(不可编译执行)。
    """
    found = []
    for m in _DECL_RE.finditer(src):
        if m.group(1) or m.group(4):          # ghost/twostate/... 或 泛型 <T>
            continue
        kind = "method" if m.group(2) == "method" else "function"
        name = m.group(3)
        params = _split_params(m.group(5))
        if m.group(6) is not None:            # returns (...)
            rets = _split_params(m.group(6))
        elif m.group(7):                      # function ... : T
            rets = [("res", m.group(7))]
        else:
            rets = []
        if not rets:
            continue
        tys = [t for _, t in params] + [t for _, t in rets]
        if any(t not in _SUPPORTED for t in tys):
            continue
        found.append(Target(kind, name, params, rets,
                            m.start(), _body_offset(src, m.end())))

    legacy = parse_signature(src)
    if legacy:
        lname, lparams, lrets = legacy
        for i, t in enumerate(found):         # legacy 目标提到最前
            if t.kind == "method" and t.name == lname and t.rets == lrets \
                    and t.params == lparams:
                found.insert(0, found.pop(i))
                break
    return found


def spec_region(src: str, method_name: str | None = None) -> tuple[int, int] | None:
    """目标声明的**子句区间** [签名末, 方法体 '{'),即它自己的 requires/ensures。

    单方法文件里该区间就是全文的 ensures 集合,故旧行为不变;多方法文件里它把
    别的方法的后置条件挡在外面 —— 探针若混入别人的 ensures 会引用未定义的
    参数名,整条任务就在"技术性失败"里被丢掉。
    """
    targets = find_targets(src)
    if not targets:
        return None
    tgt = None
    if method_name:
        tgt = next((t for t in targets if t.name == method_name), None)
    if tgt is None:
        tgt = targets[0]
    if tgt.body_start < 0:
        return None
    head_end = src.find(")", tgt.sig_start)
    return (head_end if head_end > 0 else tgt.sig_start, tgt.body_start)


def spec_source(src: str, method_name: str | None = None) -> str:
    """目标的子句文本;拿不到作用域时退回全文(与旧行为一致)。"""
    span = spec_region(src, method_name)
    return src[span[0]:span[1]] if span else src


# --------------------------------------------------------------------------- #
# 取值生成
# --------------------------------------------------------------------------- #
def _gen_scalar(ty: str, rng: random.Random):
    if ty in ("int", "nat"):
        if rng.random() < _BOUNDARY_PROB:
            cand = [b for b in _BOUNDARY_INT if not (ty == "nat" and b < 0)]
            return rng.choice(cand)
        return rng.randint(0, 12) if ty == "nat" else rng.randint(-8, 12)
    if ty == "bool":
        return rng.choice([True, False])
    if ty == "real":
        if rng.random() < _BOUNDARY_PROB:
            return rng.choice(_BOUNDARY_REAL)
        return round(rng.randint(-16, 24) / 2.0, 1)
    if ty == "char":
        return rng.choice(_ALPHABET)
    return None


def _gen_value(ty: str, rng: random.Random):
    """为类型生成一个随机字面量,返回 (Dafny 构造代码前缀行, 传参表达式, python 侧值)。

    注意:``array<int>`` 沿用旧实现的返回约定(第一位 None,由 caller 特判并
    **再抽一次**),这个怪癖被刻意保留 —— 它决定了随机数流,改了旧评测集就无法
    逐位复现。
    """
    if ty in ("int", "nat"):
        v = _gen_scalar(ty, rng)
        return [], str(v), v
    if ty == "bool":
        v = _gen_scalar(ty, rng)
        return [], ("true" if v else "false"), v
    if ty == "seq<int>":
        n = rng.randint(1, 5)
        vals = [rng.randint(-6, 9) for _ in range(n)]
        return [], "[" + ", ".join(map(str, vals)) + "]", vals
    if ty == "seq<bool>":
        n = rng.randint(1, 4)
        vals = [rng.choice([True, False]) for _ in range(n)]
        return [], "[" + ", ".join("true" if b else "false" for b in vals) + "]", vals
    if ty == "array<int>":
        n = rng.randint(1, 5)
        vals = [rng.randint(-6, 9) for _ in range(n)]
        return None, vals, vals  # 数组要在 Main 里 new,交由 caller 特殊处理
    # ---- 扩展类型(2026-08-05):走统一路径,不影响上面的随机数流 ----
    if ty in ("real", "char"):
        v = _gen_scalar(ty, rng)
        _, expr, _ = _lit(ty, v)
        return [], expr, v
    if ty == "string":
        n = rng.randint(1, 5)
        v = "".join(rng.choice(_ALPHABET) for _ in range(n))
        return [], _lit("string", v)[1], v
    et = elem_type(ty)
    if et in _ELEMS:
        n = rng.randint(1, 5)
        vals = [_gen_scalar(et, rng) for _ in range(n)]
        if is_array(ty):
            return None, vals, vals            # 同 array<int>:caller 负责 new
        return [], _lit(ty, vals)[1], vals
    return None, None, None


def _emit_print(ret_ty: str, var: str) -> str:
    """把返回值打印成可解析的一行。"""
    return f'print {var}, "\\n";'


def build_harness(src: str, inputs, name, params, rets) -> str | None:
    """生成一个 Main:用 inputs 调用目标方法并打印返回值。inputs 与 params 对齐。"""
    body = []
    args = []
    for (pname, pty), val in zip(params, inputs):
        if is_array(pty):
            et = elem_type(pty)
            arr = f"arr_{pname}"
            body.append(f"var {arr} := new {et}[{len(val)}];")
            for i, x in enumerate(val):
                body.append(f"{arr}[{i}] := {_scalar_lit(et, x)};")
            args.append(arr)
        else:
            _, expr, _ = _lit(pty, val)
            if expr is None:
                return None
            args.append(expr)
    if not rets:
        return None
    call_vars = ", ".join(r[0] for r in rets)
    body.append(f"var {call_vars} := {name}({', '.join(args)});")
    # 打印所有返回值(以制表分隔)
    prints = " print \"\\t\"; ".join(_print_expr(rty, rname) for rname, rty in rets)
    body.append(f"{prints} print \"\\n\";")
    harness = _rename_existing_main(src) + \
        "\nmethod Main() {\n  " + "\n  ".join(body) + "\n}\n"
    return harness


_MAIN_DECL_RE = re.compile(r"\b(method|function)(\s+)Main(\s*\()")


def _rename_existing_main(src: str) -> str:
    """源文件自带 ``Main`` 时先改名,否则 harness 的 Main 触发 Duplicate member name。

    DafnyBench 里带演示 ``Main`` 的文件不少,而它们与目标方法的对错无关 ——
    不改名就整份文件抽不出测试。
    """
    if not _MAIN_DECL_RE.search(src):
        return src
    return _MAIN_DECL_RE.sub(r"\1\2SFOrigMain\3", src)


def _scalar_lit(ty: str, val) -> str | None:
    if ty in ("int", "nat"):
        return str(val)
    if ty == "bool":
        return "true" if val in (True, "true", "True") else "false"
    if ty == "real":
        s = repr(float(val))
        return s if ("." in s or "e" in s) else s + ".0"
    if ty == "char":
        return "'%s'" % val
    return None


def _lit(ty: str, val):
    if ty in ("int", "nat"):
        return [], str(val), val
    if ty == "bool":
        return [], ("true" if val else "false"), val
    if ty == "seq<int>":
        return [], "[" + ", ".join(map(str, val)) + "]", val
    if ty == "seq<bool>":
        return [], "[" + ", ".join("true" if b else "false" for b in val) + "]", val
    if ty in ("real", "char"):
        return [], _scalar_lit(ty, val), val
    if ty == "string":
        return [], '"%s"' % "".join(val), val
    et = elem_type(ty)
    if et in _ELEMS and isinstance(val, (list, tuple)):
        parts = [_scalar_lit(et, x) for x in val]
        if any(p is None for p in parts):
            return None, None, None
        return [], "[" + ", ".join(parts) + "]", val
    return None, None, None


def _print_expr(ty: str, var: str) -> str:
    # Dafny print 对 int/bool/real/char/seq 都给出可解析文本(seq -> "[a, b, c]",
    # char -> "'k'", string -> 裸字符);array 需 [..] 转成 seq 再打印。
    if is_array(ty):
        return f"print {var}[..];"
    return f"print {var};"


def run_dafny_python(harness: str, dafny_path: str, timeout: int = 40) -> str | None:
    """translate 到 Python 并执行,返回 stdout(失败返回 None)。"""
    return _run_once(harness, dafny_path, timeout)[0]


def _run_once(harness: str, dafny_path: str, timeout: int = 40):
    """同 ``run_dafny_python``,但额外回报失败发生在哪一步。

    返回 ``(stdout|None, stage)``,stage ∈ {ok, translate_fail, run_fail}。
    调用方据此**早退**:translate 失败与随机输入无关,重采样 9 次纯属浪费 ——
    全量 harvest 时这是几小时和几十小时的差别。
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        dfy = os.path.join(d, "prog.dfy")
        with open(dfy, "w", encoding="utf-8") as f:
            f.write(harness)
        out_base = os.path.join(d, "out")
        try:
            # --allow-warnings 是必需的,不是宽容:Dafny 4 默认把"缩进可疑"
            # "分号已废弃"这类**纯风格警告**升级为翻译失败,DafnyBench 里大量
            # 课程作业代码因此一条测试也抽不出来,与程序对错毫无关系。
            subprocess.run([dafny_path, "translate", "py", "--include-runtime",
                            "--no-verify", "--allow-warnings",
                            dfy, "--output", out_base],
                           capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            return None, "translate_fail"
        pkg = out_base + "-py"
        main_py = os.path.join(pkg, "__main__.py")
        if not os.path.exists(main_py):
            return None, "translate_fail"
        try:
            proc = subprocess.run(["python", main_py], capture_output=True,
                                  text=True, timeout=timeout, cwd=pkg)
        except (subprocess.TimeoutExpired, OSError):
            return None, "run_fail"
        if proc.returncode != 0:
            return None, "run_fail"
        return proc.stdout.strip(), "ok"


def _tests_for_target(src, tgt: Target, dafny_path, n, seed, timeout) -> list[dict]:
    name, params, rets = tgt.signature
    rng = random.Random(seed)
    tests = []
    translate_fails = 0
    for _ in range(n * RESAMPLE_FACTOR):   # 多试几次(部分随机输入可能违反 requires)
        if len(tests) >= n:
            break
        inputs = []
        for _, pty in params:
            _, _, pyval = _gen_value(pty, rng)
            if pty == "array<int>":
                # 旧实现的怪癖:array<int> 在 _gen_value 之外**再抽一次**。
                # 保留额外抽样,以兼容旧版的随机数流和输出。
                pyval = [rng.randint(-6, 9) for _ in range(rng.randint(1, 5))]
            inputs.append(pyval)
        harness = build_harness(src, inputs, name, params, rets)
        if harness is None:
            return []
        out, stage = _run_once(harness, dafny_path, timeout=timeout)
        if stage == "translate_fail":
            translate_fails += 1
            if translate_fails >= 2:       # 编译不过与输入无关,别再重采样
                return []
            continue
        if out is None or out == "":
            continue
        tests.append({"inputs": inputs, "output": out, "params": params,
                      "rets": rets, "method": name, "kind": tgt.kind})
    return tests


def extract_tests(ground_truth_src: str, dafny_path: str, n: int = DEFAULT_N_TESTS,
                  seed: int = 0, timeout: int = DEFAULT_TIMEOUT,
                  max_targets: int = 4) -> list[dict]:
    """从 ground_truth 程序抽取至多 n 条测试用例 ``{inputs, output}``。

    按 ``find_targets`` 的优先级逐个目标尝试,第一个产出测试的目标即采用;
    全部失败返回空表(上层据此排除该任务)。
    """
    targets = find_targets(ground_truth_src)
    if not targets:
        return []
    for tgt in targets[:max_targets]:
        tests = _tests_for_target(ground_truth_src, tgt, dafny_path, n, seed, timeout)
        if tests:
            return tests
    return []
