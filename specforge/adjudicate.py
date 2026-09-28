"""归责裁决:验证失败时,判定该改 **规约** 还是 **代码**(设计文档 v2 §5)。

为什么这是空白
--------------
把相关工作按"假定哪一边正确"排开:VeriAct / AutoSpec / Loopy / DafnyPro /
Laurel 假定**代码**正确去修规约;arXiv 2507.03659 假定**规约**正确去修代码;
AxDafny 两者同时生成因而无既有制品可归责。**没有工作处理"两边都可能错、
且不知错在哪边"的情形** —— 而验证失败在信息上是对称的:``S ∧ ¬P`` 不可满足
只说明二者不一致,不说明谁错。

核心洞察:仅有 ``(S, P)`` 时"谁错"是**欠定**的,任何一边改自己都能消解冲突。
判定归责必须引入**独立于二者的第三参照** —— 自然语言需求 ``R`` 及其可执行
投影(测试用例)。这也正是本工作与上游 NL2STL 的接口:当 ``S`` 由 LLM 从
``R`` 翻译而来时,"规约可信"的传统假定不再成立。

三路证据(相互独立,均可审计、可复算):
  * **E1 规约可满足性** —— ``S`` 自身是否自洽;不可满足即定论 SPEC,与代码无关
  * **E2 可执行差分** —— ``P`` 在具体输入上是否真的违反 ``S``
  * **E3 意图锚定** —— 冲突点上,谁与 ``R`` 一致

保守优先:证据不足时输出 ``UNDECIDED`` 而不猜。误修(判错方向后动了正确的
一边)是最该压低的指标。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SPEC = "SPEC"
CODE = "CODE"
SPEC_INCOMPLETE = "SPEC-INCOMPLETE"
UNDECIDED = "UNDECIDED"


@dataclass
class Evidence:
    """三路证据的取证结果(缺省表示"未取到",而非"否")。"""
    unsat_spec: bool | None = None        # E1: S 不可满足
    concrete_conflict: bool | None = None # E2: 存在具体输入使 S 判假
    intent_side: str | None = None        # E3: 与 R 一致的一方 "spec"|"code"
    witness: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    side: str                             # SPEC | CODE | SPEC-INCOMPLETE | UNDECIDED
    reason: str
    evidence: Evidence
    confident: bool = False               # 是否为定论级证据(E1 或 E2+E3 齐全)


# --------------------------------------------------------------------------- #
# E1:规约可满足性探针
# --------------------------------------------------------------------------- #
_ENSURES_RE = re.compile(r"^[ \t]*ensures[ \t]+(.*\S)[ \t]*$", re.M)
_REQUIRES_RE = re.compile(r"^[ \t]*requires[ \t]+(.*\S)[ \t]*$", re.M)
_SIG_RE = re.compile(
    r"^[ \t]*method[ \t]+(\w+)[ \t]*(?:<[^>]*>)?[ \t]*\(([^)]*)\)"
    r"(?:[ \t]*returns[ \t]*\(([^)]*)\))?", re.M)


def probe_spec_satisfiable(program_src: str, verifier) -> tuple[bool | None, str]:
    """E1:``S`` 是否可被 **某个** 实现满足。

    做法:保留签名与 ``requires``/``ensures``,把方法体换成 ``assume false;``
    的对偶 —— 我们构造一个"只需存在性"的探针:若把后置条件当作 ``assert``
    放在一个 ``assume`` 了前置条件的空上下文中恒不可达,则 ``S`` 自相矛盾。

    实现上取一个保守而廉价的判据:若 ``requires`` 本身不可满足(前置矛盾),
    或 ``ensures`` 中存在恒假子句,则判定 ``S`` 有问题。返回
    ``(True=不可满足, 说明)``;拿不到结论返回 ``(None, 原因)``。
    """
    m = _SIG_RE.search(program_src)
    if not m:
        return None, "无法解析方法签名,E1 跳过"
    name, params, rets = m.group(1), m.group(2) or "", m.group(3) or ""
    reqs = _REQUIRES_RE.findall(program_src)
    ens = _ENSURES_RE.findall(program_src)
    if not ens:
        return None, "无 ensures,E1 不适用"

    # 探针 1:前置条件是否自相矛盾(requires 下 assert false 竟能证明)
    body_req = "\n".join(f"  requires {r}" for r in reqs)
    probe_req = (f"method SFProbeReq({params})\n{body_req}\n"
                 "{\n  assert false;\n}\n")
    try:
        r1 = verifier.verify(probe_req, None)
    except Exception as e:                                  # 探针失败不影响主流程
        return None, f"E1 探针异常:{e}"
    if r1.success:
        return True, "前置条件自相矛盾(requires 下可证 false)"

    # 探针 2:前置 ∧ 后置 是否可满足。把 **输入与输出一并作为形参**(而非
    # 局部 var —— 未初始化的局部变量在 Dafny 里不是"确定赋值",引用即报错),
    # 再把 requires ∧ ensures 全部挂成 requires:若该合取不可满足,方法体
    # 不可达,``assert false`` 平凡可证 => 规约无法被任何实现满足。
    decls = [p.strip() for grp in (params, rets) for p in grp.split(",") if p.strip()]
    conj = [f"  requires {r}" for r in reqs] + [f"  requires {e}" for e in ens]
    probe_ens = (f"method SFProbeEns({', '.join(decls)})\n" +
                 "\n".join(conj) + "\n{\n  assert false;\n}\n")
    try:
        r2 = verifier.verify(probe_ens, None)
    except Exception as e:
        return None, f"E1 探针异常:{e}"
    if r2.success:
        return True, "前置与后置合取后自相矛盾(规约不可被任何实现满足)"
    return False, "规约可满足"


# --------------------------------------------------------------------------- #
# E2':规约可实现性探针(无需执行,纯验证)
# --------------------------------------------------------------------------- #
def probe_spec_implementable(program_src: str, verifier, llm,
                             tries: int = 2) -> tuple[bool | None, str]:
    """E2':是否存在 **某个** 实现能满足 ``S'``。

    做法:把方法体清空,只留签名 + 契约,让 LLM 从零写一个新实现(**不看**
    原实现 ``P'``),再验证。若连独立重写也过不了,说明 ``S'`` 被过度约束
    (over-strong / 不可实现)—— 与代码无关地指向 SPEC。

    这一路捕获 E1 抓不到的"可满足但过强"规约(如 ``>=`` 收紧成 ``>``);
    但对"可满足且语义变味"的规约(如 ``==`` 翻成 ``!=``)仍无能为力 ——
    那类必须靠意图锚点 R(测试用例),见 E3。返回 ``(True=不可实现, 说明)``。
    """
    complete = getattr(llm, "_complete", None)
    if complete is None:
        return None, "LLM 不支持自由文本,E2' 跳过"
    m = _SIG_RE.search(program_src)
    if not m:
        return None, "无法解析签名,E2' 跳过"
    # 抽取签名行 + 契约行(到方法体 '{' 之前)
    head_lines = []
    for ln in program_src.splitlines():
        head_lines.append(ln)
        if "{" in ln:
            break
    header = "\n".join(head_lines)
    sys = ("You are a Dafny expert. Given a method signature and its "
           "specification, write a COMPLETE method body that satisfies the "
           "specification and verifies. Do not change the signature or the "
           "specification. Return one ```dafny block.")
    prompt = ("Implement this method so it verifies:\n\n```dafny\n" + header +
              "\n  // TODO: body\n}\n```")
    from .llm.client import _parse_code
    for _ in range(tries):
        try:
            raw = complete(sys, prompt)
            llm.calls += 1
        except Exception as e:
            return None, f"E2' LLM 异常:{e}"
        code = _parse_code(raw)
        if not code:
            continue
        try:
            if verifier.verify(code, None).success:
                return False, "存在可验证的独立实现,规约可实现"
        except Exception as e:
            return None, f"E2' 验证异常:{e}"
    return True, "多次独立重写均无法满足,规约过强/不可实现"


# --------------------------------------------------------------------------- #
# E2 / E3:可执行差分 + 意图锚定
# --------------------------------------------------------------------------- #
def differential_evidence(cases, run_impl, eval_spec) -> Evidence:
    """E2+E3:在 ``R`` 给出的测试用例上对质。

    :param cases: 可迭代的 ``(inputs, expected)``,来自需求 ``R``
                  (MBPP 系任务自带测试用例,是 ``R`` 的可执行投影)
    :param run_impl: ``inputs -> 实际输出``(执行 ``P``)
    :param eval_spec: ``(inputs, output) -> bool``(在具体点上求值 ``S``)
    """
    ev = Evidence()
    if not cases:
        ev.notes.append("无可用测试用例,E2/E3 跳过")
        return ev
    conflict = False
    for inputs, expected in cases:
        try:
            actual = run_impl(inputs)
        except Exception as e:
            ev.notes.append(f"执行 P 失败({e}),该用例跳过")
            continue
        try:
            spec_ok = eval_spec(inputs, actual)
        except Exception as e:
            ev.notes.append(f"求值 S 失败({e}),该用例跳过")
            continue
        if spec_ok:
            continue
        # 具体点上 S 判假 -> 二者确有冲突,用 R 定责
        conflict = True
        ev.concrete_conflict = True
        ev.witness = {"inputs": inputs, "actual": actual, "expected": expected}
        if _same(actual, expected):
            # 代码与需求一致,却被 S 判假 -> 冤枉了正确代码
            ev.intent_side = "code"
        else:
            ev.intent_side = "spec"
        break
    if not conflict:
        ev.concrete_conflict = False
    return ev


def _same(a, b) -> bool:
    if a == b:
        return True
    try:
        return list(a) == list(b)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 测试锚定证据(v3 §3):用 R=测试用例打破 S∧¬P 的对称。
#   E-code:执行(可能有 bug 的)代码 P',若在某测试输入上产出 != 期望 -> 代码错。
#   E-spec:验证给定规约 S' 是否接受正确答案(输入钉死、输出钉成期望),
#           若 ensures 被违反 -> 规约拒绝了正确答案 -> 规约错。
# 二者都以离线执行 ground_truth 得到的 (输入,期望) 为独立第三参照。
# --------------------------------------------------------------------------- #
def test_grounded_evidence(injected_src: str, tests: list, dafny_path: str,
                           verifier) -> Evidence:
    from .rextract import build_harness, parse_signature, run_dafny_python
    ev = Evidence()
    sig = parse_signature(injected_src)
    if not sig or not tests:
        ev.notes.append("无签名或无测试,测试锚定跳过")
        return ev
    name, params, rets = sig
    code_clean = True                        # 代码在所有已测输入上是否都产出期望

    for t in tests:
        # ---- E-code:执行注入后的代码,与期望对比(执行是 ground truth,最稳) ----
        harness = build_harness(injected_src, t["inputs"], name, params, rets)
        actual = run_dafny_python(harness, dafny_path) if harness else None
        if actual is not None and _norm(actual) != _norm(t["output"]):
            ev.concrete_conflict = True
            ev.intent_side = "spec"          # 代码产出 != 意图 -> 代码错(裁决表:CODE)
            ev.witness = {"inputs": t["inputs"], "actual": actual,
                          "expected": t["output"]}
            ev.notes.append("E-code:代码在测试输入上产出错值(见证 CODE)")
            return ev
        if actual is None:
            code_clean = False               # 没跑成,不能断言代码在此点正确

        # ---- E-spec:仅当代码在此点确已产出正确值,才检验 S 是否接受正确答案 ----
        # 且只有探针以 **后置条件失败**(而非解析/编译错)告败,才算真见证 ——
        # 否则是探针技术性失败,不足为凭(旧实现的 CODE 误判即源于此)。
        if actual is not None and _norm(actual) == _norm(t["output"]):
            probe = _spec_accept_probe(injected_src, name, params, rets, t)
            if probe is None:
                continue
            try:
                r = verifier.verify(probe, None)
            except Exception:
                continue
            if not r.success and _is_postcondition_failure(r):
                ev.concrete_conflict = True
                ev.intent_side = "code"      # 代码对、规约拒绝正确答案 -> 规约错(SPEC)
                ev.witness = {"inputs": t["inputs"], "expected": t["output"]}
                ev.notes.append("E-spec:代码正确但规约拒绝正确答案(见证 SPEC)")
                return ev

    # 无任何测试暴露矛盾:证据不足(可能是测试覆盖不到该缺陷)
    ev.concrete_conflict = False if code_clean else None
    ev.notes.append("测试锚定:未见证矛盾" + ("(代码在所测点均正确)" if code_clean else ""))
    return ev


def _is_postcondition_failure(res) -> bool:
    """探针告败是否为真正的后置条件违反(而非解析/类型/编译错)。"""
    fails = getattr(res, "failures", []) or []
    if not fails:
        return False
    # 只要有 post_unproved,且没有解析/类型级 'other'(未解析标识符等),即真见证
    has_post = any(getattr(f, "kind", "") == "post_unproved" for f in fails)
    has_resolution = any("unresolved" in (getattr(f, "message", "") or "").lower()
                         or "resolution" in (getattr(f, "message", "") or "").lower()
                         for f in fails)
    return has_post and not has_resolution


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip())


def _spec_accept_probe(src: str, name, params, rets, test) -> str | None:
    """构造:签名 + 原 requires/ensures + 钉死输入(requires 相等)+ 钉死输出(赋期望)。
    若该探针验证通过,说明 S' 接受(输入,期望);失败则 S' 拒绝正确答案。"""
    reqs = _REQUIRES_RE.findall(src)
    ens = _ENSURES_RE.findall(src)
    if not ens:
        return None
    out_lits = _split_outputs(test["output"], rets)
    if out_lits is None:
        return None
    # **全部参数都保留为形参**(名字必须在 ensures 里仍可见),用 requires 钉死取值:
    #   标量/seq:``requires p == lit``;array:``requires p.Length==n && p[i]==v_i``。
    # 这样 ensures 引用原参数名不会失配(旧实现把 array 移成局部 -> 未解析标识符 ->
    # 探针假性失败 -> E-spec 误判 SPEC,是 CODE 召回低的根因)。
    pin = []
    for (pn, pty), val in zip(params, test["inputs"]):
        if pty == "array<int>":
            if not isinstance(val, list):
                return None
            pin.append(f"  requires {pn}.Length == {len(val)}")
            for i, x in enumerate(val):
                pin.append(f"  requires {pn}[{i}] == {x}")
        else:
            lit = _to_dafny_literal(pty, val)
            if lit is None:
                return None
            pin.append(f"  requires {pn} == {lit}")
    param_decl = ", ".join(f"{pn}: {pty}" for pn, pty in params)
    ret_decl = ", ".join(f"{rn}: {rty}" for rn, rty in rets)
    body = [f"{rn} := {lit};" for (rn, _), lit in zip(rets, out_lits)]
    prog = (f"method SpecAccept({param_decl}) returns ({ret_decl})\n" +
            "\n".join(pin) + "\n" +
            "\n".join(f"  ensures {e}" for e in ens) + "\n{\n  " +
            "\n  ".join(body) + "\n}\n")
    return prog


def _to_dafny_literal(ty, val):
    from .rextract import _scalar_lit, elem_type

    if ty in ("int", "nat", "bool", "real", "char"):
        return _scalar_lit(ty, val)
    if ty == "string":
        return '"%s"' % "".join(val)
    et = elem_type(ty)                       # seq<T> / array<T>
    if et is not None and isinstance(val, list):
        parts = [_scalar_lit(et, x) for x in val]
        if any(p is None for p in parts):
            return None
        return "[" + ", ".join(parts) + "]"
    return None


def _split_outputs(output_str, rets):
    """把打印输出(制表/单值)拆成每个返回值的 Dafny 字面量。"""
    parts = output_str.split("\t") if "\t" in output_str else [output_str]
    if len(parts) != len(rets):
        if len(rets) == 1:
            parts = [output_str]
        else:
            return None
    from .rextract import elem_type

    out = []
    for (rn, rty), p in zip(rets, parts):
        p = p.strip()
        if rty in ("int", "nat"):
            if not re.fullmatch(r"-?\d+", p):
                return None
            out.append(p)
        elif rty == "bool":
            out.append("true" if p.lower() == "true" else "false")
        elif rty == "real":
            # Dafny 打印有理数为 5.0;除不尽时给出 (a.b/c) 之类的形式,那种拿不回
            # 字面量,直接判为不可解析而弃用该测试(宁可少一条,不可错一条)。
            if not re.fullmatch(r"-?\d+\.\d+", p):
                return None
            out.append(p)
        elif rty == "char":
            # Dafny 打印 char 时自带单引号,已经是合法字面量
            if not re.fullmatch(r"'.'", p):
                return None
            out.append(p)
        elif rty == "string":
            # 裸字符串,补引号成字面量;含引号/反斜杠的一律弃用,免得拼出坏代码
            if '"' in p or "\\" in p:
                return None
            out.append('"%s"' % p)
        elif elem_type(rty) is not None:
            # Dafny 打印的 [a, b, c] 直接可用作 seq 字面量;array 由探针再转成 new
            if not (p.startswith("[") and p.endswith("]")):
                return None
            out.append(p)
        else:
            return None
    return out


# --------------------------------------------------------------------------- #
# 裁决表(设计文档 v2 §5)
# --------------------------------------------------------------------------- #
def adjudicate(ev: Evidence) -> Verdict:
    """由三路证据得出裁决。保守优先:证据不足输出 UNDECIDED,不猜。"""
    # E1 定论:规约自身不自洽,与代码无关
    if ev.unsat_spec:
        return Verdict(SPEC, "E1:规约不可满足(自相矛盾)", ev, confident=True)

    # E2 有具体冲突 -> 由 E3 定责
    if ev.concrete_conflict:
        if ev.intent_side == "code":
            return Verdict(SPEC, "E2+E3:代码与需求一致却被规约判假,规约冤枉了正确代码",
                           ev, confident=True)
        if ev.intent_side == "spec":
            return Verdict(CODE, "E2+E3:代码输出与需求不符,实现有缺陷",
                           ev, confident=True)
        return Verdict(UNDECIDED, "E2 发现冲突但无需求可锚定,不强判", ev)

    # E2 明确无具体冲突:语义一致,验证失败多半是标注不足
    if ev.concrete_conflict is False and ev.unsat_spec is False:
        return Verdict(SPEC_INCOMPLETE,
                       "E1 可满足且 E2 无具体冲突:语义一致,缺的是证明标注", ev)

    return Verdict(UNDECIDED, "证据不足(E1/E2 均未取到结论)", ev)
