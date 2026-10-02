"""测试忠实验证率(Test-Faithful Verify Rate, TFVR)的忠实性判据(oracle)。

主卖点(叙事 v4):验证器只回答"程序满足规约吗",不回答"规约忠实于意图吗"。
当规约由 NL 翻译而来(工作① NL2STL),一次"验证通过"可能只是验证了一个**过弱/
过强的错规约**。现有全部工作只报 verify rate,无人区分"通过"与"通过了忠实规约"。

本模块给出忠实性判据 —— 用可执行测试 R 作意图的独立投影:

  * **正例接受(E-spec⁺)**:对每条测试 (x, e),规约 S 必须接受正确答案 e。
    S 拒绝 e ⇒ 规约**过强**(冤枉正确实现)⇒ 不忠实。
  * **反例拒绝(E-spec⁻)**:构造错误答案 w≠e,规约 S 必须拒绝 w。
    S 接受某个 w ⇒ 规约**过弱/空洞**(放行错误实现)⇒ 不忠实。

**测试忠实通过** = 验证通过 ∧ 正例全接受 ∧ 反例全拒绝 ∧ 未删/弱化给定规约(守卫)。
``verify_rate − TFVR`` = "验证了错规约"的比例 —— 此前无人量化的可信度水分。

判据只用验证器(探针),执行仅在离线产 R 时用一次(见 rextract),**无需 .NET**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .adjudicate import (
    _is_postcondition_failure,
    _split_outputs,
    _to_dafny_literal,
)
from .guard import _BLOCK_COMMENT_RE, _LINE_COMMENT_RE
from .rextract import _scalar_lit, elem_type, is_array, parse_signature, spec_source


ORACLE_VERSION = "preconditions-tristate-v1"


@dataclass
class Faithful:
    """三值判据:只有确定的见证才判 False,任一必需检查未知则不能判 True。

    前七个字段保持原位置,以兼容现有实验代码中的位置参数构造。
    conclusive/unknown 计数反映实际完成的检查,不把未执行的探针当成成功。
    """
    faithful: bool | None
    accepts_positives: bool | None
    rejects_negatives: bool | None
    n_tests: int
    n_neg_conclusive: int
    reason: str
    witness: dict = field(default_factory=dict)
    n_pos_conclusive: int = 0
    n_pos_unknown: int = 0
    n_neg_unknown: int = 0
    n_domain_unknown: int = 0            # 不重复计算相同输入的域检查


# --------------------------------------------------------------------------- #
# 探针:钉死输入(requires)+ 赋定输出,套用给定规约 S 的 ensures
# --------------------------------------------------------------------------- #
def _seq_elems(lit: str) -> list[str] | None:
    """把 ``[a, b, c]`` 拆成元素字面量表;非序列字面量返回 None。"""
    s = (lit or "").strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    return [x.strip() for x in inner.split(",")] if inner else []


def _assign_return(rn: str, rty: str, lit: str) -> list[str] | None:
    """探针方法体里把返回变量赋成 lit。

    数组不能直接赋序列字面量,必须 ``new T[n]`` 再逐位写入 —— 少了这一步,
    所有"返回数组"的任务(排序、映射等一大类)都会在探针编译期失败,被当成
    证据不足丢掉。"""
    if not is_array(rty):
        return [f"{rn} := {lit};"]
    et = elem_type(rty)
    xs = _seq_elems(lit)
    if xs is None or et is None:
        return None
    out = [f"{rn} := new {et}[{len(xs)}];"]
    out += [f"{rn}[{i}] := {x};" for i, x in enumerate(xs)]
    return out


def _pin_inputs(params, inputs) -> list[str] | None:
    """把一组具体输入钉成 ``requires`` 行(数组按长度 + 逐位钉)。

    从 ``_probe`` 里提出来单用,因为域检查探针需要同一套钉法而不需要 ensures。
    """
    if not isinstance(inputs, (list, tuple)) or len(inputs) != len(params):
        return None
    pin = []
    for (pn, pty), val in zip(params, inputs):
        # 不允许负 nat 等类型外输入制造矛盾的 requires,从而空洞地验证任意探针。
        values = val if isinstance(val, list) and elem_type(pty) else [val]
        scalar_ty = elem_type(pty) if elem_type(pty) and pty != "string" else pty
        if scalar_ty in ("int", "nat"):
            if any(isinstance(v, bool) or not re.fullmatch(r"-?\d+", str(v))
                   or (scalar_ty == "nat" and int(v) < 0) for v in values):
                return None
        elif scalar_ty == "bool":
            if any(v not in (True, False, "true", "false", "True", "False")
                   for v in values):
                return None
        if is_array(pty):
            et = elem_type(pty)
            if not isinstance(val, list) or et is None:
                return None
            pin.append(f"  requires {pn}.Length == {len(val)}")
            for i, x in enumerate(val):
                lit = _scalar_lit(et, x)
                if lit is None:
                    return None
                pin.append(f"  requires {pn}[{i}] == {lit}")
        else:
            lit = _to_dafny_literal(pty, val)
            if lit is None:
                return None
            pin.append(f"  requires {pn} == {lit}")
    return pin


def _contract_clauses(src: str, name: str | None, kind: str) -> list[str]:
    """按下一个规约关键字截取完整子句,避免只读首行而静默丢弃续行约束。

    ``spec_source`` 限定目标方法;关键字匹配忽略注释和字符串中的同名字样。
    这里仍使用仓库的目标定位器,不声称支持任意 Dafny AST。
    """
    segment = _BLOCK_COMMENT_RE.sub(" ", spec_source(src, name))
    segment = _LINE_COMMENT_RE.sub("", segment)
    masked = re.sub(r'"(?:\\.|[^"\\])*"', lambda m: " " * len(m.group()), segment)
    boundaries = list(re.finditer(r"\b(requires|ensures|reads|modifies|decreases)\b", masked))
    clauses = []
    for i, match in enumerate(boundaries):
        if match.group(1) != kind:
            continue
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(segment)
        expression = segment[match.end():end].strip().removesuffix(";").strip()
        if expression:
            clauses.append(expression)
    return clauses


def method_requires(src: str, name: str | None) -> list[str]:
    """目标方法**自己**的前置条件表达式(去注释、去尾分号)。

    与 ``_probe`` 取 ensures 的做法对称:只取目标那一段,多方法文件里混进别人的
    前置会引用未声明的标识符,探针在解析期就炸掉。
    """
    return _contract_clauses(src, name, "requires")


def _technical_failure(result) -> bool:
    raw = (getattr(result, "raw", "") or "").lower()
    return any(word in raw for word in (
        "timeout", "timed out", "time out", "out of resource", "resource limit",
        "out of memory", "inconclusive",
    ))


def _verified_success(result) -> bool:
    # Dafny 的汇总可能同时含 "0 errors" 和超时;不能只依赖适配器的 success。
    return (getattr(result, "success", None) is True
            and not (getattr(result, "failures", []) or [])
            and not _technical_failure(result))


def _source_verifies(src, verifier, source_verified=None) -> bool:
    """附加探针前确认候选自身可验证;显式标志可复用调用方的同一源码结果。"""
    if source_verified is not None:
        return source_verified is True
    try:
        return _verified_success(verifier.verify(src, None))
    except Exception:
        return False


def in_domain(src, name, params, inputs, verifier, *, source_verified=None) -> bool | None:
    """``inputs`` 是否落在目标方法自己的前置条件内。

    这不是吹毛求疵:Hoare 语义下 ``requires`` 为假时合约**什么都没声称**,所以
    出域输入上的 E-spec 判决(无论正例被拒还是反例被接受)都是空洞的。
    抽测试那一步用 ``dafny translate --no-verify``(见 rextract._run_once),
    前置条件在运行期不检查,出域输入照样跑出输出,所以必须在这里挡。

    双向探针:``assert P`` 通过 ⇒ 域内;``assert !P`` 通过 ⇒ 域外;都不通过 ⇒
    探针判不了(None),按本模块惯例不足为凭。

    探针**挂在原文件末尾**,不像 ``_probe`` 那样独立成篇:前置条件经常引用文件里
    自定义的谓词/函数(``sorted(a)``、``SortedSeq(a[..limit])``),独立探针连解析
    都过不去,会把一大批任务误判成"判不了"。挂在原文件上的代价是整份文件要重新
    验证一遍,所以先确认 ``src`` 自身验证通过,否则无法归因。调用方已有**同一
    源码**的验证结果时可传 ``source_verified=True`` 避免重复。

    参考规约预筛不能代替候选域检查:生成的候选可能新增更强的 requires。
    """
    if not _source_verifies(src, verifier, source_verified):
        return None
    pin = _pin_inputs(params, inputs)
    if pin is None:
        return None
    reqs = method_requires(src, name)
    if not reqs:
        return True                          # 无显式前置 ⇒ 全域(支持的输入类型内)
    decl = ", ".join(f"{pn}: {pty}" for pn, pty in params)
    conj = " && ".join(f"({e})" for e in reqs)

    def _holds(expr: str) -> bool:
        lines = [f"method DomainProbe__({decl})"] + list(pin) + ["{", f"  assert {expr};", "}"]
        prog = src.rstrip() + "\n\n" + "\n".join(lines) + "\n"
        try:
            return _verified_success(verifier.verify(prog, None))
        except Exception:
            return False

    # 失败本身不是出域证据:只有 !P 的证明成功才返回 False。
    if _holds(conj):
        return True
    return False if _holds(f"!({conj})") else None


def filter_in_domain(src, name, params, tests, verifier):
    """丢掉落在目标方法前置条件之外的测试;判不了的一并丢(保守)。"""
    keep = []
    if not _source_verifies(src, verifier):
        return keep
    for t in tests:
        if in_domain(src, name, params, t["inputs"], verifier, source_verified=True) is True:
            keep.append(t)
    return keep


def _probe(src: str, name, params, rets, inputs, out_lits) -> str | None:
    """构造 ``method SpecProbe(<所有参数>) returns(<返回>)``:
    requires 钉死输入取值,ensures 为 S 的后置,方法体把返回赋成 out_lits。
    验证通过 ⇒ S 接受 (inputs, out_lits);后置失败 ⇒ S 拒绝之。
    参数全部保留为形参(名字须在 ensures 中可见,避免未解析标识符假性失败)。

    ensures 只取**目标方法自己**的那一段(``spec_source``)。单方法文件里这与
    全文相同;多方法文件里混进别人的后置会引用未声明的参数名,探针在解析期就
    炸掉,整条任务被误判为"证据不足"。
    """
    ens = _contract_clauses(src, name, "ensures")
    if not isinstance(out_lits, (list, tuple)) or len(out_lits) != len(rets):
        return None
    pin = _pin_inputs(params, inputs)
    if pin is None:
        return None
    param_decl = ", ".join(f"{pn}: {pty}" for pn, pty in params)
    ret_decl = ", ".join(f"{rn}: {rty}" for rn, rty in rets)
    body = []
    for (rn, rty), lit in zip(rets, out_lits):
        stmts = _assign_return(rn, rty, lit)
        if stmts is None:
            return None
        body += stmts
    return (f"method SpecProbe({param_decl}) returns ({ret_decl})\n" +
            "\n".join(pin) + "\n" +
            "\n".join(f"  ensures {e}" for e in ens) + "\n{\n  " +
            "\n  ".join(body) + "\n}\n")


def _accepts(src, name, params, rets, inputs, out_lits, verifier) -> bool | None:
    """S 是否接受 (inputs, out_lits)。True=接受, False=以后置失败拒绝, None=不可判定。"""
    prog = _probe(src, name, params, rets, inputs, out_lits)
    if prog is None:
        return None
    try:
        r = verifier.verify(prog, None)
    except Exception:
        return None
    if _verified_success(r):
        return True
    failures = getattr(r, "failures", []) or []
    technical = any(getattr(f, "kind", "") != "post_unproved" for f in failures)
    technical |= _technical_failure(r)
    if not technical and _is_postcondition_failure(r):
        return False
    return None                              # 解析/类型错:探针技术性失败,不足为凭


def _perturb_scalar(ty: str, lit: str) -> list[str]:
    """把一个标量字面量改成"错误答案"候选。空表示该类型不可扰动。"""
    if ty == "int":
        try:
            v = int(lit)
        except ValueError:
            return []
        return [str(v + 1), str(v - 1)]
    if ty == "nat":
        try:                                 # nat 不往下扰动,负值是类型错而非违约
            return [str(int(lit) + 1)]
        except ValueError:
            return []
    if ty == "bool":
        return ["false" if lit == "true" else "true"]
    if ty == "real":
        try:
            return ["%s" % (float(lit) + 1.0)]
        except ValueError:
            return []
    if ty == "char":
        m = re.fullmatch(r"'(.)'", lit or "")
        if not m:
            return []
        c = m.group(1)
        return ["'%s'" % ("a" if c == "b" else "b")]
    return []


def _negatives(rets, out_lits) -> list[list[str]]:
    """由正确输出字面量构造错误答案(只扰动第一个返回值)。

    标量按类型加一/翻转;序列与数组扰动首元素(空表则给一个单元素表);字符串
    改首字符。拿不到扰动就返回空 —— 该任务的反例维度判为不可测,宁缺毋滥。"""
    if not rets:
        return []
    rty = rets[0][1]
    lit0 = out_lits[0]
    cands = _perturb_scalar(rty, lit0)

    if not cands and rty == "string":
        m = re.fullmatch(r'"(.*)"', lit0 or "", re.S)
        if m is not None:
            s = m.group(1)
            cands = ['"%s"' % (s + "z")]     # 长度变了,忠实规约必拒
            if s:
                cands.append('"%s%s"' % ("a" if s[0] == "b" else "b", s[1:]))

    if not cands:
        et = elem_type(rty)                  # seq<T> / array<T>
        xs = _seq_elems(lit0)
        if et is not None and xs is not None:
            if not xs:
                seed = _scalar_lit(et, 0 if et in ("int", "nat", "real") else
                                   (False if et == "bool" else "a"))
                cands = ["[%s]" % seed] if seed else []
            else:
                for c in _perturb_scalar(et, xs[0]):
                    cands.append("[" + ", ".join([c] + xs[1:]) + "]")

    out = []
    for c in cands:
        variant = list(out_lits)
        variant[0] = c
        out.append(variant)
    return out


def _signature_from(records: list, src: str):
    """优先用测试记录里**随任务落盘的目标签名**,拿不到才退回"文件第一个 method"。

    多方法文件里"第一个 method"往往不是被测目标;老数据集没有这些字段,退回旧
    路径,以兼容旧数据格式。"""
    for r in records or []:
        if r.get("method") and r.get("params") is not None and r.get("rets"):
            return (r["method"], [tuple(p) for p in r["params"]],
                    [tuple(t) for t in r["rets"]])
    return parse_signature(src)


def is_spec_faithful(src: str, tests: list, verifier, *, source_verified=None) -> Faithful:
    """在线判定 src 规约是否忠实于原始测试 R(即时构造反例;用于建集/独立诊断)。

    注意:反例即时构造(e±1 等),对**多值合法**的规约可能误报;主实验请改用
    ``derive_tests`` 预筛出的 (positives, negatives) + ``assess_faithful``。"""
    sig = _signature_from(tests, src)
    if not sig or not tests:
        return Faithful(None, None, None, len(tests), 0, "无签名或无测试",
                        n_pos_unknown=len(tests))
    name, params, rets = sig
    tag = {"method": name, "params": params, "rets": rets}
    positives, negatives = [], []
    for t in tests:
        try:
            out_lits = _split_outputs(t["output"], rets)
        except (KeyError, TypeError, AttributeError, ValueError):
            out_lits = None
        # 不可解码的正例仍保留为 unknown,不能静默丢掉后再声称正例全通过。
        positives.append({"inputs": t.get("inputs"), "out_lits": out_lits, **tag})
        if out_lits is None:
            continue
        for neg in _negatives(rets, out_lits):
            negatives.append({"inputs": t.get("inputs"), "out_lits": neg, **tag})
    return assess_faithful(src, positives, negatives, verifier,
                           source_verified=source_verified)


def derive_tests(src: str, tests: list, verifier):
    """建集期:用 **忠实的 ground-truth 规约** 预筛忠实性测试。

    返回 (positives, negatives),每项为 ``{inputs, out_lits}``:
      * positives —— 参考域内且原始规约**确实接受**的正确答案;
      * negatives —— 原始规约**确实以后置失败拒绝**的错误答案(排除多值合法 w)。
    这样 eval 期对任意方法输出的判定不再受"即时构造的 w 恰好合法"影响。"""
    sig = _signature_from(tests, src)
    if not sig or not tests:
        return [], []
    if not _source_verifies(src, verifier):
        return [], []
    name, params, rets = sig
    #: 目标签名随每条预筛测试落盘,eval 期就不必再猜"哪个方法是被测目标"
    tag = {"method": name, "params": [list(p) for p in params],
           "rets": [list(r) for r in rets]}
    positives, negatives = [], []
    for t in tests:
        if in_domain(src, name, params, t["inputs"], verifier,
                     source_verified=True) is not True:
            continue                         # 先限定参考域,不能把出域执行当意图证据
        out_lits = _split_outputs(t["output"], rets)
        if out_lits is None:
            continue
        if _accepts(src, name, params, rets, t["inputs"], out_lits, verifier) is not True:
            continue                         # 连真规约都不接受 -> 探针/解析有问题,弃用
        positives.append({"inputs": t["inputs"], "out_lits": out_lits, **tag})
        for neg in _negatives(rets, out_lits):
            if _accepts(src, name, params, rets, t["inputs"], neg, verifier) is False:
                negatives.append({"inputs": t["inputs"], "out_lits": neg, **tag})
    return positives, negatives


def assess_faithful(src: str, positives: list, negatives: list, verifier,
                    *, source_verified=None) -> Faithful:
    """以独立参考预筛的见证评价候选;不把未知当作接受或拒绝。

    每个正例输入必须仍在**候选**域内,其正确输出必须被后置接受。只有域内负例
    的后置探针能够提供 E-spec- 证据。未知不阻止其它见证给出 False,但至少一个
    必需检查未知、或任一见证集合为空时,不能返回 True。

    ``source_verified`` 仅可复用调用方对当前 ``src`` 的验证结果。未提供时会先
    验证候选;候选失败/不可判定无法归因于附加探针,故返回证据不足。
    """
    n_pos, n_neg = len(positives), len(negatives)
    sig = _signature_from(list(positives) + list(negatives), src)
    if not sig:
        return Faithful(None, None, None, n_pos, 0, "无签名",
                        n_pos_unknown=n_pos, n_neg_unknown=n_neg)
    if not _source_verifies(src, verifier, source_verified):
        return Faithful(None, None, None, n_pos, 0, "候选自身未确认验证通过",
                        n_pos_unknown=n_pos, n_neg_unknown=n_neg)
    name, params, rets = sig
    domains = {}

    def domain(inputs):
        # 输入记录是 JSON 值;repr 对嵌套数组可哈希且不会把 True 与 1 混为一键。
        key = repr(inputs)
        if key not in domains:
            try:
                domains[key] = in_domain(src, name, params, inputs, verifier,
                                         source_verified=True)
            except Exception:
                domains[key] = None
        return domains[key]

    def accepts(record):
        out_lits = record.get("out_lits")
        if not isinstance(out_lits, (list, tuple)) or len(out_lits) != len(rets):
            return None
        try:
            return _accepts(src, name, params, rets, record.get("inputs"),
                            out_lits, verifier)
        except Exception:
            return None

    pos_results, neg_results = [], []
    witness = {}
    for p in positives:
        inside = domain(p.get("inputs"))
        if inside is False:
            pos_results.append(False)
            if not witness:
                witness = {"kind": "precondition_rejection", "inputs": p.get("inputs"),
                           "expected": p.get("out_lits")}
            continue
        acc = accepts(p) if inside is True else None
        pos_results.append(acc)
        if acc is False and not witness:
            witness = {"kind": "positive_rejection", "inputs": p.get("inputs"),
                       "expected": p.get("out_lits")}

    for g in negatives:
        inside = domain(g.get("inputs"))
        acc = accepts(g) if inside is True else None
        neg_results.append(None if acc is None else not acc)
        if acc is True and not witness:
            witness = {"kind": "negative_acceptance", "inputs": g.get("inputs"),
                       "wrong": g.get("out_lits")}

    def conjunction(values):
        if any(v is False for v in values):
            return False
        if not values or any(v is None for v in values):
            return None
        return True

    pos_ok, neg_ok = conjunction(pos_results), conjunction(neg_results)
    faithful = conjunction([pos_ok, neg_ok])
    reasons = {
        "precondition_rejection": "候选前置条件排除参考正例输入",
        "positive_rejection": "候选后置条件拒绝正确答案",
        "negative_acceptance": "候选后置条件接受错误答案",
    }
    if faithful is False:
        reason = reasons[witness["kind"]]
    elif faithful is True:
        reason = "候选域内正例全接受且反例全拒绝(测试忠实)"
    else:
        reason = "无确定失败见证,但存在未知检查或缺少正/反例"
    return Faithful(
        faithful, pos_ok, neg_ok, n_pos,
        sum(v is not None for v in neg_results), reason, witness,
        n_pos_conclusive=sum(v is not None for v in pos_results),
        n_pos_unknown=sum(v is None for v in pos_results),
        n_neg_unknown=sum(v is None for v in neg_results),
        n_domain_unknown=sum(v is None for v in domains.values()),
    )
