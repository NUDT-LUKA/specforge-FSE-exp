"""Dafny 验证器适配:子进程调用 ``dafny verify`` 并解析输出/反例。

设计文档 8.1:子进程调用 ``dafny verify``;解析其文本输出与反例
(``/extractCounterexample``)。若环境中没有 ``dafny`` 可执行文件,
``verify`` 会抛出 :class:`DafnyNotAvailable`,上层可回退到 :class:`MockVerifier`。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time

from ..ir import SrcLoc
from ..util import vars_in
from .base import Failure, VerifyResult


class DafnyNotAvailable(RuntimeError):
    """PATH 中找不到 dafny 可执行文件。"""


# 报错关键字 -> Failure.kind(设计文档 5.5 表)。
# 注意:Dafny 4.x 措辞与旧版不同(实测 4.11):循环维持失败用
# "could not be proved to be maintained"(非旧版 "might not be maintained"),
# 入口失败用 "could not be proved on entry"。这里对新旧措辞都做覆盖,否则
# inv_maintain 会被误归类为 "other",Houdini 与 BlameLocalizer 双双失灵。
_KIND_PATTERNS = [
    (re.compile(r"postcondition.*(?:could not be proved|might not hold)", re.I), "post_unproved"),
    (re.compile(r"invariant.*(?:could not be proved on entry|might not hold on entry|not established)", re.I), "inv_entry"),
    (re.compile(r"invariant.*(?:be maintained|not be maintained|not maintained)", re.I), "inv_maintain"),
    (re.compile(r"assertion.*might not hold|assertion violation|assertion.*could not be proved", re.I), "assert"),
    (re.compile(r"(?:decreases|termination).*(?:might not decrease|cannot prove|failure to decrease|might not|could not be proved)", re.I), "decreases"),
    (re.compile(r"index out of range|out of bounds|might be (?:null|out of)", re.I), "spec_range"),
]

_LOC_RE = re.compile(r"\((\d+),(\d+)\)")
_SUMMARY_RE = re.compile(
    r"Dafny program verifier finished with\s+(\d+)\s+verified,\s+(\d+)\s+errors?([^\r\n]*)",
    re.I,
)
_INCOMPLETE_SUMMARY_RE = re.compile(
    r"\b[1-9]\d*\s+(?:time\s*outs?|inconclusives?|out of resources?|out of memory)\b",
    re.I,
)
_INCOMPLETE_DIAGNOSTIC_RE = re.compile(
    r"\btimed out\b|\bverification (?:was )?inconclusive\b|"
    r"\b(?:exceeded|reached) (?:the )?(?:resource|memory) limit\b",
    re.I,
)


class DafnyVerifier:
    # 真实 Dafny 后端不支持 Tier-0 符号探针(SMT 扰动查询编码为 M4 工程项);
    # OutputPerturbationProbe 检测到该标记为 False 时只走静态层(设计文档 6.2)。
    supports_probe = False

    def __init__(self, dafny_path: str = "dafny", timeout: int = 60, extra_args: list[str] | None = None):
        self.dafny_path = dafny_path
        self.timeout = timeout
        self.extra_args = extra_args or []
        self.calls = 0  # 验证器调用计数(预算对齐 7.3 + 调用开销指标 7.4)

    @property
    def available(self) -> bool:
        return shutil.which(self.dafny_path) is not None

    def verify(self, program_src: str, spec=None) -> VerifyResult:
        self.calls += 1
        if not self.available:
            raise DafnyNotAvailable(
                f"在 PATH 中找不到 {self.dafny_path!r};请安装 Dafny 或改用 MockVerifier。"
            )

        # ignore_cleanup_errors:Windows 上 Dafny 子进程退出后偶尔仍短暂持有
        # out/err.txt 句柄,rmtree 会抛 WinError 32;忽略清理错误(临时文件由 OS
        # 兜底回收),不影响判定。
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = os.path.join(d, "prog.dfy")
            with open(path, "w", encoding="utf-8") as f:
                f.write(program_src)
            cmd = [
                self.dafny_path,
                "verify",
                "--cores=2",
                f"--verification-time-limit={self.timeout}",
                *self.extra_args,
                path,
            ]
            # 用临时文件而非 PIPE 捕获输出:PIPE 每次调用会起 2 个读线程,长跑
            # 数千次子进程后线程累积会触发 Windows "can't start new thread";写文件
            # 无读线程,输出内容与判定完全一致。
            op = os.path.join(d, "out.txt")
            ep = os.path.join(d, "err.txt")
            # 起子进程本身可能失败(内存/页面文件不足 -> WinError 1455、句柄耗尽
            # 等)。这类失败与"程序验证不通过"是**完全不同**的事件,但如果放任
            # OSError 抛上去,分片子进程会整个崩掉,它剩下的实例全部丢失;若反过来
            # 把它当成验证失败记账,就会静默污染结果。两条路都不能走,所以退避重试:
            # 环境类失败是瞬时的,重试几乎总能过去;真过不去就抛,让它显式失败而不是
            # 变成一条假的"未通过"。
            raw = None
            for attempt in range(3):
                try:
                    with open(op, "w", encoding="utf-8") as of, \
                            open(ep, "w", encoding="utf-8") as ef:
                        subprocess.run(
                            cmd,
                            stdout=of,
                            stderr=ef,
                            timeout=self.timeout + 30,
                        )
                    with open(op, encoding="utf-8", errors="replace") as f:
                        out = f.read()
                    with open(ep, encoding="utf-8", errors="replace") as f:
                        err = f.read()
                    raw = out + "\n" + err
                    break
                except subprocess.TimeoutExpired:
                    return VerifyResult(
                        success=False,
                        failures=[Failure("other", SrcLoc(), "verifier timeout")],
                        raw="TIMEOUT",
                    )
                except OSError as e:
                    if attempt == 2:
                        raise
                    time.sleep(2.0 * (attempt + 1))
                    print(f"  [verifier] 起 Dafny 失败({e.__class__.__name__}: {e}),"
                          f"第 {attempt + 2} 次尝试", flush=True)
        return self._parse(raw, program_src)

    # ------------------------------------------------------------------ #
    def _parse(self, raw: str, program_src: str = "") -> VerifyResult:
        src_lines = program_src.splitlines()
        failures: list[Failure] = []
        for line in raw.splitlines():
            kind = self._classify(line)
            if kind is None:
                continue
            loc = SrcLoc()
            related: list[str] = []
            m = _LOC_RE.search(line)
            if m:
                ln = int(m.group(1))
                loc = SrcLoc(line=ln)
                # 从失败位置的源码行提取相关变量,供 aligner 选循环(设计文档 5.5)。
                # 注意:**不** 填 counterexample —— 真实 Dafny 默认不产出反例模型
                # (future work),位置提示不是"代码违反 S 的轨迹"。保持 None 使
                # BlameLocalizer 对真 Dafny 保守留在规约侧(DafnyBench 模式 A 代码正确,
                # 正确动作是继续补不变量而非修代码);"反例可得率"如实为 0。
                if 1 <= ln <= len(src_lines):
                    related = vars_in(src_lines[ln - 1].strip())
            failures.append(Failure(kind=kind, location=loc, message=line.strip(),
                                    related_vars=related))

        verified = errors = 0
        ms = _SUMMARY_RE.search(raw)
        if ms:
            verified, errors = int(ms.group(1)), int(ms.group(2))
        total = verified + errors
        # "0 errors" can coexist with timed-out/inconclusive obligations.
        # Missing summaries (e.g. a crashed process) are also not evidence of
        # verification success.  Keep raw output so the oracle can distinguish
        # a technical failure from a definite postcondition rejection.
        incomplete = bool(
            (ms and _INCOMPLETE_SUMMARY_RE.search(ms.group(3)))
            or _INCOMPLETE_DIAGNOSTIC_RE.search(raw)
        )
        success = bool(ms) and errors == 0 and not failures and not incomplete
        return VerifyResult(
            success=success,
            obligations_total=total,
            obligations_discharged=verified,
            failures=failures,
            raw=raw,
        )

    @staticmethod
    def _classify(line: str) -> str | None:
        for pat, kind in _KIND_PATTERNS:
            if pat.search(line):
                return kind
        if re.search(r"\berror\b", line, re.I) and "Error:" in line:
            return "other"
        return None
