"""跨模块共享的小工具。"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize_clause(s: str) -> str:
    """归一化规约子句字符串(折叠空白),用于离线 Mock 的集合比较。"""
    return _WS.sub(" ", s).strip()


def extract_clauses(src: str, keyword: str) -> list[str]:
    """从 Dafny 源码中抽取以 ``keyword`` 开头的子句(如 invariant/ensures/requires)。"""
    out: list[str] = []
    pat = re.compile(rf"^\s*{keyword}\s+(.+?)\s*$")
    for line in src.splitlines():
        m = pat.match(line)
        if m:
            out.append(normalize_clause(m.group(1)))
    return out


def method_name(src: str) -> str | None:
    m = re.search(r"\b(?:method|function|lemma|predicate)\s+(\w+)", src)
    return m.group(1) if m else None


def vars_in(text: str) -> list[str]:
    """从子句文本里粗抽标识符(给反例/aligner 当 related_vars 提示用)。"""
    toks = re.findall(r"[A-Za-z_]\w*(?:\.\w+)?", text)
    kw = {"forall", "exists", "old", "true", "false", "Length"}
    seen: list[str] = []
    for t in toks:
        base = t.split(".")[0]
        if base not in kw and base not in seen:
            seen.append(base)
    return seen
