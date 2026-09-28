"""语言无关的 TFVR 判据。

给定意图见证集 R = (Sigma+, Sigma-) 与候选规约 phi-hat:

* **E-spec+(不过强)**  ``forall sigma in Sigma+ . sigma |= phi-hat``
* **E-spec-(不过弱)**  ``forall sigma in Sigma- . sigma |/= phi-hat``
* **Verified(phi-hat)** := E-spec+ —— 验证器只查这一侧:"被验对象满足规约"。
* **Faithful(phi-hat)** := E-spec+ 且 E-spec-
* **TFVR** := Faithful 的占比;**Inflation** := VerifyRate - TFVR

**可靠性(soundness)是构造性的**:Sigma+ 里每条轨迹都被 phi* 判真,Sigma- 里每条
都被 phi* 判假(同一套语义)。所以

* 若某条 sigma in Sigma+ 被 phi-hat 判假,则 phi* |/= phi-hat —— 真的不等价;
* 若某条 sigma in Sigma- 被 phi-hat 判真,则 phi-hat |/= phi* —— 真的不等价。

也就是说**判"不忠实"永远不会误报**;判据唯一可能出错的方向是漏报(见证集没覆盖
到区别)。RQ-G2 量的就是这个漏报率,gold 用 ``smt.relation`` 的有界等价。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .monitor import Trace, Undecidable, eval_formula


@dataclass
class XVerdict:
    verified: bool | None            # E-spec+
    rejects_neg: bool | None         # E-spec-
    faithful: bool | None
    n_pos: int = 0
    n_neg: int = 0
    n_undecidable: int = 0
    reason: str = ""
    witness: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"verified": self.verified, "rejects_neg": self.rejects_neg,
                "faithful": self.faithful, "n_pos": self.n_pos, "n_neg": self.n_neg,
                "n_undecidable": self.n_undecidable, "reason": self.reason,
                "witness": self.witness}


def judge(cand, pos, neg) -> XVerdict:
    """按 R 给候选规约下忠实性判决。

    ``pos`` / ``neg`` 是 ``Trace`` 列表。任一侧上出现判不了的轨迹,该轨迹按惯例
    **保守丢弃**(不算通过也不算失败);两侧全判不了则整条判决为 None。
    """
    n_undec = 0
    ok_pos, ok_neg = 0, 0
    v = XVerdict(None, None, None, len(pos), len(neg))

    for i, tr in enumerate(pos):
        try:
            r = eval_formula(cand, tr)
        except Undecidable:
            n_undec += 1
            continue
        if not r:
            v.verified = False
            v.faithful = False
            v.n_undecidable = n_undec
            v.reason = "over-strong: rejects an intended behaviour"
            v.witness = {"side": "pos", "index": i}
            return v
        ok_pos += 1

    for i, tr in enumerate(neg):
        try:
            r = eval_formula(cand, tr)
        except Undecidable:
            n_undec += 1
            continue
        if r:
            v.verified = True if ok_pos else None
            v.rejects_neg = False
            v.faithful = False
            v.n_undecidable = n_undec
            v.reason = "too weak: admits a forbidden behaviour"
            v.witness = {"side": "neg", "index": i}
            return v
        ok_neg += 1

    v.n_undecidable = n_undec
    if ok_pos == 0 and ok_neg == 0:
        v.reason = "no decidable witness"
        return v
    v.verified = True if ok_pos else None
    v.rejects_neg = True if ok_neg else None
    v.faithful = bool(v.verified) and bool(v.rejects_neg)
    v.reason = "faithful on R" if v.faithful else "insufficient witnesses"
    return v


def load_traces(items) -> list:
    return [Trace.from_json(d) for d in items]


# --------------------------------------------------------------------------- #
def aggregate(records) -> dict:
    """把逐条判决汇总成 VerifyRate / TFVR / Inflation。

    分母是**可评实例**(判决非 None)。分母口径必须写死在这里,否则不同脚本各算
    各的,表和正文就会对不上。
    """
    n = ver = faith = 0
    for r in records:
        if r.get("verified") is None or r.get("faithful") is None:
            continue      # 见证不足,整条实例不进分母(与 Dafny 支的"证据不足"同口径)
        n += 1
        if r.get("verified"):
            ver += 1
        if r.get("faithful"):
            faith += 1
    if n == 0:
        return {"n": 0, "verify_rate": None, "tfvr": None, "inflation": None,
                "conditional_inflation": None}
    vr, tf = ver / n, faith / n
    return {"n": n, "n_verified": ver, "n_faithful": faith,
            "verify_rate": vr, "tfvr": tf, "inflation": vr - tf,
            "conditional_inflation": (vr - tf) / vr if vr else None}
