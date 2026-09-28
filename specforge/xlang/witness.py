"""见证集 R = (Sigma+, Sigma-) 的生成。

R 是"意图的独立投影"。在 Dafny 那一支里它来自**执行真值实现**;在时序语言这一支
里它来自**参考形式化 phi* 对行为的判决**。两者的共同要害是同一个:R 的构造只看
意图侧,**绝不看候选规约 phi-hat** —— 判据因此不循环。

两类反例:

* **随机反例**:随机采样后被 phi* 判否的轨迹。便宜,但往往"错得太离谱",
  过弱的候选也能拒掉,信息量低。
* **近失反例(near-miss)**:从一条正例出发,做**最小扰动**直到 phi* 翻否。
  这是 Dafny 支里"把正确输出 e 改成一个邻近的错误输出 w"的时序对应物,也是真正
  能咬住过弱规约的那一类见证。默认优先取近失反例。
"""

from __future__ import annotations

import json
import random

from .monitor import Trace, Undecidable, eval_formula


def sample_value(dom: dict, rnd: random.Random, bias: float = 0.5):
    kind = dom["kind"]
    if kind == "bool":
        return rnd.random() < bias
    if kind == "enum":
        return rnd.choice(dom["values"])
    ts = dom.get("thresholds") or []
    lo, hi = dom.get("range", [-10.0, 10.0])
    if ts and rnd.random() < 0.65:
        # 打在阈值附近:判决翻转就发生在这里,均匀采样会系统性错过
        c = rnd.choice(ts)
        step = rnd.choice([-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0])
        return round(c + step, 4)
    return round(rnd.uniform(lo, hi), 4)


def sample_trace(doms: dict, H: int, dt: float, rnd: random.Random) -> Trace:
    bias = rnd.choice([0.25, 0.5, 0.5, 0.75])
    data = {}
    for name, dom in doms.items():
        if dom["kind"] == "real" and rnd.random() < 0.4:
            # 一段分段常值的信号:比逐点独立采样更像真实的被测信号,也更容易
            # 触发 always/eventually 的窗口语义
            col, t = [], 0
            while t < H:
                v = sample_value(dom, rnd, bias)
                run = min(H - t, rnd.randint(1, max(1, H // 3)))
                col += [v] * run
                t += run
            data[name] = col[:H]
        else:
            data[name] = [sample_value(dom, rnd, bias) for _ in range(H)]
    return Trace(data=data, dt=dt)


def _flip(dom: dict, cur, rnd: random.Random):
    """给一个变量在某时刻换一个"邻近但不同"的取值。"""
    kind = dom["kind"]
    if kind == "bool":
        return not cur
    if kind == "enum":
        alts = [v for v in dom["values"] if v != cur]
        return rnd.choice(alts) if alts else cur
    ts = dom.get("thresholds") or []
    if ts:
        c = min(ts, key=lambda x: abs(x - (cur if isinstance(cur, (int, float)) else 0.0)))
        for d in (0.1, -0.1, 1.0, -1.0, 5.0, -5.0):
            cand = round(c + d, 4)
            if cand != cur:
                return cand
    lo, hi = dom.get("range", [-10.0, 10.0])
    return round(rnd.uniform(lo, hi), 4)


def near_miss(phi, tr: Trace, doms: dict, rnd: random.Random, want: bool,
              max_tries: int = 240):
    """从 tr 出发做最小扰动,找一条 phi 判决为 ``want`` 的邻近轨迹。

    先试单点扰动(改一个变量的一个时刻),不成再试双点。返回 None 表示没找到。
    """
    names = list(doms)
    H = tr.H
    cells = [(n, t) for n in names for t in range(H)]
    rnd.shuffle(cells)
    for name, t in cells[:max_tries]:
        old = tr.data[name][t]
        new = _flip(doms[name], old, rnd)
        if new == old:
            continue
        tr.data[name][t] = new
        try:
            if eval_formula(phi, tr) is want:
                out = Trace(data={k: list(v) for k, v in tr.data.items()}, dt=tr.dt)
                tr.data[name][t] = old
                return out
        except Undecidable:
            pass
        tr.data[name][t] = old
    return None


def smt_sample(phi, doms: dict, H: int, dt: float, want: bool, k: int,
               seed: int = 0, timeout_ms: int = 6000, attempts: int = 10):
    """用 Z3 直接合成 k 条判决为 ``want`` 的轨迹。

    随机采样对**算术型**需求几乎必然失手 —— FRET 里 `yout = timeStep*0.5*(xin+xinpv)+ypv`
    这种等式,随便掷点永远命不中,整条需求就会被"无正例"丢掉,而这恰恰是航空航天
    需求里最常见的一类。求解器能一步给出解,所以随机采样打空时改走这条路。

    多样性靠"随机钉住一部分格点再求解"来拿:钉得多则解更散,钉得太多会 UNSAT,
    所以钉的比例从高到低退让。
    """
    from .smt import Encoder, Unsupported
    import z3

    rnd = random.Random(seed)
    out, seen = [], set()
    fracs = [0.7, 0.5, 0.35, 0.2, 0.1, 0.0]
    for attempt in range(attempts * max(1, k)):
        if len(out) >= k:
            break
        try:
            enc = Encoder(doms, H, dt)
            s = z3.Solver()
            s.set("timeout", timeout_ms)
            for a in enc.axioms:
                s.add(a)
            root = enc.root(phi)
            s.add(root if want else z3.Not(root))
            base = sample_trace(doms, H, dt, rnd)
            cells = [(n, t) for n in doms for t in range(H)]
            rnd.shuffle(cells)
            frac = fracs[min(attempt // max(1, k), len(fracs) - 1)]
            for name, t in cells[: int(len(cells) * frac)]:
                v = base.data[name][t]
                c = enc.cells[name][t]
                if doms[name]["kind"] == "bool":
                    s.add(c if v else z3.Not(c))
                elif doms[name]["kind"] == "enum":
                    s.add(c == doms[name]["values"].index(v))
                else:
                    s.add(c == z3.RealVal(float(v)))
            for tr in out:                    # 阻塞已得到的解,避免重复
                s.add(z3.Or(*[enc.cells[n][t] != _z3lit(enc, doms, n, tr.data[n][t])
                              for n in doms for t in range(H)]))
            if s.check() != z3.sat:
                continue
            tr = enc.model_to_trace(s.model())
        except (Unsupported, z3.Z3Exception, Exception):
            continue
        key = json.dumps(tr.to_json(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            if eval_formula(phi, tr) is not want:
                continue                      # 求解器口径与监控器口径不一致,弃用
        except Undecidable:
            continue
        out.append(tr)
    return out


def _z3lit(enc, doms, name, v):
    import z3
    k = doms[name]["kind"]
    if k == "bool":
        return z3.BoolVal(bool(v))
    if k == "enum":
        return z3.IntVal(doms[name]["values"].index(v))
    return z3.RealVal(float(v))


def build_witnesses(phi, doms: dict, H: int, dt: float, n_pos: int = 4, n_neg: int = 4,
                    seed: int = 0, budget: int = 900, use_smt: bool = True):
    """返回 (pos, neg, stats)。任一侧为空即视为该任务不可评(证据不足,保守丢弃)。"""
    rnd = random.Random(seed)
    pos, neg_random = [], []
    tried = 0
    while tried < budget and (len(pos) < n_pos * 3 or len(neg_random) < n_neg * 3):
        tried += 1
        tr = sample_trace(doms, H, dt, rnd)
        try:
            v = eval_formula(phi, tr)
        except Undecidable:
            continue
        (pos if v else neg_random).append(tr)

    stats = {"sampled": tried, "n_pos_pool": len(pos), "n_neg_pool": len(neg_random),
             "smt_pos": 0, "smt_neg": 0}
    # 随机打空的那一侧交给求解器 —— 算术型需求全靠这一步才进得了评测集
    if use_smt and not pos:
        extra = smt_sample(phi, doms, H, dt, True, n_pos * 2, seed=seed)
        pos += extra
        stats["smt_pos"] = len(extra)
    if use_smt and not neg_random:
        extra = smt_sample(phi, doms, H, dt, False, n_neg * 2, seed=seed + 1)
        neg_random += extra
        stats["smt_neg"] = len(extra)

    # Sigma-:优先近失反例;不足再用随机反例补齐
    neg = []
    for tr in pos[: n_neg * 4]:
        if len(neg) >= n_neg:
            break
        nm = near_miss(phi, tr, doms, rnd, want=False)
        if nm is not None:
            neg.append(nm)
    stats["n_near_miss"] = len(neg)
    for tr in neg_random:
        if len(neg) >= n_neg:
            break
        neg.append(tr)

    # Sigma+:同样掺入"从反例最小扰动回来"的正例,让正侧也贴着判决边界
    out_pos = pos[:n_pos]
    if len(out_pos) < n_pos:
        for tr in neg_random[: n_pos * 4]:
            if len(out_pos) >= n_pos:
                break
            nm = near_miss(phi, tr, doms, rnd, want=True)
            if nm is not None:
                out_pos.append(nm)
    stats["n_pos"], stats["n_neg"] = len(out_pos), len(neg)
    return out_pos, neg, stats
