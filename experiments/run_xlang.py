"""跨语言 TFVR 主实验驱动(受控变异臂)。

对每个任务:
  1. 从 ``spec_json`` 还原参考公式 phi*,从 ``pos``/``neg`` 还原见证集 R;
  2. 用 ``mutate.mutants`` 造候选规约(W/S/N 三族,模拟"机器翻译出来的规约");
  3. 用 ``tfvr_x.judge`` 在 R 上给 Verified / Faithful 判决(**这就是 TFVR**);
  4. 可选地用 ``smt.relation`` 算有界语义关系作为 gold(RQ-G2 的效度检验)。

输出逐候选一行 JSON,聚合留给 ``agg_xlang.py`` —— 原始判决一次算完存盘,后面换
口径重算不必重跑求解器。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

from specforge.xlang import smt                       # noqa: E402
from specforge.xlang.monitor import Trace             # noqa: E402
from specforge.xlang.mutate import mutants            # noqa: E402
from specforge.xlang.tfvr_x import judge              # noqa: E402
from specforge.xlang.tlast import from_json, to_json, to_str  # noqa: E402


def load_tasks(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def run_task(task, k, do_smt, smt_timeout, seed):
    ref = from_json(task["spec_json"])
    pos = [Trace.from_json(d) for d in task["pos"]]
    neg = [Trace.from_json(d) for d in task["neg"]]
    doms, H, dt = task["vars"], task["horizon"], task["dt"]
    recs = []

    def one(op, fam, cand):
        v = judge(cand, pos, neg)
        r = {"task_id": task["task_id"], "lang": task["lang"], "source": task["source"],
             "op": op, "family": fam, "spec_cand": to_str(cand),
             # 候选的 IR 原样存盘。下游脚本(|R| 敏感性、视界敏感性)必须从这里
             # 还原候选,**不要**去重新解析 spec_cand 那个给人看的字符串 ——
             # 那条路会引入解析漂移,SVA 上更是直接解不动。
             "spec_cand_json": to_json(cand),
             "n_pos": len(pos), "n_neg": len(neg)}
        r.update(v.to_json())
        if do_smt:
            t0 = time.time()
            rel = smt.relation(ref, cand, doms, H, dt, timeout_ms=smt_timeout)
            r["gold"] = rel["verdict"]
            r["gold_ref_entails_cand"] = rel["ref_entails_cand"]
            r["gold_cand_entails_ref"] = rel["cand_entails_ref"]
            r["smt_secs"] = round(time.time() - t0, 3)
        recs.append(r)

    one("identity", "REF", ref)          # 完美翻译对照组:必须 verified 且 faithful
    for op, fam, cand in mutants(ref, family=None, limit=k, seed=seed):
        one(op, fam, cand)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=12, help="每任务候选变异体数量")
    ap.add_argument("--smt", action="store_true", help="同时算有界语义 gold")
    ap.add_argument("--smt-timeout", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = load_tasks(args.data, args.limit or None)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            try:
                recs = run_task(task, args.k, args.smt, args.smt_timeout, args.seed + i)
            except Exception as e:
                print("  task %s failed: %r" % (task.get("task_id"), e)[:200])
                continue
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += len(recs)
            if (i + 1) % 25 == 0:
                print("  %4d/%4d tasks  %6d records  %.1fs"
                      % (i + 1, len(tasks), n, time.time() - t0), flush=True)
    print("done: %d tasks -> %d records in %.1fs  (%s)"
          % (len(tasks), n, time.time() - t0, args.out))


if __name__ == "__main__":
    main()
