#!/usr/bin/env python

"""Training-site selection against random sets matched on training-set size."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_s = importlib.util.spec_from_file_location("dn", HERE / "53_deploy_network.py")
dn = importlib.util.module_from_spec(_s)
_s.loader.exec_module(dn)
i38 = dn.i38
ARMS = ["near_mean", "far_mean", "near_membership", "far_membership"]
N_REF = {"held_out": 200, "mimic": 500}
N_B = {"held_out": 1000, "mimic": 2000}
TOL, TOL_WIDE, MIN_REF, MAX_DRAW = 0.10, 0.20, 50, 20000


def matched_sets(pool, sizes, target_n, n_ref, rng):
    """Random sets of 10 from pool with total stays within TOL (else TOL_WIDE) of target_n."""
    for tol in (TOL, TOL_WIDE):
        found, draws = [], 0
        while len(found) < n_ref and draws < MAX_DRAW:
            s = rng.choice(len(pool), i38.K, replace=False)
            draws += 1
            if abs(sizes[s].sum() - target_n) <= tol * target_n:
                found.append(pool[s])
        if len(found) >= MIN_REF:
            return found, tol
    return found, np.nan


def run_target(t, task, ti, X, y, rk, Xt, yt, pool, sets, args, seed):
    rng = np.random.default_rng([seed, 58, dn.HCODE[args.half], ti, int(t) + 1])
    sizes = np.array([len(rk[h]) for h in pool])
    jobs, ref, tol = [], {}, {}
    for a in ARMS:
        jobs.append((a, sets[a]))
        found, tol[a] = matched_sets(pool, sizes, sum(len(rk[h]) for h in sets[a]), N_REF[args.half], rng)
        ref[a] = [len(jobs) + i for i in range(len(found))]
        jobs += [(f"ref_{a}", s) for s in found]
    fits = Parallel(n_jobs=args.jobs, batch_size=8)(
        delayed(dn.fit_pred)(X, y, np.concatenate([rk[h] for h in s]), Xt) for _, s in jobs)
    ok = np.array([f is not None for f in fits])
    P = np.vstack([f if f is not None else np.zeros(len(yt), np.float32) for f in fits])
    A = dn.auc_boot(P, yt, N_B[args.half], rng)
    row = dict(target=t, task=task, stays=len(yt), events=int(yt.sum()), pool=len(pool))
    for i, a in enumerate([j[0] for j in jobs if not j[0].startswith("ref_")]):
        k = [j for j in ref[a] if ok[j]]
        ai = [n for n, (lab, _) in enumerate(jobs) if lab == a][0]
        if not ok[ai] or len(k) < MIN_REF:
            continue
        g = A[:, ai] - A[:, k].mean(axis=1)
        row.update({f"{a}_auc": A[0, ai], f"{a}_ref_auc": A[0, k].mean(), f"{a}_gain": g[0], f"{a}_se": g[1:].std(ddof=1),
                    f"{a}_lo": np.percentile(g[1:], 2.5), f"{a}_hi": np.percentile(g[1:], 97.5),
                    f"{a}_pct": 100 * (A[0, k] < A[0, ai]).mean(), f"{a}_nref": len(k), f"{a}_tol": tol[a],
                    f"{a}_stays": sum(len(rk[h]) for h in sets[a])})
    return row


def cmd_run(args):
    d = dn.load(args.config)
    od = d["res"] / "size_matched" / ("_test" if args.limit else "")
    od.mkdir(parents=True, exist_ok=True)
    members, targets, D, mt, mlook = dn.setting(d, args.half, args.limit)
    if args.half == "mimic":
        memb_of = lambda t, hs: pd.Series(mlook(t, hs), index=hs)
    else:
        mem = pd.read_csv(d["res"] / "deploy_network" / f"membership_{args.half}.csv")
        memb_of = lambda t, hs: mem[mem.target == t].set_index("source").membership_c.reindex(hs)
    out = od / f"size_matched_targets_{args.half}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["target", "task"])
    keys, seed = set(zip(done.target, done.task)), d["seed"]
    for t in targets:
        pool = np.array([h for h in members if h != t])
        ang = pd.Series(D.loc[t, pool].to_numpy(float), index=pool).dropna()
        pool = ang.index.to_numpy()
        sets = i38.make_sets({"mean": ang}, memb_of(t, pool))
        rows = []
        for ti, task in enumerate(dn.TASKS):
            if (t, task) in keys:
                continue
            X, y, rk, Xm, ym = mt[task]
            Xt, yt = (Xm, ym) if args.half == "mimic" else (X[rk[t]], y[rk[t]])
            if len(yt) == 0 or yt.min() == yt.max():
                continue
            rows.append(run_target(t, task, ti, X, y, rk, Xt, yt, pool, sets, args, seed))
            if args.half == "mimic":
                print(pd.DataFrame([rows[-1]]).filter(regex="task|_gain|_pct|_tol").round(4).to_string(index=False), flush=True)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(out, index=False, float_format="%.6f")
        print(f"size-matched {args.half} target {t} done ({done.target.nunique()}/{len(targets)})", flush=True)


def cmd_pool(args):
    cfg, st, res, _ = i38.setup(args.config)
    od = res / "size_matched"
    df = pd.read_csv(od / "size_matched_targets_held_out.csv")
    dn.pool_cols(df, ["task"], ARMS).to_csv(od / "size_matched_pooled_all_targets.csv", index=False, float_format="%.6f")
    df = df[df.events >= i38.MIN_EVENTS_MAIN]
    p = dn.pool_cols(df, ["task"], ARMS)
    p = p.merge(pd.DataFrame([dict(task=t, arm=a, pct_median=g[f"{a}_pct"].median(), tol_wide=(g[f"{a}_tol"] > TOL).mean())
                              for t, g in df.groupby("task") for a in ARMS]), on=["task", "arm"])
    p.to_csv(od / "size_matched_pooled.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    print(p[["task", "arm", "k", "pooled_gain", "lo", "hi", "share_positive", "pct_median", "tol_wide"]].round(4).to_string(index=False))
    m = od / "size_matched_targets_mimic.csv"
    if m.exists():
        print(pd.read_csv(m).filter(regex="task|_gain$|_lo$|_hi$|_pct$|_tol$").round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "pool"])
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--half", choices=["held_out", "mimic"], default="held_out")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"run": cmd_run, "pool": cmd_pool}[args.step](args)


if __name__ == "__main__":
    main()
