#!/usr/bin/env python

"""Random-half check combined over five random splits."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_s = importlib.util.spec_from_file_location("i38", HERE / "38_iecv.py")
i38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(i38)

R = HERE.parent / "results_v7" / "v7_2026-11-16"
ARMS = ["near_mean", "far_mean", "near_membership"]
comb, by = [], []
for analysis, minev in [("main", 10), ("all_targets", 0)]:
  for sample in ["all", "with_medication"]:
    parts = []
    for k, d in enumerate(["random_half"] + [f"random_half_r{i}" for i in range(1, 5)]):
        t = pd.read_csv(R / d / f"iecv_targets_held_out_{sample}.csv").assign(split=k)
        t = t[t.events >= minev]
        parts.append(t)
        for task, s in t.groupby("task"):
            for a in ARMS:
                ok = s[f"{a}_gain"].notna() & (s[f"{a}_se"] > 0)
                r = i38.reml(s.loc[ok, f"{a}_gain"], s.loc[ok, f"{a}_se"] ** 2)
                by.append(dict(analysis=analysis, sample=sample, split=k, task=task, arm=a, k=r["k"], pooled_gain=r["pooled_gain"], lo=r["lo"], hi=r["hi"]))
    df = pd.concat(parts)
    for task, s in df.groupby("task"):
        for a in ARMS:
            s2 = s[s[f"{a}_gain"].notna() & (s[f"{a}_se"] > 0)].assign(v=lambda x: x[f"{a}_se"] ** 2)
            g = s2.groupby("target").agg(gain=(f"{a}_gain", "mean"), v=("v", "mean"), n=("split", "nunique"))
            g = g[g.n == 5]
            r = i38.reml(g.gain, g.v)
            comb.append(dict(analysis=analysis, sample=sample, task=task, arm=a, hospitals=r["k"], pooled_gain=r["pooled_gain"], lo=r["lo"], hi=r["hi"],
                             share_positive=r["share_positive"]))
comb, by = pd.DataFrame(comb), pd.DataFrame(by)
comb.to_csv(R / "random_half_combined.csv", index=False, float_format="%.6f")
by.to_csv(R / "random_half_by_split.csv", index=False, float_format="%.6f")
pd.set_option("display.width", 250)
print(comb[comb.analysis == "main"].round(4).to_string(index=False))
w = by[(by.arm == "near_mean") & (by.analysis == "main")].assign(cell=lambda x: x.pooled_gain.round(4).astype(str) + x.apply(lambda r: "*" if r.lo > 0 else "", axis=1))
print(w.pivot_table(index=["sample", "task"], columns="split", values="cell", aggfunc="first").to_string())
