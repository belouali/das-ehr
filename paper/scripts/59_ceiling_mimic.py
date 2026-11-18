#!/usr/bin/env python

"""Selection ceiling in MIMIC-IV."""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("iecv", HERE / "38_iecv.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
OUT = os.environ.get("EHRDOC_OUT", "v7_2026-11-16")
rows = []


def hook(task, ti, arms, P, ym, seed):
    rng = np.random.default_rng([seed, 5900, ti])
    pos, neg = rng.permutation(np.flatnonzero(ym == 1)), rng.permutation(np.flatnonzero(ym == 0))
    S = np.sort(np.r_[pos[: len(pos) // 2], neg[: len(neg) // 2]])
    E = np.sort(np.r_[pos[len(pos) // 2:], neg[len(neg) // 2:]])
    full, fr, _, _, _ = M.score_target(arms, P, ym, 0, rng)
    ri = [i for i, a in enumerate(arms) if a.startswith("rand_")]
    aS = np.array([M.s19.auc(ym[S], P[i, S]) for i in ri])
    first = np.array([int(arms[i].split("_")[1]) < 200 for i in ri])
    pick = {"set_best": ri[int(np.argmax(aS))], "set_worst": ri[int(np.argmin(aS))],
            "set_best200": ri[int(np.argmax(np.where(first, aS, -np.inf)))],
            "set_worst200": ri[int(np.argmin(np.where(first, aS, np.inf)))]}
    arms_e = arms + list(pick)
    PE = np.vstack([P[:, E]] + [P[i, E] for i in pick.values()])
    point, rmean, rauc, draws, _ = M.score_target(arms_e, PE, ym[E], M.N_BOOT["mimic"], rng)
    for a in point:
        rows.append(dict(task=task, arm=a, auroc=point[a], random_mean=rmean, gain=point[a] - rmean,
                         lo=np.percentile(draws[a], 2.5), hi=np.percentile(draws[a], 97.5),
                         percentile=100 * (rauc < point[a]).mean(),
                         full_gain=full[a] - fr if a in full else np.nan, stays=len(E), events=int(ym[E].sum())))
    print(pd.DataFrame(rows)[lambda x: x.task == task].round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    M.TASKS = M.TASKS + ["icu_readmit"]
    M.DAS_MIN[0] = 30
    M.setup.out = OUT
    M.HOOK[0] = hook
    M.cmd_mimic(argparse.Namespace(config="configs/external.yaml", measure="mean", sample="all", spec="v6",
                                   jobs=a.jobs, tasks=""))
    res = Path(M.setup("configs/external.yaml")[2])
    pd.DataFrame(rows).to_csv(res / "ceiling_mimic.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
