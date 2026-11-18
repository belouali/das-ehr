#!/usr/bin/env python

"""DAS from all eligible stays against DAS from the modeled stays."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import os
import yaml
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s02", HERE / "02_das_external.py")
s02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s02)
cfg = yaml.safe_load(open(HERE.parent / os.environ.get("EHRDOC_CONFIG", "configs/external.yaml")))
RL, RES = Path(cfg["paths"]["rowlevel"]), HERE.parent / cfg["paths"]["results"]
CONDITIONS = ["diabetes", "hypertension", "chf", "af"]

flags = pd.read_parquet(RL / "eicu_flags.parquet")
m1 = pd.read_parquet(RL / "eicu_m1_features.parquet", columns=["patientunitstayid", "hospitalid"])
size = m1.hospitalid.value_counts()
study = sorted(size[size >= 100].index)
flags = flags[flags.hospitalid.isin(study)]
sub = flags[flags.patientunitstayid.isin(m1.patientunitstayid)]


def mean_das(fl, min_n):
    H = len(study); tot = np.zeros((H, H)); k = np.zeros((H, H))
    for c in CONDITIONS:
        cnt = s02.hospital_counts(fl, s02.lobe_codes(fl, c, ("D", "H", "M")), 7)
        cnt = cnt[cnt.sum(axis=1) >= min_n].reindex(study)
        v = cnt.div(cnt.sum(axis=1), axis=0).to_numpy(float)
        u = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        a = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))
        ok = ~np.isnan(a)
        tot[ok] += a[ok]; k[ok] += 1
    with np.errstate(invalid="ignore"):
        return np.where(k > 0, tot / k, np.nan)


rows = []
for min_n, label in [(100, "documentation analysis (at least 100 stays with evidence)"), (30, "training-site selection (at least 30)")]:
    A, B = mean_das(flags, min_n), mean_das(sub, min_n)
    iu = np.triu_indices(len(study), 1)
    a, b = A[iu], B[iu]; ok = ~np.isnan(a) & ~np.isnan(b)
    d = np.abs(a[ok] - b[ok])
    ov = []
    for i in range(len(study)):
        ra, rb = A[i].copy(), B[i].copy(); ra[i] = rb[i] = np.nan
        if np.isnan(ra).all() or np.isnan(rb).all(): continue
        na = set(np.argsort(np.where(np.isnan(ra), np.inf, ra))[:10]); nb = set(np.argsort(np.where(np.isnan(rb), np.inf, rb))[:10])
        ov.append(len(na & nb))
    ov = np.array(ov)
    rows.append(dict(setting=label, stays_all=len(flags), stays_modeled=len(sub), hospitals_all=int((~np.isnan(A)).any(axis=1).sum()), hospitals_modeled=int((~np.isnan(B)).any(axis=1).sum()),
                     pairs=int(ok.sum()), spearman=round(spearmanr(a[ok], b[ok])[0], 4), median_das_all=round(np.median(a[ok]), 2),
                     median_das_modeled=round(np.median(b[ok]), 2), abs_diff_median=round(np.median(d), 2), abs_diff_p95=round(np.quantile(d, .95), 2),
                     nearest10_shared_mean=round(ov.mean(), 2), nearest10_shared_median=float(np.median(ov)), hospitals_sharing_8plus=round((ov >= 8).mean(), 3)))
out = pd.DataFrame(rows)
print(f"stays: all {len(flags)}, modeled {len(sub)}")
print(out.T.to_string())
out.to_csv(RES / "_explore" / "das_all_vs_modeled_stays.csv", index=False)
