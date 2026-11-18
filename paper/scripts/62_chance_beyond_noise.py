#!/usr/bin/env python

"""Chance rate of the beyond-noise rule."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("s40", HERE / "40_das_variation.py")
s40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s40)
s38, s02 = s40.s38, s40.s02
SPLITS = 200


def ang(a, b):
    """Angle in degrees between count vectors, row by row."""
    a, b = a.astype(float), b.astype(float)
    c = (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def halves(x, rng):
    """200 split-half DAS values of one group of stays given as seven lobe counts."""
    a = rng.multivariate_hypergeometric(x, int(x.sum()) // 2, size=SPLITS)
    return ang(a, x[None, :] - a)


cfg, _, res, rowlevel = s38.setup("configs/external.yaml")
split = pd.read_csv(res / "iecv_split.csv")
e, X, XM, Y, rows_of, V, no_med = s38.load_eicu(cfg, rowlevel)
flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
C = s40.CONDITIONS
hosp_141 = np.sort(split.hospitalid.to_numpy())
hosp = hosp_141[np.array([any(h in V[c].index for c in C) for h in hosp_141])]
fl = flags[flags.hospitalid.isin(hosp)]
cnt = {}
for c in C:
    codes = s02.lobe_codes(fl, c, ("D", "H", "M"))
    keep = codes >= 0
    g = pd.Series(codes[keep]).groupby(fl.hospitalid.to_numpy()[keep]).apply(lambda v: np.bincount(v, minlength=7))
    cnt[c] = {h: g[h].astype(np.int64) for h in hosp if h in V[c].index}
rng = np.random.default_rng([6200, 40])
own = {h: {c: halves(cnt[c][h], rng) for c in C if h in cnt[c]} for h in hosp}
rows = []
for i, a in enumerate(hosp):
    for b in hosp[i + 1:]:
        shared = [c for c in C if a in cnt[c] and b in cnt[c]]
        if not shared:
            continue
        r = dict(a=a, b=b, nomed=(a in no_med) or (b in no_med))
        da, db, dd = {}, {}, []
        for c in shared:
            xa, xb = cnt[c][a], cnt[c][b]
            ga = rng.multivariate_hypergeometric(xa + xb, int(xa.sum()))
            gb = xa + xb - ga
            d = float(ang(ga[None, :], gb[None, :])[0])
            da[c], db[c] = halves(ga, rng), halves(gb, rng)
            r[c] = d > max(np.percentile(da[c], 95), np.percentile(db[c], 95))
            dd.append(d)
        ma = np.mean([da.get(c, own[a].get(c)) for c in C if a in cnt[c]], axis=0)
        mb = np.mean([db.get(c, own[b].get(c)) for c in C if b in cnt[c]], axis=0)
        r["mean"] = np.mean(dd) > max(np.percentile(ma, 95), np.percentile(mb, 95))
        rows.append(r)
d = pd.DataFrame(rows)
out = []
for sample, sub in [("all", d), ("with_medication", d[~d.nomed])]:
    for m in ["mean"] + C:
        v = sub[m].dropna()
        out.append(dict(sample=sample, measure=m, pairs=len(v), chance_share_beyond_noise=float(v.astype(bool).mean())))
o = pd.DataFrame(out)
(res / "_explore").mkdir(exist_ok=True)
o.to_csv(res / "_explore" / "chance_beyond_noise.csv", index=False, float_format="%.6f")
print(o.round(4).to_string(index=False))
