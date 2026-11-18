#!/usr/bin/env python

"""DAS variation with the hospitals outside the study added back."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import os
import yaml

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


s02, s40 = load("s02", "02_das_external.py"), load("s40", "40_das_variation.py")
cfg = yaml.safe_load(open(HERE.parent / os.environ.get("EHRDOC_CONFIG", "configs/external.yaml")))
RL, RES = Path(cfg["paths"]["rowlevel"]), HERE.parent / cfg["paths"]["results"]
CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
rng = np.random.default_rng([cfg["das"]["seed"], 65])

flags = pd.read_parquet(RL / "eicu_flags.parquet")
split = pd.read_csv(RES / "iecv_split.csv")
study = set(split.hospitalid)
V = {}
for c in CONDITIONS:
    cnt = s02.hospital_counts(flags, s02.lobe_codes(flags, c, ("D", "H", "M")), 7)
    cnt = cnt[cnt.sum(axis=1) >= cfg["das"]["min_records_per_hospital"]]
    V[c] = cnt.div(cnt.sum(axis=1), axis=0)
hosp = np.array(sorted(set().union(*[set(V[c].index) for c in CONDITIONS])))
in_study = np.array([h in study for h in hosp])
no_med = flags.groupby("hospitalid").any_M.max().reindex(hosp).to_numpy() == 0
stays = flags.hospitalid.value_counts().reindex(hosp).to_numpy()
print(f"hospitals with a DAS measure: {len(hosp)} (study {in_study.sum()}, added {(~in_study).sum()}); "
      f"added hospitals: {stays[~in_study].sum()} stays, {no_med[~in_study].sum()} with no medication record "
      f"(study: {no_med[in_study].sum()} of {in_study.sum()})")

n = len(hosp); mats = {}
for c in CONDITIONS:
    m = np.full((n, n), np.nan); have = np.array([h in V[c].index for h in hosp]); idx = np.flatnonzero(have)
    m[np.ix_(idx, idx)] = s02.pairwise_angles(V[c].loc[hosp[have]].to_numpy(float)); mats[c] = m
with np.errstate(all="ignore"):
    import warnings; warnings.simplefilter("ignore")
    mats["mean"] = np.nanmean(np.stack([mats[c] for c in CONDITIONS]), axis=0)
nf = s40.noise_floor(flags, V, hosp, 200, rng)
iu = np.triu_indices(n, 1)
rows = []
for meas in ["mean"] + CONDITIONS:
    q95 = nf[nf.condition == meas].set_index("hospitalid").split_half_q95.reindex(hosp).to_numpy()
    thr = np.maximum.outer(q95, q95)[iu]; d = mats[meas][iu]
    a, b = in_study[iu[0]], in_study[iu[1]]
    for label, keep in [("study hospitals", a & b), ("all hospitals", np.ones(len(d), bool)),
                        ("added to study", a ^ b), ("added to added", ~a & ~b)]:
        k = keep & np.isfinite(d) & np.isfinite(thr)
        if not k.any():
            rows.append(dict(measure=meas, pairs_of=label, hospitals=0, pairs=0)); continue
        hs = np.unique(np.concatenate([iu[0][k], iu[1][k]]))
        rows.append(dict(measure=meas, pairs_of=label, hospitals=len(hs), pairs=int(k.sum()), das_median=round(np.median(d[k]), 1),
                         das_q25=round(np.percentile(d[k], 25), 1), das_q75=round(np.percentile(d[k], 75), 1),
                         share_pairs_beyond_noise=round((d[k] > thr[k]).mean(), 3)))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(RES / "_explore" / "das_with_dropped_hospitals.csv", index=False)
