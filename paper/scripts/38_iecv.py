#!/usr/bin/env python

"""Training-site selection by internal-external cross-validation: hospital split, development screen, held-out test and MIMIC-IV external test.

Usage (from paper/): python scripts/38_iecv.py {split,loop,pool,mimic} --spec v6 --out v7_2026-11-16 --config configs/external_v7.yaml [--half held_out] [--sample with_medication] [--measure mean]"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from joblib import Parallel, delayed
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_model import C_M1, design_columns


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_das_external.py")
s19 = _load("s19", "19_ceiling.py")
s24 = _load("s24", "24_w1_mimic.py")
s35 = _load("s35", "35_membership_eicu.py")

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
MEASURES = CONDITIONS + ["mean"]
FLAGS = [f"{c}_{s}" for c in CONDITIONS for s in ("D", "H", "M")]
TASKS = ["mortality", "A3_los_gt7d", "A6_hospital_los_gt14d", "A7_los_gt5d"]
MIMIC_TASK = {"mortality": "death", "A3_los_gt7d": "los_gt7d", "A6_hospital_los_gt14d": "hosp_los_gt14d",
              "A7_los_gt5d": "los_gt5d"}
K, MIN_STAYS, N_REPEAT, MIN_EVENTS_SENS = 10, 100, 10, 20
MIN_EVENTS_MAIN = 10
N_RAND = {"eicu": 200, "mimic": 500}
N_BOOT = {"eicu": 1000, "mimic": 2000}
HOOK = [None]


def setup(config):
    config = os.environ.get("EHRDOC_CONFIG", config)
    cfg_path = Path(config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    st = yaml.safe_load(open(base / "configs" / "screen_settings.yaml"))["S_site_selection_iecv"]
    assert st["status"] == "locked", "S_site_selection_iecv must be locked"
    res = base / cfg["paths"]["results"]
    res.mkdir(exist_ok=True)
    if getattr(setup, "out", ""):
        sub = res / setup.out
        sub.mkdir(exist_ok=True)
        if not (sub / "iecv_split.csv").exists():
            (sub / "iecv_split.csv").write_text((res / "iecv_split.csv").read_text())
        res = sub
    return cfg, st, res, Path(cfg["paths"]["rowlevel"])


def flag_dir(rowlevel):
    """Folder holding the evidence-flag files; a medication-definition variant (EHRDOC_VARIANT) has its own folder."""
    v = os.environ.get("EHRDOC_VARIANT", "")
    return Path(str(rowlevel) + f"_{v}") if v else rowlevel


def load_eicu(cfg, rowlevel):
    E = cfg["paths"]["eicu"]
    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    if os.environ.get("EHRDOC_APACHE_OFFICIAL") and "apache_source" in e.columns:
        e = e[e.apache_source == "official"].reset_index(drop=True)
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    flags = pd.read_parquet(flag_dir(rowlevel) / "eicu_flags.parquet")
    if "copd" in CONDITIONS:
        flags = flags.merge(pd.read_parquet(rowlevel / "eicu_flags_extra.parquet").drop(columns="hospitalid"),
                            on="patientunitstayid", how="left")
    e = e.merge(flags[["patientunitstayid", "any_M"] + FLAGS], on="patientunitstayid", how="left")
    e[FLAGS + ["any_M"]] = e[FLAGS + ["any_M"]].fillna(0).astype(float)
    pt = duckdb.execute(f"""SELECT patientunitstayid, unitdischargeoffset / 1440.0 AS los_d,
                                   (hospitaldischargeoffset - hospitaladmitoffset) / 1440.0 AS hosp_los_d
                            FROM '{E}/patient.parquet'""").df()
    e = e.merge(pt, on="patientunitstayid", how="left")
    assert e[["death", "los_d", "hosp_los_d"]].notna().all().all()
    epf = rowlevel / "eicu_icu_episode.parquet"
    if epf.exists():
        n0 = len(e)
        e = e.merge(pd.read_parquet(epf)[["patientunitstayid", "episode_d", "episode_alive", "readmit_after"]], on="patientunitstayid", how="left")
        assert len(e) == n0 and e.episode_d.notna().all()
        e["unit_los_d"] = e.los_d
        e["los_d"] = e.episode_d
    Y = {"mortality": e.death.to_numpy(int), "A3_los_gt7d": (e.los_d > 7).to_numpy(int),
         "A6_hospital_los_gt14d": (e.hosp_los_d > 14).to_numpy(int), "A7_los_gt5d": (e.los_d > 5).to_numpy(int)}
    X = e[design_columns() + ["charlson"] + FLAGS].to_numpy(float)
    XM = e[design_columns()].to_numpy(float)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}

    flags_das = flags
    if os.environ.get("EHRDOC_DAS_24H"):
        f24d = pd.read_parquet(flag_dir(rowlevel) / "eicu_flags_24h.parquet")
        flags_das = flags[["patientunitstayid", "hospitalid"]].merge(f24d[["patientunitstayid"] + FLAGS], on="patientunitstayid", how="left")
        flags_das[FLAGS] = flags_das[FLAGS].fillna(False).astype(bool)
    V, blocks = {}, []
    for cond in CONDITIONS:
        raw = s02.hospital_counts(flags_das, s02.lobe_codes(flags_das, cond, ("D", "H", "M")), 7)
        cnt = raw[raw.sum(axis=1) >= (DAS_MIN[0] or cfg["das"]["min_records_per_hospital"])]
        V[cond] = cnt.div(cnt.sum(axis=1), axis=0)
        NREC[cond] = cnt.sum(axis=1)
        raw = raw.reindex(np.unique(flags.hospitalid), fill_value=0)
        p0 = raw.sum(axis=0) / raw.to_numpy().sum()
        b = (raw + SHRINK * p0).div(raw.sum(axis=1) + SHRINK, axis=0)
        blocks.append(b.div(np.sqrt((b ** 2).sum(axis=1)), axis=0).add_prefix(f"{cond}_"))
    JOINT["v"] = pd.concat(blocks, axis=1)
    no_med = set(e.groupby("hospitalid").any_M.max().loc[lambda s: s == 0].index)
    if "icu_readmit" in TASKS:
        p = duckdb.execute(f"""SELECT patientunitstayid, patienthealthsystemstayid, unitvisitnumber, unitstaytype,
                                      unitdischargestatus, hospitaldischargestatus, hospitaldischargelocation
                               FROM '{E}/patient.parquet'""").df()
        later = p[p.unitstaytype == "readmit"].groupby("patienthealthsystemstayid").unitvisitnumber.max().rename("last_readmit")
        q = e[["patientunitstayid"]].merge(p, on="patientunitstayid", how="left").merge(
            later, on="patienthealthsystemstayid", how="left")
        assert (q.patientunitstayid.to_numpy() == e.patientunitstayid.to_numpy()).all()
        Y["icu_readmit"] = (q.last_readmit > q.unitvisitnumber).to_numpy(int)
        ELIG["icu_readmit"] = (q.unitdischargestatus == "Alive").to_numpy()
        if "episode_d" in e.columns:
            Y["icu_readmit"] = e.readmit_after.to_numpy(int)
            ELIG["icu_readmit"] = e.episode_alive.to_numpy(bool)


        fac = q.hospitaldischargelocation.isin(["Skilled Nursing Facility", "Rehabilitation", "Nursing Home"])
        el = (q.hospitaldischargestatus == "Alive") & (fac | (q.hospitaldischargelocation == "Home"))
        share = pd.Series(fac[el].to_numpy(), index=e.hospitalid[el].to_numpy()).groupby(level=0).mean()
        DROP_FAC.clear(); DROP_FAC.update(share[share < FAC_MIN].index)
        Y["disch_facility"] = fac.to_numpy(int)
        ELIG["disch_facility"] = (el & ~e.hospitalid.isin(DROP_FAC)).to_numpy()
        XDIS[0] = np.hstack([XM, e[FLAGS].to_numpy(float), np.log1p(e.los_d.to_numpy(float))[:, None]])
    return e, X, XM, Y, rows_of, V, no_med


NREC, JOINT, SHRINK = {}, {}, 100
DAS_MIN, ELIG, XDIS = [0], {}, [None]
COMBINED = [False]
DROP_FAC, FAC_MIN = set(), 0.02


def task_pool(task, pool, sets, ang, memb):
    """disch_facility: candidates without the hospitals that almost never record a facility; sets rebuilt on that pool."""
    if task != "disch_facility" or not DROP_FAC:
        return pool, sets
    keep = np.array([h for h in pool if h not in DROP_FAC])
    a2 = {k: v[v.index.isin(keep)] for k, v in ang.items()}
    s2 = make_sets({k: v for k, v in a2.items() if len(v) >= 2 * K}, memb[memb.index.isin(keep)] if memb is not None else None)
    return keep, {k: v for k, v in s2.items() if k in sets}
SOFA_COLS = ["sofa", "sofa_resp", "sofa_coag", "sofa_liver", "sofa_cv", "sofa_cns", "sofa_renal"]


def rcs(x, q=(5, 35, 65, 95)):
    """Nonlinear terms of a restricted cubic spline (Harrell), knots at percentiles of x; the linear term is in X."""
    k = np.percentile(x, q)
    p = lambda u: np.clip(u, 0, None) ** 3
    d = (k[-1] - k[0]) ** 2
    return np.column_stack([(p(x - k[j]) - p(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                             + p(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / d for j in range(len(k) - 2)])


def variant_rankings(V, t, cand):
    """v5: record-weighted mean of condition angles, and the angle between stacked shrunken vectors."""
    out = {}
    num, den = pd.Series(0.0, index=cand), pd.Series(0.0, index=cand)
    for cond in CONDITIONS:
        if t not in V[cond].index:
            continue
        cc = np.array([h for h in cand if h in V[cond].index])
        if len(cc) == 0:
            continue
        a = s02.angles(V[cond].loc[cc].to_numpy(float), V[cond].loc[t].to_numpy(float))
        w = 1.0 / (1.0 / NREC[cond][t] + 1.0 / NREC[cond].loc[cc].to_numpy(float))
        num.loc[cc] += a * w
        den.loc[cc] += w
    wm = (num / den)[den > 0]
    if len(wm) >= 2 * K:
        out["wmean"] = wm
    J = JOINT["v"]
    if t in J.index:
        cj = np.array([h for h in cand if h in J.index])
        out["joint"] = pd.Series(s02.angles(J.loc[cj].to_numpy(float), J.loc[t].to_numpy(float)), index=cj)
    return out


def das_rankings(V, target_vecs, cand):
    """Angles (degrees) of candidate hospitals to the target, per condition and their mean."""
    ang = {}
    for cond in CONDITIONS:
        tv = target_vecs.get(cond)
        if tv is None:
            continue
        c = np.array([h for h in cand if h in V[cond].index])
        if len(c) >= 2 * K:
            ang[cond] = pd.Series(s02.angles(V[cond].loc[c].to_numpy(float), tv), index=c)
    if ang:
        m = pd.concat(ang.values(), axis=1).mean(axis=1, skipna=True)
        if len(m) >= 2 * K:
            ang["mean"] = m
    return ang


def make_sets(ang, memb):
    sets = {}
    for name, a in list(ang.items()) + ([("membership", memb)] if memb is not None else []):
        sets[f"near_{name}"] = a.sort_values(kind="stable").index[:K].to_numpy()
        sets[f"far_{name}"] = a.sort_values(ascending=False, kind="stable").index[:K].to_numpy()
    if COMBINED[0] and "mean" in ang and memb is not None:
        c = (ang["mean"].rank() + memb.reindex(ang["mean"].index).rank()) / 2
        sets["near_combined"] = c.sort_values(kind="stable").index[:K].to_numpy()
        sets["far_combined"] = c.sort_values(ascending=False, kind="stable").index[:K].to_numpy()
    return sets


def size_jobs(sets, rand, rows_of, rng, spec="v1"):
    size = lambda hs: sum(len(rows_of[h]) for h in hs)
    if spec in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"):
        n = int(min(min(size(h) for h in sets.values()), min(size(r) for r in rand)))
    else:
        n = int(min(min(size(h) for h in sets.values()), np.percentile([size(r) for r in rand], 5)))
    out = []
    for name, hs in sets.items():
        rows = np.concatenate([rows_of[h] for h in hs])
        for r in range(N_REPEAT):
            out.append((name, rows if len(rows) <= n else rng.choice(rows, n, replace=False)))
    for i, hs in enumerate(rand):
        rows = np.concatenate([rows_of[h] for h in hs])
        out.append((f"rand_{i}", rows if len(rows) <= n else rng.choice(rows, n, replace=False)))
    return out, n


def calib(y, p):
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    except Exception:
        citl = slope = np.nan
    return citl, slope


XGB = [False]


def fit_one(X, y, rows, Xt, yt):
    if y[rows].min() == y[rows].max():
        return None, np.nan, np.nan
    if XGB[0]:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, tree_method="hist", n_jobs=1, random_state=0, verbosity=0)
        p = m.fit(X[rows], y[rows]).predict_proba(Xt)[:, 1].astype(np.float32)
        return p, *calib(yt, p)
    p = s19.fit_scores(X, y, rows, Xt, C_M1)
    return p, *calib(yt, p)


def presort(P):
    """Sort order and tie-group ids of each prediction vector (rows of P)."""
    order = np.argsort(P, axis=1, kind="stable")
    Ps = np.take_along_axis(P, order, axis=1)
    new = np.ones_like(Ps, dtype=bool)
    new[:, 1:] = Ps[:, 1:] != Ps[:, :-1]
    gid = np.cumsum(new, axis=1) - 1
    return order, gid


def weighted_auc(order, gid, y, w):
    """AUROC of every row for labels y and case weights w (ties count one half)."""
    m, n = order.shape
    ws, ys = w[order], y[order]
    off = (np.arange(m) * n)[:, None]
    flat = (gid + off).ravel()
    wn = np.bincount(flat, (ws * (1 - ys)).ravel(), minlength=m * n).reshape(m, n)
    wp = np.bincount(flat, (ws * ys).ravel(), minlength=m * n).reshape(m, n)
    below = np.cumsum(wn, axis=1) - wn
    W1, W0 = (w * y).sum(), (w * (1 - y)).sum()
    return (wp * (below + 0.5 * wn)).sum(axis=1) / (W1 * W0)


def score_target(arms, P, yt, n_boot, rng):
    """Point AUROC per arm (mean over repeats), random mean, and bootstrap draws of each arm's gain."""
    order, gid = presort(P)
    names = np.array(arms)
    is_rand = np.char.startswith(names.astype(str), "rand_")
    sel = [a for a in dict.fromkeys(names[~is_rand])]
    idx_of = {a: np.flatnonzero(names == a) for a in sel}
    def summarize(auc):
        r = auc[is_rand].mean()
        return {a: auc[idx_of[a]].mean() for a in sel}, r
    auc0 = weighted_auc(order, gid, yt, np.ones(len(yt)))
    point, rmean = summarize(auc0)
    pos, neg = np.flatnonzero(yt == 1), np.flatnonzero(yt == 0)
    draws = {a: np.empty(n_boot) for a in sel}
    for b in range(n_boot):
        w = np.zeros(len(yt))
        np.add.at(w, rng.choice(pos, len(pos)), 1)
        np.add.at(w, rng.choice(neg, len(neg)), 1)
        pb, rb = summarize(weighted_auc(order, gid, yt, w))
        for a in sel:
            draws[a][b] = pb[a] - rb
    return point, rmean, auc0[is_rand], draws, auc0


def pct_per_repeat(arms, auc0):
    """Share of random-set AUROCs below each trim of a selected set, averaged over the trims (percent).

    The paper reports the percentile of the reported trim-mean AUROC instead ({arm}_pctm in loop rows,
    percentile_mean in MIMIC-IV rows), so that it matches the plotted gain."""
    names = np.array(arms)
    rnd = auc0[np.char.startswith(names.astype(str), "rand_")]
    return {a: 100 * np.mean([(rnd < v).mean() for v in auc0[names == a]])
            for a in dict.fromkeys(names) if not str(a).startswith("rand_")}


def cmd_split(args):
    cfg, st, res, rowlevel = setup(args.config)
    out = res / "iecv_split.csv"
    if out.exists():
        print(f"{out} exists; not overwritten"); return
    e, X, XM, Y, rows_of, V, no_med = load_eicu(cfg, rowlevel)
    size = pd.Series({h: len(r) for h, r in rows_of.items()})
    elig = np.array(sorted(size[size >= MIN_STAYS].index))
    hosp = duckdb.execute(f"SELECT hospitalid, region FROM '{cfg['paths']['eicu']}/hospital.parquet'").df()
    hosp = hosp.set_index("hospitalid")
    hosp["region"] = hosp.region.replace({"nan": np.nan})
    t = s19.split_targets(elig, hosp, size, np.random.default_rng(int(st["seed"])))
    t["half"] = t.half.replace({"confirmation": "held_out"})
    t["no_medication"] = t.hospitalid.isin(no_med)
    t.to_csv(out, index=False)
    print(t.groupby("half").agg(hospitals=("hospitalid", "size"), stays=("stays", "sum"),
                                no_medication=("no_medication", "sum")).to_string())
    print(pd.crosstab([t.region, t.size_tertile], t.half).to_string())


def cmd_loop(args):
    cfg, st, res, rowlevel = setup(args.config)
    if args.half == "held_out":
        lock = res / f"iecv_locked_choice_{args.sample}.yaml"
        assert lock.exists(), "screening for this sample has not written its locked choice"
        assert yaml.safe_load(open(lock))["decision"] == "lock", "screening stopped; no held-out run"
    split = pd.read_csv(res / "iecv_split.csv")
    e, X, XM, Y, rows_of, V, no_med = load_eicu(cfg, rowlevel)
    half = split[split.half == args.half]
    if args.sample == "with_medication":
        half = half[~half.no_medication]
    members = half.hospitalid.to_numpy()
    if os.environ.get("EHRDOC_APACHE_OFFICIAL"):
        members = np.array([h for h in members if h in rows_of and len(rows_of[h]) >= 100])
    das_ok = set().union(*[set(V[c].index) for c in CONDITIONS])
    Xfit = {"v1": X, "v2": XM}.get(args.spec)
    if args.spec in ("v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"):
        f24 = pd.read_parquet(flag_dir(rowlevel) / "eicu_flags_24h.parquet")
        if args.spec == "v4":
            f24 = f24.merge(pd.read_parquet(rowlevel / "eicu_flags_extra_24h.parquet").drop(columns="hospitalid"),
                            on="patientunitstayid", how="left")
        g = e[["patientunitstayid"]].merge(f24, on="patientunitstayid", how="left")
        assert (g.patientunitstayid.to_numpy() == e.patientunitstayid.to_numpy()).all()
        Xfit = np.hstack([XM, g[FLAGS].fillna(False).astype(float).to_numpy()])
    if getattr(args, "harmonized", False):


        Xfit = XM
        XDIS[0] = np.hstack([XM, e[[f for f in FLAGS if f.endswith("_M")]].to_numpy(float), np.log1p(e.los_d.to_numpy(float))[:, None]])
    if args.spec == "v8":
        S = np.hstack([rcs(e[v].to_numpy(float)) for v in ("apachescore", "agenum")])
        Xfit, XDIS[0] = np.hstack([Xfit, S]), np.hstack([XDIS[0], S])
    if args.spec == "v9":
        so = e[["patientunitstayid"]].merge(pd.read_parquet(rowlevel / "eicu_sofa_lastday.parquet"), on="patientunitstayid", how="left")
        assert (so.patientunitstayid.to_numpy() == e.patientunitstayid.to_numpy()).all()
        XDIS[0] = np.hstack([XDIS[0], so[SOFA_COLS].fillna(0).to_numpy(float)])
    if args.limit:
        (res / "_test").mkdir(exist_ok=True)
    base = res / "_test" if args.limit else res
    if args.save_random:
        base = base / args.save_random
        base.mkdir(parents=True, exist_ok=True)
    TEMPORAL = getattr(args, "temporal", False)
    if TEMPORAL:


        yr = duckdb.execute(f"SELECT patientunitstayid, hospitaldischargeyear yr FROM '{cfg['paths']['eicu']}/patient.parquet' ORDER BY 1").df()
        if getattr(args, "split_mode", "year") == "random":
            yr["yr"] = np.random.default_rng(int(st["seed"]) + 2026 + 1000 * int(os.environ.get("EHRDOC_SPLIT_REP", "0"))).integers(0, 2, len(yr))
        y0, y1 = int(yr.yr.min()), int(yr.yr.max())
        year_of = e[["patientunitstayid"]].merge(yr, on="patientunitstayid", how="left").yr.to_numpy()
        fl = pd.read_parquet(flag_dir(rowlevel) / "eicu_flags.parquet").merge(yr, on="patientunitstayid", how="left")
        fl = fl[fl.yr == y0]
        VT = {}
        for cond in CONDITIONS:
            raw = s02.hospital_counts(fl, s02.lobe_codes(fl, cond, ("D", "H", "M")), 7)
            cnt = raw[raw.sum(axis=1) >= (DAS_MIN[0] or cfg["das"]["min_records_per_hospital"])]
            VT[cond] = cnt.div(cnt.sum(axis=1), axis=0)
        early_rows = {h: r[year_of[r] == y0] for h, r in rows_of.items()}
        late_rows = {h: r[year_of[r] == y1] for h, r in rows_of.items()}
        members = np.array([h for h in members if len(early_rows[h]) >= 50 and len(late_rows[h]) >= 50
                            and any(h in VT[c].index for c in CONDITIONS)])
        print(f"earlier year {y0}, later year {y1}: {len(members)} targets with at least 50 modeled stays in each year", flush=True)
    out = base / f"iecv_targets_{args.half}_{args.sample}.csv"
    rand_out = base / f"iecv_random_aucs_{args.half}_{args.sample}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["target", "task"])
    done_keys = set(zip(done.target, done.task))
    targets = members[: args.limit] if args.limit else members
    seed = int(st["seed"])
    hcode, scode = {"development": 1, "held_out": 2}[args.half], {"all": 1, "with_medication": 2}[args.sample]
    for t in targets:
        if all((t, task) in done_keys for task in TASKS):
            continue
        pool = members[members != t]
        if args.spec in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"):
            pool = np.array([h for h in pool if h in das_ok])
        rt = late_rows[t] if TEMPORAL else rows_of[t]
        rt_ref = early_rows[t] if TEMPORAL else rt
        VS = VT if TEMPORAL else V
        tvec = {c: VS[c].loc[t].to_numpy(float) for c in CONDITIONS if t in VS[c].index}
        ang = das_rankings(V, tvec, pool)
        if args.spec == "v5":
            ang.update(variant_rankings(V, t, pool))
        cs = Parallel(n_jobs=args.jobs)(delayed(s35.membership_c)(XM[rows_of[h]], XM[rt_ref], seed + int(h))
                                        for h in pool)
        memb = pd.Series(cs, index=pool)
        sets = make_sets(ang, memb)
        rows = []
        for ti, task in enumerate(TASKS):
            if (t, task) in done_keys:
                continue
            if task in ELIG:
                el = ELIG[task]
                rtk, rows_k, Xk = rt[el[rt]], {h: rows_of[h][el[rows_of[h]]] for h in pool}, XDIS[0]
            else:
                rtk, rows_k, Xk = rt, rows_of, Xfit
            y, yt = Y[task], Y[task][rtk]
            pool_k, sets_k = task_pool(task, pool, sets, ang, memb)
            row = dict(target=t, task=task, half=args.half, sample=args.sample, stays=len(rtk), events=int(yt.sum()),
                       pool=len(pool_k))
            if len(yt) == 0 or yt.min() == yt.max():
                rows.append({**row, "skipped": "no events or no non-events"}); continue
            rng = np.random.default_rng([seed, hcode, scode, ti, int(t)])
            rand = [rng.choice(pool_k, K, replace=False) for _ in range(N_RAND["eicu"])]
            jobs, n = size_jobs(sets_k, rand, rows_k, rng, args.spec)
            fits = Parallel(n_jobs=args.jobs, batch_size=8)(delayed(fit_one)(Xk, y, r, Xk[rtk], yt) for _, r in jobs)
            ok = [i for i, f in enumerate(fits) if f[0] is not None]
            arms = [jobs[i][0] for i in ok]
            P = np.vstack([fits[i][0] for i in ok])
            point, rmean, rauc, draws, auc0 = score_target(arms, P, yt, args.boot or N_BOOT["eicu"], rng)
            pct = pct_per_repeat(arms, auc0)
            if args.spec in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"):
                sa = base / f"iecv_set_aucs_{args.half}_{args.sample}.csv"
                pd.DataFrame(dict(target=t, task=task, set=arms, auroc=auc0)).to_csv(
                    sa, mode="a", header=not sa.exists(), index=False, float_format="%.6f")
            if args.save_random:
                pd.DataFrame(dict(target=t, task=task, set=[a for a in arms if a.startswith("rand_")], auroc=rauc)).to_csv(
                    rand_out, mode="a", header=not rand_out.exists(), index=False, float_format="%.6f")
            cal = pd.DataFrame([(jobs[i][0], fits[i][1], fits[i][2]) for i in ok], columns=["arm", "citl", "slope"])
            cal["grp"] = np.where(cal.arm.str.startswith("rand_"), "random", cal.arm)
            cal = cal.groupby("grp")[["citl", "slope"]].mean()
            row.update(n_train=n, random_auc=rmean, random_sd=rauc.std(), random_sets=len(rauc),
                       random_citl=cal.loc["random", "citl"], random_slope=cal.loc["random", "slope"])
            for a in point:
                row.update({f"{a}_auc": point[a], f"{a}_gain": point[a] - rmean, f"{a}_se": draws[a].std(ddof=1),
                            f"{a}_pct": pct[a],

                            f"{a}_pctm": 100 * (rauc < point[a]).mean(),
                            f"{a}_citl": cal.loc[a, "citl"], f"{a}_slope": cal.loc[a, "slope"],
                            f"{a}_hospitals": ";".join(map(str, sets_k[a]))})
            rows.append(row)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(out, index=False, float_format="%.6f")
        print(f"target {t} done ({done.target.nunique()}/{len(members)})", flush=True)


def reml(y, v):
    """Random-effects meta-analysis: REML tau2, Hartung-Knapp CI, prediction interval, I2."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    Q = (w * (y - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    for _ in range(500):
        w = 1 / (v + tau2)
        mu = (w * y).sum() / w.sum()
        new = max(0.0, ((w ** 2) * ((y - mu) ** 2 - v)).sum() / (w ** 2).sum() + 1 / w.sum())
        if abs(new - tau2) < 1e-12:
            tau2 = new; break
        tau2 = new
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    q = (w * (y - mu) ** 2).sum() / (k - 1)
    se = np.sqrt(max(q, 1e-300) / w.sum())
    tq = stats.t.ppf(0.975, k - 1)
    se_pi = np.sqrt(tau2 + 1 / w.sum())
    tp = stats.t.ppf(0.975, k - 2) if k > 2 else np.nan
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(k=k, pooled_gain=mu, lo=mu - tq * se, hi=mu + tq * se, pi_lo=mu - tp * se_pi, pi_hi=mu + tp * se_pi,
                tau2=tau2, I2=I2, mean_gain=y.mean(), share_positive=(y > 0).mean())


def pool_file(df, half="development"):
    arms = sorted({c[:-5] for c in df.columns if c.endswith("_gain")})
    out = []
    for task, s in df.groupby("task"):
        sets = [("main", s)] if half == "development" else [("main", s[s.events >= MIN_EVENTS_MAIN]), ("all_targets", s)]
        for analysis, d in sets + [("ge20_events", s[s.events >= MIN_EVENTS_SENS])]:
            for a in arms:
                ok = d[f"{a}_gain"].notna() & (d[f"{a}_se"] > 0)
                if ok.sum() < 3:
                    continue
                r = reml(d.loc[ok, f"{a}_gain"], d.loc[ok, f"{a}_se"] ** 2)
                out.append(dict(task=task, analysis=analysis, arm=a, citl=d.loc[ok, f"{a}_citl"].median(),
                                slope=d.loc[ok, f"{a}_slope"].median(),
                                random_citl=d.loc[ok, "random_citl"].median(),
                                random_slope=d.loc[ok, "random_slope"].median(), **r))
    return pd.DataFrame(out)


def cmd_pool(args):
    cfg, st, res, rowlevel = setup(args.config)
    split = pd.read_csv(res / "iecv_split.csv")
    allp = []
    for f in sorted(res.glob("iecv_targets_*.csv")):
        half, sample = f.stem.replace("iecv_targets_", "").rsplit("_", 1) if not f.stem.endswith("with_medication") \
            else (f.stem.replace("iecv_targets_", "").replace("_with_medication", ""), "with_medication")
        p = pool_file(pd.read_csv(f), half)
        p.insert(0, "sample", sample); p.insert(0, "half", half)
        allp.append(p)
        if half == "development":
            want = split[split.half == "development"]
            if sample == "with_medication":
                want = want[~want.no_medication]
            got = pd.read_csv(f).target.nunique()
            assert got == len(want), f"development {sample}: {got} of {len(want)} targets; screening needs all"
            lock = res / f"iecv_locked_choice_{sample}.yaml"
            cand = p[(p.analysis == "main") & p.arm.isin([f"near_{m}" for m in MEASURES])]
            best = cand.sort_values("pooled_gain", ascending=False).iloc[0]
            choice = dict(sample=sample, rule=st["step2_screening"]["rule"], candidates=int(len(cand)),
                          best_measure=best.arm.replace("near_", ""), best_task=best.task,
                          pooled_gain=float(best.pooled_gain), lo=float(best.lo), hi=float(best.hi),
                          decision="lock" if best.lo > 0 else "stop")
            if lock.exists():
                prev = yaml.safe_load(open(lock))
                assert (prev["best_measure"], prev["best_task"], prev["decision"]) == \
                    (choice["best_measure"], choice["best_task"], choice["decision"]), "locked choice changed"
            else:
                yaml.safe_dump(choice, open(lock, "w"), sort_keys=False)
            print(f"screening ({sample}): {choice}")
    pooled = pd.concat(allp, ignore_index=True)
    pooled.to_csv(res / "iecv_pooled.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    print(pooled[pooled.analysis == "main"].drop(columns=["tau2", "citl", "slope", "random_citl", "random_slope"])
          .round(4).to_string(index=False))


def cmd_mimic(args):
    cfg, st, res, rowlevel = setup(args.config)
    assert args.spec not in ("v3", "v4", "v5", "v7"), "v3 and v4 are eICU only: MIMIC-IV diagnosis codes and past history have no times"
    lock = yaml.safe_load(open(res / "iecv_locked_choice_all.yaml"))
    assert lock["decision"] == "lock", "screening stopped; no external test"
    measure = args.measure or lock["best_measure"]
    tag = f"_{args.measure}" if args.measure else ""
    if args.sample == "with_medication":
        tag += "_withmed"
        lock = yaml.safe_load(open(res / "iecv_locked_choice_with_medication.yaml"))
        measure = args.measure or lock["best_measure"]
    split = pd.read_csv(res / "iecv_split.csv")
    e, X, XM, Y, rows_of, V, no_med = load_eicu(cfg, rowlevel)
    M = cfg["paths"]["mimic"]
    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet")
    m = m.merge(pd.read_parquet(rowlevel / "mimic_m2_features.parquet")[["stay_id", "charlson"]], on="stay_id", how="left")
    mf = pd.read_parquet(flag_dir(rowlevel) / "mimic_flags.parquet")
    mh = pd.read_parquet(rowlevel / "mimic_history_flags.parquet")
    dm = [f for f in FLAGS if not f.endswith("_H")]
    hf = [f for f in FLAGS if f.endswith("_H")]
    m = m.merge(mf[["stay_id"] + dm], on="stay_id", how="left").merge(mh[["stay_id"] + hf], on="stay_id", how="left")
    m[FLAGS] = m[FLAGS].fillna(0).astype(float)
    TEMPORAL = getattr(args, "temporal", False)
    if TEMPORAL:

        early_ids = set(mf.stay_id[mf.anchor_year_group.isin(["2008 - 2010", "2011 - 2013"])])
        if getattr(args, "split_mode", "year") == "random":
            ids = np.sort(mf.stay_id.to_numpy()); early_ids = set(ids[np.random.default_rng(int(st["seed"]) + 2026 + 1000 * int(os.environ.get("EHRDOC_SPLIT_REP", "0"))).integers(0, 2, len(ids)) == 0])
        XMm_ref = m.loc[m.stay_id.isin(early_ids), design_columns()].to_numpy(float)
        m = m[~m.stay_id.isin(early_ids)].reset_index(drop=True)
        print(f"MIMIC-IV: documentation from {len(early_ids):,} earlier stays, evaluation on {len(m):,} later stays", flush=True)
    Xm, XMm = m[design_columns() + ["charlson"] + FLAGS].to_numpy(float), m[design_columns()].to_numpy(float)

    flagsM = mf[["stay_id"] + dm].merge(mh[["stay_id"] + hf], on="stay_id")
    if TEMPORAL:
        flagsM = flagsM[flagsM.stay_id.isin(early_ids)]
    flagsM["hospitalid"] = -1
    tvec = {}
    for cond in CONDITIONS:
        cnt = s02.hospital_counts(flagsM, s02.lobe_codes(flagsM, cond, ("D", "H", "M")), 7)
        tvec[cond] = (cnt.iloc[0] / cnt.iloc[0].sum()).to_numpy(float)
    pool = split.hospitalid.to_numpy()
    if os.environ.get("EHRDOC_APACHE_OFFICIAL"):
        pool = np.array([h for h in pool if h in rows_of and len(rows_of[h]) >= 100])
    if args.sample == "with_medication":
        pool = split.loc[~split.no_medication, "hospitalid"].to_numpy()
    if args.spec in ("v2", "v6", "v10"):
        das_ok = set().union(*[set(V[c].index) for c in CONDITIONS])
        pool = np.array([h for h in pool if h in das_ok])
        X, Xm = XM, XMm
    ELIG_M, XD, XDm = {}, None, None
    if args.spec in ("v6", "v10"):
        q = duckdb.execute(f"""SELECT i.stay_id, i.hadm_id, i.intime, i.outtime, i.los, a.deathtime, a.hospital_expire_flag,
                                      a.discharge_location,
                                      EXISTS (SELECT 1 FROM '{M}/icu/icustays.parquet' j
                                              WHERE j.hadm_id = i.hadm_id AND j.intime > i.outtime AND EXISTS (
                                                  SELECT 1 FROM '{M}/hosp/transfers.parquet' t   -- a stay outside the ICU, other than the recovery room or discharge lounge, between the two ICU stays
                                                  WHERE t.hadm_id = i.hadm_id AND t.intime >= i.outtime - INTERVAL 1 MINUTE AND t.intime < j.intime
                                                    AND t.eventtype <> 'discharge' AND t.careunit IS NOT NULL AND t.careunit NOT IN ('PACU', 'Discharge Lounge'))) AS readmit
                               FROM '{M}/icu/icustays.parquet' i JOIN '{M}/hosp/admissions.parquet' a USING (hadm_id)""").df()
        q = m[["stay_id"]].merge(q, on="stay_id", how="left")
        assert q.los.notna().all()
        ELIG_M["icu_readmit"] = (q.deathtime.isna() | (q.deathtime > q.outtime)).to_numpy()


        mfac = q.discharge_location.isin(["SKILLED NURSING FACILITY", "REHAB", "ASSISTED LIVING", "CHRONIC/LONG TERM ACUTE CARE"])
        ELIG_M["disch_facility"] = ((q.hospital_expire_flag == 0)
                                    & (mfac | q.discharge_location.isin(["HOME", "HOME HEALTH CARE"]))).to_numpy()
        YM = {"icu_readmit": q.readmit.to_numpy(int), "disch_facility": mfac.to_numpy(int)}
        mf_ = [f for f in FLAGS if f.endswith("_M")]
        XD = np.hstack([XM, e[mf_].to_numpy(float), np.log1p(e.los_d.to_numpy(float))[:, None]])
        XDm = np.hstack([XMm, m[mf_].to_numpy(float), np.log1p(q.los.to_numpy(float))[:, None]])
    ang = das_rankings(V, tvec, pool)
    seed = int(st["seed"])
    cs = Parallel(n_jobs=args.jobs, batch_size=4)(delayed(s35.membership_c)(XM[rows_of[h]], XMm_ref if TEMPORAL else XMm, seed + 7 + int(h))
                                                  for h in pool)
    memb = pd.Series(cs, index=pool)
    if HOOK[0] is None:
      pd.DataFrame({"hospitalid": pool, "membership_c": cs,
                  **{f"das_{k}": ang[k].reindex(pool).values for k in ang}}).round(5) \
        .to_csv(res / f"iecv_mimic_relatedness{tag}.csv", index=False)
    sets = {k: v for k, v in make_sets({measure: ang[measure]}, memb).items()}
    prev = pd.read_csv(res / f"iecv_mimic{tag}.csv") if (res / f"iecv_mimic{tag}.csv").exists() else pd.DataFrame(columns=["task"])
    run = args.tasks.split(",") if args.tasks else TASKS
    out = prev[~prev.task.isin(run)].to_dict("records")
    for ti, task in enumerate(TASKS):
        if task not in run:
            continue
        rng = np.random.default_rng([seed, 9, ti])
        if task in ELIG_M:
            el, elm = ELIG[task], ELIG_M[task]
            y, ym = Y[task], YM[task][elm]
            rows_k, Xk, Xmk = {h: rows_of[h][el[rows_of[h]]] for h in pool}, XD, XDm[elm]
        else:
            y, ym = Y[task], s24.mimic_outcome(MIMIC_TASK[task], m, M)
            rows_k, Xk, Xmk = rows_of, X, Xm
        pool_k, sets_k = task_pool(task, pool, sets, {measure: ang[measure]}, memb)
        rand = [rng.choice(pool_k, K, replace=False) for _ in range(N_RAND["mimic"])]
        jobs, n = size_jobs(sets_k, rand, rows_k, rng, args.spec)
        fits = Parallel(n_jobs=args.jobs, batch_size=8)(delayed(fit_one)(Xk, y, r, Xmk, ym) for _, r in jobs)
        ok = [i for i, f in enumerate(fits) if f[0] is not None]
        arms = [jobs[i][0] for i in ok]
        P = np.vstack([fits[i][0] for i in ok])
        if HOOK[0] is not None:
            HOOK[0](task, ti, arms, P, ym, seed)
            continue
        point, rmean, rauc, draws, auc0 = score_target(arms, P, ym, N_BOOT["mimic"], rng)
        pct = pct_per_repeat(arms, auc0)
        if args.spec in ("v2", "v6", "v10"):
            sa = res / f"iecv_set_aucs_mimic{tag}.csv"
            pd.DataFrame(dict(task=task, set=arms, auroc=auc0)).to_csv(sa, mode="a", header=not sa.exists(),
                                                                      index=False, float_format="%.6f")
        cal = pd.DataFrame([(jobs[i][0], fits[i][1], fits[i][2]) for i in ok], columns=["arm", "citl", "slope"])
        cal["grp"] = np.where(cal.arm.str.startswith("rand_"), "random", cal.arm)
        cal = cal.groupby("grp")[["citl", "slope"]].mean()
        for a in point:
            out.append(dict(task=task, arm=a, auroc=point[a], random_mean=rmean, random_sd=rauc.std(),
                            gain=point[a] - rmean, lo=np.percentile(draws[a], 2.5), hi=np.percentile(draws[a], 97.5),
                            percentile=pct[a] if args.spec in ("v2", "v6", "v10") else 100 * (rauc < point[a]).mean(),
                            percentile_mean=100 * (rauc < point[a]).mean(),
                            citl=cal.loc[a, "citl"],
                            slope=cal.loc[a, "slope"], random_citl=cal.loc["random", "citl"],
                            random_slope=cal.loc["random", "slope"], n_train=n, stays=len(ym), events=int(ym.sum()),
                            hospitals=";".join(map(str, sets_k[a]))))
        print(pd.DataFrame(out)[lambda x: x.task == task].drop(columns="hospitals").round(4).to_string(index=False),
              flush=True)
    if HOOK[0] is None:
        pd.DataFrame(out).to_csv(res / f"iecv_mimic{tag}.csv", index=False, float_format="%.6f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "loop", "pool", "mimic"])
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--half", choices=["development", "held_out"], default="development")
    ap.add_argument("--sample", choices=["all", "with_medication"], default="all")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-random", default="", help="loop: subfolder of results/ for a rerun that saves each random set's AUROC")
    ap.add_argument("--boot", type=int, default=0, help="loop: bootstrap resamples (default N_BOOT); fewer only for --save-random reruns")
    ap.add_argument("--measure", default="", help="mimic: DAS measure to test instead of the locked one (own output files)")
    ap.add_argument("--tasks", default="", help="mimic: comma-separated tasks to (re)compute; other rows kept")
    ap.add_argument("--spec", choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"], default="v1", help="v2: corrected; v3: v2 plus first-24-h flags; v4: v3 with six conditions")
    ap.add_argument("--out", default="", help="results subfolder for this run (required with --spec v2)")
    ap.add_argument("--k", type=int, default=10, help="hospitals per training set (5 or 20 for the size sensitivity analysis, each in its own subfolder)")
    ap.add_argument("--temporal", action="store_true", help="loop, mimic: the target's DAS and membership reference from its earlier stays, evaluation on its later stays")
    ap.add_argument("--split-mode", choices=["year", "random"], default="year", help="with --temporal: earlier and later year (default), or a random half of the target's stays")
    ap.add_argument("--harmonized", action="store_true", help="loop: model inputs as in the MIMIC-IV test (covariates only for 24-h outcomes; covariates, medication flags and ICU length of stay for discharge-time outcomes)")
    ap.add_argument("--no-facility", action="store_true", help="v6 without disch_facility (dropped from the paper); last task, so other seeds unchanged")
    args = ap.parse_args()
    assert args.spec == "v1" or args.out, "--spec v2 and v3 write to their own --out folder"
    assert args.k == 10 or "/k" in args.out, "another training-set size writes to its own subfolder, e.g. --out v7_2026-11-16/k5"
    setup.out = args.out
    global CONDITIONS, MEASURES, FLAGS, TASKS, K
    K = args.k
    XGB[0] = args.spec == "v7"
    if args.spec in ("v6", "v7", "v8", "v9"):
        TASKS = TASKS + ["icu_readmit"] + ([] if args.no_facility else ["disch_facility"])
        DAS_MIN[0] = 30
    if args.spec == "v10":
        TASKS = TASKS + ["icu_readmit"]
        DAS_MIN[0] = 30
        COMBINED[0] = True
    if args.spec == "v4":
        CONDITIONS = CONDITIONS + ["copd", "hypothyroidism"]
        MEASURES = CONDITIONS + ["mean"]
        FLAGS = [f"{c}_{s}" for c in CONDITIONS for s in ("D", "H", "M")]
    if args.spec == "v5":
        MEASURES = MEASURES + ["wmean", "joint"]
    {"split": cmd_split, "loop": cmd_loop, "pool": cmd_pool, "mimic": cmd_mimic}[args.step](args)


if __name__ == "__main__":
    main()
