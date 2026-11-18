#!/usr/bin/env python

"""Helpers for the held-out and MIMIC-IV settings, imported by the size-matched analysis."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_s = importlib.util.spec_from_file_location("i38", HERE / "38_iecv.py")
i38 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(i38)
from m1_model import design_columns

i38.TASKS = i38.TASKS + ["icu_readmit"]
i38.DAS_MIN[0] = 30
i38.setup.out = os.environ.get("EHRDOC_OUT", "v7_2026-11-16")
TASKS = i38.TASKS
MIN_EV, MENUS, NETS, DEV_CUTS, LOSS = 20, (3, 5, 10), (5, 10), (15, 25, 45), 0.05
N_MENU, B_DEPLOY, B_NET = 200, 1000, 200
N_NET = {"development": 100, "held_out": 200, "mimic": 200}
HCODE = {"development": 1, "held_out": 2, "mimic": 3}


def load(config):
    cfg, st, res, rowlevel = i38.setup(config)
    e, X, XM, Y, rows_of, V, no_med = i38.load_eicu(cfg, rowlevel)
    f24 = pd.read_parquet(rowlevel / "eicu_flags_24h.parquet")
    g = e[["patientunitstayid"]].merge(f24, on="patientunitstayid", how="left")
    assert (g.patientunitstayid.to_numpy() == e.patientunitstayid.to_numpy()).all()
    Xfit = np.hstack([XM, g[i38.FLAGS].fillna(False).astype(float).to_numpy()])
    return dict(cfg=cfg, seed=int(st["seed"]), res=res, rowlevel=rowlevel, e=e, XM=XM, Xfit=Xfit, Y=Y,
                rows_of=rows_of, V=V, no_med=no_med)


def eicu_task(d, task):
    """Inputs, labels and per-hospital rows of one outcome, as the paper's eICU loop (v6)."""
    if task in i38.ELIG:
        el = i38.ELIG[task]
        return i38.XDIS[0], d["Y"][task], {h: r[el[r]] for h, r in d["rows_of"].items()}
    return d["Xfit"], d["Y"][task], d["rows_of"]


def load_mimic(d):
    """MIMIC-IV inputs and labels per outcome, and its documentation vectors, as cmd_mimic --spec v6."""
    cfg, rowlevel, e, XM, Y, rows_of = d["cfg"], d["rowlevel"], d["e"], d["XM"], d["Y"], d["rows_of"]
    M, FL = cfg["paths"]["mimic"], i38.FLAGS
    m = pd.read_parquet(rowlevel / "mimic_m1_features.parquet")
    m = m.merge(pd.read_parquet(rowlevel / "mimic_m2_features.parquet")[["stay_id", "charlson"]], on="stay_id", how="left")
    mf = pd.read_parquet(rowlevel / "mimic_flags.parquet")
    mh = pd.read_parquet(rowlevel / "mimic_history_flags.parquet")
    dm, hf = [f for f in FL if not f.endswith("_H")], [f for f in FL if f.endswith("_H")]
    m = m.merge(mf[["stay_id"] + dm], on="stay_id", how="left").merge(mh[["stay_id"] + hf], on="stay_id", how="left")
    m[FL] = m[FL].fillna(0).astype(float)
    XMm = m[design_columns()].to_numpy(float)
    flagsM = mf[["stay_id"] + dm].merge(mh[["stay_id"] + hf], on="stay_id")
    flagsM["hospitalid"] = -1
    tvec = {}
    for cond in i38.CONDITIONS:
        cnt = i38.s02.hospital_counts(flagsM, i38.s02.lobe_codes(flagsM, cond, ("D", "H", "M")), 7)
        tvec[cond] = (cnt.iloc[0] / cnt.iloc[0].sum()).to_numpy(float)
    q = duckdb.execute(f"""SELECT i.stay_id, i.outtime, i.los, a.deathtime,
                                  EXISTS (SELECT 1 FROM '{M}/icu/icustays.parquet' j
                                          WHERE j.hadm_id = i.hadm_id AND j.intime > i.outtime AND EXISTS (
                                                  SELECT 1 FROM '{M}/hosp/transfers.parquet' t   -- a stay outside the ICU, other than the recovery room or discharge lounge, between the two ICU stays
                                                  WHERE t.hadm_id = i.hadm_id AND t.intime >= i.outtime - INTERVAL 1 MINUTE AND t.intime < j.intime
                                                    AND t.eventtype <> 'discharge' AND t.careunit IS NOT NULL AND t.careunit NOT IN ('PACU', 'Discharge Lounge'))) AS readmit
                           FROM '{M}/icu/icustays.parquet' i JOIN '{M}/hosp/admissions.parquet' a USING (hadm_id)""").df()
    q = m[["stay_id"]].merge(q, on="stay_id", how="left")
    assert q.los.notna().all()
    elm = (q.deathtime.isna() | (q.deathtime > q.outtime)).to_numpy()
    mfl = [f for f in FL if f.endswith("_M")]
    XD = np.hstack([XM, e[mfl].to_numpy(float), np.log1p(e.los_d.to_numpy(float))[:, None]])
    XDm = np.hstack([XMm, m[mfl].to_numpy(float), np.log1p(q.los.to_numpy(float))[:, None]])
    out = {}
    for task in TASKS:
        if task == "icu_readmit":
            el = i38.ELIG[task]
            out[task] = (XD, Y[task], {h: r[el[r]] for h, r in rows_of.items()}, XDm[elm], q.readmit.to_numpy(int)[elm])
        else:
            out[task] = (XM, Y[task], rows_of, XMm, i38.s24.mimic_outcome(i38.MIMIC_TASK[task], m, M))
    return out, tvec


def das_matrix(V, hosp, extra=None):
    """Mean DAS (degrees) between hospitals over the conditions where both have a vector; extra = outside target (-1)."""
    idx = list(hosp) + ([-1] if extra else [])
    S, N = np.zeros((len(idx), len(idx))), np.zeros((len(idx), len(idx)))
    for c in i38.CONDITIONS:
        vec = {h: V[c].loc[h].to_numpy(float) for h in hosp if h in V[c].index}
        if extra and c in extra:
            vec[-1] = extra[c]
        ks = [i for i, h in enumerate(idx) if h in vec]
        if len(ks) < 2:
            continue
        S[np.ix_(ks, ks)] += i38.s02.pairwise_angles(np.vstack([vec[idx[i]] for i in ks]))
        N[np.ix_(ks, ks)] += 1
    D = np.where(N > 0, S / np.maximum(N, 1), np.nan)
    return pd.DataFrame(D, index=idx, columns=idx)


def auc_boot(P, y, B, rng, chunk=150):
    """AUROC of every row of P: row 0 unweighted, rows 1..B stratified patient-bootstrap resamples shared by all rows."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    W = np.zeros((B, len(y)), np.uint16)
    for b in range(B):
        np.add.at(W[b], rng.choice(pos, len(pos)), 1)
        np.add.at(W[b], rng.choice(neg, len(neg)), 1)
    out = np.empty((B + 1, len(P)))
    for i in range(0, len(P), chunk):
        o, g = i38.presort(P[i:i + chunk])
        out[0, i:i + chunk] = i38.weighted_auc(o, g, y, np.ones(len(y)))
        for b in range(B):
            out[b + 1, i:i + chunk] = i38.weighted_auc(o, g, y, W[b].astype(float))
    return out


def fit_pred(X, y, rows, Xt):
    if y[rows].min() == y[rows].max():
        return None
    return i38.s19.fit_scores(X, y, rows, Xt, i38.C_M1)


def own_cv(X, y, Xt, yt, seed):
    """Target's own model, 5-fold cross-validated AUROC; NaN below 20 events or non-events."""
    if yt.sum() < MIN_EV or (len(yt) - yt.sum()) < MIN_EV:
        return np.nan
    p = np.empty(len(yt), np.float32)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xt, yt):
        p[te] = i38.s19.fit_scores(Xt, yt, tr, Xt[te], i38.C_M1)
    return float(roc_auc_score(yt, p))


def menus_eval(A, das, memb, size, rng):
    n, out = A.shape[1], []
    for m in MENUS:
        if n < m:
            continue
        Mn = np.array([rng.choice(n, m, replace=False) for _ in range(N_MENU)])
        mean, best = A[:, Mn].mean(axis=2), A[0, Mn].max(axis=1)
        r = dict(menu=m, models=n, menu_auc=mean[0].mean(), oracle_gain=(best - mean[0]).mean())
        for arm, score in [("das", das), ("membership", memb), ("largest", -size)]:
            if score is None or not np.isfinite(score).all():
                continue
            pick = Mn[np.arange(N_MENU), np.argmin(score[Mn], axis=1)]
            g = (A[:, pick] - mean).mean(axis=1)
            r.update({f"{arm}_gain": g[0], f"{arm}_se": g[1:].std(ddof=1), f"{arm}_auc": A[0, pick].mean(),
                      f"{arm}_best": (A[0, pick] >= best - 1e-12).mean(), f"{arm}_above": (A[0, pick] > mean[0]).mean(),
                      f"{arm}_regret": (best - A[0, pick]).mean()})
        out.append(r)
    return out


def membership_table(d, half, members, targets, out_dir, jobs):
    f = out_dir / f"membership_{half}.csv"
    if f.exists():
        return pd.read_csv(f)
    XM, ro, seed = d["XM"], d["rows_of"], d["seed"]
    pairs = [(t, h) for t in targets for h in members if h != t]
    cs = Parallel(n_jobs=jobs, batch_size=8)(delayed(i38.s35.membership_c)(XM[ro[h]], XM[ro[t]], seed + int(h))
                                             for t, h in pairs)
    tab = pd.DataFrame(pairs, columns=["target", "source"]).assign(membership_c=cs)
    tab.to_csv(f, index=False, float_format="%.6f")
    return tab


def setting(d, half, limit):
    """Members, targets, DAS matrix, per-outcome data and membership lookups for a half or MIMIC-IV."""
    split = pd.read_csv(d["res"] / "iecv_split.csv")
    das_ok = set().union(*[set(d["V"][c].index) for c in i38.CONDITIONS])
    if half == "mimic":
        members = np.array([h for h in split.hospitalid if h in das_ok])
        mt, tvec = load_mimic(d)
        D = das_matrix(d["V"], members, extra=tvec)
        rel = pd.read_csv(d["res"] / "iecv_mimic_relatedness_mean.csv").set_index("hospitalid").membership_c
        return members, np.array([-1]), D, mt, lambda t, hs: rel.reindex(hs).to_numpy(float)
    members = np.array([h for h in split.loc[split.half == half, "hospitalid"] if h in das_ok])
    targets = members[:limit] if limit else members
    D = das_matrix(d["V"], members)
    mt = {}
    for task in TASKS:
        X, y, rk = eicu_task(d, task)
        mt[task] = (X, y, rk, None, None)
    return members, targets, D, mt, None


def cmd_deploy(args):
    d = load(args.config)
    out_dir = d["res"] / "deploy_network" / ("_test" if args.limit else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    members, targets, D, mt, mlook = setting(d, args.half, args.limit)
    if mlook is None:
        mem = membership_table(d, args.half, members, targets, out_dir, args.jobs).set_index(["target", "source"]).membership_c
        mlook = lambda t, hs: mem.reindex(list(zip([t] * len(hs), hs))).to_numpy(float)
    rows, pairs, seed = [], [], d["seed"]
    for ti, task in enumerate(TASKS):
        X, y, rk, Xm, ym = mt[task]
        src = [h for h in members if MIN_EV <= y[rk[h]].sum() <= len(rk[h]) - MIN_EV]
        if args.half == "mimic":
            Xt_all, yt_of, sl = Xm, {-1: ym}, {-1: slice(0, len(ym))}
            owns = {-1: own_cv(None, None, Xm, ym, seed + 1)}
        else:
            allr = np.concatenate([rk[t] for t in targets])
            Xt_all, cut = X[allr], np.cumsum([0] + [len(rk[t]) for t in targets])
            sl = {t: slice(cut[i], cut[i + 1]) for i, t in enumerate(targets)}
            yt_of = {t: y[rk[t]] for t in targets}
            ov = Parallel(n_jobs=args.jobs)(delayed(own_cv)(None, None, X[rk[t]], y[rk[t]], seed + int(t)) for t in targets)
            owns = dict(zip(targets, ov))
        preds = Parallel(n_jobs=args.jobs, batch_size=4)(delayed(fit_pred)(X, y, rk[h], Xt_all) for h in src)
        preds = {h: p for h, p in zip(src, preds) if p is not None}
        for t in targets:
            yt = yt_of[t]
            if len(yt) == 0 or yt.min() == yt.max():
                continue
            cand = [h for h in preds if h != t and np.isfinite(D.loc[t, h])]
            P = np.vstack([preds[h][sl[t]] for h in cand])
            rng = np.random.default_rng([seed, 53, HCODE[args.half], ti, int(t) + 1])
            A = auc_boot(P, yt, B_DEPLOY, rng)
            das, memb = D.loc[t, cand].to_numpy(float), mlook(t, cand)
            size = np.array([len(rk[h]) for h in cand], float)
            base = dict(target=t, task=task, stays=len(yt), events=int(yt.sum()), own_cv=owns[t])
            rows += [{**base, **r} for r in menus_eval(A, das, memb, size, rng)]
            pairs.append(pd.DataFrame(dict(target=t, task=task, source=cand, das=das, membership_c=memb, source_stays=size,
                                           auroc=A[0], outside_mean=A[0].mean(), own_cv=owns[t], events=int(yt.sum()))))
        print(f"deploy {args.half} {task}: {len(preds)} models, {len(targets)} targets", flush=True)
    pd.DataFrame(rows).to_csv(out_dir / f"deploy_targets_{args.half}.csv", index=False, float_format="%.6f")
    pd.concat(pairs).to_csv(out_dir / f"deploy_pairs_{args.half}.csv", index=False, float_format="%.6f")


def cmd_network(args):
    d = load(args.config)
    out_dir = d["res"] / "deploy_network" / ("_test" if args.limit else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.half == "development":
        cuts = DEV_CUTS
    else:
        lock = d["res"] / "deploy_network" / "network_cutoff.yaml"
        assert lock.exists(), "development network has not written its locked cutoff"
        cuts = (yaml.safe_load(open(lock))["cutoff"],)
    members, targets, D, mt, _ = setting(d, args.half, args.limit)
    out = out_dir / f"network_targets_{args.half}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["target", "task"])
    keys, seed, nomed = set(zip(done.target, done.task)), d["seed"], d["no_med"]
    for t in targets:
        rows = []
        for ti, task in enumerate(TASKS):
            if (t, task) in keys:
                continue
            X, y, rk, Xm, ym = mt[task]
            Xt, yt = (Xm, ym) if args.half == "mimic" else (X[rk[t]], y[rk[t]])
            if len(yt) == 0 or yt.min() == yt.max():
                continue
            cand = np.array([h for h in members if h != t and np.isfinite(D.loc[t, h])])
            dist = D.loc[t, cand].to_numpy(float)
            rng = np.random.default_rng([seed, 54, HCODE[args.half], ti, int(t) + 1])
            plan, sets = [], {}
            key = lambda ix: sets.setdefault(tuple(sorted(cand[ix])), len(sets))
            for M in NETS:
                for _ in range(N_NET[args.half]):
                    net = rng.choice(len(cand), M, replace=False)
                    has_nomed = any(cand[i] in nomed for i in net)
                    for T in cuts:
                        kept = net[dist[net] < T]
                        fb = len(kept) == 0
                        if fb:
                            kept = net[[np.argmin(dist[net])]]
                        rnd = rng.choice(net, len(kept), replace=False)
                        plan.append((M, T, key(net), key(kept), key(rnd), len(kept), fb, len(kept) == M, has_nomed))
            sk = list(sets)
            fits = Parallel(n_jobs=args.jobs, batch_size=4)(
                delayed(fit_pred)(X, y, np.concatenate([rk[h] for h in s]), Xt) for s in sk)
            okf = np.array([f is not None for f in fits])
            P = np.vstack([f if f is not None else np.zeros(len(yt), np.float32) for f in fits])
            A = auc_boot(P, yt, B_NET, rng)
            pl = pd.DataFrame(plan, columns=["net", "cutoff", "all", "kept", "rand", "n_kept", "fallback", "same", "nomed"])
            pl = pl[okf[pl["all"]] & okf[pl["kept"]] & okf[pl["rand"]]]
            for (M, T), g in pl.groupby(["net", "cutoff"]):
                r = dict(target=t, task=task, stays=len(yt), events=int(yt.sum()), net=M, cutoff=T, draws=len(g),
                         fits=len(sk), auc_all=A[0, g["all"]].mean(), auc_kept=A[0, g.kept].mean(),
                         n_kept=g.n_kept.mean(), fallback=g.fallback.mean(), same=g.same.mean(), nomed_share=g.nomed.mean())
                for nm, sub in [("", g), ("_nomed", g[g.nomed]), ("_clean", g[~g.nomed])]:
                    if len(sub) == 0:
                        continue
                    for cmp in ("all", "rand"):
                        gg = (A[:, sub.kept.to_numpy()] - A[:, sub[cmp].to_numpy()]).mean(axis=1)
                        r[f"vs_{cmp}{nm}_gain"], r[f"vs_{cmp}{nm}_se"] = gg[0], gg[1:].std(ddof=1)
                    r[f"draws{nm}"] = len(sub)
                rows.append(r)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(out, index=False, float_format="%.6f")
        print(f"network {args.half} target {t} done ({done.target.nunique()}/{len(targets)})", flush=True)


def pool_cols(df, by, prefix_cols):
    out = []
    for k, g in df.groupby(by):
        for c in prefix_cols:
            if f"{c}_gain" not in g:
                continue
            ok = g[f"{c}_gain"].notna() & (g[f"{c}_se"] > 0)
            if ok.sum() < 3:
                continue
            r = i38.reml(g.loc[ok, f"{c}_gain"], g.loc[ok, f"{c}_se"] ** 2)
            out.append({**dict(zip(by, k if isinstance(k, tuple) else (k,))), "arm": c, **r})
    return pd.DataFrame(out)


def gonogo(p, T):
    flag, bad = p.das >= T, p.loss > LOSS
    return dict(sens=(flag & bad).sum() / max(bad.sum(), 1), spec=(~flag & ~bad).sum() / max((~bad).sum(), 1),
                ppv=(flag & bad).sum() / max(flag.sum(), 1), npv=(~flag & ~bad).sum() / max((~flag).sum(), 1),
                flagged=flag.mean(), bad=bad.mean())


def cmd_pool(args):
    cfg, st, res, _ = i38.setup(args.config)
    od = res / "deploy_network"
    parts = []
    for half in ("development", "held_out", "mimic"):
        f = od / f"deploy_targets_{half}.csv"
        if f.exists():
            df = pd.read_csv(f)
            if half == "mimic":
                for _, r in df.iterrows():
                    for a in ("das", "membership", "largest"):
                        parts.append(dict(analysis="deploy", half=half, task=r.task, menu=r.menu, arm=a, k=1,
                                          pooled_gain=r[f"{a}_gain"], lo=r[f"{a}_gain"] - 1.96 * r[f"{a}_se"],
                                          hi=r[f"{a}_gain"] + 1.96 * r[f"{a}_se"], best=r[f"{a}_best"],
                                          regret=r[f"{a}_regret"], oracle_gain=r.oracle_gain))
            else:
                p = pool_cols(df, ["task", "menu"], ["das", "membership", "largest"])
                ext = df.groupby(["task", "menu"]).agg(**{f"{a}_{s}": (f"{a}_{s}", "mean") for a in ("das", "membership", "largest")
                                                          for s in ("best", "regret")}, oracle_gain=("oracle_gain", "mean")).reset_index()
                p = p.merge(ext, on=["task", "menu"])
                p["best"] = p.apply(lambda r: r[f"{r.arm}_best"], axis=1)
                p["regret"] = p.apply(lambda r: r[f"{r.arm}_regret"], axis=1)
                parts.append(p.assign(analysis="deploy", half=half)[["analysis", "half", "task", "menu", "arm", "k", "pooled_gain",
                                                                     "lo", "hi", "pi_lo", "pi_hi", "I2", "share_positive",
                                                                     "best", "regret", "oracle_gain"]])
        f = od / f"network_targets_{half}.csv"
        if f.exists():
            df = pd.read_csv(f)
            cols = [f"vs_{c}{n}" for c in ("all", "rand") for n in ("", "_nomed", "_clean")]
            if half == "mimic":
                for _, r in df.iterrows():
                    for c in cols:
                        if pd.notna(r.get(f"{c}_gain")):
                            parts.append(dict(analysis="network", half=half, task=r.task, net=r.net, cutoff=r.cutoff, arm=c, k=1,
                                              pooled_gain=r[f"{c}_gain"], lo=r[f"{c}_gain"] - 1.96 * r[f"{c}_se"],
                                              hi=r[f"{c}_gain"] + 1.96 * r[f"{c}_se"], n_kept=r.n_kept, same=r.same))
            else:
                p = pool_cols(df, ["task", "net", "cutoff"], cols)
                ext = df.groupby(["task", "net", "cutoff"])[["n_kept", "same", "fallback", "nomed_share"]].mean().reset_index()
                parts.append(p.merge(ext, on=["task", "net", "cutoff"]).assign(analysis="network", half=half))
            if half == "development":
                p = parts[-1]
                m = p[p.arm == "vs_all"].groupby("cutoff").pooled_gain.mean()
                choice = dict(cutoff=int(m.idxmax()), rule="largest mean pooled gain of kept over everyone, outcomes x network sizes",
                              mean_gain={int(k): round(float(v), 5) for k, v in m.items()})
                lk = od / "network_cutoff.yaml"
                if lk.exists():
                    assert yaml.safe_load(open(lk))["cutoff"] == choice["cutoff"], "locked network cutoff changed"
                else:
                    yaml.safe_dump(choice, open(lk, "w"), sort_keys=False)
                print("network cutoff:", choice)
    pooled = pd.concat([x if isinstance(x, pd.DataFrame) else pd.DataFrame([x]) for x in parts], ignore_index=True)
    pooled.to_csv(od / "deploy_network_pooled.csv", index=False, float_format="%.6f")

    g_rows = []
    for half in ("development", "held_out", "mimic"):
        f = od / f"deploy_pairs_{half}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        p = p[p.own_cv.notna()].assign(loss=lambda x: x.own_cv - x.auroc, loss_avg=lambda x: x.outside_mean - x.auroc)
        if half == "development":
            grid = {T: gonogo(p, T) for T in range(5, 95, 5)}
            T = max(grid, key=lambda k: grid[k]["sens"] + grid[k]["spec"])
            ch = dict(cutoff=int(T), rule=f"5-degree grid, max Youden J for loss > {LOSS} AUROC vs own model, all outcomes")
            lk = od / "gonogo_cutoff.yaml"
            if lk.exists():
                assert yaml.safe_load(open(lk))["cutoff"] == ch["cutoff"], "locked go/no-go cutoff changed"
            else:
                yaml.safe_dump(ch, open(lk, "w"), sort_keys=False)
            print("go/no-go cutoff:", ch)
        T = yaml.safe_load(open(od / "gonogo_cutoff.yaml"))["cutoff"]
        for task, q in list(p.groupby("task")) + [("all", p)]:
            bad, bad2 = (q.loss > LOSS).astype(int), (q.loss_avg > LOSS).astype(int)
            c = lambda b, s: roc_auc_score(b, s) if 0 < b.sum() < len(b) else np.nan
            rho = q.groupby("target").apply(lambda z: z.das.corr(z.loss, method="spearman") if len(z) > 3 else np.nan).median()
            g_rows.append(dict(half=half, task=task, pairs=len(q), targets=q.target.nunique(), cutoff=T,
                               median_loss=q.loss.median(), c_das=c(bad, q.das), c_membership=c(bad, q.membership_c),
                               c_das_vs_avg=c(bad2, q.das), rho_within_target=rho, **gonogo(q, T)))
    gg = pd.DataFrame(g_rows)
    gg.to_csv(od / "gonogo.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(gg.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["deploy", "network", "pool"])
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--half", choices=["development", "held_out", "mimic"], default="development")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="first n targets, outputs in deploy_network/_test")
    args = ap.parse_args()
    {"deploy": cmd_deploy, "network": cmd_network, "pool": cmd_pool}[args.step](args)


if __name__ == "__main__":
    main()
