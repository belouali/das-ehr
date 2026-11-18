#!/usr/bin/env python

"""Stay-level evidence flags under the condition definitions in configs/definitions_v7.yaml."""
import importlib.util
from pathlib import Path
import re
import duckdb, pandas as pd, yaml

HERE = Path(__file__).resolve().parent
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v7, s01 = load("v7_rules"), load("01_build_flags")
cfg, DEF, ph, CONDITIONS = v7.CFG, v7.DEF["conditions"], v7.PH, v7.CONDITIONS
E, M, RL = cfg["paths"]["eicu"], cfg["paths"]["mimic"], Path(cfg["paths"]["rowlevel"])
OUT = Path(v7.DEF["rowlevel_v7"] + v7.variant_suffix()); OUT.mkdir(parents=True, exist_ok=True)
TAB = HERE.parent / cfg["paths"]["results"] / "definitions_v7" / (f"variant_{v7.VARIANT}" if v7.VARIANT else ""); TAB.mkdir(parents=True, exist_ok=True)
hicl = yaml.safe_load(open(HERE.parent / "configs" / f"eicu_hicl{v7.variant_suffix()}.yaml"))["hicl"]
LM = 60 * cfg["cohort"]["landmark_hours"]
con = duckdb.connect()
base = pd.read_parquet(RL / "eicu_flags.parquet"); base24 = pd.read_parquet(RL / "eicu_flags_24h.parquet")
con.register("cohdf", base[["patientunitstayid"]]); con.execute("CREATE TABLE ecoh AS SELECT patientunitstayid FROM cohdf")


dxl = con.execute(f"""SELECT diagnosisstring label, coalesce(trim(icd9code), '') icd, count(DISTINCT patientunitstayid) stays
                      FROM '{E}/diagnosis.parquet' d JOIN ecoh USING (patientunitstayid) GROUP BY 1, 2""").df()
hxl = con.execute(f"""SELECT pasthistorypath path, pasthistoryvalue value, count(DISTINCT patientunitstayid) stays
                      FROM '{E}/pastHistory.parquet' h JOIN ecoh USING (patientunitstayid) GROUP BY 1, 2""").df()
mdl = con.execute(f"""SELECT upper(trim(drugname)) name, count(*) n, count(DISTINCT patientunitstayid) stays
                      FROM '{E}/medication.parquet' m JOIN ecoh USING (patientunitstayid) WHERE drugname IS NOT NULL AND trim(drugname) <> '' GROUP BY 1""").df()
def term_re(terms):
    return r"(?<![A-Za-z])(?:" + "|".join(re.escape(x) for x in terms) + ")"


def med_decision(names, counted, c):
    """counted, no longer counted, or counted only together with the partner drug of a combination"""
    d = counted.map({True: "counted", False: "no longer counted"})
    cb = DEF[c]["medication"].get("combination")
    if cb:
        part = names.astype(str).str.contains(term_re(cb["a"]["terms"] + cb["b"]["terms"]), case=False, regex=True) & ~counted
        d[part] = "counted only with its partner drug (" + cb["name"] + ")"
    return d


dx_tab, hx_tab, md_tab, dx_ok, hx_ok, md_ok = [], [], [], {}, {}, {}
for c in CONDITIONS:
    old_t = s01.text_mask(dxl.label, ph[c]["diagnosis"]); old_i = dxl.icd.map(lambda x: s01.eicu_icd_match(x, cfg["icd"][c]))
    d = v7.decide(dxl.label, DEF[c]["eicu_diagnosis"], relevant=old_t | old_i)
    keep = old_t | old_i | (d.status != "not counted")
    dx_tab.append(dxl[keep].assign(condition=c, decision=d.status[keep], reason=d.reason[keep], previous_text_rule=old_t[keep], icd_rule=old_i[keep]))
    dx_ok[c] = set(dxl.label[d.status == "counted"])
    h = v7.decide(hxl.path, DEF[c]["eicu_history_path"]); old_h = s01.text_mask(hxl.value, ph[c]["past_history"])
    keeph = old_h | (h.status != "not counted")
    hx_tab.append(hxl[keeph].assign(condition=c, decision=h.status[keeph], reason=h.reason[keeph], previous_rule=old_h[keeph]))
    hx_ok[c] = set(zip(hxl.path[h.status == "counted"], hxl.value[h.status == "counted"]))
    new_m = v7.med_mask(mdl.name, c); old_m = s01.contains_any(mdl.name, ph[c]["medication"]["include"])
    keepm = new_m | old_m
    md_tab.append(mdl[keepm].assign(condition=c, decision=med_decision(mdl.name, new_m, c)[keepm], previous_rule=old_m[keepm], dataset="eICU"))
    md_ok[c] = set(mdl.name[new_m])
cols = ["condition", "decision", "reason", "stays", "label", "icd", "previous_text_rule", "icd_rule"]
pd.concat(dx_tab)[cols].sort_values(["condition", "decision", "stays"], ascending=[True, True, False]).to_csv(TAB / "eicu_diagnosis_labels.csv", index=False)
pd.concat(hx_tab)[["condition", "decision", "reason", "stays", "value", "path", "previous_rule"]].sort_values(["condition", "stays"], ascending=[True, False]).to_csv(TAB / "eicu_history_entries.csv", index=False)


def eicu_combination(c, limit, strict=False):
    """stays with both drugs of a combination ordered with overlapping periods (a missing stop time counts as one day)"""
    cb = DEF[c]["medication"].get("combination")
    if not cb: return set()
    q = lambda side: f"""SELECT patientunitstayid id, drugstartoffset s, CASE WHEN drugstopoffset IS NULL OR drugstopoffset <= drugstartoffset THEN {"NULL" if strict else "drugstartoffset + 1440"} ELSE drugstopoffset END e
        FROM '{E}/medication.parquet' m JOIN ecoh USING (patientunitstayid)
        WHERE (regexp_matches(lower(coalesce(drugname, '')), '{term_re(cb[side]["terms"]).lower().replace("(?<![a-za-z])", "(^|[^a-z])")}') OR drughiclseqno IN ({", ".join(str(x) for x in cb[side]["hicl"])}))
          {"AND drugorderoffset <= " + str(limit) if limit is not None else ""}"""
    return set(con.cursor().execute(f"SELECT DISTINCT a.id FROM ({q('a')}) a JOIN ({q('b')}) b ON a.id = b.id AND a.s <= b.e AND b.s <= a.e").df().id)


def eicu_flags(b, wd, wh, wm, limit=None):
    out = b.copy()
    dx = con.cursor().execute(f"SELECT DISTINCT patientunitstayid id, diagnosisstring v FROM '{E}/diagnosis.parquet' d JOIN ecoh USING (patientunitstayid) {wd}").df()
    hx = con.cursor().execute(f"SELECT DISTINCT patientunitstayid id, pasthistorypath p, pasthistoryvalue v FROM '{E}/pastHistory.parquet' h JOIN ecoh USING (patientunitstayid) {wh}").df()
    md = con.cursor().execute(f"""SELECT DISTINCT patientunitstayid id, upper(trim(drugname)) v, CAST(drughiclseqno AS BIGINT) h
                                  FROM '{E}/medication.parquet' m JOIN ecoh USING (patientunitstayid) {wm}""").df()
    hx["k"] = list(zip(hx.p, hx.v))
    for c in CONDITIONS:
        out[f"{c}_D"] = out.patientunitstayid.isin(set(dx.id[dx.v.isin(dx_ok[c])]))
        out[f"{c}_H"] = out.patientunitstayid.isin(set(hx.id[hx.k.isin(hx_ok[c])]))
        ex = DEF[c]["medication"].get("name_exclude")
        not_excluded = ~md.v.fillna("").str.contains(ex, case=False, regex=True) if ex else True
        out[f"{c}_M"] = out.patientunitstayid.isin(set(md.id[md.v.isin(md_ok[c]) | (md.h.isin(list(hicl[c])) & not_excluded)]))
        comb = eicu_combination(c, limit, strict=bool((DEF[c]["medication"].get("combination") or {}).get("strict"))); COMB[(c, "whole stay" if limit is None else "first 24 hours")] = int(out.patientunitstayid.isin(comb).sum())
        if DEF[c]["medication"].get("combination"):
            COMB[(c, ("whole stay" if limit is None else "first 24 hours") + ", recorded stop times only")] = int(out.patientunitstayid.isin(eicu_combination(c, limit, strict=True)).sum())
        out[f"{c}_M"] |= out.patientunitstayid.isin(comb)
    return out


COMB = {}
whole = eicu_flags(base, "", "", "")
first = eicu_flags(base24, f"WHERE d.diagnosisoffset <= {LM}", f"WHERE h.pasthistoryenteredoffset <= {LM}", f"WHERE m.drugorderoffset <= {LM}", LM)
assert list(whole.columns) == list(base.columns) and list(first.columns) == list(base24.columns)
whole.to_parquet(OUT / "eicu_flags.parquet", index=False); first.to_parquet(OUT / "eicu_flags_24h.parquet", index=False)


mim = pd.read_parquet(RL / "mimic_flags.parquet")
cur = con.cursor(); cur.register("mdf", mim[["stay_id", "hadm_id"]])
cur.execute(f"CREATE TABLE mcoh AS SELECT d.stay_id, d.hadm_id, i.intime, i.outtime FROM mdf d JOIN '{M}/icu/icustays.parquet' i USING (stay_id)")
mdx = cur.execute(f"SELECT c.stay_id, x.icd_code code, x.icd_version ver FROM '{M}/hosp/diagnoses_icd.parquet' x JOIN mcoh c USING (hadm_id)").df()
mrx = cur.execute(f"""SELECT c.stay_id, upper(trim(r.drug)) name, r.starttime s, coalesce(r.stoptime, r.starttime + INTERVAL 1 DAY) e, r.stoptime e0 FROM '{M}/hosp/prescriptions.parquet' r JOIN mcoh c ON c.hadm_id = r.hadm_id
                      WHERE r.drug IS NOT NULL AND r.starttime < c.outtime AND coalesce(r.stoptime, r.starttime) >= c.intime""").df()
titles = cur.execute(f"SELECT icd_code code, icd_version ver, long_title title FROM '{M}/hosp/d_icd_diagnoses.parquet'").df()
codes = mdx.groupby(["code", "ver"]).stay_id.nunique().reset_index(name="stays").merge(titles, on=["code", "ver"], how="left")
rxn = mrx.groupby("name").agg(n=("stay_id", "size"), stays=("stay_id", "nunique")).reset_index()
mout, icd_tab = mim.copy(), []
for c in CONDITIONS:
    new = pd.Series([v7.icd_match(k, v, DEF[c]["mimic_icd"]) for k, v in zip(codes.code, codes.ver)], index=codes.index)
    old = pd.Series([s01.icd_match(k, int(v), cfg["icd"][c]) for k, v in zip(codes.code, codes.ver)], index=codes.index)
    keep = new | old
    icd_tab.append(codes[keep].assign(condition=c, decision=[("counted" if n else "no longer counted") for n in new[keep]], previous_rule=old[keep]))
    ok = codes[new][["code", "ver"]]
    mout[f"{c}_D"] = mim.stay_id.isin(set(mdx.merge(ok, on=["code", "ver"]).stay_id))
    newm = v7.med_mask(rxn.name, c); oldm = s01.contains_any(rxn.name, ph[c]["medication"]["include"]); keepm = newm | oldm
    md_tab.append(rxn[keepm].assign(condition=c, decision=med_decision(rxn.name, newm, c)[keepm], previous_rule=oldm[keepm], dataset="MIMIC-IV"))
    mout[f"{c}_M"] = mim.stay_id.isin(set(mrx.stay_id[mrx.name.isin(set(rxn.name[newm]))]))
    cb = DEF[c]["medication"].get("combination")
    if cb:
        A = mrx[mrx.name.str.contains(term_re(cb["a"]["terms"]), case=False, regex=True)]; B = mrx[mrx.name.str.contains(term_re(cb["b"]["terms"]), case=False, regex=True)]
        pr = A.merge(B, on="stay_id", suffixes=("_a", "_b")); pr = pr[(pr.s_a <= pr.e_b) & (pr.s_b <= pr.e_a)]
        if cb.get("strict"): pr = pr[pr.e0_a.notna() & pr.e0_b.notna()]
        COMB[(c, "MIMIC-IV")] = int(mim.stay_id.isin(set(pr.stay_id)).sum()); mout[f"{c}_M"] |= mim.stay_id.isin(set(pr.stay_id))
        prs = pr[pr.e0_a.notna() & pr.e0_b.notna()]; COMB[(c, "MIMIC-IV, recorded stop times only")] = int(mim.stay_id.isin(set(prs.stay_id)).sum())
assert list(mout.columns) == list(mim.columns)
mout.to_parquet(OUT / "mimic_flags.parquet", index=False)
pd.concat(icd_tab)[["condition", "decision", "stays", "code", "ver", "title", "previous_rule"]].sort_values(["condition", "decision", "stays"], ascending=[True, True, False]).to_csv(TAB / "mimic_icd_codes.csv", index=False)
pd.concat(md_tab)[["dataset", "condition", "decision", "n", "stays", "name", "previous_rule"]].sort_values(["dataset", "condition", "decision", "n"], ascending=[True, True, True, False]).to_csv(TAB / "drug_names.csv", index=False)


rows = []
for ds, old, new, idc in [("eICU whole stay", base, whole, "patientunitstayid"), ("eICU first 24 hours", base24, first, "patientunitstayid"), ("MIMIC-IV", mim, mout, "stay_id")]:
    for c in CONDITIONS:
        for s in ("D", "H", "M"):
            if f"{c}_{s}" not in old: continue
            a, b = old[f"{c}_{s}"], new[f"{c}_{s}"]
            rows.append(dict(dataset=ds, condition=c, source=s, stays=len(old), previous=int(a.sum()), version7=int(b.sum()), gained=int((b & ~a).sum()), lost=int((a & ~b).sum())))
res = pd.DataFrame(rows); res.to_csv(TAB / "flag_counts_previous_vs_v7.csv", index=False)
pd.DataFrame([dict(condition=k[0], dataset_or_window=k[1], stays_with_both_drugs_overlapping=v) for k, v in COMB.items()]).to_csv(TAB / "medication_combination_counts.csv", index=False)
print("combination:", COMB)
pd.set_option("display.width", 200); print(res.to_string(index=False)); print("wrote", OUT)
