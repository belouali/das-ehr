#!/usr/bin/env python

"""Check that the eICU diagnosis rule stated codes-first matches the label rule stay for stay."""
import importlib.util
from pathlib import Path
import duckdb, pandas as pd, yaml

HERE = Path(__file__).resolve().parent
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v7, s01 = load("v7_rules"), load("01_build_flags")
cfg, DEF, CONDS = v7.CFG, v7.DEF["conditions"], v7.CONDITIONS
E, V7 = cfg["paths"]["eicu"], Path(v7.DEF["rowlevel_v7"]); TAB = HERE.parent / cfg["paths"]["results"] / "definitions_v7"
LM = 60 * cfg["cohort"]["landmark_hours"]


def code_on_list(field, rule):
    for code in str(field).split(","):
        code = code.strip().upper()
        if code and v7.icd_match(code, 9 if s01.ICD9_SHAPE.match(code) else 10, rule): return True
    return False


con = duckdb.connect()
whole = pd.read_parquet(V7 / "eicu_flags.parquet"); first = pd.read_parquet(V7 / "eicu_flags_24h.parquet")
con.register("cohdf", whole[["patientunitstayid"]]); con.execute("CREATE TABLE ecoh AS SELECT patientunitstayid FROM cohdf")
rows = con.execute(f"""SELECT d.patientunitstayid id, d.diagnosisstring label, coalesce(trim(d.icd9code), '') icd, d.diagnosisoffset off
                       FROM '{E}/diagnosis.parquet' d JOIN ecoh USING (patientunitstayid)""").df()
lab = rows.groupby(["label", "icd"]).id.nunique().reset_index(name="stays")
tab, spec, check = [], {}, []
for c in CONDS:
    dec = v7.decide(lab.label, DEF[c]["eicu_diagnosis"])
    lab["counted"], lab["reason"], lab["code"] = (dec.status == "counted").to_numpy(), dec.reason.to_numpy(), lab.icd.map(lambda x: code_on_list(x, DEF[c]["mimic_icd"]))
    g = lab.groupby("label").agg(counted=("counted", "first"), reason=("reason", "first"), any_code=("code", "any"), all_code=("code", "all"), stays=("stays", "sum"),
                                 icd=("icd", lambda x: "; ".join(sorted(set(i for i in x if i))))).reset_index()
    add = g[g.counted & ~g.all_code]; rem = g[~g.counted & g.any_code]; by_code = g[g.counted & g.all_code]
    for name, d, why in [("1. counted by its attached ICD code", by_code, None), ("2. added: states the condition, no listed code", add, None), ("3. removed: listed code, text does not state the condition", rem, None)]:
        for r in d.sort_values("stays", ascending=False).itertuples():
            tab.append(dict(condition=c, step=name, stays=int(r.stays), label=r.label, icd=r.icd, reason=r.reason or "label does not state the condition"))
    spec[c] = {"icd_list": DEF[c]["mimic_icd"], "added_labels": sorted(add.label), "removed_labels": sorted(rem.label)}

    rows["code"] = rows.icd.map(dict(zip(lab.icd.unique(), [code_on_list(x, DEF[c]["mimic_icd"]) for x in lab.icd.unique()])))
    hit = (rows.code & ~rows.label.isin(set(rem.label))) | rows.label.isin(set(add.label))
    for name, fl, m in [("whole stay", whole, hit), ("first 24 hours", first, hit & (rows.off <= LM))]:
        new = fl.patientunitstayid.isin(set(rows.id[m])); old = fl[f"{c}_D"]
        check.append(dict(condition=c, window=name, stays=len(fl), label_list=int(old.sum()), codes_first=int(new.sum()), differ=int((new != old).sum()),
                          labels_by_code=len(by_code), labels_added=len(add), labels_removed=len(rem),
                          stays_through_code=int(fl.patientunitstayid.isin(set(rows.id[rows.code & ~rows.label.isin(set(rem.label)) & (rows.off <= (LM if name != "whole stay" else 10**9))])).sum())))
pd.DataFrame(tab).to_csv(TAB / "eicu_diagnosis_codes_first.csv", index=False)
yaml.safe_dump({"description": "eICU diagnosis rule, codes-first form. A diagnosis counts when its attached ICD code is on icd_list, or its label is in added_labels, and its label is not in removed_labels. Built and verified by scripts/80_dx_codes_first.py.", "conditions": spec},
               open(HERE.parent / "configs" / "eicu_diagnosis_codes_first.yaml", "w"), sort_keys=False, width=200)
ck = pd.DataFrame(check); ck.to_csv(TAB / "eicu_diagnosis_codes_first_check.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 100, "display.max_rows", 300); print(ck.to_string(index=False))
t = pd.DataFrame(tab); print(t[~t.step.str.startswith("1")].assign(label=lambda d: d.label.str.split("|").str[1:].str.join("|").str[:80])[["condition", "step", "stays", "icd", "label"]].assign(step=lambda d: d.step.str[:9]).to_string(index=False))
