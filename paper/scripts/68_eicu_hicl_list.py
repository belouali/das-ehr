#!/usr/bin/env python

"""eICU drug codes (HICL) for each condition's medication rule."""
import re
from pathlib import Path

import duckdb
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
cfg = yaml.safe_load(open(HERE.parent / "configs" / "external.yaml"))
import importlib.util
_s = importlib.util.spec_from_file_location("v7_rules", HERE / "v7_rules.py"); v7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(v7)
E, RL, RES = cfg["paths"]["eicu"], cfg["paths"]["rowlevel"], HERE.parent / cfg["paths"]["results"]
CONDITIONS = ["diabetes", "hypertension", "chf", "af"]

INCLUDE = {
    "diabetes": {9690: "glyburide-metformin", 12257: "glyburide (Glynase)", 33202: "pioglitazone-metformin",
                 34665: "sitagliptin-metformin", 38464: "linagliptin-metformin"},
    "hypertension": {120: "guanfacine", 131: "lisinopril-HCTZ", 147: "atenolol-chlorthalidone", 181: "nifedipine", 2900: "spironolactone-HCTZ",
                     3647: "triamterene-HCTZ", 8268: "nisoldipine", 8715: "bisoprolol-HCTZ", 9863: "losartan-HCTZ", 9934: "moexipril",
                     10041: "amlodipine-benazepril", 13237: "enalaprilat", 13911: "perindopril", 18963: "irbesartan-HCTZ",
                     25951: "amlodipine-atorvastatin", 26810: "diltiazem", 33009: "isosorbide dinitrate-hydralazine", 34245: "carvedilol phosphate",
                     34433: "amlodipine-valsartan", 35042: "amlodipine-olmesartan", 37444: "azilsartan"},
    "chf": {130: "enalapril", 131: "lisinopril-HCTZ", 132: "lisinopril",
            6323: "metoprolol succinate", 2900: "spironolactone-HCTZ", 6080: "ramipril", 7631: "quinapril",
            8715: "bisoprolol-HCTZ", 8991: "trandolapril", 9829: "losartan", 9863: "losartan-HCTZ", 12204: "valsartan", 13237: "enalaprilat",
            13911: "perindopril", 15576: "irbesartan", 16913: "candesartan", 17084: "valsartan-HCTZ", 18963: "irbesartan-HCTZ",
            34245: "carvedilol phosphate", 34433: "amlodipine-valsartan"},
    "af": {147: "atenolol-chlorthalidone", 2101: "propranolol", 2104: "atenolol", 4833: "propafenone", 7396: "bisoprolol",
           8715: "bisoprolol-HCTZ", 13795: "carvedilol", 26810: "diltiazem", 34245: "carvedilol phosphate", 37678: "amiodarone (Nexterone)"},
}
EXCLUDE = {
    20334: "insulin administration supplies", 23037: "insulin syringe", 23038: "insulin syringe", 23041: "insulin syringe", 24100: "insulin syringe",
    35487: "insulin pump", 2807: "heparin flush", 3770: "lactose (matches 'actos')", 885: "enteral feeds", 3441: "apraclonidine eye drops (matches 'clonidine')",
    2854: "unrelated drugs under one code", 13633: "unrelated drugs under one code", 18084: "unrelated drugs under one code",
    2159: "unrelated drugs under one code", 920: "two named rows, different drugs", 1694: "morphine",
}

con = duckdb.connect()
names = con.execute(f"""SELECT h, v, sum(n) n FROM (
   SELECT drughiclseqno h, upper(trim(drugname)) v, count(*) n FROM '{E}/medication.parquet'
     WHERE drughiclseqno IS NOT NULL AND drugname IS NOT NULL AND trim(drugname) <> '' GROUP BY 1, 2
   UNION ALL
   SELECT drughiclseqno h, upper(trim(drugname)) v, count(*) n FROM '{E}/admissionDrug.parquet'
     WHERE drughiclseqno > 0 AND drugname IS NOT NULL AND trim(drugname) <> '' GROUP BY 1, 2) GROUP BY 1, 2""").df()
names["h"] = names.h.astype(int)
nl = con.execute(f"""WITH c AS (SELECT patientunitstayid FROM '{RL}/eicu_flags.parquet')
   SELECT drughiclseqno h, count(*) n FROM '{E}/medication.parquet' m JOIN c USING (patientunitstayid)
   WHERE drughiclseqno IS NOT NULL AND (drugname IS NULL OR trim(drugname) = '') GROUP BY 1""").df()
nl = nl.assign(h=nl.h.astype(int)).set_index("h").n

final, review = {}, []
for c in CONDITIONS:
    names["hit"] = v7.med_mask(names.v, c)
    final[c] = {}
    for h, d in names.groupby("h"):
        named, share = int(d.n.sum()), d.n[d.hit].sum() / d.n.sum()
        gen = d.assign(g=d.v.str.replace(r"[\d(].*$", "", regex=True).str.strip().str[:30]).groupby("g").n.sum().sort_values(ascending=False)
        top = "; ".join(f"{g} ({int(n)})" for g, n in gen.head(3).items() if g)
        auto = named >= 20 and share >= 0.95
        if h in EXCLUDE: status = "excluded: " + EXCLUDE[h]
        elif h in INCLUDE[c] and (not v7.VARIANT or share >= 0.5): status = "included by review: " + INCLUDE[c][h]
        elif auto: status = "included: names match the rule"
        elif d.hit.any(): status = "not included: too few or too mixed names"
        else: continue
        if status.startswith("included"):
            assert named >= 1
            final[c][int(h)] = (gen.index[0] or gen.index[min(1, len(gen) - 1)]).title()
        review.append(dict(condition=c, hicl=int(h), status=status, named_rows=named, share_matching_rule=round(share, 3),
                           nameless_rows_in_cohort=int(nl.get(h, 0)), names=top))
    missing = set(INCLUDE[c]) - set(final[c]); assert v7.VARIANT or not missing, (c, missing)
rev = pd.DataFrame(review); rev.to_csv(RES / "_explore" / f"eicu_hicl_review{v7.variant_suffix()}.csv", index=False)
doc = {"description": "eICU drughiclseqno codes counted as medication evidence, whether or not the drug name is filled in (the paper's drug lists). Built by scripts/68_eicu_hicl_list.py; "
                      "evidence in results/_explore/eicu_hicl_review.csv.",
       "hicl": {c: {k: final[c][k] for k in sorted(final[c])} for c in CONDITIONS},
       "excluded": {int(k): v for k, v in sorted(EXCLUDE.items())}}
yaml.safe_dump(doc, open(HERE.parent / "configs" / f"eicu_hicl{v7.variant_suffix()}.yaml", "w"), sort_keys=False, width=120)
inc = rev[rev.status.str.startswith("included")]
print(inc.groupby("condition").agg(codes=("hicl", "size"), by_review=("status", lambda s: s.str.contains("review").sum()),
                                   nameless_rows=("nameless_rows_in_cohort", "sum")).to_string())
ni = rev[rev.status.str.startswith("not included")]
print("not included (too few or mixed):", len(ni), "codes,", int(ni.drop_duplicates("hicl").nameless_rows_in_cohort.sum()), "nameless rows")
print(ni[ni.nameless_rows_in_cohort > 20][["condition", "hicl", "named_rows", "share_matching_rule", "nameless_rows_in_cohort", "names"]].to_string(index=False))
allc = set().union(*[set(final[c]) for c in CONDITIONS])
print("nameless coded rows in cohort:", int(nl.sum()), "| on a listed code:", int(nl[nl.index.isin(allc)].sum()),
      "| code has no name anywhere in eICU:", int(nl[~nl.index.isin(names.h)].sum()))
