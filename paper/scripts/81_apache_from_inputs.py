#!/usr/bin/env python

"""APACHE IVa score rebuilt from the APACHE inputs recorded in eICU, checked against the official score."""
from pathlib import Path
import duckdb, numpy as np, pandas as pd, yaml
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
cfg = yaml.safe_load(open(HERE.parent / "configs" / "external.yaml")); DEF = yaml.safe_load(open(HERE.parent / "configs" / "definitions_v7.yaml"))
E, RL, OUT = cfg["paths"]["eicu"], Path(cfg["paths"]["rowlevel"]), Path(DEF["rowlevel_v7"]); TAB = HERE.parent / cfg["paths"]["results"] / "definitions_v7"
CHRONIC = {"aids": 23, "hepaticfailure": 16, "lymphoma": 13, "metastaticcancer": 11, "leukemia": 10, "immunosuppression": 10, "cirrhosis": 4}
COMP = ["heartrate", "meanbp", "temperature", "respiratoryrate", "oxygenation", "hematocrit", "wbc", "creatinine", "urine", "bun", "sodium", "albumin", "bilirubin", "glucose", "acidbase", "gcs"]


def table(x, cuts, pts):
    """points by the first cut the value is below; the last entry of pts is for values at or above the last cut"""
    out = np.select([x < c for c in cuts], pts[:-1], default=pts[-1]).astype(float); out[np.isnan(x)] = np.nan; return out


def aps(d):
    v = {c: d[c].astype(float).where(d[c] != -1) for c in ["heartrate", "meanbp", "temperature", "respiratoryrate", "hematocrit", "wbc", "creatinine", "urine", "bun", "sodium", "albumin", "bilirubin", "glucose", "ph", "pco2", "pao2", "fio2", "eyes", "motor", "verbal"]}
    vent, intub = (d.vent == 1), (d.intubated == 1)
    p = pd.DataFrame(index=d.index)
    p["heartrate"] = table(v["heartrate"], [40, 50, 100, 110, 120, 140, 155], [8, 5, 0, 1, 5, 7, 13, 17])
    p["meanbp"] = table(v["meanbp"], [40, 60, 70, 80, 100, 120, 130, 140], [23, 15, 7, 6, 0, 4, 7, 9, 10])
    p["temperature"] = table(v["temperature"], [33.0, 33.5, 34.0, 35.0, 36.0, 40.0], [20, 16, 13, 8, 2, 0, 4])
    rr = table(v["respiratoryrate"], [6, 12, 14, 25, 35, 40, 50], [17, 8, 7, 0, 6, 9, 11, 18]); rr[(vent & (v["respiratoryrate"] >= 6) & (v["respiratoryrate"] < 14)).to_numpy()] = 0; p["respiratoryrate"] = rr
    p["hematocrit"] = table(v["hematocrit"], [41.0, 50.0], [3, 0, 3])
    p["wbc"] = table(v["wbc"], [1.0, 3.0, 20.0, 25.0], [19, 5, 0, 1, 5])
    arf = (v["creatinine"] >= 1.5) & (v["urine"] < 410) & (d.dialysis != 1)
    cr = table(v["creatinine"], [0.5, 1.5, 1.95], [3, 0, 4, 7]); cr[arf.to_numpy()] = 10; p["creatinine"] = cr
    p["urine"] = table(v["urine"], [400, 600, 900, 1500, 2000, 4000], [15, 8, 7, 5, 4, 0, 1])
    p["bun"] = table(v["bun"], [17.0, 20.0, 40.0, 80.0], [0, 2, 7, 11, 12])
    p["sodium"] = table(v["sodium"], [120, 135, 155], [3, 2, 0, 4])
    p["albumin"] = table(v["albumin"], [2.0, 2.5, 4.5], [11, 6, 0, 4])
    p["bilirubin"] = table(v["bilirubin"], [2.0, 3.0, 5.0, 8.0], [0, 5, 6, 8, 16])
    p["glucose"] = table(v["glucose"], [40, 60, 200, 350], [8, 9, 0, 3, 5])

    aado2 = v["fio2"] / 100 * 713 - v["pco2"] - v["pao2"]
    use_aa = intub & (v["fio2"] >= 50) & aado2.notna()
    ox = table(v["pao2"], [50, 70, 80], [15, 5, 2, 0]); ox_aa = table(aado2, [100, 250, 350, 500], [0, 7, 9, 11, 14]); ox[use_aa.to_numpy()] = ox_aa[use_aa.to_numpy()]; p["oxygenation"] = ox
    ph, pc = v["ph"], v["pco2"]
    ab = np.select([ph < 7.20, ph < 7.30, ph < 7.35, ph < 7.45, ph < 7.50, ph < 7.60],
                   [np.where(pc < 50, 12, 4), np.select([pc < 30, pc < 40, pc < 50], [9, 6, 3], 2), np.select([pc < 30, pc < 45], [9, 0], 1), np.select([pc < 30, pc < 45], [5, 0], 1),
                    np.select([pc < 30, pc < 35, pc < 45], [5, 0, 2], 12), np.where(pc < 40, 3, 12)], default=np.select([pc < 25, pc < 40], [0, 3], 12)).astype(float)
    ab[(ph.isna() | pc.isna()).to_numpy()] = np.nan; p["acidbase"] = ab
    e, m, vb = v["eyes"], v["motor"], v["verbal"]
    g1 = np.select([(vb == 1) & (m <= 2), (vb == 1) & (m <= 4), (vb == 1), (vb <= 3) & (m <= 2), (vb <= 3) & (m <= 4)], [48, 33, 16, 29, 24], default=0)
    g2 = np.select([(vb == 1) & (m <= 2), (vb == 1) & (m <= 4), (vb == 1), (vb <= 3) & (m <= 2), (vb <= 3) & (m <= 4), (vb <= 3) & (m == 5), (vb <= 3) & (m == 6),
                    (vb == 4) & (m <= 4), (vb == 4) & (m == 5), (vb == 4) & (m == 6), (vb == 5) & (m <= 5)], [29, 24, 15, 29, 24, 13, 10, 13, 8, 3, 3], default=0)
    gcs = np.where(e == 1, g1, g2).astype(float); gcs[(e.isna() | m.isna() | vb.isna()).to_numpy()] = np.nan; gcs[(d.meds == 1).to_numpy()] = 0; p["gcs"] = gcs
    return p[COMP]


con = duckdb.connect()
coh = pd.read_parquet(RL / "eicu_flags.parquet", columns=["patientunitstayid", "hospitalid"]); con.register("coh", coh)
d = con.execute(f"""SELECT c.patientunitstayid, c.hospitalid, CASE WHEN p.age = '> 89' THEN 90 ELSE TRY_CAST(p.age AS INTEGER) END agenum,
      r.apachescore::DOUBLE official, r.aps::DOUBLE official_aps, a.*, v.aids, v.hepaticfailure, v.lymphoma, v.metastaticcancer, v.leukemia, v.immunosuppression, v.cirrhosis, v.electivesurgery,
      (a.patientunitstayid IS NOT NULL) has_inputs
    FROM coh c JOIN '{E}/patient.parquet' p USING (patientunitstayid)
    LEFT JOIN (SELECT patientunitstayid, max(apachescore) apachescore, max(acutephysiologyscore) aps FROM '{E}/apachePatientResult.parquet' WHERE apacheversion = 'IVa' GROUP BY 1) r USING (patientunitstayid)
    LEFT JOIN (SELECT * EXCLUDE (apacheapsvarid) FROM '{E}/apacheApsVar.parquet' QUALIFY row_number() OVER (PARTITION BY patientunitstayid ORDER BY apacheapsvarid) = 1) a USING (patientunitstayid)
    LEFT JOIN (SELECT * FROM '{E}/apachePredVar.parquet' QUALIFY row_number() OVER (PARTITION BY patientunitstayid ORDER BY apachepredvarid) = 1) v USING (patientunitstayid)""").df()
d = d.loc[:, ~d.columns.duplicated()]
inp = d[d.has_inputs].copy()
pts = aps(inp); inp["aps_rebuilt"] = pts.fillna(0).sum(axis=1); inp["missing_components"] = pts.isna().sum(axis=1)
inp["gcs_not_assessable"] = (inp.meds == 1)
chron = pd.DataFrame({k: inp[k].fillna(0).astype(float) * w for k, w in CHRONIC.items()})
agep = pd.cut(inp.agenum, bins=[-np.inf, 44, 59, 64, 69, 74, 84, np.inf], labels=[0, 5, 11, 13, 16, 17, 24]).astype(float)
nonel = (inp.electivesurgery.fillna(0) != 1)
val = inp[(inp.official > 0) & (inp.official_aps > 0)]
rule = max(["max", "sum"], key=lambda r: np.mean(np.isclose((agep + (chron.max(axis=1) if r == "max" else chron.sum(axis=1)) * nonel)[val.index], val.official - val.official_aps)))
inp["score_rebuilt"] = inp.aps_rebuilt + agep + (chron.max(axis=1) if rule == "max" else chron.sum(axis=1)) * nonel
val = inp.loc[val.index]

rows = []
for name, a, b in [("physiology score", val.aps_rebuilt, val.official_aps), ("APACHE IVa score", val.score_rebuilt, val.official)]:
    ok = a.notna() & b.notna(); diff = (a - b)[ok]
    hosp = diff.groupby(val.hospitalid[ok]).agg(["mean", "size"]); hosp = hosp[hosp["size"] >= 50]
    rows.append(dict(measure=name, stays=int(ok.sum()), exact=round(float((diff == 0).mean()), 4), within_1=round(float((diff.abs() <= 1).mean()), 4), within_5=round(float((diff.abs() <= 5).mean()), 4),
                     spearman=round(spearmanr(a[ok], b[ok])[0], 4), median_abs_diff=float(diff.abs().median()), mean_diff=round(float(diff.mean()), 3), p95_abs_diff=float(diff.abs().quantile(.95)),
                     hospitals_50plus=len(hosp), hospital_mean_diff_min=round(float(hosp["mean"].min()), 2), hospital_mean_diff_max=round(float(hosp["mean"].max()), 2),
                     hospitals_mean_diff_over_1pt=int((hosp["mean"].abs() > 1).sum()), hospitals_mean_diff_over_2pt=int((hosp["mean"].abs() > 2).sum())))
res = pd.DataFrame(rows); res.to_csv(TAB / "apache_rebuild_validation.csv", index=False)
pd.set_option("display.width", 220); print("chronic-health rule:", rule); print(res.T.to_string())
dd = (val.aps_rebuilt - val.official_aps)
print("physiology difference distribution:", dd.round().value_counts(normalize=True).sort_index().loc[-6:6].round(4).to_dict())


d = d.merge(inp[["patientunitstayid", "aps_rebuilt", "score_rebuilt", "missing_components", "gcs_not_assessable"]], on="patientunitstayid", how="left")
official = (d.official > 0)
d["apache_score"] = np.where(official, d.official, d.score_rebuilt)
d["apache_source"] = np.select([official, d.score_rebuilt.notna()], ["official", "rebuilt from eICU inputs"], default="none")
d.loc[d.apache_score.isna(), "apache_source"] = "none"
out = d[["patientunitstayid", "hospitalid", "apache_score", "apache_source", "aps_rebuilt", "missing_components", "gcs_not_assessable"]]
out.to_parquet(OUT / "eicu_apache.parquet", index=False)
src = out.apache_source.value_counts(); print(src.to_dict())
h = out[out.apache_source != "none"].groupby("hospitalid").size(); print("hospitals with at least 100 scored stays:", int((h >= 100).sum()), "| stays there:", int(h[h >= 100].sum()))
reb = out[out.apache_source == "rebuilt from eICU inputs"]
pd.DataFrame([dict(official=int(src.get("official", 0)), rebuilt=int(src.get("rebuilt from eICU inputs", 0)), none=int(src.get("none", 0)), hospitals_100=int((h >= 100).sum()),
                   median_score_official=float(out.apache_score[out.apache_source == "official"].median()), median_score_rebuilt=float(reb.apache_score.median()),
                   median_missing_components_official=float(inp.loc[val.index, "missing_components"].median()), median_missing_components_rebuilt=float(reb.missing_components.median()))]).to_csv(TAB / "apache_sources.csv", index=False)
print("median score official %.0f, rebuilt %.0f | median missing components: scored %.0f, rebuilt %.0f" % (out.apache_score[out.apache_source == "official"].median(), reb.apache_score.median(), inp.loc[val.index, "missing_components"].median(), reb.missing_components.median()))
