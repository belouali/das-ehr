#!/usr/bin/env python

"""Aggregate numbers quoted in the Discussion and Limitations."""
import os
from pathlib import Path
import duckdb, pandas as pd, yaml
HERE = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(HERE / "configs/external_v7.yaml")); E, M, RL = cfg["paths"]["eicu"], cfg["paths"]["mimic"], Path(cfg["paths"]["rowlevel"])
R = HERE / "results_v7"; RUN = R / "v7_2026-11-16"
split = pd.read_csv(RUN / "iecv_split.csv")
e = pd.read_parquet(RL / "eicu_m1_features.parquet")[["patientunitstayid", "hospitalid"]]
e = e[e.hospitalid.isin(split.hospitalid)]
ep = pd.read_parquet(RL / "eicu_icu_episode.parquet")
x = e.merge(ep, on="patientunitstayid")
rd = x[x.episode_alive].groupby("hospitalid").readmit_after.sum()
duckdb.register("ec", e)
pre = duckdb.execute(f"SELECT -p.hospitaladmitoffset / 1440.0 AS pre FROM ec JOIN '{E}/patient.parquet' p USING (patientunitstayid)").df().pre
m = pd.read_parquet(RL / "mimic_m1_features.parquet")[["stay_id"]]
q = m.merge(duckdb.execute(f"SELECT i.stay_id, date_diff('second', a.admittime, i.intime) / 86400.0 pre FROM '{M}/icu/icustays.parquet' i JOIN '{M}/hosp/admissions.parquet' a USING (hadm_id)").df(), on="stay_id")
dv = pd.read_csv(RUN / "iecv_das_vs_membership.csv")
dv = dv[(dv.half == "held_out") & (dv["sample"] == "all") & (dv.analysis == "main")]
w = dv.pivot_table(index="task", columns="se_rule", values=["lo", "hi"])
dep = [t for t in w.index if (w.loc[t, ("lo", "independent")] > 0 or w.loc[t, ("hi", "independent")] < 0) != (w.loc[t, ("lo", "corr_0.5")] > 0 or w.loc[t, ("hi", "corr_0.5")] < 0)]
rows = dict(study_hospitals=len(split), hospitals_no_readmission=int((rd == 0).sum()), hospitals_under5_readmission=int((rd < 5).sum()),
            eicu_modeled=len(e), eicu_hosp14_at_24h=int((pre > 13).sum()), mimic_modeled=len(q), mimic_hosp14_at_24h=int((q.pre > 13).sum()),
            dvm_outcomes_dependent=";".join(dep))
print(rows); pd.DataFrame([rows]).to_csv(R / "discussion_numbers.csv", index=False)
print(dv[["task", "se_rule", "diff", "lo", "hi"]].round(4).to_string(index=False))
