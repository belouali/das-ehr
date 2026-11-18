#!/usr/bin/env python

"""Continuous ICU episode for every eICU stay: its length, survival at its end and readmission after it."""
import os
from pathlib import Path
import duckdb, pandas as pd, yaml
HERE = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(HERE / os.environ.get("EHRDOC_CONFIG", "configs/external_v7.yaml")))
E, RL = cfg["paths"]["eicu"], Path(cfg["paths"]["rowlevel"])
GAP = 60
coh = pd.read_parquet(RL / "eicu_flags.parquet", columns=["patientunitstayid"])
allp = duckdb.execute(f"""SELECT patientunitstayid, patienthealthsystemstayid hs, unitstaytype t, unitdischargestatus st,
                                 -hospitaladmitoffset AS start, -hospitaladmitoffset + unitdischargeoffset AS stop, unitdischargeoffset d
                          FROM '{E}/patient.parquet'""").df()
p = coh.merge(allp, on="patientunitstayid", how="left")
assert p.d.notna().all()
by = {k: g.sort_values("start") for k, g in allp[allp.hs.isin(p.hs)].groupby("hs")}
ep, units, alive, readmit = [], [], [], []
for r in p.itertuples():
    end, last, members = r.stop, r.st, {r.patientunitstayid}
    g = by[r.hs]
    for s in g.itertuples():
        if s.patientunitstayid not in members and s.t != "stepdown/other" and s.start >= r.stop - 1 and abs(s.start - end) <= GAP:
            end, last = max(end, s.stop), s.st; members.add(s.patientunitstayid)
    after = g[(g.start > end - 1) & ~g.patientunitstayid.isin(members)]
    ep.append((end - r.start) / 1440); units.append(len(members))
    alive.append(last == "Alive")
    readmit.append(bool((after.t == "readmit").any()))
out = pd.DataFrame({"patientunitstayid": p.patientunitstayid, "los_d": p.d / 1440, "episode_d": ep, "units": units,
                    "episode_alive": alive, "readmit_after": readmit})
assert (out.episode_d >= out.los_d - 1e-9).all()
out.to_parquet(RL / "eicu_icu_episode.parquet", index=False)
m = pd.read_parquet(RL / "eicu_m1_features.parquet", columns=["patientunitstayid"]).merge(out, on="patientunitstayid")
print(f"cohort stays {len(out):,}; extended {int((out.units > 1).sum()):,}")
print(f"modeled stays {len(m):,}; extended {int((m.units > 1).sum()):,}")
for lim in (5, 7):
    print(f"modeled, ICU stay > {lim} d: unit stay {int((m.los_d > lim).sum()):,}; episode {int((m.episode_d > lim).sum()):,}; labels changed {int(((m.los_d > lim) != (m.episode_d > lim)).sum()):,}")
print(f"modeled, alive at end of episode {int(m.episode_alive.sum()):,}; readmissions among them {int((m.episode_alive & m.readmit_after).sum()):,}")
