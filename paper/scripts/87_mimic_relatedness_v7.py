#!/usr/bin/env python

"""MIMIC-IV's mean DAS to each eICU study hospital."""
import importlib.util, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("i38", HERE / "38_iecv.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
cfg, st, res, rowlevel = M.setup("configs/external.yaml")
split = pd.read_csv(res / "iecv_split.csv")
e, X, XM, Y, rows_of, V, no_med = M.load_eicu(cfg, rowlevel)
mf = pd.read_parquet(rowlevel / "mimic_flags.parquet"); mh = pd.read_parquet(rowlevel / "mimic_history_flags.parquet")
dm = [f for f in M.FLAGS if not f.endswith("_H")]; hf = [f for f in M.FLAGS if f.endswith("_H")]
flagsM = mf[["stay_id"] + dm].merge(mh[["stay_id"] + hf], on="stay_id"); flagsM["hospitalid"] = -1
tvec = {}
for cond in M.CONDITIONS:
    cnt = M.s02.hospital_counts(flagsM, M.s02.lobe_codes(flagsM, cond, ("D", "H", "M")), 7)
    tvec[cond] = (cnt.iloc[0] / cnt.iloc[0].sum()).to_numpy(float)
pool = split.hospitalid.to_numpy()
ang = M.das_rankings(V, tvec, pool)
out = pd.DataFrame({"hospitalid": pool})
run = res / os.environ.get("EHRDOC_OUT", "") / "iecv_mimic_relatedness.csv"
memb = pd.read_csv(run).set_index("hospitalid").membership_c if run.exists() else pd.Series(dtype=float)
out["membership_c"] = out.hospitalid.map(memb)
for k in M.CONDITIONS + ["mean"]:
    out[f"das_{k}"] = out.hospitalid.map(ang[k]) if k in ang else np.nan
out.round(5).to_csv(res / "iecv_mimic_relatedness.csv", index=False)
print(out.notna().sum().to_string()); print(out.filter(like="das_").describe().round(1).loc[["50%", "min", "max"]].to_string())
