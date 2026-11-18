#!/usr/bin/env python

"""Quantiles of between-hospital DAS for Figure 2B."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("s40", HERE / "40_das_variation.py")
s40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s40)
s38 = s40.s38

cfg, _, res, rowlevel = s38.setup("configs/external.yaml")
split = pd.read_csv(res / "iecv_split.csv")
e, X, XM, Y, rows_of, V, no_med = s38.load_eicu(cfg, rowlevel)
hosp_141 = np.sort(split.hospitalid.to_numpy())
has_das = np.array([any(h in V[c].index for c in s40.CONDITIONS) for h in hosp_141])
hosp_all = hosp_141[has_das]
mats_all = s40.pairwise(V, hosp_all)
old = pd.read_csv(res / "partA_spread.csv").set_index(["sample", "measure"])
rows = []
for sample in ["all", "with_medication"]:
    keep = np.array([h not in no_med for h in hosp_all]) if sample == "with_medication" else np.ones(len(hosp_all), bool)
    for meas in s40.CONDITIONS + ["mean"]:
        m = s40.sub(mats_all[meas], keep)
        have = np.isfinite(m).sum(1) > 1
        b = s40.upper(s40.sub(m, have))
        q1, q2, q3 = np.percentile(b, [25, 50, 75])
        o = old.loc[(sample, meas)]
        assert len(b) == o.pairs and np.allclose([q1, q2, q3], [o.das_q25, o.das_median, o.das_q75], atol=1e-4), (sample, meas)
        iqr = q3 - q1
        rows.append(dict(sample=sample, measure=meas, pairs=len(b), das_min=b.min(), das_p5=np.percentile(b, 5), das_q25=q1,
                         das_median=q2, das_q75=q3, das_p95=np.percentile(b, 95), das_max=b.max(),
                         whisker_lo=b[b >= q1 - 1.5 * iqr].min(), whisker_hi=b[b <= q3 + 1.5 * iqr].max(),
                         pairs_beyond_whiskers=int(((b < q1 - 1.5 * iqr) | (b > q3 + 1.5 * iqr)).sum())))
d = pd.DataFrame(rows)
d.to_csv(res / "partA_spread_quantiles.csv", index=False, float_format="%.6f")
print(d.round(2).to_string(index=False))
