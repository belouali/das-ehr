#!/usr/bin/env python
"""Check that the das package reproduces the paper's DAS distribution from the stay-level evidence flags.

Usage: python paper/scripts/check_das_against_paper.py <eicu_flags.parquet> <study_hospitals.csv>
The flags file has one row per eligible stay with hospitalid and <condition>_D/_H/_M booleans (built from eICU under
a PhysioNet credentialed-access agreement; not distributed here). study_hospitals.csv lists the study hospitals
(column hospitalid). Prints the median and IQR of pairwise DAS per measure, to compare with Supplementary Table S1.
"""
import sys
import numpy as np
import pandas as pd
import das

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
flags = pd.read_parquet(sys.argv[1]).rename(columns={"hospitalid": "hospital_id"})
study = set(pd.read_csv(sys.argv[2]).hospitalid)
flags = flags[flags.hospital_id.isin(study)]
v = das.documentation_vectors(flags, CONDITIONS, min_stays=100)
hosp = sorted(set().union(*[x.index for x in v.values()]))
iu = np.triu_indices(len(hosp), 1)
for name, m in [*((c, das.das_matrix(v[c])) for c in CONDITIONS), ("mean", das.mean_das_matrix(v, hosp))]:
    x = m.to_numpy()[np.triu_indices(len(m), 1)]
    x = x[np.isfinite(x)]
    print(f"{name:12s} hospitals {len(m):3d}  median {np.median(x):5.1f}  IQR {np.percentile(x, 25):5.1f} to {np.percentile(x, 75):5.1f}")
