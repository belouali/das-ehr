"""DAS on simulated data: three hospitals, one condition. Run: python examples/quickstart.py"""
import numpy as np
import pandas as pd

import das

rng = np.random.default_rng(0)
rows = []
# Hospitals A and B record diabetes the same way; hospital C records it mostly as diagnoses and rarely orders insulin.
for hospital, (p_dx, p_hx, p_rx) in {"A": (0.6, 0.3, 0.5), "B": (0.6, 0.3, 0.5), "C": (0.7, 0.05, 0.0)}.items():
    for _ in range(1000):
        rows.append({"hospital_id": hospital, "diabetes_D": rng.random() < p_dx,
                     "diabetes_H": rng.random() < p_hx, "diabetes_M": rng.random() < p_rx})
stays = pd.DataFrame(rows)

vectors = das.documentation_vectors(stays, ["diabetes"], min_stays=100)
print("Documentation vectors (share of evidenced stays in each Venn lobe):")
print(vectors["diabetes"].round(2), "\n")

print("DAS between hospitals (degrees):")
print(das.mean_das_matrix(vectors).round(1), "\n")

noise = das.split_half_noise(stays, ["diabetes"], vectors, splits=100)
print("Pairs that differ beyond split-half noise:")
print(das.beyond_noise(das.mean_das_matrix(vectors), noise), "\n")

ranking = das.rank_candidates({"diabetes": vectors["diabetes"].loc["A"]}, {"diabetes": vectors["diabetes"].drop(index="A")})
print("Candidate training hospitals for A, nearest first:", das.nearest(ranking, k=2))
