"""From stay-level evidence flags to one seven-element documentation vector per hospital and condition.

Input: one row per stay, a hospital identifier, and for each condition three booleans saying whether the stay has
evidence of the condition in diagnoses (D), past history (H) and medications (M). A stay with evidence in at least one
source falls into exactly one of the seven Venn lobes below. The vector is the share of a hospital's evidenced stays
in each lobe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCES = ("D", "H", "M")
# Lobe order used in the paper: all three sources first, single sources last.
LOBES = ("D+H+M", "D+H", "D+M", "D only", "H+M", "H only", "M only")


def lobe_codes(diagnosis, history, medication) -> np.ndarray:
    """Lobe index (0-6, in LOBES order) for each stay; -1 for a stay with no evidence in any source."""
    d, h, m = (np.asarray(x, dtype=bool) for x in (diagnosis, history, medication))
    code = d.astype(int) * 4 + h.astype(int) * 2 + m.astype(int)
    return np.where(code > 0, 7 - code, -1)


def lobe_counts(stays: pd.DataFrame, condition: str, hospital: str = "hospital_id",
                columns: tuple[str, str, str] | None = None) -> pd.DataFrame:
    """Hospitals x 7 lobe counts for one condition.

    By default the evidence columns are named "<condition>_D", "<condition>_H" and "<condition>_M"; pass
    `columns` to use other names.
    """
    cols = columns or tuple(f"{condition}_{s}" for s in SOURCES)
    codes = lobe_codes(*(stays[c] for c in cols))
    keep = codes >= 0
    counts = pd.crosstab(stays.loc[keep, hospital].to_numpy(), codes[keep])
    counts = counts.reindex(columns=range(7), fill_value=0)
    counts.columns = list(LOBES)
    counts.index.name = hospital
    return counts


def documentation_vectors(stays: pd.DataFrame, conditions, hospital: str = "hospital_id",
                          min_stays: int = 100) -> dict[str, pd.DataFrame]:
    """Documentation vector (lobe shares summing to 1) per hospital, for each condition.

    A hospital gets a vector for a condition only if at least `min_stays` of its stays have evidence of it
    (100 in the paper's documentation-variation analyses, 30 for training-site selection).
    """
    out = {}
    for c in conditions:
        counts = lobe_counts(stays, c, hospital)
        counts = counts[counts.sum(axis=1) >= min_stays]
        out[c] = counts.div(counts.sum(axis=1), axis=0)
    return out


def vectors_from_counts(counts: pd.DataFrame, min_stays: int = 0) -> pd.DataFrame:
    """Documentation vectors from a hospitals x 7 table of lobe counts (for sites that share counts only)."""
    counts = counts[counts.sum(axis=1) >= min_stays]
    return counts.div(counts.sum(axis=1), axis=0)
