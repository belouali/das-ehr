"""Choosing training hospitals for a target hospital by DAS, without the target's outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .angle import das


def rank_candidates(target: dict, candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """DAS of every candidate hospital to the target, per condition and their mean (the main measure).

    target: {condition: documentation (or count) vector of the target}; candidates: {condition: hospitals x 7 vectors}.
    Sorted from nearest to farthest by mean DAS.
    """
    cols = {}
    for c, tv in target.items():
        if c in candidates and tv is not None:
            v = candidates[c]
            cols[c] = pd.Series([das(row, tv) for row in v.to_numpy(float)], index=v.index)
    out = pd.DataFrame(cols)
    out["mean"] = out.mean(axis=1, skipna=True)
    return out.sort_values("mean", kind="stable")


def nearest(ranking: pd.DataFrame, k: int = 10, measure: str = "mean") -> list:
    """The k candidates with the smallest DAS to the target."""
    return ranking[measure].dropna().sort_values(kind="stable").index[:k].tolist()


def farthest(ranking: pd.DataFrame, k: int = 10, measure: str = "mean") -> list:
    """The k candidates with the largest DAS to the target."""
    return ranking[measure].dropna().sort_values(ascending=False, kind="stable").index[:k].tolist()
