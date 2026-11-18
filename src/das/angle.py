"""DAS: the angle between two hospitals' documentation vectors, in degrees (0 = same spread, 90 = no shared lobe)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def das(a, b) -> float:
    """DAS between two documentation vectors (or two count vectors: the angle does not depend on scale)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def das_matrix(vectors: pd.DataFrame) -> pd.DataFrame:
    """Pairwise DAS between all hospitals (rows of `vectors`)."""
    v = vectors.to_numpy(float)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    m = np.degrees(np.arccos(np.clip(u @ u.T, -1.0, 1.0)))
    np.fill_diagonal(m, 0.0)
    return pd.DataFrame(m, index=vectors.index, columns=vectors.index)


def mean_das_matrix(vectors: dict[str, pd.DataFrame], hospitals=None) -> pd.DataFrame:
    """Mean DAS between hospitals: for each pair, the mean of the condition DAS over the conditions both have."""
    hospitals = pd.Index(hospitals if hospitals is not None else sorted(set().union(*[v.index for v in vectors.values()])))
    stack = np.stack([das_matrix(v).reindex(index=hospitals, columns=hospitals).to_numpy() for v in vectors.values()])
    m = _nanmean(stack)  # NaN where a pair shares no condition
    return pd.DataFrame(m, index=hospitals, columns=hospitals)


def _nanmean(stack):
    cnt = np.isfinite(stack).sum(0)
    tot = np.nansum(stack, axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
