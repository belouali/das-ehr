"""Split-half noise: how far a hospital's DAS moves when its own stays are split in two at random."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .angle import das
from .vectors import lobe_codes


def split_half_noise(stays: pd.DataFrame, conditions, vectors: dict[str, pd.DataFrame], hospital: str = "hospital_id",
                     splits: int = 200, seed: int = 0) -> pd.DataFrame:
    """For each hospital and condition (and for the mean over the hospital's conditions): the mean and 95th
    percentile of DAS between two random halves of its evidenced stays, over `splits` random splits."""
    rng = np.random.default_rng(seed)
    hosp = sorted(set().union(*[v.index for v in vectors.values()]))
    rows, mean_draws, n_cond = [], {h: np.zeros(splits) for h in hosp}, {h: 0 for h in hosp}
    for c in conditions:
        codes = lobe_codes(*(stays[f"{c}_{s}"] for s in ("D", "H", "M")))
        keep = codes >= 0
        by_h = pd.Series(codes[keep]).groupby(stays[hospital].to_numpy()[keep]).apply(np.asarray)
        for h in hosp:
            if h not in vectors[c].index:
                continue
            x = by_h[h]
            half = len(x) // 2
            d = np.empty(splits)
            for s in range(splits):
                p = rng.permutation(x)
                d[s] = das(np.bincount(p[:half], minlength=7), np.bincount(p[half:2 * half], minlength=7))
            rows.append(dict(hospital=h, condition=c, stays=len(x), split_half_mean=d.mean(), split_half_q95=np.percentile(d, 95)))
            mean_draws[h] += d
            n_cond[h] += 1
    for h in hosp:
        if n_cond[h]:
            d = mean_draws[h] / n_cond[h]
            rows.append(dict(hospital=h, condition="mean", stays=np.nan, split_half_mean=d.mean(), split_half_q95=np.percentile(d, 95)))
    return pd.DataFrame(rows)


def beyond_noise(das_between: pd.DataFrame, noise: pd.DataFrame, condition: str = "mean") -> pd.DataFrame:
    """True where a pair's DAS exceeds the larger of the two hospitals' 95th percentiles of split-half DAS."""
    q = noise[noise.condition == condition].set_index("hospital").split_half_q95.reindex(das_between.index).to_numpy()
    thr = np.maximum.outer(q, q)
    out = das_between.to_numpy() > thr
    np.fill_diagonal(out, False)
    return pd.DataFrame(out, index=das_between.index, columns=das_between.columns)
