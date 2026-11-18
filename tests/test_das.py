import numpy as np
import pandas as pd
import pytest

import das


def test_lobe_codes_order():
    # D+H+M, D+H, D+M, D only, H+M, H only, M only, none
    d = [1, 1, 1, 1, 0, 0, 0, 0]; h = [1, 1, 0, 0, 1, 1, 0, 0]; m = [1, 0, 1, 0, 1, 0, 1, 0]
    assert das.lobe_codes(d, h, m).tolist() == [0, 1, 2, 3, 4, 5, 6, -1]


def test_angle_bounds_and_scale():
    assert das.das([1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]) == pytest.approx(0)
    assert das.das([1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]) == pytest.approx(90)
    a, b = np.array([5, 1, 2, 3, 0, 1, 4]), np.array([1, 2, 3, 1, 1, 0, 2])
    assert das.das(a, b) == pytest.approx(das.das(10 * a, b))   # counts or shares give the same angle


def _toy(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for h, p in [("A", (0.6, 0.3, 0.5)), ("B", (0.6, 0.3, 0.5)), ("C", (0.7, 0.05, 0.0))]:
        for _ in range(2000):
            rows.append(dict(hospital_id=h, dm_D=rng.random() < p[0], dm_H=rng.random() < p[1], dm_M=rng.random() < p[2]))
    return pd.DataFrame(rows)


def test_vectors_matrix_and_ranking():
    stays = _toy()
    v = das.documentation_vectors(stays, ["dm"], min_stays=100)
    assert np.allclose(v["dm"].sum(axis=1), 1)
    m = das.mean_das_matrix(v)
    assert m.loc["A", "B"] < 10 < 40 < m.loc["A", "C"]   # A and B document alike, C does not
    r = das.rank_candidates({"dm": v["dm"].loc["A"]}, {"dm": v["dm"].drop(index="A")})
    assert das.nearest(r, k=1) == ["B"] and das.farthest(r, k=1) == ["C"]


def test_noise_and_beyond():
    stays = _toy()
    v = das.documentation_vectors(stays, ["dm"], min_stays=100)
    nf = das.split_half_noise(stays, ["dm"], v, splits=50, seed=0)
    b = das.beyond_noise(das.mean_das_matrix(v), nf)
    assert not b.loc["A", "B"] and b.loc["A", "C"]
