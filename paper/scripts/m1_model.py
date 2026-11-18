"""Prediction model: patient covariates, design columns and the logistic regression pipeline."""

from __future__ import annotations

import pandas as pd

C_M1 = 1e6
NUMERIC = ["apachescore", "agenum"]
CATS = {
    "gender": ["Male", "Female", "Unknown", "Other"],
    "ethnicity": ["Caucasian", "African American", "Other/Unknown", "Hispanic", "Asian", "Native American"],
    "unitstaytype": ["admit", "transfer", "stepdown/other", "readmit"],
    "unitadmitsource": ["Emergency Department", "Operating Room", "Floor", "Direct Admit", "Recovery Room",
                        "Other Hospital", "Acute Care/Floor", "Step-Down Unit (SDU)", "ICU to SDU", "Other ICU",
                        "PACU", "ICU", "Chest Pain Center", "Observation", "Other"],
}


def design_columns() -> list[str]:
    return NUMERIC + [f"{c}={v}" for c, vals in CATS.items() for v in vals]


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric inputs plus one-hot columns for the fixed categories."""
    out = df[NUMERIC].astype(float).copy()
    for c, vals in CATS.items():
        unknown = set(df[c].dropna().unique()) - set(vals)
        assert not unknown, f"{c}: values outside the eICU categories: {sorted(unknown)}"
        for v in vals:
            out[f"{c}={v}"] = (df[c] == v).astype(float)
    return out
