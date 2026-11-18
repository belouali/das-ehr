"""Shared helpers for the condition definitions."""
import re
from pathlib import Path
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(HERE.parent / "configs" / "external.yaml"))
DEF = yaml.safe_load(open(HERE.parent / "configs" / "definitions_v7.yaml"))
PH = yaml.safe_load(open(HERE.parent / CFG["paths"]["phenotypes"]))
CONDITIONS = ["diabetes", "hypertension", "chf", "af"]

import os
VARIANT = os.environ.get("EHRDOC_VARIANT", "")
if VARIANT:
    _v = yaml.safe_load(open(HERE.parent / "configs" / "definitions_v7_variants.yaml"))["variants"][VARIANT]
    for _c in CONDITIONS:
        _o, _m = _v.get(_c) or {}, DEF["conditions"][_c]["medication"]
        for _k, _val in (_o.get("set") or {}).items():
            if _val is None: _m.pop(_k, None)
            else: _m[_k] = _val
        _m["remove"] = list(_m.get("remove", [])) + list(_o.get("remove_extra", []))
        _m["add"] = [x for x in list(_m.get("add", [])) + list(_o.get("add_extra", [])) if x not in _m["remove"]]
        if _o.get("combination_strict") and _m.get("combination"): _m["combination"]["strict"] = True


def variant_suffix():
    return f"_{VARIANT}" if VARIANT else ""


def med_terms(c):
    m = DEF["conditions"][c]["medication"]
    base = [t for t in PH[c]["medication"]["include"] if t not in m.get("remove", [])]
    return base + [t for t in m.get("add", []) if t not in base]


def med_mask(values: pd.Series, c: str) -> pd.Series:
    """A term counts only at the start of a word; names matching the condition's name_exclude never count."""
    v = values.astype(str)
    m = v.str.contains(r"(?<![A-Za-z])(?:" + "|".join(re.escape(t) for t in med_terms(c)) + ")", case=False, na=False, regex=True)
    ex = DEF["conditions"][c]["medication"].get("name_exclude")
    return m & ~v.str.contains(ex, case=False, na=False, regex=True) if ex else m


def decide(text: pd.Series, rule: dict, relevant: pd.Series = None) -> pd.DataFrame:
    """status and reason for each menu text. An exclusion applies to entries an include matched (or that `relevant`
    marks, e.g. counted by the previous rule). `notes` give the reason for entries deliberately left uncounted."""
    t = text.astype(str).str.lower()
    status = pd.Series("not counted", index=text.index); reason = pd.Series("", index=text.index)
    for inc in rule.get("include", []):
        m = t.str.contains(inc["pattern"], regex=True) & (status == "not counted")
        status[m] = "counted"; reason[m] = inc["reason"]
    scope = (status == "counted") | (relevant if relevant is not None else False)
    for exc in rule.get("exclude", []) or []:
        m = t.str.contains(exc["pattern"], regex=True) & scope
        reason[m] = exc["reason"]; status[m] = "excluded"
    for note in rule.get("notes", []) or []:
        m = t.str.contains(note["pattern"], regex=True) & (status == "not counted")
        reason[m] = note["reason"]
    return pd.DataFrame({"status": status, "reason": reason})


def icd_match(code: str, version: int, rule: dict) -> bool:
    code = str(code).replace(".", "").strip().upper()
    return bool(re.search(rule["icd10" if int(version) == 10 else "icd9"], code))
