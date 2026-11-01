"""Load and prepare the UCI Diabetes 130-US Hospitals data for SafeDischarge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "diabetic_data.csv"
SEED = 42

# Discharge outcomes where readmission is impossible or not relevant (death, hospice).
EXCLUDE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

DIABETES_MEDS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "n_med_changes", "prior_stays", "prior_readmit_30d",
]
CATEGORICAL = [
    "gender", "admission_type", "discharge_group", "admission_source", "diag_1_group",
    "diag_2_group", "diag_3_group", "max_glu_serum", "A1Cresult", "insulin",
    "metformin", "med_change", "on_diabetes_med", "specialty",
]
AUDIT = ["race", "gender", "age_group", "payer"]


def _icd9_group(code) -> str:
    """Group ICD-9 codes into the broad categories used by Strack et al. (2014)."""
    if pd.isna(code):
        return "missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "other"
    try:
        v = float(s)
    except ValueError:
        return "other"
    if 390 <= v <= 459 or int(v) == 785:
        return "circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "digestive"
    if int(v) == 250:
        return "diabetes"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def _admission_type(i: int) -> str:
    return {1: "emergency", 2: "urgent", 3: "elective", 7: "emergency"}.get(i, "other/unknown")


def _admission_source(i: int) -> str:
    if i == 7:
        return "emergency_room"
    if i in (1, 2, 3):
        return "referral"
    if i in (4, 5, 6, 10, 18, 22, 25, 26):
        return "transfer"
    return "other/unknown"


def _discharge_group(i: int) -> str:
    if i == 1:
        return "home"
    if i in (6, 8):
        return "home_with_care"
    if i in (3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30):
        return "care_facility"
    if i == 7:
        return "left_against_advice"
    return "other/unknown"


def _age_group(years: float) -> str:
    if years < 40:
        return "<40"
    if years < 60:
        return "40-59"
    if years < 80:
        return "60-79"
    return "80+"


def _add_history(d: pd.DataFrame) -> pd.DataFrame:
    """Add each patient's earlier stays in the dataset, using only stays before the current one.

    The data has no dates, so encounter_id order is used as time order (IDs are assigned
    sequentially). Computed before any exclusions, so stays ending in death or hospice still
    count as history. An earlier stay's 30-day readmission outcome is always known by the time
    of the current discharge: either the current stay is that readmission or it came later.
    """
    d = d.sort_values("encounter_id")
    g = d.groupby("patient_nbr")
    d["prior_stays"] = g.cumcount()
    readmit = (d["readmitted"] == "<30").astype(int)
    d["prior_readmit_30d"] = readmit.groupby(d["patient_nbr"]).cumsum() - readmit
    return d.sort_index()


def load_clean(path: Path = RAW) -> pd.DataFrame:
    """Return one row per eligible encounter with engineered features, target and audit columns."""
    d = pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)
    d = _add_history(d)
    d = d[~d["discharge_disposition_id"].isin(EXCLUDE_DISPOSITIONS)]
    d = d[d["gender"].isin(["Female", "Male"])].copy()

    out = pd.DataFrame(index=d.index)
    out["encounter_id"] = d["encounter_id"]
    out["patient_nbr"] = d["patient_nbr"]
    out["y"] = (d["readmitted"] == "<30").astype(int)

    out["age_years"] = d["age"].str.extract(r"\[(\d+)-").astype(float)[0] + 5
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
              "prior_stays", "prior_readmit_30d"]:
        out[c] = d[c].astype(float)
    changed = d[DIABETES_MEDS].isin(["Up", "Down"])
    out["n_med_changes"] = changed.sum(axis=1).astype(float)

    out["gender"] = d["gender"]
    out["admission_type"] = d["admission_type_id"].map(_admission_type)
    out["discharge_group"] = d["discharge_disposition_id"].map(_discharge_group)
    out["admission_source"] = d["admission_source_id"].map(_admission_source)
    for k in (1, 2, 3):
        out[f"diag_{k}_group"] = d[f"diag_{k}"].map(_icd9_group)
    out["max_glu_serum"] = d["max_glu_serum"].replace({"None": "not_measured", "": "not_measured"})
    out["A1Cresult"] = d["A1Cresult"].replace({"None": "not_measured", "": "not_measured"})
    out["insulin"] = d["insulin"]
    out["metformin"] = d["metformin"]
    out["med_change"] = d["change"].map({"Ch": "yes", "No": "no"})
    out["on_diabetes_med"] = d["diabetesMed"].str.lower()
    spec = d["medical_specialty"].fillna("Unknown")
    top = spec.value_counts().index[:10]
    out["specialty"] = spec.where(spec.isin(top), "Other")

    # Audit-only columns (race may optionally be used as a feature; payer never is).
    out["race"] = d["race"].fillna("Unknown")
    out["age_group"] = out["age_years"].map(_age_group)
    out["payer"] = d["payer_code"].fillna("Unknown")
    return out.reset_index(drop=True)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def patient_split(df: pd.DataFrame, seed: int = SEED) -> Split:
    """70/15/15 split by patient so no patient appears in more than one part."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, rest_idx = next(g1.split(df, groups=df["patient_nbr"]))
    rest = df.iloc[rest_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va_idx, te_idx = next(g2.split(rest, groups=rest["patient_nbr"]))
    return Split(df.iloc[tr_idx].reset_index(drop=True),
                 rest.iloc[va_idx].reset_index(drop=True),
                 rest.iloc[te_idx].reset_index(drop=True))


class Encoder:
    """One-hot encode categoricals and standardise numerics, with readable column names."""

    def __init__(self, use_race: bool = False):
        self.cats = CATEGORICAL + (["race"] if use_race else [])
        self.columns: list[str] = []
        self.mean = self.std = None

    def fit(self, df: pd.DataFrame) -> "Encoder":
        X = self._raw(df)
        self.columns = list(X.columns)
        self.mean = df[NUMERIC].mean()
        self.std = df[NUMERIC].std().replace(0, 1)
        return self

    def _raw(self, df: pd.DataFrame) -> pd.DataFrame:
        oh = pd.get_dummies(df[self.cats].astype(str), prefix_sep="=", dtype=float)
        return pd.concat([df[NUMERIC].astype(float), oh], axis=1)

    def transform(self, df: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        X = self._raw(df).reindex(columns=self.columns, fill_value=0.0)
        if scale:
            X[NUMERIC] = (X[NUMERIC] - self.mean) / self.std
        return X


def summary(df: pd.DataFrame) -> dict:
    return {
        "encounters": int(len(df)),
        "patients": int(df["patient_nbr"].nunique()),
        "readmit_30d_rate": round(float(df["y"].mean()), 4),
        "race_share": (df["race"].value_counts(normalize=True).round(4)).to_dict(),
    }


def dirichlet_hospitals(df: pd.DataFrame, n_hospitals: int = 5, alpha: float = 0.5,
                        seed: int = SEED) -> np.ndarray:
    """Assign each patient to a virtual hospital. Strata = race x age group; within each stratum,
    hospital shares are drawn from Dirichlet(alpha), giving each hospital a different case mix."""
    rng = np.random.default_rng(seed)
    pats = df.groupby("patient_nbr").first()[["race", "age_group"]]
    strata = pats["race"] + "|" + pats["age_group"]
    assign = pd.Series(0, index=pats.index)
    for _, idx in strata.groupby(strata).groups.items():
        p = rng.dirichlet([alpha] * n_hospitals)
        assign.loc[idx] = rng.choice(n_hospitals, size=len(idx), p=p)
    return df["patient_nbr"].map(assign).to_numpy()
