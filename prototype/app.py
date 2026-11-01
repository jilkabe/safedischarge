"""SafeDischarge nurse dashboard (Streamlit prototype).

Run:  uv run streamlit run app.py
Uses held-out test patients as a simulated "today's discharge list".
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from safedischarge import metrics as M  # noqa: E402
from safedischarge.data import NUMERIC, load_clean, patient_split  # noqa: E402

st.set_page_config(page_title="SafeDischarge", layout="wide")

# Plain-language reason and suggested follow-up for each feature group.
REASONS = {
    "number_inpatient": ("{v:.0f} inpatient admission(s) in the past year", "Refer to case management"),
    "number_emergency": ("{v:.0f} emergency visit(s) in the past year", "Book clinic follow-up within 7 days"),
    "prior_stays": ("{v:.0f} earlier stay(s) in this hospital system", "Refer to case management"),
    "prior_readmit_30d": ("{v:.0f} earlier 30-day readmission(s)", "Refer to case management; review why earlier discharges failed"),
    "number_outpatient": ("{v:.0f} outpatient visit(s) in the past year", "Coordinate with outpatient team"),
    "discharge_group": ("Discharge destination: {v}", "Confirm home support / hand-over to receiving facility"),
    "number_diagnoses": ("{v:.0f} recorded diagnoses", "Pharmacist medication reconciliation"),
    "num_medications": ("{v:.0f} medications during stay", "Pharmacist medication reconciliation"),
    "time_in_hospital": ("Length of stay {v:.0f} days", "Nurse phone call within 72 hours"),
    "age_years": ("Age group {age}", "Check caregiver support at home"),
    "num_lab_procedures": ("{v:.0f} lab tests during stay", "Review outstanding results before discharge"),
    "num_procedures": ("{v:.0f} procedures during stay", "Review procedure aftercare plan"),
    "A1Cresult": ("HbA1c: {v}", "Order HbA1c / diabetes clinic review"),
    "max_glu_serum": ("Max glucose serum: {v}", "Diabetes clinic review"),
    "insulin": ("Insulin: {v}", "Diabetes nurse educator (insulin teaching)"),
    "metformin": ("Metformin: {v}", "Diabetes nurse educator"),
    "med_change": ("Diabetes medication changed: {v}", "Diabetes education on new regimen"),
    "n_med_changes": ("{v:.0f} diabetes medication change(s)", "Diabetes education on new regimen"),
    "on_diabetes_med": ("On diabetes medication: {v}", "Diabetes clinic review"),
    "admission_type": ("Admission type: {v}", "Review in discharge huddle"),
    "admission_source": ("Admitted via: {v}", "Review in discharge huddle"),
    "specialty": ("Admitting specialty: {v}", "Specialty follow-up appointment"),
    "gender": ("Sex: {v}", "Review in discharge huddle"),
}
for k in (1, 2, 3):
    REASONS[f"diag_{k}_group"] = (("Primary" if k == 1 else "Secondary") + " diagnosis group: {v}",
                                  "Disease-specific follow-up appointment")


@st.cache_resource
def load():
    art = joblib.load(ROOT / "results" / "artifacts.joblib")
    sp = patient_split(load_clean())
    test = sp.test
    enc = art["encoder"]
    X = enc.transform(test)
    raw = art["model"].predict_proba(X)[:, 1]          # continuous score, used for ranking
    risk = art["calibrator"].predict(raw)              # calibrated probability, shown to nurses
    fair = json.loads((ROOT / "results" / "fairness.json").read_text())
    shift = fair["group_thresholds"]["score_shift"]
    central = json.loads((ROOT / "results" / "central.json").read_text())
    fed_path = ROOT / "results" / "federated.json"
    fed = json.loads(fed_path.read_text()) if fed_path.exists() else None
    return art, test, X, raw, risk, shift, fair, central, fed


art, test, X, raw, risk, shift, fair, central, fed = load()
group_of = {c: c.split("=")[0] for c in X.columns}

with st.sidebar:
    st.title("SafeDischarge")
    st.caption("30-day readmission risk for diabetic inpatients · prototype on UCI Diabetes 130-US data")
    day = st.number_input("Simulated day", 1, 200, 1)
    n_today = st.slider("Discharges today", 20, 100, 50, 5)
    capacity = st.slider("Follow-up capacity (share of discharges)", 0.05, 0.5, M.CAPACITY, 0.05)
    fair_mode = st.toggle("Age-fair ranking (approved group thresholds)", value=True)
    st.caption("Race is **not** a model input. It is used only for fairness auditing.")


def ranking_score(r, ages):
    return r - np.vectorize(shift.get)(ages) if fair_mode else r


tab1, tab2, tab3 = st.tabs(["Today's discharges", "Fairness monitor", "Model card"])

with tab1:
    rng = np.random.default_rng(int(day))
    idx = rng.choice(len(test), n_today, replace=False)
    today = test.iloc[idx].copy()
    today["risk"] = risk[idx]
    today["rank_score"] = ranking_score(raw[idx], today["age_group"].to_numpy())
    k = int(round(capacity * n_today))
    today = today.sort_values("rank_score", ascending=False)
    today["band"] = ["High: follow-up" if i < k else ("Medium" if i < 2 * k else "Low") for i in range(n_today)]

    sv = art["explainer"].shap_values(X.iloc[today.index])
    sv = sv[1] if isinstance(sv, list) else sv
    contrib = pd.DataFrame(sv, columns=X.columns, index=today.index).T.groupby(group_of).sum().T

    def reasons(row_idx, n=3):
        row, c = test.loc[row_idx], contrib.loc[row_idx]
        out = []
        for feat in c.sort_values(ascending=False).index[:n]:
            if c[feat] <= 0 or feat not in REASONS:
                continue
            text, action = REASONS[feat]
            v = row["age_group"] if feat == "age_years" else row[feat]
            out.append((text.format(v=v, age=row["age_group"]) if feat in NUMERIC or feat == "age_years"
                        else text.format(v=str(v).replace("_", " ")), action, float(c[feat])))
        return out

    c1, c2, c3 = st.columns(3)
    c1.metric("Discharges today", n_today)
    c2.metric("Follow-up slots", k)
    c3.metric("Mean risk of flagged patients", f"{today['risk'].iloc[:k].mean():.0%}" if k else "–")

    table = pd.DataFrame({
        "Encounter": today["encounter_id"].astype(str),
        "Band": today["band"],
        "30-day risk": (today["risk"] * 100).round(0).astype(int).astype(str) + "%",
        "Age": today["age_group"], "Sex": today["gender"],
        "Top reason": [(reasons(i, 1) or [("—", "", 0)])[0][0] for i in today.index],
    })
    st.dataframe(table, hide_index=True, use_container_width=True, height=380)

    st.subheader("Flagged patients: reasons and suggested actions")
    for i in today.index[:k]:
        row = today.loc[i]
        with st.expander(f"Encounter {row['encounter_id']} · {row['age_group']} {row['gender']} · "
                         f"risk {row['risk']:.0%}"):
            for text, action, val in reasons(i):
                st.markdown(f"- **{text}** → _{action}_  (SHAP +{val:.2f})")
            st.selectbox("Nurse decision", ["Accept follow-up plan", "Override: not needed", "Override: other plan"],
                         key=f"dec{i}")

with tab2:
    st.subheader("Recall at capacity by group (full held-out test set)")
    st.caption("Recall = share of patients actually readmitted within 30 days who were flagged. "
               f"Groups with fewer than {M.MIN_POS} readmissions are shown but excluded from the gap.")
    rs = ranking_score(raw, test["age_group"].to_numpy())
    flagged = M.flag_top_k(rs, capacity)
    cols = st.columns(3)
    for col, attr in zip(cols, ["age_group", "race", "gender"]):
        tab = M.group_recall(test["y"], flagged, test[attr].to_numpy())
        p = M.gap_pvalue(test["y"], flagged, test[attr].to_numpy(), n_perm=500)
        col.markdown(f"**{attr.replace('_', ' ').title()}**: gap {M.recall_gap(tab):.1%} "
                     f"(permutation p = {p:.3f})")
        col.dataframe(tab[["group", "n_pos", "recall", "ci_low", "ci_high", "reportable"]].round(3),
                      hide_index=True, use_container_width=True)
    st.info(f"Release gate: no statistically significant recall gap (permutation p ≥ 0.05) for any protected attribute. "
            f"Current mode: {'age-fair ranking' if fair_mode else 'baseline ranking'}.")

with tab3:
    st.subheader("Model card")
    mc = central["models"]["gradient_boosting"]
    st.markdown(f"""
- **Intended use:** prioritise diabetic inpatients for post-discharge follow-up. Decision support only; the nurse decides.
- **Model:** gradient-boosted trees (scikit-learn), isotonic calibration, age-group threshold adjustment.
- **Data:** UCI Diabetes 130-US Hospitals 1999–2008; {central['data']['encounters']:,} encounters after excluding deaths/hospice; patient-level 70/15/15 split.
- **Performance (test):** AUROC {mc['auroc']:.3f}, AUPRC {mc['auprc']:.3f}, recall at 20% capacity {fair['group_thresholds']['recall_at_capacity']:.1%} (age-fair) vs {fair['baseline']['recall_at_capacity']:.1%} (baseline).
- **Fairness:** age recall gap {fair['baseline']['age_group_gap']:.1%} → {fair['group_thresholds']['age_group_gap']:.1%} with group thresholds; race gap {fair['group_thresholds']['race_gap']:.1%}; sex gap {fair['group_thresholds']['gender_gap']:.1%}.
- **Limitations:** old US data; modest discrimination; small race groups (Asian, Other, Unknown) too small to audit reliably.
""")
    if fed:
        st.markdown("**Federated + differential privacy experiment**")
        st.dataframe(pd.DataFrame(fed["settings"]).T[["auroc", "recall_at_capacity", "age_group_gap",
                                                       "membership_auc"]], use_container_width=True)
