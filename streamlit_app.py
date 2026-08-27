from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import logit
from sklearn.experimental import enable_iterative_imputer  # noqa: F401


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "palf_he_lr_model.joblib"


st.set_page_config(
    page_title="Pediatric PALF–HE Research Calculator",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #102A43;
        --muted: #5C6F82;
        --blue: #2166AC;
        --blue-mid: #4393C3;
        --blue-pale: #D1E5F0;
        --paper: #F7FAFC;
        --line: #C7D7E5;
        --warning: #B2182B;
    }

    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247, 250, 252, 0.92); }
    [data-testid="stDeployButton"], [data-testid="stToolbar"], #MainMenu { display: none; }
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] { display: none; }
    [data-testid="stSidebar"] { background: #EEF5F9; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.72rem; }
    [data-testid="stSidebar"] .stNumberInput { margin-bottom: -0.18rem; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--ink); }
    .block-container { max-width: 1180px; padding-top: 2.25rem; padding-bottom: 4rem; }

    h1, h2, h3 { color: var(--ink); }
    h1 { font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.03em; }

    .eyebrow {
        color: var(--blue);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .hero-title {
        font-family: Georgia, 'Times New Roman', serif;
        color: var(--ink);
        font-size: clamp(2.2rem, 5vw, 4.2rem);
        line-height: 0.98;
        letter-spacing: -0.045em;
        margin: 0;
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 780px;
        line-height: 1.65;
        margin-top: 1rem;
    }
    .time-ribbon {
        display: grid;
        grid-template-columns: auto minmax(70px, 1fr) auto minmax(70px, 1fr) auto;
        align-items: center;
        gap: 0.65rem;
        border: 1px solid var(--line);
        background: white;
        padding: 0.9rem 1.05rem;
        margin: 1.55rem 0 1.75rem;
        border-radius: 8px;
    }
    .time-node { color: var(--ink); font-weight: 750; font-size: 0.86rem; white-space: nowrap; }
    .time-line { height: 2px; background: var(--blue-mid); position: relative; }
    .time-line::after {
        content: '';
        position: absolute;
        right: -1px;
        top: -3px;
        width: 0;
        height: 0;
        border-top: 4px solid transparent;
        border-bottom: 4px solid transparent;
        border-left: 7px solid var(--blue-mid);
    }

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        margin: 1.2rem 0 2rem;
    }
    .metric-item { padding: 0.95rem 1.1rem; border-right: 1px solid var(--line); }
    .metric-item:last-child { border-right: none; }
    .metric-value { color: var(--ink); font-size: 1.35rem; font-weight: 800; font-variant-numeric: tabular-nums; }
    .metric-label { color: var(--muted); font-size: 0.78rem; margin-top: 0.15rem; }

    .section-kicker { color: var(--blue); font-weight: 800; font-size: 0.8rem; letter-spacing: 0.09em; text-transform: uppercase; }
    .section-copy { color: var(--muted); margin: -0.3rem 0 1.1rem; }

    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 3rem;
        background: var(--blue);
        color: white;
        border: 1px solid var(--blue);
        border-radius: 6px;
        font-weight: 800;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #174D82;
        border-color: #174D82;
        color: white;
    }

    .result-panel {
        background: white;
        border: 1px solid var(--line);
        border-top: 5px solid var(--blue);
        border-radius: 8px;
        padding: 1.35rem 1.45rem 1.2rem;
        margin: 0.5rem 0 1rem;
    }
    .result-label { color: var(--muted); font-size: 0.82rem; font-weight: 750; text-transform: uppercase; letter-spacing: 0.09em; }
    .result-number { color: var(--ink); font-size: clamp(2.8rem, 7vw, 5.3rem); line-height: 1; font-weight: 850; letter-spacing: -0.055em; font-variant-numeric: tabular-nums; margin: 0.35rem 0 1rem; }
    .ready-panel { min-height: 205px; display: flex; flex-direction: column; justify-content: center; }
    .ready-title { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; font-size: 1.8rem; font-weight: 750; margin: 0.45rem 0 0.2rem; }
    .prob-track { height: 12px; background: #E8EFF4; border-radius: 999px; position: relative; overflow: visible; }
    .prob-fill { height: 12px; background: linear-gradient(90deg, var(--blue-mid), var(--blue)); border-radius: 999px; }
    .cohort-marker { position: absolute; top: -5px; width: 2px; height: 22px; background: var(--warning); }
    .cohort-note { color: var(--muted); font-size: 0.78rem; margin-top: 0.65rem; }
    .result-interpretation { color: var(--ink); line-height: 1.55; margin-top: 1rem; }

    .disclaimer {
        border-left: 4px solid var(--warning);
        background: #FFF8F7;
        padding: 0.9rem 1rem;
        color: #5E2527;
        font-size: 0.9rem;
        line-height: 1.55;
        margin: 1.1rem 0;
    }
    .small-note { color: var(--muted); font-size: 0.82rem; line-height: 1.55; }

    @media (max-width: 760px) {
        .block-container { padding-top: 1.25rem; }
        .time-ribbon { grid-template-columns: 1fr; gap: 0.4rem; }
        .time-line { width: 2px; height: 24px; margin-left: 0.35rem; }
        .time-line::after { right: -3px; top: auto; bottom: -1px; transform: rotate(90deg); }
        .time-node { white-space: normal; }
        .metric-strip { grid-template-columns: 1fr; }
        .metric-item { border-right: none; border-bottom: 1px solid var(--line); }
        .metric-item:last-child { border-bottom: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


def predict_probability(model: dict, values: dict[str, float]) -> tuple[float, float]:
    frame = pd.DataFrame([values], columns=model["features"])
    raw_probability = float(model["estimator"].predict_proba(frame)[0, 1])
    log_odds = logit(np.clip(raw_probability, 1e-6, 1 - 1e-6))
    calibrated_probability = float(
        model["calibrator"].predict_proba(np.array([[log_odds]]))[0, 1]
    )
    return raw_probability, calibrated_probability


def outside_training_range(model: dict, values: dict[str, float]) -> list[str]:
    warnings = []
    for feature, value in values.items():
        limits = model["observed_ranges"][feature]
        if value < limits["min"] or value > limits["max"]:
            label = model["feature_labels"][feature]
            warnings.append(
                f"{label}: {value:g} is outside the observed training range "
                f"({limits['min']:g}–{limits['max']:g} {model['units'][feature]})."
            )
    return warnings


try:
    model = load_model()
except Exception as exc:
    st.error(f"The prediction model could not be loaded: {exc}")
    st.stop()


if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.markdown("### Admission measurements")
    st.caption("Enter the first available laboratory values from the index admission.")

    with st.form("prediction_form"):
        inr = st.number_input(
            "INR",
            min_value=0.1,
            max_value=30.0,
            value=None,
            step=0.01,
            format="%.2f",
            placeholder="Enter value",
            help="International normalized ratio; dimensionless.",
        )
        tc = st.number_input(
            "Total cholesterol (mmol/L)",
            min_value=0.01,
            max_value=25.0,
            value=None,
            step=0.01,
            format="%.2f",
            placeholder="Enter value",
        )
        tbil = st.number_input(
            "Total bilirubin (μmol/L)",
            min_value=0.0,
            max_value=3000.0,
            value=None,
            step=1.0,
            format="%.1f",
            placeholder="Enter value",
        )
        alt = st.number_input(
            "Alanine aminotransferase (U/L)",
            min_value=0.0,
            max_value=50000.0,
            value=None,
            step=10.0,
            format="%.0f",
            placeholder="Enter value",
        )
        phos = st.number_input(
            "Phosphorus (mmol/L)",
            min_value=0.01,
            max_value=15.0,
            value=None,
            step=0.01,
            format="%.2f",
            placeholder="Enter value",
        )
        che = st.number_input(
            "Cholinesterase (U/L)",
            min_value=0.0,
            max_value=30000.0,
            value=None,
            step=10.0,
            format="%.0f",
            placeholder="Enter value",
        )
        submitted = st.form_submit_button("Calculate estimated probability")

    st.markdown("---")
    with st.expander("Study context", expanded=False):
        st.markdown(
            """
            **Population**  
            Children with acute hepatic dysfunction who fulfilled PALF criteria during the index hospitalization and had no HE at admission.

            **Outcome**  
            New-onset HE during hospitalization.
            """
        )
        st.caption(model["model_version"] + " · Single-center internal validation")
    st.markdown(
        '<div class="small-note">No patient identifiers are requested. Calculation history is stored only in the current browser session.</div>',
        unsafe_allow_html=True,
    )


st.markdown('<div class="eyebrow">Research calculator · logistic regression</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Pediatric PALF–HE<br>risk estimation</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-subtitle">
    Estimate the probability of new-onset hepatic encephalopathy using six routinely available admission laboratory measurements. The calculator mirrors the final model reported in the revised study.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="time-ribbon">
        <div class="time-node">Index admission</div><div class="time-line"></div>
        <div class="time-node">Six admission measurements</div><div class="time-line"></div>
        <div class="time-node">New-onset HE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

validation = model["internal_validation"]
st.markdown(
    f"""
    <div class="metric-strip">
        <div class="metric-item"><div class="metric-value">270 / 43</div><div class="metric-label">Children / incident HE events</div></div>
        <div class="metric-item"><div class="metric-value">{validation['AUROC']:.3f}</div><div class="metric-label">Internally validated AUROC</div></div>
        <div class="metric-item"><div class="metric-value">{validation['AUPRC']:.3f}</div><div class="metric-label">Internally validated AUPRC</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_values = {
    "INR": inr,
    "TC": tc,
    "TBIL": tbil,
    "ALT": alt,
    "PHOS": phos,
    "CHE": che,
}
values = (
    {name: float(value) for name, value in input_values.items()}
    if all(value is not None for value in input_values.values())
    else None
)

st.markdown('<div class="section-kicker">Model output</div>', unsafe_allow_html=True)
st.subheader("Estimated new-onset HE probability")

if submitted and values is not None:
    raw_probability, probability = predict_probability(model, values)
    event_rate = float(model["cohort"]["event_rate"])
    relation = "above" if probability >= event_rate else "below"
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-label">Calibrated probability</div>
            <div class="result-number">{probability * 100:.1f}%</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{probability * 100:.2f}%"></div>
                <div class="cohort-marker" style="left:{event_rate * 100:.2f}%"></div>
            </div>
            <div class="cohort-note">Red marker: study-cohort HE incidence ({event_rate * 100:.1f}%).</div>
            <div class="result-interpretation">This estimate is <strong>{relation}</strong> the overall event rate observed in the study cohort. This comparison is descriptive and is not a treatment threshold.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    range_messages = outside_training_range(model, values)
    if range_messages:
        st.warning(
            "One or more inputs fall outside the observed training range. The estimate involves extrapolation:\n\n"
            + "\n".join(f"- {message}" for message in range_messages)
        )

    st.session_state.history.insert(
        0,
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Estimated probability": f"{probability * 100:.1f}%",
            "INR": inr,
            "TC": tc,
            "TBIL": tbil,
            "ALT": alt,
            "PHOS": phos,
            "CHE": che,
        },
    )
else:
    if submitted:
        st.warning("Please enter all six admission measurements before calculation.")
    st.markdown(
        """
        <div class="result-panel ready-panel">
            <div class="result-label">Ready to calculate</div>
            <div class="ready-title">Six measurements, one calibrated estimate</div>
            <div class="result-interpretation">Enter INR, total cholesterol, total bilirubin, ALT, phosphorus, and cholinesterase in the left panel, then calculate the estimated probability.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="disclaimer">
    <strong>Research use only.</strong> This estimate does not diagnose hepatic encephalopathy, recommend treatment, or replace clinical assessment. The model was developed at one center and requires external validation before clinical implementation.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("---")
st.subheader("Session calculation history")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    if st.button("Clear session history", type="secondary"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("No calculations in this session.")


with st.expander("Model and study details"):
    st.markdown(
        f"""
        - **Algorithm:** Logistic regression with balanced class weights
        - **Predictors:** INR, total cholesterol, total bilirubin, ALT, phosphorus, and cholinesterase
        - **Preprocessing:** Iterative imputation and standardization fitted within model development
        - **Probability calibration:** Platt calibration
        - **Development cohort:** {model['cohort']['n']} children; {model['cohort']['events']} incident HE events
        - **Internal validation:** AUROC {validation['AUROC']:.3f} (95% CI {validation['AUROC_95CI'][0]:.3f}–{validation['AUROC_95CI'][1]:.3f}); AUPRC {validation['AUPRC']:.3f} (95% CI {validation['AUPRC_95CI'][0]:.3f}–{validation['AUPRC_95CI'][1]:.3f})
        - **Validation status:** No independent external validation
        """
    )

st.caption("PALF–HE research calculator · Model version " + model["model_version"])
