import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data_gen import generate_synthetic_healthcare_data
from model import RiskModel

st.set_page_config(
    page_title="Real-Time Healthcare Analytics Dashboard",
    layout="wide",
)

@st.cache_data
def load_data():
    df = generate_synthetic_healthcare_data(
        n_patients=300,
        days=21,
    )
    return df


@st.cache_resource
def load_model(df):
    model = RiskModel()
    model.fit(df)
    return model


df = load_data()
model = load_model(df)

st.title("🏥 Real-Time Healthcare Analytics Dashboard (Synthetic Data)")
st.markdown(
    """
This dashboard uses **HIPAA-safe synthetic data** to simulate real-time healthcare monitoring.

- Each row = a patient-day observation  
- Features: vitals (HR, BP, temp, resp)  
- Outputs: **risk score** + **high-risk flag** + **anomaly score**  

All data is fully synthetic and generated on the fly.
"""
)

# Sidebar filters
st.sidebar.header("Filters")
selected_date = st.sidebar.selectbox(
    "Select date",
    sorted(df["date"].unique()),
    index=len(df["date"].unique()) - 1,
)
selected_patient = st.sidebar.selectbox(
    "Focus on patient (optional)",
    options=["All"] + sorted(df["patient_id"].unique().tolist()),
)

# Filter by date
df_day = df[df["date"] == selected_date].copy()

# Model predictions for the selected day
df_day["predicted_risk_proba"] = model.predict_risk_proba(df_day)
df_day["anomaly_score"] = model.predict_anomaly_score(df_day)

# Global KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Total Patients (Today)",
        value=df_day["patient_id"].nunique(),
    )
with col2:
    st.metric(
        "High-Risk Patients (True)",
        value=int(df_day["high_risk"].sum()),
    )
with col3:
    st.metric(
        "High-Risk (Predicted ≥ 0.5)",
        value=int((df_day["predicted_risk_proba"] >= 0.5).sum()),
    )
with col4:
    st.metric(
        "Mean Anomaly Score",
        value=round(df_day["anomaly_score"].mean(), 2),
    )

st.markdown("---")

# Risk distribution
st.subheader("Risk Score Distribution (Predicted)")
fig_risk = px.histogram(
    df_day,
    x="predicted_risk_proba",
    nbins=20,
    title="Predicted Risk Probability Distribution",
)
st.plotly_chart(fig_risk, use_container_width=True)

# Anomaly vs risk scatter
st.subheader("Anomaly vs Risk")
fig_scatter = px.scatter(
    df_day,
    x="anomaly_score",
    y="predicted_risk_proba",
    color=df_day["high_risk"].map({0: "Low Risk (True)", 1: "High Risk (True)"}),
    hover_data=["patient_id"],
    labels={
        "anomaly_score": "Anomaly Score (higher = more anomalous)",
        "predicted_risk_proba": "Predicted Risk Probability",
        "color": "True Label",
    },
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Patient-level time series
st.subheader("Patient-Level Time Series")

if selected_patient != "All":
    pid = int(selected_patient)
    df_patient = df[df["patient_id"] == pid].copy()
    df_patient["predicted_risk_proba"] = model.predict_risk_proba(df_patient)
    df_patient["anomaly_score"] = model.predict_anomaly_score(df_patient)

    col_ts1, col_ts2 = st.columns(2)

    with col_ts1:
        fig_hr = px.line(
            df_patient,
            x="date",
            y="heart_rate",
            title=f"Heart Rate Over Time (Patient {pid})",
        )
        st.plotly_chart(fig_hr, use_container_width=True)

        fig_temp = px.line(
            df_patient,
            x="date",
            y="temperature",
            title=f"Temperature Over Time (Patient {pid})",
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_ts2:
        fig_risk_ts = px.line(
            df_patient,
            x="date",
            y="predicted_risk_proba",
            title=f"Predicted Risk Probability Over Time (Patient {pid})",
        )
        st.plotly_chart(fig_risk_ts, use_container_width=True)

        fig_anom_ts = px.line(
            df_patient,
            x="date",
            y="anomaly_score",
            title=f"Anomaly Score Over Time (Patient {pid})",
        )
        st.plotly_chart(fig_anom_ts, use_container_width=True)

    st.dataframe(
        df_patient[
            [
                "date",
                "heart_rate",
                "bp_systolic",
                "bp_diastolic",
                "temperature",
                "resp_rate",
                "high_risk",
                "predicted_risk_proba",
                "anomaly_score",
            ]
        ].sort_values("date"),
        use_container_width=True,
    )
else:
    st.info("Select a specific patient in the sidebar to view time-series details.")

st.markdown("---")
st.caption(
    "All data is synthetic and generated for demonstration purposes only. "
    "No real patient data is used."
)
