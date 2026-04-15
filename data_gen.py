import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

RNG_SEED = 42
np.random.seed(RNG_SEED)


def generate_synthetic_healthcare_data(
    n_patients: int = 500,
    days: int = 30
) -> pd.DataFrame:
    """
    Generate HIPAA-safe synthetic time-series healthcare data.
    Each row = patient-day observation with vitals + lab-like features.
    """
    patient_ids = np.arange(1, n_patients + 1)
    start_date = datetime.today() - timedelta(days=days)

    records = []
    for pid in patient_ids:
        baseline_hr = np.random.normal(75, 5)
        baseline_bp_sys = np.random.normal(120, 10)
        baseline_bp_dia = np.random.normal(80, 5)
        baseline_temp = np.random.normal(36.8, 0.2)
        baseline_resp = np.random.normal(16, 2)

        for d in range(days):
            date = start_date + timedelta(days=d)

            # Add some temporal drift and noise
            hr = baseline_hr + np.random.normal(0, 4)
            bp_sys = baseline_bp_sys + np.random.normal(0, 8)
            bp_dia = baseline_bp_dia + np.random.normal(0, 5)
            temp = baseline_temp + np.random.normal(0, 0.3)
            resp = baseline_resp + np.random.normal(0, 2)

            # Simple synthetic "risk" logic
            risk_score = (
                0.03 * (hr - 70)
                + 0.02 * (bp_sys - 120)
                + 0.05 * (temp - 37.0) * 10
                + 0.03 * (resp - 16)
                + np.random.normal(0, 0.5)
            )

            # Binary label: high risk if score above threshold
            high_risk = int(risk_score > 2.0)

            records.append(
                {
                    "patient_id": pid,
                    "date": date.date(),
                    "heart_rate": round(hr, 1),
                    "bp_systolic": round(bp_sys, 1),
                    "bp_diastolic": round(bp_dia, 1),
                    "temperature": round(temp, 2),
                    "resp_rate": round(resp, 1),
                    "risk_score": round(risk_score, 2),
                    "high_risk": high_risk,
                }
            )

    df = pd.DataFrame(records)
    return df


def train_test_split_healthcare(df: pd.DataFrame):
    feature_cols = [
        "heart_rate",
        "bp_systolic",
        "bp_diastolic",
        "temperature",
        "resp_rate",
    ]
    X = df[feature_cols]
    y = df["high_risk"]
    return train_test_split(X, y, test_size=0.2, random_state=RNG_SEED)


if __name__ == "__main__":
    df = generate_synthetic_healthcare_data()
    print(df.head())
    print(df["high_risk"].value_counts(normalize=True))
