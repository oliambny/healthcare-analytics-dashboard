import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score


class RiskModel:
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        self.iso = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self.feature_cols = [
            "heart_rate",
            "bp_systolic",
            "bp_diastolic",
            "temperature",
            "resp_rate",
        ]

    def fit(self, df):
        X = df[self.feature_cols]
        y = df["high_risk"]
        self.clf.fit(X, y)
        self.iso.fit(X)

    def predict_risk_proba(self, df):
        X = df[self.feature_cols]
        return self.clf.predict_proba(X)[:, 1]

    def predict_anomaly_score(self, df):
        X = df[self.feature_cols]
        # IsolationForest: lower score = more anomalous
        return -self.iso.score_samples(X)

    def evaluate_auc(self, df):
        X = df[self.feature_cols]
        y = df["high_risk"]
        proba = self.clf.predict_proba(X)[:, 1]
        return roc_auc_score(y, proba)
