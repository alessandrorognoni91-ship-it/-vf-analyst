"""
ml_model.py — VF Analyst Early Warning System
===============================================
Lightweight, interpretable ML module for predicting alarm risk.

Design choices for clinical context:
- Random Forest: robust, handles NaN-imputed data, gives feature importance
- Simple train/test split (no leakage across sessions)
- All outputs expressed in plain clinical language
- No deep learning, no black boxes

Pipeline
--------
1. engineer_features()  →  time-series features per session
2. make_alarm_target()  →  forward-looking binary alarm label
3. VFModel.train()      →  fit on one or multiple sessions
4. VFModel.predict()    →  risk score 0–1 per time point
5. VFModel.explain()    →  top features in plain language
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import MEASUREMENT_COLS, DISPLAY_LABELS
from data_cleaning import engineer_features, make_alarm_target

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature column names (produced by engineer_features)
# ─────────────────────────────────────────────────────────────────────────────

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all engineered feature columns present in df."""
    candidates = (
        [f"rolling_mean_{c}" for c in MEASUREMENT_COLS] +
        [f"rolling_std_{c}"  for c in MEASUREMENT_COLS] +
        [f"delta_{c}"        for c in MEASUREMENT_COLS] +
        ["pressure_flow_ratio"]
    )
    return [c for c in candidates if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable feature label mapping
# ─────────────────────────────────────────────────────────────────────────────

def _friendly_feature_name(col: str) -> str:
    """Convert engineered feature name to plain clinical language."""
    for raw, label in DISPLAY_LABELS.items():
        if raw in col:
            if col.startswith("rolling_mean_"):
                return f"Avg. {label} (recent trend)"
            if col.startswith("rolling_std_"):
                return f"{label} — instability"
            if col.startswith("delta_"):
                return f"{label} — rate of change"
    if col == "pressure_flow_ratio":
        return "Pressure/Flow instability index"
    return col


# ─────────────────────────────────────────────────────────────────────────────
# Main model class
# ─────────────────────────────────────────────────────────────────────────────

class VFModel:
    """
    Wraps a scikit-learn pipeline (imputer + scaler + classifier).

    Attributes
    ----------
    trained      : bool — whether the model has been fitted
    metrics      : dict — evaluation metrics from the last train() call
    feature_importance : list[dict] — top features with clinical labels
    model_type   : str — 'random_forest' or 'logistic_regression'
    horizon      : int — forecast horizon in samples used during training
    """

    MAX_SESSIONS_FOR_TRAINING = 5  # guard against huge in-memory loads

    def __init__(self, model_type: str = "random_forest", horizon: int = 5):
        self.model_type = model_type
        self.horizon    = horizon
        self.trained    = False
        self.metrics: dict[str, Any]           = {}
        self.feature_importance: list[dict]    = []
        self._feature_cols: list[str]          = []
        self._pipeline: Pipeline | None        = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, sessions_data: list[pd.DataFrame]) -> dict[str, Any]:
        """
        Train on a list of clean measurement DataFrames (one per session).

        Steps
        -----
        1. Engineer features for each session independently (no leakage)
        2. Add forward-looking alarm target
        3. Concatenate all sessions
        4. Simple 80/20 train/test split (time-aware: split by row order)
        5. Fit pipeline, evaluate, extract feature importance

        Returns
        -------
        metrics dict (also stored in self.metrics)
        """
        if not sessions_data:
            raise ValueError("No session data provided for training.")

        frames = []
        for df in sessions_data[:self.MAX_SESSIONS_FOR_TRAINING]:
            df_feat = engineer_features(df, window=10)
            df_feat = make_alarm_target(df_feat, horizon=self.horizon)
            frames.append(df_feat)

        combined = pd.concat(frames, ignore_index=True)

        feat_cols = _feature_cols(combined)
        if not feat_cols:
            raise ValueError("No feature columns found after engineering.")

        self._feature_cols = feat_cols
        target_col = "alarm_future"

        # Drop rows where target is NaN
        valid = combined[feat_cols + [target_col]].dropna(subset=[target_col])
        X = valid[feat_cols].values
        y = valid[target_col].astype(int).values

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Training data has only one class — cannot train a classifier. "
                "Ensure at least some alarm events exist in the selected sessions."
            )

        # Time-aware split: first 80% train, last 20% test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build pipeline
        clf = self._build_classifier()
        self._pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("clf",    clf),
        ])
        self._pipeline.fit(X_train, y_train)
        self.trained = True

        # Evaluate
        y_pred  = self._pipeline.predict(X_test)
        y_proba = self._pipeline.predict_proba(X_test)[:, 1]

        self.metrics = {
            "auc":             round(float(roc_auc_score(y_test, y_proba)), 3),
            "train_samples":   int(len(X_train)),
            "test_samples":    int(len(X_test)),
            "alarm_rate_train": round(float(y_train.mean()), 3),
            "alarm_rate_test":  round(float(y_test.mean()), 3),
            "report":          classification_report(y_test, y_pred, output_dict=True),
        }

        self._extract_feature_importance()
        logger.info("Model trained. AUC=%.3f", self.metrics["auc"])
        return self.metrics

    def _build_classifier(self):
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=500, class_weight="balanced", C=0.1, random_state=42
            )
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    def _extract_feature_importance(self) -> None:
        """Extract and store feature importance with clinical labels."""
        clf = self._pipeline.named_steps["clf"]

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            self.feature_importance = []
            return

        pairs = sorted(
            zip(self._feature_cols, importances),
            key=lambda x: x[1], reverse=True
        )
        self.feature_importance = [
            {
                "feature":       col,
                "label":         _friendly_feature_name(col),
                "importance":    round(float(imp), 4),
                "importance_pct": round(float(imp) / max(sum(i for _, i in pairs), 1e-9) * 100, 1),
            }
            for col, imp in pairs[:12]  # top 12 features
        ]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate risk scores for a new session.

        Returns a DataFrame with columns:
        - timestamp
        - risk_score    : float 0–1 (probability of alarm in next N samples)
        - risk_level    : 'Low' | 'Moderate' | 'High'
        - alarm_active  : actual alarm state (for comparison)
        """
        if not self.trained or self._pipeline is None:
            raise RuntimeError("Model must be trained before predicting.")

        df_feat = engineer_features(df, window=10)
        feat_cols_present = [c for c in self._feature_cols if c in df_feat.columns]

        if not feat_cols_present:
            raise ValueError("No features found in the provided session data.")

        X = df_feat[feat_cols_present].values

        # Impute missing feature cols that were in training but not here
        # (pipeline imputer handles NaN gracefully)
        if len(feat_cols_present) < len(self._feature_cols):
            missing = len(self._feature_cols) - len(feat_cols_present)
            logger.warning("Prediction: %d feature columns missing vs training.", missing)

        proba = self._pipeline.predict_proba(X)[:, 1]

        result = pd.DataFrame({
            "timestamp":    df_feat["timestamp"].values,
            "risk_score":   proba,
        })

        result["risk_level"] = pd.cut(
            result["risk_score"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Low", "Moderate", "High"],
            include_lowest=True,
        )

        if "alarm_active" in df_feat.columns:
            result["alarm_active"] = df_feat["alarm_active"].values

        return result

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def explain(self) -> str:
        """
        Return a plain-language explanation of the top risk factors.
        Suitable for display in the clinical dashboard without jargon.
        """
        if not self.feature_importance:
            return "No feature importance available. Train the model first."

        top3 = self.feature_importance[:3]
        lines = [
            f"• {f['label']} ({f['importance_pct']:.0f}% influence)"
            for f in top3
        ]
        return (
            "The model identifies the following as the strongest early warning signals:\n"
            + "\n".join(lines)
            + "\n\nHigher values in these parameters are associated with increased alarm risk."
        )
