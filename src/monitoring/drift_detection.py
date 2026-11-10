"""
src/monitoring/drift_detection.py
=================================
Evidently AI-inspired data drift and model performance monitoring.

Implements monitoring reports for:
- Feature drift detection (statistical tests on feature distributions)
- Prediction drift detection (distribution shift in model outputs)
- Data quality monitoring (missing values, outliers, new categories)
- Model performance degradation alerts

Aligned with RVCE MLOps AI254TA:
- Evidently AI tool presentations
- Unit V: Model monitoring, logging, and observability
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("questioncraft.monitoring.drift")


class DriftReport:
    """
    Evidently AI-style drift detection report for QuestionCraft.
    
    Compares a reference dataset (training data) against current production
    data to detect statistical drift in features and predictions.
    """

    def __init__(self, reference_data: pd.DataFrame, column_mapping: dict | None = None):
        self.reference = reference_data
        self.column_mapping = column_mapping or {}
        self.report: dict[str, Any] = {}

    def run(self, current_data: pd.DataFrame) -> dict:
        """Generate a comprehensive drift report."""
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(self.reference),
            "current_size": len(current_data),
            "feature_drift": {},
            "prediction_drift": {},
            "data_quality": {},
            "overall_drift_detected": False,
        }

        # Feature drift analysis
        numerical_cols = self.reference.select_dtypes(include=[np.number]).columns.tolist()
        drift_count = 0

        for col in numerical_cols:
            if col in current_data.columns:
                drift_result = self._check_numerical_drift(
                    self.reference[col].dropna(),
                    current_data[col].dropna(),
                    col,
                )
                self.report["feature_drift"][col] = drift_result
                if drift_result["drift_detected"]:
                    drift_count += 1

        # Categorical drift
        categorical_cols = self.reference.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            if col in current_data.columns:
                drift_result = self._check_categorical_drift(
                    self.reference[col].dropna(),
                    current_data[col].dropna(),
                    col,
                )
                self.report["feature_drift"][col] = drift_result
                if drift_result["drift_detected"]:
                    drift_count += 1

        # Data quality checks
        self.report["data_quality"] = self._check_data_quality(current_data)

        # Overall drift flag
        total_features = len(numerical_cols) + len(categorical_cols)
        self.report["overall_drift_detected"] = bool(
            drift_count > max(1, total_features * 0.3)  # >30% features drifted
        )
        self.report["drift_summary"] = {
            "total_features_checked": total_features,
            "features_with_drift": drift_count,
            "drift_ratio": drift_count / max(1, total_features),
        }

        logger.info(
            "Drift report: %d/%d features drifted — overall: %s",
            drift_count, total_features,
            "DRIFT DETECTED" if self.report["overall_drift_detected"] else "stable",
        )
        return self.report

    def _check_numerical_drift(
        self, reference: pd.Series, current: pd.Series, col_name: str
    ) -> dict:
        """Kolmogorov-Smirnov test for numerical feature drift."""
        try:
            ks_stat, p_value = stats.ks_2samp(reference, current)
            drift_detected = bool(p_value < 0.05)  # 5% significance level

            return {
                "type": "numerical",
                "test": "Kolmogorov-Smirnov",
                "statistic": round(float(ks_stat), 6),
                "p_value": round(float(p_value), 6),
                "drift_detected": drift_detected,
                "reference_mean": round(float(reference.mean()), 4),
                "current_mean": round(float(current.mean()), 4),
                "reference_std": round(float(reference.std()), 4),
                "current_std": round(float(current.std()), 4),
            }
        except Exception as e:
            return {"type": "numerical", "error": str(e), "drift_detected": False}

    def _check_categorical_drift(
        self, reference: pd.Series, current: pd.Series, col_name: str
    ) -> dict:
        """Chi-squared test for categorical feature drift."""
        try:
            ref_counts = reference.value_counts(normalize=True)
            cur_counts = current.value_counts(normalize=True)

            all_categories = set(ref_counts.index) | set(cur_counts.index)
            ref_freq = [ref_counts.get(c, 0) for c in all_categories]
            cur_freq = [cur_counts.get(c, 0) for c in all_categories]

            # Ensure non-zero for chi2
            ref_freq = [max(f, 1e-10) for f in ref_freq]
            cur_freq = [max(f, 1e-10) for f in cur_freq]

            chi2_stat, p_value = stats.chisquare(cur_freq, f_exp=ref_freq)
            drift_detected = bool(p_value < 0.05)

            new_categories = set(cur_counts.index) - set(ref_counts.index)

            return {
                "type": "categorical",
                "test": "Chi-Squared",
                "statistic": round(float(chi2_stat), 6),
                "p_value": round(float(p_value), 6),
                "drift_detected": drift_detected,
                "new_categories": list(new_categories),
                "reference_distribution": ref_counts.to_dict(),
                "current_distribution": cur_counts.to_dict(),
            }
        except Exception as e:
            return {"type": "categorical", "error": str(e), "drift_detected": False}

    def _check_data_quality(self, df: pd.DataFrame) -> dict:
        """Run data quality checks on the current dataset."""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": {
                col: int(df[col].isnull().sum())
                for col in df.columns
                if df[col].isnull().sum() > 0
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "empty_strings": {
                col: int((df[col].astype(str).str.strip() == "").sum())
                for col in df.select_dtypes(include=["object"]).columns
            },
        }

    def save_report(self, output_path: str | Path) -> None:
        """Save the drift report as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        logger.info("Drift report saved to %s", path)


class PredictionDriftMonitor:
    """
    Monitors prediction distribution drift over time.
    
    Tracks:
    - Difficulty prediction distribution shifts
    - Bloom's level prediction imbalances
    - CT skill co-occurrence pattern changes
    """

    def __init__(self):
        self._prediction_log: list[dict] = []

    def log_prediction(self, prediction: dict) -> None:
        """Log a single prediction for drift monitoring."""
        self._prediction_log.append({
            "timestamp": datetime.now().isoformat(),
            **prediction,
        })

    def get_prediction_distribution(self) -> dict:
        """Get current prediction distribution across all dimensions."""
        if not self._prediction_log:
            return {"status": "no_predictions_logged"}

        df = pd.DataFrame(self._prediction_log)
        result = {}

        if "difficulty" in df.columns:
            result["difficulty_distribution"] = df["difficulty"].value_counts(normalize=True).to_dict()
        if "blooms_level" in df.columns:
            result["blooms_distribution"] = df["blooms_level"].value_counts(normalize=True).to_dict()

        result["total_predictions"] = len(self._prediction_log)
        return result

    def check_prediction_drift(self, expected_distribution: dict) -> dict:
        """
        Compare current predictions against expected distribution.
        Alerts if any class proportion deviates by more than 15%.
        """
        current = self.get_prediction_distribution()
        alerts = []

        for dimension in ["difficulty_distribution", "blooms_distribution"]:
            if dimension in current and dimension in expected_distribution:
                for label, expected_ratio in expected_distribution[dimension].items():
                    actual_ratio = current[dimension].get(label, 0.0)
                    deviation = abs(actual_ratio - expected_ratio)
                    if deviation > 0.15:
                        alerts.append({
                            "dimension": dimension,
                            "label": label,
                            "expected": expected_ratio,
                            "actual": actual_ratio,
                            "deviation": round(deviation, 4),
                        })

        return {
            "drift_detected": len(alerts) > 0,
            "alerts": alerts,
            "current_distribution": current,
        }
