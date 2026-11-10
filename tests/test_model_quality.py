"""
tests/test_model_quality.py
===========================
Unit tests for model quality gates and data validation suites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestDataValidationSuite:
    """Test Great Expectations-style data validation."""

    def test_column_exists_pass(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"question_text": ["test"], "difficulty": ["Easy"]})
        result = suite.expect_column_to_exist(df, "question_text")
        assert result.passed is True

    def test_column_exists_fail(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"question_text": ["test"]})
        result = suite.expect_column_to_exist(df, "missing_column")
        assert result.passed is False

    def test_not_null_pass(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        result = suite.expect_column_values_not_null(df, "col")
        assert result.passed is True

    def test_not_null_fail(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"col": ["a", None, None, None]})
        result = suite.expect_column_values_not_null(df, "col", max_null_ratio=0.01)
        assert result.passed is False

    def test_values_in_set(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"difficulty": ["Easy", "Medium", "Hard"]})
        result = suite.expect_column_values_in_set(df, "difficulty", {"Easy", "Medium", "Hard"})
        assert result.passed is True

    def test_values_not_in_set(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"difficulty": ["Easy", "Unknown"]})
        result = suite.expect_column_values_in_set(df, "difficulty", {"Easy", "Medium", "Hard"})
        assert result.passed is False

    def test_row_count_range(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"x": range(100)})
        result = suite.expect_table_row_count_between(df, 50, 200)
        assert result.passed is True

    def test_min_length(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"q": ["What is recursion?", "Hi"]})
        result = suite.expect_column_min_length(df, "q", min_length=5)
        assert result.passed is False

    def test_no_duplicates(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"q": ["a", "b", "c"]})
        result = suite.expect_no_duplicate_rows(df)
        assert result.passed is True

    def test_suite_summary(self):
        from src.monitoring.model_tests import DataValidationSuite
        suite = DataValidationSuite()
        df = pd.DataFrame({"question_text": ["test question here"], "difficulty": ["Easy"]})
        suite.expect_column_to_exist(df, "question_text")
        suite.expect_column_to_exist(df, "difficulty")
        summary = suite.summary
        assert summary["total_tests"] == 2
        assert summary["all_passed"] is True


class TestModelQualityGates:
    """Test model risk/quality gate checks."""

    def test_accuracy_gate_pass(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_minimum_accuracy(0.95)
        assert result.passed is True

    def test_accuracy_gate_fail(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_minimum_accuracy(0.50)
        assert result.passed is False

    def test_f1_gate_pass(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_minimum_f1(0.94)
        assert result.passed is True

    def test_regression_gate_pass(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_no_regression(new_f1=0.95, production_f1=0.94)
        assert result.passed is True

    def test_regression_gate_fail(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_no_regression(new_f1=0.85, production_f1=0.94)
        assert result.passed is False

    def test_cv_stability_pass(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_cv_stability(0.003)
        assert result.passed is True

    def test_cv_stability_fail(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_cv_stability(0.15)
        assert result.passed is False

    def test_data_hash_pass(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_data_hash_match("abc123", "abc123")
        assert result.passed is True

    def test_data_hash_fail(self):
        from src.monitoring.quality_gates import ModelQualityGates
        gates = ModelQualityGates()
        result = gates.check_data_hash_match("abc123", "xyz789")
        assert result.passed is False

    def test_full_quality_check(self):
        from src.monitoring.quality_gates import run_full_quality_check
        result = run_full_quality_check(
            metrics={
                "test_accuracy": 0.97,
                "test_f1_macro": 0.95,
                "cv_f1_macro_std": 0.003,
                "test_size_n": 5000,
            },
            production_metrics={"test_f1_macro": 0.94},
            data_hash="abc123",
            expected_data_hash="abc123",
        )
        assert result["all_passed"] is True
        assert result["promotion_allowed"] is True


class TestDriftDetection:
    """Test Evidently AI-style drift detection."""

    def test_numerical_drift_stable(self):
        from src.monitoring.drift_detection import DriftReport
        np.random.seed(42)
        ref = pd.DataFrame({"x": np.random.normal(10, 2, 500)})
        cur = pd.DataFrame({"x": np.random.normal(10, 2, 500)})
        report = DriftReport(ref)
        result = report.run(cur)
        assert result["overall_drift_detected"] is False

    def test_numerical_drift_detected(self):
        from src.monitoring.drift_detection import DriftReport
        np.random.seed(42)
        ref = pd.DataFrame({"x": np.random.normal(10, 2, 500)})
        cur = pd.DataFrame({"x": np.random.normal(50, 2, 500)})  # massive shift
        report = DriftReport(ref)
        result = report.run(cur)
        assert result["feature_drift"]["x"]["drift_detected"] is True

    def test_categorical_drift(self):
        from src.monitoring.drift_detection import DriftReport
        ref = pd.DataFrame({"cat": ["A"] * 300 + ["B"] * 200})
        cur = pd.DataFrame({"cat": ["A"] * 100 + ["B"] * 400})  # distribution shift
        report = DriftReport(ref)
        result = report.run(cur)
        assert "cat" in result["feature_drift"]

    def test_prediction_monitor(self):
        from src.monitoring.drift_detection import PredictionDriftMonitor
        monitor = PredictionDriftMonitor()
        monitor.log_prediction({"difficulty": "Easy", "blooms_level": "L1"})
        monitor.log_prediction({"difficulty": "Hard", "blooms_level": "L5"})
        dist = monitor.get_prediction_distribution()
        assert dist["total_predictions"] == 2

