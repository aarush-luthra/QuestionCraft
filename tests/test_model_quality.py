"""
tests/test_model_quality.py
===========================
Unit tests for model quality gates and data validation suites.
"""

from __future__ import annotations

import pandas as pd


class TestDataValidationSuite:
    """Test the rule-based data validation suite."""

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


class TestDatasetCleaning:
    def test_clean_removes_short_duplicate_and_conflicting(self):
        import pandas as pd

        from src.monitoring.model_tests import clean_text_dataset

        df = pd.DataFrame(
            {
                "question_text": [
                    "What is a stack?",
                    "What is a stack?",
                    "hi",
                    "Define a queue.",
                    "Define a queue.",
                ],
                "difficulty": ["Easy", "Easy", "Easy", "Easy", "Hard"],
            }
        )
        clean, report = clean_text_dataset(df, "question_text", "difficulty")
        assert list(clean["question_text"]) == ["What is a stack?"]
        assert report["removed_too_short"] == 1
        assert report["removed_conflicting_label_texts"] == 1
        assert report["removed_duplicates"] == 1


class TestRvceEval:
    def test_single_label_metrics_and_baseline(self, tmp_path):
        import pandas as pd
        from sklearn.dummy import DummyClassifier

        from src.models.rvce_eval import evaluate_single_label

        p = tmp_path / "eval.parquet"
        pd.DataFrame(
            {
                "question_text": ["a", "b", "c", "d"],
                "label": ["Easy", "Easy", "Easy", "Hard"],
                "label_source": ["llm_draft"] * 4,
            }
        ).to_parquet(p)
        clf = DummyClassifier(strategy="constant", constant="Easy").fit([[0]] * 2, ["Easy", "Hard"])
        clf.predict = lambda X: ["Easy"] * len(X)
        m = evaluate_single_label(clf, str(p), ["Easy", "Medium", "Hard"])
        assert m["rvce_n"] == 4
        assert m["rvce_accuracy"] == 0.75
        assert m["rvce_majority_baseline_accuracy"] == 0.75

    def test_missing_eval_returns_empty(self):
        from src.models.rvce_eval import evaluate_single_label

        assert evaluate_single_label(None, None, ["Easy"]) == {}


class TestBuildSplitsLoader:
    def test_load_bank_normalises_labels(self, tmp_path):
        import json

        from src.data.build_splits import load_bank

        p = tmp_path / "bank.json"
        p.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "question": "Define a stack.",
                            "btl": "Remember",
                            "ct_skill": "Abstraction",
                            "difficulty": "Easy",
                            "label_source": {"difficulty": "llm_draft"},
                        }
                    ]
                }
            )
        )
        df = load_bank(p, "rvce_bank")
        row = df.iloc[0]
        assert row["blooms_level"] == "L1"
        assert row["ct_skills"] == "Abstraction"
        assert row["ls_difficulty"] == "llm_draft"
        assert row["ls_blooms"] == "unknown"


class TestPromotionIsRefused:
    """Phase 2 scope §3: a model that fails any gate must be refused promotion."""

    def test_failing_gates_block_promotion(self):
        from src.monitoring.quality_gates import run_full_quality_check

        result = run_full_quality_check(
            {
                "test_accuracy": 0.70,
                "test_f1_macro": 0.25,
                "cv_f1_macro_std": 0.2,
                "test_size_n": 50,
            },
            production_metrics={"test_f1_macro": 0.60},
            data_hash="aaa",
            expected_data_hash="bbb",
        )
        assert result["promotion_allowed"] is False
        assert {g["gate"] for g in result["details"] if not g["passed"]} == {
            "minimum_accuracy",
            "minimum_f1_macro",
            "no_regression",
            "cv_stability",
            "data_hash_verification",
            "test_sample_size",
        }

    def test_cli_refuses_regressing_candidate_and_keeps_production(self, tmp_path):
        import mlflow
        from click.testing import CliRunner

        import scripts.promote_model as promote
        from src.registry.model_manager import REGISTRY_MODELS

        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        name = REGISTRY_MODELS["difficulty"]
        client.create_registered_model(name)
        exp = client.create_experiment("gate-test", artifact_location=str(tmp_path / "art"))

        def version(acc, f1):
            run = client.create_run(exp)
            for k, v in {
                "test_accuracy": acc,
                "test_f1_macro": f1,
                "cv_f1_macro_std": 0.01,
            }.items():
                client.log_metric(run.info.run_id, k, v)
            client.log_param(run.info.run_id, "data_hash", "same")
            client.log_param(run.info.run_id, "test_size_n", "500")
            client.set_terminated(run.info.run_id)
            return client.create_model_version(
                name, f"runs:/{run.info.run_id}/model", run.info.run_id
            ).version

        prod = version(0.95, 0.90)
        cand = version(0.94, 0.70)  # 20-point F1 regression -> must be refused
        client.transition_model_version_stage(name, prod, "Production")
        client.transition_model_version_stage(name, cand, "Staging")

        res = CliRunner().invoke(
            promote.main, ["--model", "difficulty", "--to", "production", "--tracking-uri", uri]
        )

        assert res.exit_code == 1
        stages = {
            v.version: v.current_stage for v in client.search_model_versions(f"name='{name}'")
        }
        assert stages[prod] == "Production"
        assert stages[cand] == "Staging"
