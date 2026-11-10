"""
src/monitoring/model_tests.py
=============================
Great Expectations-style data validation and model quality tests.

Implements data validation suites for:
- Input data schema validation
- Feature distribution checks
- Label distribution verification
- Data quality gates (missing values, outliers, duplicates)

Aligned with RVCE MLOps AI254TA Unit III: Reproducibility, auditability,
data/model versioning, quality assurance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("questioncraft.monitoring.model_tests")


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    test_name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class DataValidationSuite:
    """
    Great Expectations-inspired data validation suite for QuestionCraft.
    
    Validates incoming data against expected schemas, distributions,
    and quality thresholds before model training or inference.
    """

    def __init__(self):
        self.results: list[ValidationResult] = []

    def reset(self):
        self.results = []

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def summary(self) -> dict:
        return {
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "all_passed": self.all_passed,
            "details": [
                {"test": r.test_name, "passed": r.passed, "message": r.message}
                for r in self.results
            ],
        }

    def expect_column_to_exist(self, df: pd.DataFrame, column: str) -> ValidationResult:
        """Validate that a column exists in the DataFrame."""
        result = ValidationResult(
            test_name=f"column_exists:{column}",
            passed=bool(column in df.columns),
            message=f"Column '{column}' {'exists' if column in df.columns else 'is missing'}",
        )
        self.results.append(result)
        return result

    def expect_column_values_not_null(
        self, df: pd.DataFrame, column: str, max_null_ratio: float = 0.01
    ) -> ValidationResult:
        """Validate that null values are below threshold."""
        if column not in df.columns:
            result = ValidationResult(
                test_name=f"not_null:{column}",
                passed=False,
                message=f"Column '{column}' does not exist",
            )
        else:
            null_ratio = df[column].isnull().mean()
            result = ValidationResult(
                test_name=f"not_null:{column}",
                passed=bool(null_ratio <= max_null_ratio),
                message=f"Null ratio for '{column}': {null_ratio:.4f} (threshold: {max_null_ratio})",
                details={"null_ratio": float(null_ratio), "threshold": max_null_ratio},
            )
        self.results.append(result)
        return result

    def expect_column_values_in_set(
        self, df: pd.DataFrame, column: str, expected_values: set
    ) -> ValidationResult:
        """Validate that column values are within expected set."""
        if column not in df.columns:
            result = ValidationResult(
                test_name=f"values_in_set:{column}",
                passed=False,
                message=f"Column '{column}' does not exist",
            )
        else:
            actual = set(df[column].dropna().unique())
            unexpected = actual - expected_values
            result = ValidationResult(
                test_name=f"values_in_set:{column}",
                passed=bool(len(unexpected) == 0),
                message=(
                    f"All values in expected set"
                    if not unexpected
                    else f"Unexpected values: {unexpected}"
                ),
                details={"expected": list(expected_values), "unexpected": list(unexpected)},
            )
        self.results.append(result)
        return result

    def expect_column_min_length(
        self, df: pd.DataFrame, column: str, min_length: int = 5
    ) -> ValidationResult:
        """Validate minimum text length for string columns."""
        if column not in df.columns:
            result = ValidationResult(
                test_name=f"min_length:{column}",
                passed=False,
                message=f"Column '{column}' does not exist",
            )
        else:
            lengths = df[column].astype(str).str.len()
            too_short = (lengths < min_length).sum()
            result = ValidationResult(
                test_name=f"min_length:{column}",
                passed=bool(too_short == 0),
                message=f"{too_short} values shorter than {min_length} chars in '{column}'",
                details={"too_short_count": int(too_short), "min_length": min_length},
            )
        self.results.append(result)
        return result

    def expect_table_row_count_between(
        self, df: pd.DataFrame, min_rows: int, max_rows: int
    ) -> ValidationResult:
        """Validate DataFrame size is within expected range."""
        n = len(df)
        result = ValidationResult(
            test_name="row_count_range",
            passed=bool(min_rows <= n <= max_rows),
            message=f"Row count: {n} (expected {min_rows}–{max_rows})",
            details={"actual": n, "min": min_rows, "max": max_rows},
        )
        self.results.append(result)
        return result

    def expect_column_mean_between(
        self, df: pd.DataFrame, column: str, min_val: float, max_val: float
    ) -> ValidationResult:
        """Validate that column mean is within expected range (distribution check)."""
        if column not in df.columns:
            result = ValidationResult(
                test_name=f"mean_range:{column}",
                passed=False,
                message=f"Column '{column}' does not exist",
            )
        else:
            mean = float(df[column].mean())
            result = ValidationResult(
                test_name=f"mean_range:{column}",
                passed=bool(min_val <= mean <= max_val),
                message=f"Mean of '{column}': {mean:.4f} (expected {min_val}–{max_val})",
                details={"mean": mean, "min": min_val, "max": max_val},
            )
        self.results.append(result)
        return result

    def expect_no_duplicate_rows(
        self, df: pd.DataFrame, subset: list[str] | None = None
    ) -> ValidationResult:
        """Validate no duplicate rows exist."""
        dupes = df.duplicated(subset=subset).sum()
        result = ValidationResult(
            test_name=f"no_duplicates:{subset or 'all'}",
            passed=bool(dupes == 0),
            message=f"{dupes} duplicate rows found",
            details={"duplicate_count": int(dupes)},
        )
        self.results.append(result)
        return result


def validate_difficulty_dataset(df: pd.DataFrame) -> dict:
    """Run full validation suite on the difficulty classification dataset."""
    suite = DataValidationSuite()
    suite.expect_column_to_exist(df, "question_text")
    suite.expect_column_to_exist(df, "difficulty")
    suite.expect_column_values_not_null(df, "question_text")
    suite.expect_column_values_not_null(df, "difficulty")
    suite.expect_column_values_in_set(df, "difficulty", {"Easy", "Medium", "Hard"})
    suite.expect_column_min_length(df, "question_text", min_length=5)
    suite.expect_table_row_count_between(df, min_rows=100, max_rows=500_000)
    suite.expect_no_duplicate_rows(df, subset=["question_text"])
    logger.info("Difficulty dataset validation: %s", "PASSED" if suite.all_passed else "FAILED")
    return suite.summary


def validate_blooms_dataset(df: pd.DataFrame) -> dict:
    """Run full validation suite on the Bloom's taxonomy dataset."""
    suite = DataValidationSuite()
    suite.expect_column_to_exist(df, "question_text")
    suite.expect_column_to_exist(df, "blooms_level")
    suite.expect_column_values_not_null(df, "question_text")
    suite.expect_column_values_not_null(df, "blooms_level")
    suite.expect_column_min_length(df, "question_text", min_length=5)
    suite.expect_table_row_count_between(df, min_rows=100, max_rows=500_000)
    logger.info("Blooms dataset validation: %s", "PASSED" if suite.all_passed else "FAILED")
    return suite.summary


def validate_inference_input(questions: list[str]) -> dict:
    """Validate inference request inputs."""
    suite = DataValidationSuite()
    df = pd.DataFrame({"question_text": questions})
    suite.expect_column_values_not_null(df, "question_text")
    suite.expect_column_min_length(df, "question_text", min_length=5)
    suite.expect_table_row_count_between(df, min_rows=1, max_rows=100)
    return suite.summary
