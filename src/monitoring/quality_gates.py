"""
src/monitoring/quality_gates.py
===============================
Model risk and quality gate checks for QuestionCraft.

Implements quality gates that must pass before a model can be promoted
from Staging to Production. Aligned with RVCE MLOps AI254TA Unit III:
- Model governance and compliance
- Quality assurance and risk management

Quality gates:
1. Minimum accuracy threshold
2. Minimum F1-macro threshold  
3. No regression vs. current production model
4. Cross-validation stability check
5. Data hash verification (reproducibility)
6. Feature schema compatibility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("questioncraft.monitoring.quality_gates")


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    message: str
    threshold: float | None = None
    actual_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ModelQualityGates:
    """
    Enforces quality gates before model promotion.
    
    All gates must pass for a model to be promoted from Staging → Production.
    This prevents deploying models that regress on key metrics.
    """

    # Default thresholds aligned with QuestionCraft Phase 1 results
    DEFAULT_THRESHOLDS = {
        "min_accuracy": 0.85,
        "min_f1_macro": 0.30,
        "max_regression_pct": 0.02,  # Allow max 2% regression
        "min_cv_stability": 0.05,    # CV std must be < 0.05
        "min_test_samples": 100,
    }

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.results: list[QualityGateResult] = []

    def reset(self):
        self.results = []

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def summary(self) -> dict:
        return {
            "total_gates": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "all_passed": self.all_passed,
            "promotion_allowed": self.all_passed,
            "details": [
                {
                    "gate": r.gate_name,
                    "passed": r.passed,
                    "message": r.message,
                    "threshold": r.threshold,
                    "actual": r.actual_value,
                }
                for r in self.results
            ],
        }

    def check_minimum_accuracy(self, accuracy: float) -> QualityGateResult:
        """Gate 1: Model must meet minimum accuracy threshold."""
        threshold = self.thresholds["min_accuracy"]
        result = QualityGateResult(
            gate_name="minimum_accuracy",
            passed=accuracy >= threshold,
            message=f"Accuracy {accuracy:.4f} {'>=' if accuracy >= threshold else '<'} {threshold}",
            threshold=threshold,
            actual_value=accuracy,
        )
        self.results.append(result)
        return result

    def check_minimum_f1(self, f1_macro: float) -> QualityGateResult:
        """Gate 2: Model must meet minimum F1-macro threshold."""
        threshold = self.thresholds["min_f1_macro"]
        result = QualityGateResult(
            gate_name="minimum_f1_macro",
            passed=f1_macro >= threshold,
            message=f"F1-macro {f1_macro:.4f} {'>=' if f1_macro >= threshold else '<'} {threshold}",
            threshold=threshold,
            actual_value=f1_macro,
        )
        self.results.append(result)
        return result

    def check_no_regression(
        self, new_f1: float, production_f1: float
    ) -> QualityGateResult:
        """Gate 3: New model must not regress more than threshold vs production."""
        max_regression = self.thresholds["max_regression_pct"]
        regression = production_f1 - new_f1
        allowed = regression <= max_regression

        result = QualityGateResult(
            gate_name="no_regression",
            passed=allowed,
            message=(
                f"Regression: {regression:.4f} "
                f"({'within' if allowed else 'exceeds'} {max_regression} threshold) | "
                f"New: {new_f1:.4f} vs Production: {production_f1:.4f}"
            ),
            threshold=max_regression,
            actual_value=regression,
            details={"new_f1": new_f1, "production_f1": production_f1},
        )
        self.results.append(result)
        return result

    def check_cv_stability(self, cv_std: float) -> QualityGateResult:
        """Gate 4: Cross-validation standard deviation must be below threshold."""
        threshold = self.thresholds["min_cv_stability"]
        result = QualityGateResult(
            gate_name="cv_stability",
            passed=cv_std <= threshold,
            message=f"CV std {cv_std:.4f} {'<=' if cv_std <= threshold else '>'} {threshold}",
            threshold=threshold,
            actual_value=cv_std,
        )
        self.results.append(result)
        return result

    def check_data_hash_match(
        self, training_hash: str, expected_hash: str
    ) -> QualityGateResult:
        """Gate 5: Training data hash must match expected hash (reproducibility)."""
        result = QualityGateResult(
            gate_name="data_hash_verification",
            passed=training_hash == expected_hash,
            message=(
                "Data hash verified"
                if training_hash == expected_hash
                else f"Hash mismatch: {training_hash[:8]}… ≠ {expected_hash[:8]}…"
            ),
            details={"training_hash": training_hash, "expected_hash": expected_hash},
        )
        self.results.append(result)
        return result

    def check_test_sample_size(self, test_size: int) -> QualityGateResult:
        """Gate 6: Test set must have minimum number of samples."""
        threshold = self.thresholds["min_test_samples"]
        result = QualityGateResult(
            gate_name="test_sample_size",
            passed=test_size >= threshold,
            message=f"Test samples: {test_size} {'>=' if test_size >= threshold else '<'} {threshold}",
            threshold=threshold,
            actual_value=float(test_size),
        )
        self.results.append(result)
        return result


def run_full_quality_check(
    metrics: dict,
    production_metrics: dict | None = None,
    data_hash: str | None = None,
    expected_data_hash: str | None = None,
    thresholds: dict | None = None,
) -> dict:
    """
    Run all quality gates on a candidate model.
    
    Parameters
    ----------
    metrics : dict
        Candidate model metrics (test_accuracy, test_f1_macro, cv_f1_macro_std, test_size_n)
    production_metrics : dict, optional
        Current production model metrics for regression check
    data_hash : str, optional
        Hash of training data for reproducibility check
    expected_data_hash : str, optional
        Expected data hash
    thresholds : dict, optional
        Custom threshold overrides

    Returns
    -------
    dict
        Quality gate summary with pass/fail for each gate
    """
    gates = ModelQualityGates(thresholds=thresholds)

    # Gate 1 & 2: Minimum performance
    gates.check_minimum_accuracy(metrics.get("test_accuracy", 0.0))
    gates.check_minimum_f1(metrics.get("test_f1_macro", 0.0))

    # Gate 3: No regression (if production model exists)
    if production_metrics:
        gates.check_no_regression(
            metrics.get("test_f1_macro", 0.0),
            production_metrics.get("test_f1_macro", 0.0),
        )

    # Gate 4: CV stability
    if "cv_f1_macro_std" in metrics:
        gates.check_cv_stability(metrics["cv_f1_macro_std"])

    # Gate 5: Data hash (reproducibility)
    if data_hash and expected_data_hash:
        gates.check_data_hash_match(data_hash, expected_data_hash)

    # Gate 6: Test sample size
    if "test_size_n" in metrics:
        gates.check_test_sample_size(int(metrics["test_size_n"]))

    logger.info(
        "Quality gates: %d/%d passed — promotion %s",
        gates.summary["passed"],
        gates.summary["total_gates"],
        "ALLOWED" if gates.all_passed else "BLOCKED",
    )
    return gates.summary
