"""
tests/test_serving.py
=====================
Unit tests for the QuestionCraft FastAPI inference service.
Tests endpoint responses, request validation, and model loading.
"""

from __future__ import annotations

import pytest


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_question_input_valid(self):
        from src.serving.schemas import QuestionInput
        q = QuestionInput(question_text="What is a binary search tree?")
        assert q.question_text == "What is a binary search tree?"

    def test_question_input_too_short(self):
        from src.serving.schemas import QuestionInput
        with pytest.raises(Exception):  # Pydantic validation error
            QuestionInput(question_text="Hi")

    def test_batch_input(self):
        from src.serving.schemas import BatchQuestionInput, QuestionInput
        batch = BatchQuestionInput(
            questions=[
                QuestionInput(question_text="What is a binary search tree?"),
                QuestionInput(question_text="Design a cache-aware B-Tree for storage."),
            ]
        )
        assert len(batch.questions) == 2

    def test_classification_result(self):
        from src.serving.schemas import ClassificationResult
        r = ClassificationResult(
            question_text="Test question here",
            difficulty="Easy",
            difficulty_confidence=0.95,
            blooms_level="L1",
            blooms_confidence=0.88,
            ct_skills=["Pattern Recognition"],
            ct_confidences={"Pattern Recognition": 0.88},
        )
        assert r.difficulty == "Easy"
        assert r.blooms_level == "L1"

    def test_health_response(self):
        from src.serving.schemas import HealthResponse
        h = HealthResponse(
            status="healthy",
            models_loaded={"difficulty": True, "blooms": True, "ct": False},
        )
        assert h.status == "healthy"
        assert h.models_loaded["difficulty"] is True

    def test_artifact_manifest(self):
        from src.serving.schemas import ArtifactManifest
        manifest = ArtifactManifest(
            dataset="test_data",
            preprocessing_pipeline="test_pipeline",
            trained_model="test_model",
            model_configuration="test_config",
            evaluation_results="test_results",
            training_logs="test_logs",
            feature_engineering_artifacts="test_features",
            tokenizer="test_tokenizer",
            model_metadata="test_metadata",
            deployment_artifacts="test_deploy",
            monitoring_artifacts="test_monitoring",
        )
        assert manifest.dataset == "test_data"

    def test_duplicate_check_input(self):
        from src.serving.schemas import DuplicateCheckInput
        d = DuplicateCheckInput(
            question_a="What is a binary tree?",
            question_b="What is a BST?",
        )
        assert len(d.question_a) > 5


class TestModelStore:
    """Test the ModelStore class."""

    def test_model_store_init(self):
        from src.serving.model_loader import ModelStore
        store = ModelStore(models_dir="models")
        assert store.models_dir.name == "models"

    def test_model_store_status_empty(self):
        from src.serving.model_loader import ModelStore
        store = ModelStore(models_dir="nonexistent_dir")
        store.load_all()
        status = store.get_all_status()
        assert isinstance(status, dict)
        assert "difficulty" in status

    def test_canary_ratio(self):
        from src.serving.model_loader import ModelStore
        store = ModelStore(models_dir="models")
        store.set_canary_ratio(0.2)
        assert store._canary_ratio == 0.2

    def test_canary_ratio_clamping(self):
        from src.serving.model_loader import ModelStore
        store = ModelStore(models_dir="models")
        store.set_canary_ratio(1.5)
        assert store._canary_ratio == 1.0
        store.set_canary_ratio(-0.5)
        assert store._canary_ratio == 0.0

    def test_artifact_manifest(self):
        from src.serving.model_loader import ModelStore
        store = ModelStore(models_dir="models")
        manifest = store.get_artifact_manifest()
        assert "dataset" in manifest
        assert "preprocessing_pipeline" in manifest
        assert "trained_model" in manifest
        assert "model_configuration" in manifest
        assert "evaluation_results" in manifest
        assert "training_logs" in manifest
        assert "feature_engineering_artifacts" in manifest
        assert "tokenizer" in manifest
        assert "model_metadata" in manifest
        assert "deployment_artifacts" in manifest
        assert "monitoring_artifacts" in manifest

