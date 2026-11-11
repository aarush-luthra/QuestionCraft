"""
src/serving/schemas.py
======================
Pydantic request/response schemas for the QuestionCraft inference API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Request Schemas ────────────────────────────────────────────────────────────


class QuestionInput(BaseModel):
    """Single question for classification."""

    question_text: str = Field(..., min_length=5, description="The question text to classify")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question_text": "Design a cache-aware B-Tree for large-scale key-value storage."}
            ]
        }
    }


class BatchQuestionInput(BaseModel):
    """Batch of questions for classification."""

    questions: list[QuestionInput] = Field(..., min_length=1, max_length=100)


class DuplicateCheckInput(BaseModel):
    """Pair of questions to check for semantic similarity."""

    question_a: str = Field(..., min_length=5)
    question_b: str = Field(..., min_length=5)


class BatchDuplicateInput(BaseModel):
    """List of questions to check pairwise for duplicates."""

    questions: list[str] = Field(..., min_length=2, max_length=50)
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)


# ── Response Schemas ───────────────────────────────────────────────────────────


class ClassificationResult(BaseModel):
    """Classification result for a single question."""

    question_text: str
    difficulty: str = Field(..., description="Easy / Medium / Hard")
    difficulty_confidence: float = Field(..., ge=0.0, le=1.0)
    blooms_level: str = Field(..., description="L1 through L6")
    blooms_confidence: float = Field(..., ge=0.0, le=1.0)
    ct_skills: list[str] = Field(
        default_factory=list, description="Computational Thinking skills detected"
    )
    ct_confidences: dict[str, float] = Field(default_factory=dict)


class BatchClassificationResult(BaseModel):
    """Batch classification response."""

    results: list[ClassificationResult]
    model_versions: dict[str, str] = Field(default_factory=dict)
    inference_time_ms: float


class DuplicateResult(BaseModel):
    """Duplicate check result for a pair."""

    question_a: str
    question_b: str
    cosine_similarity: float
    is_duplicate: bool


class BatchDuplicateResult(BaseModel):
    """Batch duplicate detection response."""

    pairs: list[DuplicateResult]
    threshold_used: float
    total_pairs_checked: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "2.0.0"
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    mlflow_connected: bool = False
    uptime_seconds: float = 0.0


class ModelInfoResponse(BaseModel):
    """Model registry info response."""

    model_name: str
    current_version: str
    stage: str
    registered_at: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ArtifactManifest(BaseModel):
    """ML Artifact manifest — tracks all artifacts as required by RVCE MLOps evaluation."""

    dataset: str = Field(..., description="Data used for training, validation, and testing")
    preprocessing_pipeline: str = Field(..., description="Transformations applied to raw data")
    trained_model: str = Field(..., description="Model parameters and learned weights")
    model_configuration: str = Field(..., description="Parameters used to train the model")
    evaluation_results: str = Field(..., description="Metrics used to measure model performance")
    training_logs: str = Field(..., description="Records of training progress")
    feature_engineering_artifacts: str = Field(
        ..., description="Definitions and transformations of input features"
    )
    tokenizer: str = Field(
        ..., description="Vocabulary and tokenization configuration for NLP models"
    )
    model_metadata: str = Field(..., description="Information about the model and its development")
    deployment_artifacts: str = Field(..., description="Files required to serve a model")
    monitoring_artifacts: str = Field(..., description="Records used to monitor model behaviour")
