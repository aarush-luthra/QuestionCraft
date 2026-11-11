"""
scripts/generate_pdf_report.py
==============================
Generates the QuestionCraft Phase 2 PDF report (QuestionCraft_Phase2_Report.pdf).

The content mirrors the "QuestionCraft Phase 2 Scope" document: eight in-scope
deliverables, Kubernetes and drift monitoring out of scope, custom (not Great
Expectations / Evidently) validation, and honest evaluation numbers.

Usage:
    python scripts/generate_pdf_report.py [--out QuestionCraft_Phase2_Report.pdf]
"""

from __future__ import annotations

import argparse
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PRIMARY = colors.HexColor("#1e3a8a")
MUTED = colors.HexColor("#475569")
BORDER = colors.HexColor("#cbd5e1")
LIGHT = colors.HexColor("#f8fafc")
HEAD_BG = colors.HexColor("#e0e7ff")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self.setFont("Helvetica", 8)
            self.setFillColor(MUTED)
            self.drawString(54, 28, "QuestionCraft — Phase 2 Report (RVCE AI254TA MLOps)")
            self.drawRightString(A4[0] - 54, 28, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=base["Title"], textColor=PRIMARY, fontSize=20, spaceAfter=4
        ),
        "sub": ParagraphStyle("s", parent=base["Normal"], textColor=MUTED, fontSize=10, leading=13),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            textColor=PRIMARY,
            fontSize=14,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            textColor=PRIMARY,
            fontSize=11,
            spaceBefore=6,
            spaceAfter=2,
        ),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=13, spaceAfter=4),
        "bullet": ParagraphStyle(
            "bl", parent=base["Normal"], fontSize=9.5, leading=13, leftIndent=12, bulletIndent=2
        ),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=8.5, leading=11),
        "cellb": ParagraphStyle(
            "cb", parent=base["Normal"], fontSize=8.5, leading=11, fontName="Helvetica-Bold"
        ),
    }


def _table(rows, widths, st):
    data = [
        [Paragraph(str(c), st["cellb"] if i == 0 else st["cell"]) for c in r]
        for i, r in enumerate(rows)
    ]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _section(story, st, title, intro=None, bullets=(), table=None, widths=None, done=None):
    block = [Paragraph(title, st["h1"]), HRFlowable(width="100%", thickness=0.8, color=PRIMARY)]
    if intro:
        block.append(Paragraph(intro, st["body"]))
    for b in bullets:
        block.append(Paragraph(b, st["bullet"], bulletText="•"))
    if table:
        block += [Spacer(1, 3), _table(table, widths, st)]
    if done:
        block += [Spacer(1, 3), Paragraph(f"<b>Done when:</b> {done}", st["body"])]
    story.append(KeepTogether(block[:4]))
    story.extend(block[4:])


def build_pdf(output_path: str) -> None:
    st = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4, leftMargin=54, rightMargin=54, topMargin=48, bottomMargin=48
    )
    W = A4[0] - 108
    story = [
        Paragraph("QuestionCraft — Phase 2 Report", st["title"]),
        Paragraph(
            "Productionization &amp; Deployment · RV College of Engineering · AI254TA Machine Learning "
            f"Operations (5th Sem) · {date.today():%B %Y}",
            st["sub"],
        ),
        Spacer(1, 8),
    ]

    _section(
        story,
        st,
        "1. Scope",
        "Phase 2 covers eight deliverables. Kubernetes and statistical drift monitoring are out of scope "
        "and have been removed from the repository; Docker Compose is the deployment target.",
        table=[
            ["Area", "In Phase 2?", "Where"],
            ["Model testing and data validation", "In", "src/monitoring/model_tests.py, tests/"],
            ["Model-risk and quality gates", "In", "src/monitoring/quality_gates.py"],
            [
                "Model registry and promotion",
                "In",
                "src/registry/model_manager.py, scripts/promote_model.py",
            ],
            ["FastAPI inference service", "In", "src/serving/"],
            ["Deployment strategies (blue-green, canary)", "In", "Registry + API (no Kubernetes)"],
            ["11 ML artifacts", "In", "src/registry/artifacts.py, GET /artifacts"],
            ["Observability (Prometheus, Grafana)", "In", "/metrics, monitoring/"],
            ["Docker and Compose", "In", "Dockerfile, docker-compose.yml"],
            ["CI/CD", "In", ".github/workflows/ci.yml"],
            ["Kubernetes", "<b>Out</b>", "Removed"],
            ["Drift monitoring", "<b>Out</b>", "Removed"],
        ],
        widths=[W * 0.42, W * 0.14, W * 0.44],
    )

    _section(
        story,
        st,
        "2. Model testing and data validation",
        "Rule-based checks implemented in the project (Great Expectations is not used).",
        bullets=[
            "Column existence, null ratio ≤ 1%, allowed label set, minimum question length 5, row count "
            "100–500,000, feature mean bounds, no duplicate rows.",
            "<b>Cleaning before splitting</b> (clean_text_dataset): the processed difficulty data fails the raw "
            "checks (22 short texts, 87,772 duplicate rows); cleaning takes it from 191,775 to 103,989 rows and "
            "all 8 checks pass.",
            "src/data/build_splits.py trains on public/proxy data and holds out real RVCE exam papers; any text "
            "that appears in the held-out set is removed from training.",
        ],
        done="pytest tests/ passes and the pipeline's data-validation step reports all checks passed.",
    )

    _section(
        story,
        st,
        "3. Model-risk and quality gates",
        "No model reaches Production unless every gate passes. Metrics come from the candidate's real MLflow "
        "run and are compared with the current Production version.",
        table=[
            ["Gate", "Rule"],
            ["Minimum accuracy", "≥ 0.85"],
            ["Minimum F1-macro", "≥ 0.30"],
            ["No regression", "F1 drop vs Production ≤ 2.0%"],
            ["Stability", "5-fold CV std ≤ 0.05"],
            ["Reproducibility", "Training-data MD5 matches Production's"],
            ["Sample size", "≥ 100 test samples"],
        ],
        widths=[W * 0.35, W * 0.65],
        done="a model that fails any gate is refused promotion — covered by TestPromotionIsRefused, which "
        "runs the promotion CLI against a temporary registry and checks Production is unchanged.",
    )

    _section(
        story,
        st,
        "4. Model registry and promotion",
        bullets=[
            "MLflow Model Registry (sqlite:///mlflow.db): None → Staging → Production → Archived.",
            "Each version is tagged with data_hash, data_file, model_type, run_name and key metrics.",
            "CLI: --list, --to staging|production (gated), --compare A B, --rollback, --sync-tags.",
        ],
        done="a model can be registered, staged, promoted, compared and rolled back from the CLI.",
    )

    _section(
        story,
        st,
        "5. FastAPI inference service",
        table=[
            ["Endpoint", "Purpose"],
            ["GET /health", "Status, uptime, loaded models"],
            ["POST /predict, /classify, /batch/classify", "Difficulty, Bloom's level, CT skills"],
            ["POST /dedup, /batch/dedup", "Semantic duplicate check"],
            ["GET /models/info", "Serving versions, canary version and ratio, registry stages"],
            ["POST /models/reload", "Pick up a promotion or rollback without restarting"],
            ["GET /artifacts", "The 11-artifact manifest"],
            ["GET /deploy/status, POST /deploy/canary", "Deployment strategy and canary ratio"],
            ["GET /metrics", "Prometheus text format"],
        ],
        widths=[W * 0.42, W * 0.58],
        done="all endpoints answer, input is validated, and tests/test_serving.py passes.",
    )

    _section(
        story,
        st,
        "6. Deployment strategies",
        bullets=[
            "<b>Blue-green:</b> Production serves 100%; the Staging candidate is promoted only after the gates "
            "pass; POST /models/reload swaps it in; --rollback restores the previous version.",
            "<b>Canary:</b> ModelStore routes a configurable share of requests (0.0–1.0) to the Staging model; "
            "POST /deploy/canary?ratio=0.1 sets 10%.",
            "The evaluator demo drives the real API and prints /models/info at baseline, canary 10%, after "
            "promotion (v6 → Production) and after rollback (v5 restored).",
        ],
        done="a demo shows a canary at 10%, a promotion and a rollback, with /models/info reflecting each step.",
    )

    _section(
        story,
        st,
        "7. The 11 ML artifacts",
        "Defined once in src/registry/artifacts.py, served by GET /artifacts and verified by the pipeline.",
        table=[
            ["#", "Artifact", "Location"],
            ["1", "Dataset", "DVC-versioned Parquet in data/processed/ (+ data/splits/)"],
            ["2", "Preprocessing pipeline", "src/data/preprocess.py, src/data/build_splits.py"],
            ["3", "Trained models", "models/*.pkl (difficulty, Bloom's, CT)"],
            ["4", "Model configuration", "configs/model_config.yaml"],
            ["5", "Evaluation results", "MLflow metrics, incl. rvce_* held-out metrics"],
            ["6", "Training logs", "MLflow run telemetry"],
            ["7", "Feature engineering", "src/data/feature_engineering.py"],
            ["8", "Tokenizer / vectorizer", "TF-IDF saved inside each model pipeline"],
            ["9", "Model metadata", "MLflow registry version tags"],
            ["10", "Deployment artifacts", "Dockerfile, docker-compose.yml, CI workflow"],
            [
                "11",
                "Monitoring artifacts",
                "monitoring/prometheus.yml, Grafana dashboard + provisioning",
            ],
        ],
        widths=[W * 0.06, W * 0.28, W * 0.66],
    )

    _section(
        story,
        st,
        "8. Observability",
        "Prometheus scrapes /metrics every 10 s; Grafana shows a provisioned dashboard on port 3000. "
        "Observability covers service health and traffic only.",
        bullets=[
            "questioncraft_requests_total, questioncraft_request_latency_seconds_total, "
            "questioncraft_predictions_total, questioncraft_models_loaded, questioncraft_uptime_seconds."
        ],
    )

    _section(
        story,
        st,
        "9. Docker",
        bullets=[
            "Multi-stage Dockerfile (python:3.13-slim); runtime dependencies pinned to the versions the models "
            "were trained with (requirements-serving.txt); CPU-only torch; one Uvicorn worker; HEALTHCHECK.",
            "docker-compose.yml: API :8000, MLflow :5000, Prometheus :9090, Grafana :3000. API and MLflow share "
            "the host registry so the API serves Production and can canary Staging.",
        ],
        done="docker compose up -d brings up all four services healthy and /docs loads on port 8000.",
    )

    _section(
        story,
        st,
        "10. CI/CD",
        bullets=[
            "Ruff and Black (strict) → PyTest with coverage on Python 3.10 / 3.11 → Phase 2 pipeline → Docker "
            "image build → container smoke test (/health 200 with all models loaded, /predict, /metrics).",
            "Pushing the image to a registry on main is an open question and not part of Phase 2 yet.",
        ],
        done="the workflow is green on main and the smoke test passes.",
    )

    _section(
        story,
        st,
        "11. Honest evaluation",
        "Earlier headline numbers (0.97 accuracy / 0.95 F1) came from duplicate questions leaking between train "
        "and test. After de-duplication, and on real RVCE papers the models never trained on:",
        table=[
            ["Model", "In-domain test", "RVCE held-out", "Majority baseline"],
            ["Difficulty — LR (Production)", "acc 0.92 · F1 0.66", "acc 0.41", "0.46"],
            ["Difficulty — XGBoost (Staging)", "acc 0.95 · F1 0.69", "acc 0.46", "0.46"],
            ["Bloom's level", "acc 0.58", "acc 0.24", "0.47"],
            ["CT skills (multi-label)", "F1-micro 0.70", "F1-micro 0.36", "—"],
        ],
        widths=[W * 0.34, W * 0.24, W * 0.2, W * 0.22],
    )
    story.append(
        Paragraph(
            "RVCE difficulty and CT labels are LLM drafts pending faculty review, so these numbers are reported "
            "rather than used as promotion gates. Faculty-labelled RVCE data is the main lever for improving them.",
            st["body"],
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] PDF report written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default="QuestionCraft_Phase2_Report.pdf")
    build_pdf(parser.parse_args().out)
