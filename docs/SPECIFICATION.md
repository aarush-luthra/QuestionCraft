# 🎓 QuestionCraft

> **An End-to-End MLOps Pipeline for Automated Question Classification & OBE-Compliant Test Paper Assembly**
> Built for **RV College of Engineering**

---

## 📌 Project Overview

QuestionCraft is a production-grade MLOps pipeline designed to automate the classification of examination questions and intelligently assemble **Outcome-Based Education (OBE)** compliant test papers. It leverages machine learning and NLP to classify questions across three educational dimensions, enabling faculty to generate well-balanced, pedagogically sound assessments with minimal manual effort.

---

## 🧠 Core Classification Dimensions

The system uses **three independent ML models** to tag every question:

| Dimension | Labels | Purpose |
|---|---|---|
| **Difficulty** | Easy, Medium, Hard | Ensures balanced cognitive load across the paper |
| **Bloom's Taxonomy Level** | L1 (Remember) → L6 (Create) | Maps questions to higher-order thinking skills |
| **Computational Thinking (CT) Skill** | Decomposition, Pattern Recognition, Abstraction, Algorithmic Thinking, Evaluation & Debugging | Measures CS-specific reasoning skills |

---

## 🏗️ System Architecture

```
QuestionCraft
├── Phase 1 — Development & Reproducibility  ← (Current Phase)
│   ├── Data Acquisition & Feature Engineering
│   ├── Model Training (Difficulty, Bloom's, CT)
│   ├── MLflow Experiment Tracking & Model Registry
│   └── DVC Pipeline for Data & Code Versioning
│
└── Phase 2 — Serving & Assembly  (Upcoming)
    ├── FastAPI Backend (REST API for inference)
    ├── Evidently AI for Data Drift Monitoring
    └── ILP Solver for Constraint-Based Paper Assembly
```

---

## 🛠️ Phase 1 — Technical Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **ML / NLP** | scikit-learn, XGBoost, Transformers, sentence-transformers, textstat, NLTK |
| **MLOps — Tracking** | MLflow (experiment tracking, model registry) |
| **MLOps — Versioning** | DVC (Data Version Control for datasets & pipelines) |
| **Data Formats** | Parquet (processed datasets), JSON (RVCE question bank) |

---

## 📂 Repository Structure

```
QuestionCraft/
├── .github/workflows/ci.yml       # GitHub Actions CI workflow (linting & pytest)
├── configs/
│   └── model_config.yaml          # Pinned hyperparameters for all models
├── data/
│   ├── raw/
│   │   └── rvce/
│   │       ├── question_bank.json # Curated RVCE DSA questions with manual tags
│   │       └── uhv_hs248at.json   # Curated RVCE UHV questions
│   └── processed/                 # Processed Parquet files (DVC-tracked)
├── models/                        # Serialized model artifacts (.pkl)
├── notebooks/                     # Exploratory data analysis notebooks & generator
├── reports/                       # Standalone EDA HTML report & figure PNGs
├── scripts/                       # Environment verification & pipeline runners
├── src/
│   ├── data/
│   │   ├── download.py            # Downloads ARC, RACE, SciQ, Bloom's datasets
│   │   ├── feature_engineering.py # FeatureExtractor class (readability, POS, etc.)
│   │   └── preprocess.py          # Data normalizer and feature parquet builder
│   ├── models/
│   │   ├── difficulty_classifier.py   # TF-IDF + Logistic Regression / XGBoost
│   │   ├── blooms_classifier.py       # L1–L6 Bloom's Taxonomy classifier
│   │   ├── ct_classifier.py           # Multi-label CT skill classifier (OvR LR)
│   │   └── duplicate_detector.py      # Cosine similarity via sentence-transformers
│   └── utils/                     # Config loader, seed setter, dataframe MD5 hashing
├── docs/                          # Architecture guides & specifications
│   ├── QuestionCraft_Phase1_DeepDive.pdf # Phase 1 architecture & viva guide
│   └── SPECIFICATION.md           # Phase 1 specification & rubric guide
├── dvc.yaml                       # DVC pipeline definition
├── pyproject.toml                 # Pinned dependencies & random seeds
└── README.md                      # Primary architecture documentation
```

---

## 🔬 Phase 1 — Tasks Breakdown

### 1. Project Scaffolding & Configuration

- **`pyproject.toml`** — All dependencies pinned with fixed random seeds for full reproducibility.
- **`configs/model_config.yaml`** — Central config file pinning all model hyperparameters (learning rates, regularization, n-estimators, etc.).

### 2. Data Acquisition & Feature Engineering

- **`src/data/download.py`** — Downloads open-access question datasets using the HuggingFace `datasets` library:
  - [ARC](https://huggingface.co/datasets/ai2_arc) — Grade school science questions
  - [RACE](https://huggingface.co/datasets/race) — Reading comprehension
  - [SciQ](https://huggingface.co/datasets/sciq) — Science question bank
  - Bloom's Kaggle dataset
  - Saves all datasets as `.parquet` files for efficient downstream processing.

- **`data/raw/rvce/question_bank.json`** — Sample RVCE question bank with mock questions manually tagged with:
  - Course Outcomes (CO1–CO5)
  - Bloom's Taxonomy Level
  - CT Skill
  - Marks

- **`src/data/feature_engineering.py`** — `FeatureExtractor` class that computes:
  - **Readability metrics**: Flesch Reading Ease, Coleman-Liau Index
  - **Structural features**: Word count, sentence length, POS tag distributions
  - **Semantic indicators**: Bloom's taxonomy verb detection, CT keyword matching

### 3. Model Training & MLflow Integration

All models log the following to a **local MLflow experiment**:
- Dataset hash (for traceability)
- All hyperparameters
- Evaluation metrics (accuracy, F1, precision, recall)
- Trained model artifacts

| Script | Model | Task |
|---|---|---|
| `difficulty_classifier.py` | TF-IDF + Logistic Regression | 3-class classification (Easy / Medium / Hard) |
| `blooms_classifier.py` | TF-IDF + Logistic Regression | 6-class classification (L1–L6) |
| `ct_classifier.py` | TF-IDF + One-Vs-Rest LR | Multi-label classification (5 CT skills) |
| `duplicate_detector.py` | `all-MiniLM-L6-v2` + Cosine Similarity | Semantic duplicate detection |

### 4. DVC Pipeline

**`dvc.yaml`** defines a fully reproducible pipeline with the following stages:

```
download → preprocess → train_difficulty
                     → train_blooms
                     → train_ct
```

DVC tracks all raw and processed data directories, ensuring that any data change triggers re-execution of only the affected downstream stages.

---

## 🔄 MLflow Experiment Tracking

Each training run logs:

```
mlflow/
└── experiments/
    ├── difficulty_classifier/
    │   ├── params/   (tfidf__max_features, C, solver, seed, data_hash, ...)
    │   ├── metrics/  (accuracy, f1_macro, precision, recall)
    │   └── artifacts/ (model.pkl, confusion_matrix.png)
    ├── blooms_classifier/
    └── ct_classifier/
```

Run the MLflow UI locally:
```bash
mlflow ui --port 5000
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python --version  # 3.10+
pip install uv    # recommended for fast dependency resolution
```

### Installation

```bash
git clone <repo-url>
cd quespaper
uv pip install -e ".[dev]"  # installs from pyproject.toml
```

### Run the DVC Pipeline

```bash
# Initialize DVC (first time only)
dvc init

# Run the full pipeline
dvc repro

# Check pipeline status
dvc status
```

### View Experiment Results

```bash
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

---

## 📊 Datasets Used

| Dataset | Source | Used For |
|---|---|---|
| ARC (Easy + Challenge) | HuggingFace / Allen AI | Difficulty classification baseline |
| RACE | HuggingFace | Reading-level difficulty features |
| SciQ | HuggingFace | Science domain question features |
| Bloom's Dataset | Kaggle | Bloom's taxonomy level supervision |
| RVCE Question Bank | Internal (mock) | Domain-specific fine-tuning & validation |

---

## 🔭 Phase 2 Roadmap (Upcoming)

| Feature | Technology | Description |
|---|---|---|
| REST API | FastAPI | Inference endpoints for question classification |
| Drift Monitoring | Evidently AI | Detect feature/prediction drift in production |
| Paper Assembly | PuLP / OR-Tools (ILP) | Constraint-based test paper generation respecting CO coverage, difficulty distribution, and marks |
| CI/CD | GitHub Actions | Automated pipeline re-training on data updates |

---

## 🏫 About

QuestionCraft is developed in the context of **Outcome-Based Education (OBE)** as mandated by accreditation bodies like NBA/NAAC for engineering colleges in India. The goal is to reduce manual effort in question paper setting while ensuring:

- ✅ Balanced cognitive difficulty distribution
- ✅ Coverage of all Course Outcomes (CO1–CO5)
- ✅ Alignment with Bloom's Taxonomy
- ✅ Assessment of Computational Thinking competencies
- ✅ Zero duplicate or near-duplicate questions

---

*Built with ❤️ for RV College of Engineering | Phase 1 — Development & Reproducibility*
