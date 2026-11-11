"""
src/data/build_splits.py
========================
Build leak-free train / RVCE-held-out evaluation sets for every task.

Design
------
* TRAIN  = public + proxy data (ARC / RACE / SciQ / Bloom's CSV / Bebras / RVCE mock bank)
* EVAL   = real RVCE exam questions (AI254TA MLOps, HS251TA POME, HS248AT UHV), never trained on
* Every frame is cleaned BEFORE splitting (short texts, conflicting labels, exact duplicates),
  and any RVCE-eval text is removed from TRAIN so the held-out set never leaks.
* Rows carry `label_source`; sources listed in `splits.exclude_label_sources` are dropped from
  TRAIN (default: keyword_heuristic, i.e. Mendeley's keyword-derived Bloom's levels).

Outputs (data/splits/):
  {difficulty,blooms,ct}_train.parquet   question_text, label, source, label_source
  {difficulty,blooms,ct}_rvce_eval.parquet
  split_report.json                      counts, removals, label-source mix, hashes

Usage:
    python -m src.data.build_splits --config configs/model_config.yaml
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import click
import pandas as pd
import yaml

from src.monitoring.model_tests import clean_text_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("questioncraft.build_splits")

BLOOMS_NORM = {
    "1": "L1",
    "l1": "L1",
    "remember": "L1",
    "knowledge": "L1",
    "2": "L2",
    "l2": "L2",
    "understand": "L2",
    "comprehension": "L2",
    "3": "L3",
    "l3": "L3",
    "apply": "L3",
    "application": "L3",
    "4": "L4",
    "l4": "L4",
    "analyze": "L4",
    "analysis": "L4",
    "5": "L5",
    "l5": "L5",
    "evaluate": "L5",
    "evaluation": "L5",
    "6": "L6",
    "l6": "L6",
    "create": "L6",
    "creation": "L6",
    "synthesis": "L6",
}
CT_LABELS = [
    "Decomposition",
    "Pattern Recognition",
    "Abstraction",
    "Algorithmic Thinking",
    "Evaluation & Debugging",
]


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def md5(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def load_bank(path: Path, bank_name: str) -> pd.DataFrame:
    """Load a QuestionCraft-schema JSON bank into a flat frame (all three label columns)."""
    rows = []
    for q in json.loads(path.read_text()).get("questions", []):
        ct = q.get("ct_skills", q.get("ct_skill"))
        if isinstance(ct, str):
            ct = [ct]
        ls = q.get("label_source") or {}
        rows.append(
            {
                "question_text": q["question"],
                "difficulty": q.get("difficulty"),
                "blooms_level": BLOOMS_NORM.get(
                    str(q.get("blooms_level") or q.get("btl") or "").strip().lower()
                ),
                "ct_skills": None if ct is None else "|".join(c for c in ct if c),
                "source": bank_name,
                "ls_difficulty": ls.get("difficulty", "unknown"),
                "ls_blooms": ls.get("blooms_level", "unknown"),
                "ls_ct": ls.get("ct_skills", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def task_frame(df: pd.DataFrame, label_col: str, ls_col: str) -> pd.DataFrame:
    out = df[["question_text", label_col, "source", ls_col]].rename(
        columns={label_col: "label", ls_col: "label_source"}
    )
    return out.dropna(subset=["question_text", "label"])


def clean(df: pd.DataFrame, name: str, report: dict) -> pd.DataFrame:
    if df.empty:
        report[name] = {"rows_before": 0, "rows_after": 0}
        return df
    out, rep = clean_text_dataset(df, "question_text", "label")
    report[name] = rep
    return out


@click.command()
@click.option("--config", default="configs/model_config.yaml", show_default=True)
def main(config: str) -> None:
    cfg = yaml.safe_load(open(config))
    data_cfg = cfg["data"]
    split_cfg = cfg.get("splits", {})
    processed = Path(data_cfg["processed_dir"])
    out_dir = Path(split_cfg.get("out_dir", "data/splits"))
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded = set(split_cfg.get("exclude_label_sources", ["keyword_heuristic"]))
    eval_banks = set(split_cfg.get("rvce_eval_banks", []))

    banks = {
        Path(p).stem: load_bank(Path(p), f"rvce_{Path(p).stem}")
        for p in data_cfg.get("rvce_banks", [])
        if Path(p).exists()
    }
    public = {
        Path(p).stem: load_bank(Path(p), f"public_{Path(p).stem}")
        for p in data_cfg.get("public_banks", [])
        if Path(p).exists()
    }

    eval_df = pd.concat(
        [banks[b] for b in banks if b in eval_banks] or [pd.DataFrame()], ignore_index=True
    )
    train_banks = pd.concat(
        [banks[b] for b in banks if b not in eval_banks] + list(public.values()), ignore_index=True
    )
    eval_texts = set(eval_df["question_text"].map(norm_text)) if not eval_df.empty else set()
    logger.info(
        "RVCE eval banks: %s (%d questions) | train banks: %s",
        sorted(eval_banks),
        len(eval_df),
        sorted(set(banks) - eval_banks) + sorted(public),
    )

    # Existing proxy datasets from preprocess (exclude any RVCE rows - they are re-added from JSON)
    def proxy(name: str, label: str) -> pd.DataFrame:
        f = processed / f"{name}_dataset.parquet"
        if not f.exists():
            return pd.DataFrame(columns=["question_text", "label", "source", "label_source"])
        p = pd.read_parquet(f, columns=["question_text", label, "source"])
        p = p[~p["source"].astype(str).str.startswith("rvce_")]
        p = p.rename(columns={label: "label"})
        p["label_source"] = "proxy_dataset"
        return p

    report: dict = {
        "excluded_label_sources": sorted(excluded),
        "eval_banks": sorted(eval_banks),
        "tasks": {},
    }
    for task, label_col, ls_col in (
        ("difficulty", "difficulty", "ls_difficulty"),
        ("blooms", "blooms_level", "ls_blooms"),
        ("ct", "ct_skills", "ls_ct"),
    ):
        tr = pd.concat(
            [
                proxy(
                    task,
                    {"difficulty": "difficulty", "blooms": "blooms_level", "ct": "ct_skills"}[task],
                ),
                task_frame(train_banks, label_col, ls_col),
            ],
            ignore_index=True,
        )
        if task == "blooms":
            tr["label"] = tr["label"].astype(str).str.strip().str.lower().map(BLOOMS_NORM)
            tr = tr.dropna(subset=["label"])
        if task == "ct":
            tr = tr[tr["label"].astype(str).str.strip() != ""]
        if task == "difficulty":
            tr = tr[tr["label"].isin(["Easy", "Medium", "Hard"])]
        rep: dict = {"train_raw": len(tr)}
        tr = tr[~tr["label_source"].isin(excluded)]
        rep["train_after_label_source_filter"] = len(tr)
        leak = tr["question_text"].map(norm_text).isin(eval_texts)
        rep["removed_eval_overlap"] = int(leak.sum())
        tr = tr[~leak]
        cleaning: dict = {}
        tr = clean(tr, "train", cleaning)
        ev = clean(
            task_frame(eval_df, label_col, ls_col) if not eval_df.empty else pd.DataFrame(),
            "eval",
            cleaning,
        )
        if task == "ct" and not ev.empty:
            ev = ev[ev["label"].astype(str).str.strip() != ""]
        rep["cleaning"] = cleaning
        rep["train_rows"], rep["eval_rows"] = len(tr), len(ev)
        rep["train_label_dist"] = (
            tr["label"].value_counts().to_dict()
            if task != "ct"
            else {c: int(tr["label"].str.contains(re.escape(c)).sum()) for c in CT_LABELS}
        )
        rep["eval_label_dist"] = (
            ev["label"].value_counts().to_dict()
            if task != "ct"
            else (
                {c: int(ev["label"].str.contains(re.escape(c)).sum()) for c in CT_LABELS}
                if not ev.empty
                else {}
            )
        )
        rep["train_label_sources"] = tr["label_source"].value_counts().to_dict()
        rep["eval_label_sources"] = (
            ev["label_source"].value_counts().to_dict() if not ev.empty else {}
        )
        tr.to_parquet(out_dir / f"{task}_train.parquet", index=False)
        ev.to_parquet(out_dir / f"{task}_rvce_eval.parquet", index=False)
        rep["train_hash"], rep["eval_hash"] = md5(tr), (md5(ev) if not ev.empty else None)
        report["tasks"][task] = rep
        logger.info(
            "%-10s train=%6d eval=%4d (removed eval overlap: %d)",
            task,
            len(tr),
            len(ev),
            rep["removed_eval_overlap"],
        )

    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    logger.info("Split report → %s", out_dir / "split_report.json")


if __name__ == "__main__":
    main()
