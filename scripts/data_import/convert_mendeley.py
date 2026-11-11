"""Convert Mendeley w5zt9n6vsc (CS student questions) to QuestionCraft JSON.
The dataset only ships a cumulative keyword score, so the Bloom's level is recomputed from the
dataset's own Blooms_Taxonomy.csv keyword list: level = highest level whose keyword appears.
These are WEAK labels (label_source = keyword_heuristic)."""

import csv
import json
import re
import sys
from pathlib import Path

LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
src, out = Path(sys.argv[1]), Path(sys.argv[2])
kw = {lvl: set() for lvl in LEVELS}
with open(src / "Blooms_Taxonomy.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        for lvl in LEVELS:
            if row.get(lvl):
                kw[lvl].add(row[lvl].strip().lower())
pat = {
    lvl: re.compile(
        r"\b(" + "|".join(map(re.escape, sorted(kw[lvl], key=len, reverse=True))) + r")\b"
    )
    for lvl in LEVELS
}
rows, seen, nomatch = [], set(), 0
for fname, course in [
    ("Data_Structure.csv", "Data Structures"),
    ("Intro_Computers_Research.csv", "Introduction to Computers and Research"),
]:
    with open(src / fname, encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            q = (r.get("Questions") or "").strip()
            if len(q) < 10 or q.lower() in seen:
                continue
            seen.add(q.lower())
            hit = [idx for idx, lvl in enumerate(LEVELS) if pat[lvl].search(q.lower())]
            if not hit:
                nomatch += 1
                continue
            top = max(hit)
            rows.append(
                {
                    "id": f"MENDELEY-{fname.split('_')[0].upper()}-{i:04d}",
                    "subject": course,
                    "topic": r.get("Topic"),
                    "exam_type": "student_query",
                    "question": q,
                    "marks": None,
                    "blooms_level": f"L{top + 1}",
                    "blooms_label": LEVELS[top],
                    "ct_skills": None,
                    "difficulty": None,
                    "keyword_score": r.get("Score"),
                    "label_source": {"blooms_level": "keyword_heuristic"},
                    "source": "https://data.mendeley.com/datasets/w5zt9n6vsc/4",
                }
            )
data = {
    "metadata": {
        "dataset": "Mendeley w5zt9n6vsc v4 - CS course student queries (Independent University, Bangladesh)",
        "doi": "10.17632/w5zt9n6vsc.4",
        "license_note": "CC BY 4.0 per Mendeley Data listing; verify before redistribution.",
        "label_note": "Bloom's level recomputed as the highest level whose keyword (from Blooms_Taxonomy.csv) appears. Weak labels.",
        "dropped_no_keyword": nomatch,
        "total_questions": len(rows),
    },
    "questions": rows,
}
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
from collections import Counter

print(len(rows), "kept |", nomatch, "no keyword |", Counter(r["blooms_level"] for r in rows))
