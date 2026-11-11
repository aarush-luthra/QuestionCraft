"""Parse Bebras answer booklets (text extracted with pypdf) into QuestionCraft JSON.

CT mapping (Bebras skill tags -> QuestionCraft CT labels):
  AB Abstraction -> Abstraction, AL Algorithmic Thinking -> Algorithmic Thinking,
  DE Decomposition -> Decomposition, EV Evaluation -> Evaluation & Debugging,
  GE Generalisation -> Pattern Recognition (Bebras' "generalisation" = spotting/reusing patterns).
Difficulty: Bebras rates each task A/B/C per age group (Kits..Elite). We take the most frequent
letter (ties -> harder), A=Easy, B=Medium, C=Hard. This is a heuristic and is recorded as such.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

CT = {
    "AB": "Abstraction",
    "AL": "Algorithmic Thinking",
    "DE": "Decomposition",
    "EV": "Evaluation & Debugging",
    "GE": "Pattern Recognition",
}
DIFF = {"A": "Easy", "B": "Medium", "C": "Hard"}
SOURCES = {
    "uk_2014": "https://www.curriculumonline.ie/getmedia/f87eb61c-b7dd-4171-9401-9cc414f228ae/ukbebras2014-answers_1.pdf",
    "uk_2015": "https://www.curriculumonline.ie/getmedia/9c84edf3-3bef-421d-8c7b-f8682d6ffe8e/ukbebras2015-answers_1.pdf",
    "uk_2016": "https://www.curriculumonline.ie/getmedia/da0c349c-d205-47ff-9b43-6c820a62807c/uk-bebras-2016-answers.pdf",
    "au_2014": "https://amt.edu.au/wp-content/uploads/2024/02/2014-Bebras-Solution-Guide.pdf",
    "au_2015": "https://amt.edu.au/wp-content/uploads/2024/02/2015-Bebras-Solution-Guide.pdf",
}
AGE_ROW = re.compile(
    r"((?:[ABC]\s*){0,6})Kits:\s*Castors:\s*Juniors:\s*Intermediates?:\s*Seniors:\s*Elite:[ \t]*((?:[ABC][ \t\n]*){0,6})"
)
STOP = re.compile(r"Answer:|Explanation:|It’s Computational|It's Computational")


def clean(s: str) -> str:
    s = s.replace("!", " ").replace("ﬁ", "fi").replace("ﬂ", "fl").replace(" ", " ")
    s = re.sub(r"-\s?(?=[a-z])", "", s) if False else s
    return re.sub(r"\s+", " ", s).strip()


def parse_page(page: str):
    m = AGE_ROW.search(page)
    page = page.replace("\u2028", "\n").replace("\xa0", " ")
    skills_m = re.search(
        r"(?:Skills|Concepts)\s*-\s*((?:[A-Za-z ]+\((?:AB|AL|DE|EV|GE)\),?\s*)+)", page
    )
    if not m or not skills_m:
        return None
    letters = re.findall(r"[ABC]", m.group(1)) + ([m.group(2)] if m.group(2) else [])
    codes = re.findall(r"\((AB|AL|DE|EV|GE)\)", skills_m.group(1))
    body = page[m.end() :]
    stop = STOP.search(body)
    q = re.search(
        r"Question:\s*(.*?)(?=Explanation:|Answer:|It’s Computational|\n\n|$)", body, re.S
    )
    head = body[: stop.start()] if stop else body.split("\n\n")[0]
    if q and (not stop or q.start() > stop.start()):
        statement = head + " Question: " + q.group(1)
    elif q:
        statement = body[: q.start()] + " Question: " + q.group(1)
    else:
        statement = head
    title = clean(page.strip().split("\n")[0])
    title = re.sub(r"^\d+\s*", "", title)
    statement = clean(statement)
    if len(statement) < 40:
        return None
    cnt = Counter(letters)
    top = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return {
        "title": title,
        "question": statement,
        "ct_skills": sorted({CT[c] for c in codes}),
        "difficulty": DIFF[top],
        "bebras_levels_raw": "".join(letters),
    }


TAG = re.compile(r"(?:Skills|Concepts)\s*-\s*((?:[A-Za-z ]+\((?:AB|AL|DE|EV|GE)\),?\s*)+)")


def parse_stream(text: str):
    text = text.replace("\u2028", "\n").replace("\xa0", " ")
    page_starts = [m.start() for m in re.finditer("=====PAGE=====", text)]
    anchors = list(AGE_ROW.finditer(text))
    tags = list(TAG.finditer(text))
    out = []
    for k, m in enumerate(anchors):
        lo = anchors[k - 1].end() if k else 0
        hi = anchors[k + 1].start() if k + 1 < len(anchors) else len(text)
        cands = [t for t in tags if lo <= t.start() < hi]
        if not cands:
            continue
        t = min(cands, key=lambda t: abs(t.start() - m.start()))
        codes = re.findall(r"\((AB|AL|DE|EV|GE)\)", t.group(1))
        after = m.group(2)
        body_start = m.end()
        if after and text[m.end() : m.end() + 1].islower():  # "Elite:Aprincess": "A" is an article
            after = after.rstrip()[:-1]
            body_start = m.start(2) + len(after)
        letters = re.findall(r"[ABC]", m.group(1) + after)
        if not letters:
            continue
        body = text[body_start:hi].replace("=====PAGE=====", "\n\n")
        stop = STOP.search(body)
        q = re.search(
            r"Question:\s*(.*?)(?=Explanation:|Answer:|It\u2019s Computational|\n\n|$)", body, re.S
        )
        head = body[: stop.start()] if stop else body.split("\n\n")[0]
        if q and (not stop or q.start() > stop.start()):
            statement = head + " Question: " + q.group(1)
        elif q:
            statement = body[: q.start()] + " Question: " + q.group(1)
        else:
            statement = head
        statement = re.sub(r"^(?:[ABC]\s+)+(?=[A-Z])", "", clean(statement))
        if len(statement) < 40:
            continue
        cnt = Counter(letters)
        top = max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
        page = sum(1 for ps in page_starts if ps < m.start())
        out.append(
            (
                page,
                {
                    "question": statement,
                    "ct_skills": sorted({CT[c] for c in codes}),
                    "difficulty": DIFF[top],
                    "bebras_levels_raw": "".join(letters),
                },
            )
        )
    return out


def main(txt_dir: str, out_path: str):
    rows, seen = [], set()
    per_file = {}
    for name, url in SOURCES.items():
        f = Path(txt_dir) / f"{name}.txt"
        if not f.exists():
            continue
        n = 0
        for i, r in parse_stream(f.read_text()):
            key = re.sub(r"\W", "", r["question"].lower())[:200]
            if key in seen:  # AU 2015 reuses UK 2015 tasks
                continue
            seen.add(key)
            n += 1
            rows.append(
                {
                    "id": f"BEBRAS-{name.upper()}-P{i:03d}-{n:02d}",
                    "subject": "Bebras Computational Thinking Challenge",
                    "exam_type": "Bebras",
                    "exam": f"Bebras {name[:2].upper()} {name[3:]}",
                    "question": r["question"],
                    "marks": None,
                    "blooms_level": None,
                    "ct_skills": r["ct_skills"],
                    "difficulty": r["difficulty"],
                    "bebras_levels_raw": r["bebras_levels_raw"],
                    "label_source": {
                        "ct_skills": "bebras_official_tag",
                        "difficulty": "bebras_age_group_level_heuristic",
                    },
                    "source": f"{url}#page={i + 1}",
                }
            )
        per_file[name] = n
    data = {
        "metadata": {
            "dataset": "Bebras Computational Thinking tasks (UK 2014-2016, AU 2014-2015 answer booklets)",
            "license_note": "Bebras tasks are shared by the international Bebras community, typically under CC BY-NC-SA. Non-commercial educational/research use; check each booklet before redistribution.",
            "ct_mapping": CT,
            "difficulty_rule": "most frequent A/B/C age-group level (ties -> harder); A=Easy, B=Medium, C=Hard",
            "per_file_tasks": per_file,
            "total_questions": len(rows),
        },
        "questions": rows,
    }
    Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(per_file, "total", len(rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
