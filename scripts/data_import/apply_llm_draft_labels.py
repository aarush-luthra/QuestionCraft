"""Apply LLM-drafted difficulty / CT labels to the real RVCE question banks.

These labels were drafted by Claude (an LLM) from the question text, marks and printed
Bloom's level, using the rubric below. They are DRAFTS for faculty review and are tagged
label_source = "llm_draft" so the pipeline can weight or exclude them.

Difficulty rubric
  E (Easy)   - recall / define / fill-in / list; typically 1-2 marks, L1-L2.
  M (Medium) - explain or discuss one concept, short applied question; typically 4-10 marks, L2-L3.
  H (Hard)   - multi-part scenario, design or analyse/evaluate (L3+ with >= 6-10 marks).
CT codes (MLOps only; management/economics questions are left unlabelled)
  D Decomposition, P Pattern Recognition, A Abstraction, G Algorithmic Thinking,
  E Evaluation & Debugging, "-" no CT skill dominant.

Usage: python scripts/data_import/apply_llm_draft_labels.py
"""

import json
from pathlib import Path

DIFF = {"E": "Easy", "M": "Medium", "H": "Hard"}
CT = {
    "D": "Decomposition",
    "P": "Pattern Recognition",
    "A": "Abstraction",
    "G": "Algorithmic Thinking",
    "E": "Evaluation & Debugging",
}

# Strings are in the same question order as the bank (per paper key).
MLOPS_DIFF = {
    "CIE1-B22": "EEEEMEMMMMHMMHM",
    "CIE2-B22": "EEMEMMMMMHMMMM",
    "CIE3-B22": "HMMMMHMM",
    "CIE1-B23": "MHMMHMHM",
    "CIE2-B23": "MMMHMMHMH",
    "CIE3-B23": "MHEMHH",
    "SEE-B22": "EEEEMMEEEE" + "MHMMMMHHMMHMMMMMMM",
    "SEE-B23": "EEEEMEEEMEE" + "HMMMMHMMMHMMMHMMMMM",
}
MLOPS_CT = {
    "CIE1-B22": "- - - - PE - - E - E DGA E D DA E",
    "CIE2-B22": "- A PA A D - - - E DGE A - E DE",
    "CIE3-B22": "DG E E E P PE - -",
    "CIE1-B23": "- D D - DE E A AP",
    "CIE2-B23": "- GE - E D DG E A DE",
    "CIE3-B23": "E - - DA E DGA",
    "SEE-B22": "P E - - - - - - - E - - AP E D A DE - E E PE D DG PE - E E -",
    "SEE-B23": "- - - - E - E - P - - DGA - A D E DA - E - DE E G GE E PE E E A E",
}
POME_DIFF = {
    "CIE1-B22": "EEEEEMEEE" + "MMMMHH",
    "CIE2-B22": "EEEEEEEEE" + "MMMMH",
    "CIE3-B22": "EEEEEEEEEE" + "HMMMM",
    "CIE1-B22E": "EEEEE" + "MMHHM",
    "CIE2-B22E": "EEEEM" + "MHHMH",
    "CIE1-B23": "EEEEE" + "MMMHM",
    "CIE2-B23": "EEEEEEEEEE" + "MMMMM",
    "CIE3-B23": "EEEEE" + "MMMMM",
    "SEE-B22": "E" * 14 + "M" * 18,
    "SEE-B23": "EEEEMMEEEE" + "MMHMHHMMMMMMMMMMMH",
    "SEE-MODEL22": "E" * 20 + "MMMMMMMMMMMHHMMMMM",
    "SEE-18HEM51-2021JAN": "E" * 20 + "HHHMHMMMHMMMMM",
    "SEE-18HEM51-2020DEC": "E" * 19 + "M" * 14,
    "SEE-18HEM61-2021NOV": "E" * 19 + "MMMHH" + "M" * 15,
    "SEE-16HEM51-2019DEC": "E" * 10 + "M" * 7,
    "SEE-16HEM51-2021JAN": "E" * 5 + "M" * 7,
}


def paper_key(qid: str, prefix: str) -> str:
    rest = qid[len(prefix) :]
    return rest.rsplit("-", 1)[0]


def apply(path: Path, prefix: str, diff: dict, ct: dict | None) -> None:
    data = json.loads(path.read_text())
    order: dict[str, list] = {}
    for q in data["questions"]:
        order.setdefault(paper_key(q["id"], prefix), []).append(q)
    for key, qs in order.items():
        d = diff[key]
        assert len(d) == len(qs), f"{path.name} {key}: {len(d)} labels for {len(qs)} questions"
        c = ct[key].split() if ct else None
        if c is not None:
            assert len(c) == len(
                qs
            ), f"{path.name} {key}: {len(c)} CT labels for {len(qs)} questions"
        for i, q in enumerate(qs):
            q["difficulty"] = DIFF[d[i]]
            q.setdefault("label_source", {})["difficulty"] = "llm_draft"
            if c is not None:
                q["ct_skills"] = [] if c[i] == "-" else sorted(CT[ch] for ch in c[i])
                q["label_source"]["ct_skills"] = "llm_draft"
    data["metadata"]["llm_draft_note"] = (
        "difficulty"
        + (" and ct_skills" if ct else "")
        + " drafted by an LLM (Claude) from question text, marks and printed BT; pending faculty review."
    )
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"{path.name}: labelled {sum(len(v) for v in order.values())} questions")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "data" / "raw" / "rvce"
    apply(root / "ai254ta_mlops_qp.json", "RVCE-AI254TA-", MLOPS_DIFF, MLOPS_CT)
    apply(root / "hs251ta_pome_qp.json", "RVCE-POME-", POME_DIFF, None)
