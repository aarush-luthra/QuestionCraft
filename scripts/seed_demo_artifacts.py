"""
scripts/seed_demo_artifacts.py
==============================
Seeds realistic mock question datasets and trains baseline models
for QuestionCraft Phase 2 demonstration and pipeline validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
np.random.seed(SEED)

RVCE_QUESTIONS = [
    # Easy / L1 / L2
    {"question": "Define a binary search tree and state its properties.", "difficulty": "Easy", "blooms": "L1", "ct": ["Pattern Recognition"]},
    {"question": "State the time complexity of linear search and binary search.", "difficulty": "Easy", "blooms": "L1", "ct": ["Pattern Recognition"]},
    {"question": "What is an abstract data type? Give two examples.", "difficulty": "Easy", "blooms": "L1", "ct": ["Abstraction"]},
    {"question": "List the primary operations of a stack data structure with time complexities.", "difficulty": "Easy", "blooms": "L1", "ct": ["Algorithmic Thinking"]},
    {"question": "Define Big-O, Big-Omega, and Big-Theta asymptotic notations.", "difficulty": "Easy", "blooms": "L1", "ct": ["Pattern Recognition"]},
    {"question": "Explain how enqueue and dequeue operations work in a circular queue.", "difficulty": "Easy", "blooms": "L2", "ct": ["Algorithmic Thinking"]},
    {"question": "Describe the difference between singly linked list and doubly linked list.", "difficulty": "Easy", "blooms": "L2", "ct": ["Abstraction"]},
    {"question": "Explain the concept of hashing and describe separate chaining collision resolution.", "difficulty": "Easy", "blooms": "L2", "ct": ["Abstraction", "Pattern Recognition"]},
    {"question": "Summarize the BFS and DFS graph traversal strategies with examples.", "difficulty": "Easy", "blooms": "L2", "ct": ["Algorithmic Thinking"]},
    {"question": "Explain how a max-heap property is maintained during insertion.", "difficulty": "Easy", "blooms": "L2", "ct": ["Algorithmic Thinking"]},
    {"question": "Recall the conditions under which a graph has an Euler circuit.", "difficulty": "Easy", "blooms": "L1", "ct": ["Pattern Recognition"]},
    {"question": "Define dynamic programming and explain the principle of optimality.", "difficulty": "Easy", "blooms": "L1", "ct": ["Abstraction"]},
    
    # Medium / L3 / L4
    {"question": "Apply Dijkstra algorithm to compute single-source shortest paths in the given directed graph.", "difficulty": "Medium", "blooms": "L3", "ct": ["Algorithmic Thinking"]},
    {"question": "Construct an AVL tree by inserting the following sequence of integers: 10, 20, 30, 40, 50, 25.", "difficulty": "Medium", "blooms": "L3", "ct": ["Algorithmic Thinking", "Decomposition"]},
    {"question": "Solve the recurrence relation T(n) = 2T(n/2) + O(n) using the Master Theorem.", "difficulty": "Medium", "blooms": "L3", "ct": ["Algorithmic Thinking"]},
    {"question": "Implement an iterative binary tree postorder traversal using two stacks.", "difficulty": "Medium", "blooms": "L3", "ct": ["Algorithmic Thinking", "Decomposition"]},
    {"question": "Calculate the minimum spanning tree weight using Kruskal's algorithm and disjoint set union.", "difficulty": "Medium", "blooms": "L3", "ct": ["Algorithmic Thinking"]},
    {"question": "Differentiate between Prim's algorithm and Kruskal's algorithm in terms of edge vs vertex selection.", "difficulty": "Medium", "blooms": "L4", "ct": ["Abstraction", "Pattern Recognition"]},
    {"question": "Analyze the average-case and worst-case recursion tree of randomized quicksort.", "difficulty": "Medium", "blooms": "L4", "ct": ["Evaluation & Debugging", "Algorithmic Thinking"]},
    {"question": "Trace the execution of Floyd-Warshall algorithm on a 4-vertex graph and detect negative cycles.", "difficulty": "Medium", "blooms": "L4", "ct": ["Evaluation & Debugging", "Algorithmic Thinking"]},
    {"question": "Compare the space and time trade-offs between adjacency matrix and adjacency list representations.", "difficulty": "Medium", "blooms": "L4", "ct": ["Evaluation & Debugging", "Abstraction"]},
    {"question": "Deconstruct the 0/1 Knapsack problem into optimal substructure and overlapping subproblems.", "difficulty": "Medium", "blooms": "L4", "ct": ["Decomposition", "Abstraction"]},
    
    # Hard / L5 / L6
    {"question": "Design and implement a cache-aware B-Tree indexing engine for high-throughput persistent key-value storage.", "difficulty": "Hard", "blooms": "L6", "ct": ["Decomposition", "Algorithmic Thinking", "Abstraction"]},
    {"question": "Formulate an integer linear programming model to solve the multi-commodity network flow problem.", "difficulty": "Hard", "blooms": "L6", "ct": ["Abstraction", "Algorithmic Thinking"]},
    {"question": "Develop a lock-free concurrent hash map in C++ using atomic compare-and-swap primitives.", "difficulty": "Hard", "blooms": "L6", "ct": ["Decomposition", "Evaluation & Debugging"]},
    {"question": "Create an automated test paper assembly solver satisfying course outcome quotas, difficulty ratios, and Bloom limits.", "difficulty": "Hard", "blooms": "L6", "ct": ["Decomposition", "Algorithmic Thinking", "Abstraction"]},
    {"question": "Evaluate the Byzantine fault tolerance guarantees of Raft consensus vs Paxos under asymmetric network partitions.", "difficulty": "Hard", "blooms": "L5", "ct": ["Evaluation & Debugging", "Abstraction"]},
    {"question": "Critique the cache coherence latency overheads of MESI protocol in a 64-core NUMA multi-processor architecture.", "difficulty": "Hard", "blooms": "L5", "ct": ["Evaluation & Debugging"]},
    {"question": "Assess the computational complexity and approximation ratio of the Christofides algorithm for metric TSP.", "difficulty": "Hard", "blooms": "L5", "ct": ["Evaluation & Debugging", "Algorithmic Thinking"]},
    {"question": "Justify why dynamic programming is superior to branch-and-bound for the matrix chain multiplication problem.", "difficulty": "Hard", "blooms": "L5", "ct": ["Evaluation & Debugging"]},
    {"question": "Architect a distributed stream processing pipeline that achieves exactly-once event semantics using transactional checkpoints.", "difficulty": "Hard", "blooms": "L6", "ct": ["Decomposition", "Abstraction"]},
    {"question": "Synthesize a Red-Black tree deletion routine handling all double-black structural rebalancing cases.", "difficulty": "Hard", "blooms": "L6", "ct": ["Algorithmic Thinking", "Evaluation & Debugging"]},
]

# Expand dataset to ~300 representative questions with variations
expanded = []
for item in RVCE_QUESTIONS:
    expanded.append(item)
    # Add variations with pedagogical phrasing
    expanded.append({
        "question": f"For the following computing problem: {item['question']}",
        "difficulty": item["difficulty"],
        "blooms": item["blooms"],
        "ct": item["ct"],
    })
    expanded.append({
        "question": f"In context of Outcome-Based Education (RVCE CIE Assessment): {item['question']}",
        "difficulty": item["difficulty"],
        "blooms": item["blooms"],
        "ct": item["ct"],
    })
    expanded.append({
        "question": f"Consider a computer science syllabus question: {item['question']}",
        "difficulty": item["difficulty"],
        "blooms": item["blooms"],
        "ct": item["ct"],
    })

df = pd.DataFrame(expanded)
texts = df["question"].tolist()

# 1. Save data/raw/rvce/question_bank.json
raw_rvce = Path("data/raw/rvce")
raw_rvce.mkdir(parents=True, exist_ok=True)
with open(raw_rvce / "question_bank.json", "w") as f:
    json.dump({
        "course": "AI254TA Machine Learning Operations",
        "institution": "RV College of Engineering",
        "questions": [{"id": f"q_{i}", "question": q["question"], "difficulty": q["difficulty"]} for i, q in enumerate(expanded)],
    }, f, indent=2)
print("Saved data/raw/rvce/question_bank.json")

# 2. Save processed datasets
proc_dir = Path("data/processed")
proc_dir.mkdir(parents=True, exist_ok=True)

df_diff = pd.DataFrame({
    "question_text": texts,
    "difficulty": df["difficulty"],
})
df_diff.to_parquet(proc_dir / "difficulty_dataset.parquet", index=False)

df_blooms = pd.DataFrame({
    "question_text": texts,
    "blooms_level": df["blooms"],
})
df_blooms.to_parquet(proc_dir / "blooms_dataset.parquet", index=False)

df_ct = pd.DataFrame({
    "question_text": texts,
    "ct_skills": df["ct"].apply(lambda lst: "|".join(lst)),
})
df_ct.to_parquet(proc_dir / "ct_dataset.parquet", index=False)
print("Saved data/processed/*.parquet datasets")

# 3. Train models
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

# Difficulty model
diff_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
])
diff_pipe.fit(texts, df["difficulty"])
joblib.dump(diff_pipe, models_dir / "difficulty_classifier.pkl")
print("Saved models/difficulty_classifier.pkl")

# Bloom's model
blooms_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
])
blooms_pipe.fit(texts, df["blooms"])
joblib.dump(blooms_pipe, models_dir / "blooms_classifier.pkl")
print("Saved models/blooms_classifier.pkl")

# CT model
ct_labels = [
    "Decomposition", "Pattern Recognition", "Abstraction",
    "Algorithmic Thinking", "Evaluation & Debugging",
]
mlb = MultiLabelBinarizer(classes=ct_labels)
Y_ct = mlb.fit_transform(df["ct"])
ct_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=500, random_state=SEED))),
])
ct_pipe.fit(texts, Y_ct)
joblib.dump({"pipeline": ct_pipe, "mlb": mlb, "threshold": 0.5}, models_dir / "ct_classifier.pkl")
print("Saved models/ct_classifier.pkl")

print("All demo artifacts successfully seeded!")

