"""Build the synthetic (LLM-authored) CS question banks used for the demo.

Outputs (QuestionCraft bank schema, so src/data/build_splits.py loads them unchanged):
  data/raw/public/synthetic_cs_questions.json   120 questions  -> TRAIN only (public_banks)
  data/raw/synthetic/synthetic_holdout.json      36 questions  -> never trained on; scored by
                                                                 scripts/eval_synthetic.py

Every label is an LLM draft (label_source = "synthetic_llm"), NOT faculty-verified. Scores on the
synthetic holdout are optimistic because the holdout is written in the same style as the train
bank; the real RVCE held-out set remains the honest number.

Usage: python scripts/data_import/build_synthetic_bank.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CT = {
    "D": "Decomposition",
    "P": "Pattern Recognition",
    "A": "Abstraction",
    "G": "Algorithmic Thinking",
    "E": "Evaluation & Debugging",
}
E_, M_, H_ = "Easy", "Medium", "Hard"

# (question, bloom level, difficulty, CT codes)
TRAIN = [
    # ── L1 Remember ──────────────────────────────────────────────────────────
    ("Define a linked list and name its two main variants.", "L1", E_, "A"),
    ("List the four basic properties of a database transaction.", "L1", E_, "A"),
    ("State the time complexity of binary search.", "L1", E_, "G"),
    ("Name the layers of the OSI model.", "L1", E_, "D"),
    ("What is a primary key in a relational table?", "L1", E_, "A"),
    ("Identify the data structure used to implement recursion in a program.", "L1", E_, "A"),
    ("Recall the steps of the bubble sort algorithm.", "L1", E_, "G"),
    ("Define overfitting in machine learning.", "L1", E_, "E"),
    ("List the stages of the software development life cycle.", "L1", E_, "D"),
    ("State what a hash function does.", "L1", E_, "A"),
    ("Name the three types of SQL joins.", "L1", E_, "A"),
    ("What is a confusion matrix?", "L1", E_, "E"),
    ("Define the term MLOps.", "L1", E_, "A"),
    ("List the main components of a CPU.", "L1", E_, "D"),
    ("Write the formula for the F1 score.", "L1", M_, "E"),
    ("Name the traversal orders of a binary tree.", "L1", E_, "P"),
    ("What does DNS stand for and what is it used for?", "L1", E_, "A"),
    ("Define data versioning.", "L1", E_, "A"),
    ("List the operations supported by a stack.", "L1", E_, "G"),
    ("Recall the definition of a deadlock.", "L1", E_, "A"),
    # ── L2 Understand ────────────────────────────────────────────────────────
    ("Explain how a queue differs from a stack with an example.", "L2", E_, "P"),
    ("Describe how the quicksort algorithm partitions an array around a pivot.", "L2", M_, "GD"),
    ("Summarize why normalization is used in relational database design.", "L2", M_, "A"),
    ("Explain the difference between supervised and unsupervised learning.", "L2", E_, "P"),
    ("Describe how TF-IDF weights a word in a document.", "L2", M_, "AG"),
    ("Interpret a ROC curve and explain what the area under it represents.", "L2", M_, "E"),
    ("Explain how a hash table handles collisions using chaining.", "L2", M_, "G"),
    ("Compare process and thread with respect to memory sharing.", "L2", M_, "P"),
    ("Describe how DVC tracks large data files alongside Git.", "L2", M_, "A"),
    ("Explain why cross-validation gives a more reliable estimate than a single split.", "L2", M_, "E"),
    ("Classify merge sort, quick sort and heap sort as stable or unstable.", "L2", M_, "P"),
    ("Explain the role of the model registry in an MLOps workflow.", "L2", E_, "A"),
    ("Describe how a TCP three-way handshake establishes a connection.", "L2", M_, "G"),
    ("Summarize the main advantages of containerization with Docker.", "L2", E_, "A"),
    ("Explain how breadth-first search explores a graph level by level.", "L2", M_, "G"),
    ("Describe the purpose of an index in a database.", "L2", E_, "A"),
    ("Explain what regularization does to a model's weights.", "L2", M_, "A"),
    ("Paraphrase the idea of divide and conquer in your own words.", "L2", E_, "D"),
    ("Explain the difference between precision and recall.", "L2", M_, "E"),
    ("Describe how gradient descent updates parameters.", "L2", M_, "G"),
    # ── L3 Apply ─────────────────────────────────────────────────────────────
    ("Implement a function that reverses a singly linked list.", "L3", M_, "GD"),
    ("Solve the recurrence T(n) = 2T(n/2) + n using the master theorem.", "L3", M_, "PG"),
    ("Apply Dijkstra's algorithm to find shortest paths from node A in the given weighted graph.", "L3", M_, "G"),
    ("Write a SQL query to find the top three customers by total order value.", "L3", M_, "DG"),
    ("Compute the F1 score for a classifier with 40 true positives, 10 false positives and 20 false negatives.", "L3", M_, "E"),
    ("Use a stack to evaluate the postfix expression 5 3 + 8 2 - *.", "L3", M_, "G"),
    ("Implement binary search iteratively and state its loop invariant.", "L3", M_, "GE"),
    ("Demonstrate how to insert the keys 10, 20, 5, 15 into an empty AVL tree and show the rotations.", "L3", H_, "G"),
    ("Calculate the average memory access time for a cache with a 90 percent hit rate, 2 ns hit time and 100 ns miss penalty.", "L3", M_, "G"),
    ("Apply k-fold cross-validation with k=5 to a dataset of 1000 samples and state the size of each fold.", "L3", E_, "DG"),
    ("Write a Dockerfile that packages a FastAPI service and exposes port 8000.", "L3", M_, "DA"),
    ("Use the Floyd cycle detection method to find whether a linked list has a loop.", "L3", M_, "GP"),
    ("Solve the 0/1 knapsack problem for weights 1,3,4,5 and values 1,4,5,7 with capacity 7.", "L3", H_, "DG"),
    ("Execute selection sort by hand on the array 29, 10, 14, 37, 13 showing each pass.", "L3", E_, "G"),
    ("Implement a function to check whether a string of brackets is balanced.", "L3", M_, "GP"),
    ("Calculate the cosine similarity between the vectors (1,2,3) and (2,4,6).", "L3", E_, "G"),
    ("Apply min-max scaling to the feature values 10, 20, 30, 50.", "L3", E_, "G"),
    ("Use the MLflow tracking API to log a parameter and a metric for a training run.", "L3", E_, "G"),
    ("Convert the given ER diagram into relational tables.", "L3", M_, "DA"),
    ("Write a Python function that counts word frequencies in a text and returns the top five.", "L3", M_, "DP"),
    # ── L4 Analyze ───────────────────────────────────────────────────────────
    ("Analyze why quicksort degrades to O(n^2) on already sorted input and suggest a fix.", "L4", H_, "EP"),
    ("Differentiate between a stack overflow and a memory leak, and explain how each could be diagnosed.", "L4", M_, "EP"),
    ("Examine the following code and identify why the loop never terminates.", "L4", M_, "E"),
    ("Break down the end-to-end flow of a web request from browser to database and back.", "L4", M_, "D"),
    ("Compare the time and space complexity of merge sort and heap sort and analyze when each is preferable.", "L4", H_, "PA"),
    ("Infer from a training curve with rising validation loss whether the model is overfitting, and explain your reasoning.", "L4", M_, "EP"),
    ("Distinguish between data drift and concept drift using an example for each.", "L4", M_, "PA"),
    ("Analyze how the choice of hash function affects the performance of a hash table.", "L4", H_, "AE"),
    ("Examine the trade-offs between a normalized and a denormalized schema for a reporting workload.", "L4", H_, "AE"),
    ("Decompose the problem of building a spam classifier into data, features, model and evaluation stages.", "L4", M_, "D"),
    ("Analyze the bottlenecks in a pipeline that retrains a model every night on 10 GB of data.", "L4", H_, "DE"),
    ("Identify the recurring pattern in the sequence 2, 6, 12, 20, 30 and derive its general term.", "L4", M_, "PA"),
    ("Relate the CAP theorem to the design choices of a distributed key-value store.", "L4", H_, "AE"),
    ("Analyze why accuracy is a misleading metric for a dataset with a 95 percent majority class.", "L4", M_, "EP"),
    ("Trace the execution of a recursive Fibonacci function and analyze its repeated sub-problems.", "L4", M_, "PDE"),
    ("Differentiate between depth-first and breadth-first search in terms of memory usage.", "L4", M_, "P"),
    ("Examine how class imbalance affects macro-averaged F1 compared with micro-averaged F1.", "L4", H_, "EA"),
    ("Analyze the failure modes of deploying a model without versioning its training data.", "L4", H_, "EA"),
    ("Break down a library management system into modules and identify the interfaces between them.", "L4", M_, "DA"),
    ("Compare the stability of two cross-validation runs with standard deviations 0.003 and 0.06 and infer which model is more reliable.", "L4", M_, "E"),
    # ── L5 Evaluate ──────────────────────────────────────────────────────────
    ("Evaluate whether a linked list or an array is more suitable for implementing a queue, and justify your choice.", "L5", H_, "EA"),
    ("Critique a design that stores model files directly in Git instead of using a data version control tool.", "L5", H_, "E"),
    ("Justify the use of a quality gate that blocks deployment when F1 regresses by more than two percent.", "L5", H_, "EA"),
    ("Assess the suitability of accuracy as the sole promotion criterion for a fraud detection model.", "L5", H_, "E"),
    ("Judge which sorting algorithm you would pick for a nearly sorted list of one million records and defend your answer.", "L5", H_, "EG"),
    ("Evaluate the trade-offs of blue-green deployment against canary release for a model-serving API.", "L5", H_, "EA"),
    ("Defend or refute the statement: a model with 97 percent test accuracy is ready for production.", "L5", H_, "E"),
    ("Critique the following algorithm for detecting duplicates and suggest improvements to its efficiency.", "L5", H_, "EG"),
    ("Prioritize the following technical-debt items in an ML pipeline and justify the order.", "L5", H_, "ED"),
    ("Assess the risks of using LLM-generated labels as ground truth for evaluating a classifier.", "L5", H_, "EA"),
    ("Evaluate TF-IDF with logistic regression versus a transformer for classifying exam questions with only 200 labeled examples.", "L5", H_, "EA"),
    ("Justify why a held-out test set must never overlap with training data.", "L5", M_, "E"),
    ("Judge the correctness of the proof that merge sort runs in O(n log n) and point out any gap.", "L5", H_, "E"),
    ("Argue whether microservices or a monolith is the better architecture for a small team building a question-bank tool.", "L5", H_, "ED"),
    ("Evaluate the fairness of an automated difficulty rating that is trained only on multiple-choice science questions.", "L5", H_, "EA"),
    ("Assess whether five-fold cross-validation is sufficient for a dataset of 25 test samples.", "L5", H_, "E"),
    ("Critique an index design that adds an index on every column of a write-heavy table.", "L5", M_, "E"),
    ("Justify the use of SQLite as the MLflow backend for a classroom demo and state when you would replace it.", "L5", M_, "EA"),
    ("Evaluate the reliability of a monitoring system that checks only server uptime and ignores model accuracy.", "L5", M_, "E"),
    ("Defend the choice of macro-F1 over accuracy as the primary metric for imbalanced classes.", "L5", M_, "E"),
    # ── L6 Create ────────────────────────────────────────────────────────────
    ("Design a data structure that supports insert, delete and get-random in O(1) time.", "L6", H_, "AG"),
    ("Create an algorithm to detect duplicate questions in a bank of 10,000 questions.", "L6", H_, "GDP"),
    ("Design an MLOps pipeline that ingests new data, validates it, retrains a model and promotes it only if quality gates pass.", "L6", H_, "DGE"),
    ("Develop a schema for a question bank that stores questions, Bloom's levels, course outcomes and revision history.", "L6", H_, "AD"),
    ("Construct a test plan to verify that a classifier does not regress between releases.", "L6", H_, "ED"),
    ("Formulate a strategy to collect 500 faculty-verified labels for Bloom's taxonomy with limited reviewer time.", "L6", H_, "DE"),
    ("Design a caching layer for a model-serving API that reduces latency for repeated questions.", "L6", H_, "AG"),
    ("Propose an algorithm that generates a balanced question paper covering all course outcomes within 100 marks.", "L6", H_, "GD"),
    ("Create a monitoring dashboard design that tracks request latency, error rate and prediction distribution.", "L6", M_, "DA"),
    ("Design a recursive function that generates all permutations of a string and explain its base case.", "L6", M_, "GD"),
    ("Develop a plan to migrate a monolithic grading application to containers with zero downtime.", "L6", H_, "DG"),
    ("Construct a decision procedure that chooses between a stack and a queue for a given problem.", "L6", M_, "AG"),
    ("Design a feature set that captures the difficulty of a programming question beyond word counts.", "L6", H_, "AP"),
    ("Compose a rubric that scores student answers to a design question on correctness, clarity and efficiency.", "L6", M_, "AE"),
    ("Create a program that reads a CSV of marks and produces a ranked list with ties handled fairly.", "L6", M_, "GD"),
    ("Design an experiment to compare a TF-IDF classifier and a sentence-embedding classifier fairly.", "L6", H_, "ED"),
    ("Plan a rollback procedure for a bad model release in production.", "L6", M_, "GE"),
    ("Develop an abstraction for a generic sorting interface that works with any comparable type.", "L6", M_, "A"),
    ("Design an algorithm to schedule exam halls so that no two students from the same batch sit adjacent.", "L6", H_, "GD"),
    ("Formulate a reproducible procedure for versioning datasets, code and models together.", "L6", M_, "AG"),
]

HOLDOUT = [
    ("Define a binary search tree.", "L1", E_, "A"),
    ("List the four pillars of object-oriented programming.", "L1", E_, "D"),
    ("State what a foreign key is.", "L1", E_, "A"),
    ("Name two sorting algorithms with O(n log n) worst-case time.", "L1", E_, "P"),
    ("What is a Docker image?", "L1", E_, "A"),
    ("Recall the formula for precision.", "L1", E_, "E"),
    ("Explain how merge sort divides and combines an array.", "L2", M_, "GD"),
    ("Describe the difference between a primary key and a unique key.", "L2", E_, "P"),
    ("Summarize how a model registry supports rollback.", "L2", M_, "A"),
    ("Explain why feature scaling matters for gradient descent.", "L2", M_, "AE"),
    ("Describe how a queue is used in breadth-first search.", "L2", E_, "G"),
    ("Explain what a data hash tells you about a dataset.", "L2", E_, "A"),
    ("Implement a function to find the second largest number in a list.", "L3", E_, "G"),
    ("Apply the Euclidean algorithm to find the gcd of 48 and 18.", "L3", E_, "G"),
    ("Write a SQL query to count orders per customer.", "L3", E_, "DG"),
    ("Compute the recall for 30 true positives and 10 false negatives.", "L3", E_, "E"),
    ("Use insertion sort to arrange 7, 3, 9, 1 showing each step.", "L3", E_, "G"),
    ("Implement a queue using two stacks.", "L3", M_, "GA"),
    ("Analyze why a hash table's lookup can degrade to O(n).", "L4", M_, "EA"),
    ("Differentiate between a classifier's training accuracy and its held-out accuracy.", "L4", M_, "PE"),
    ("Break down the steps involved in deploying a model as a REST API.", "L4", M_, "D"),
    ("Infer from a confusion matrix which class the model confuses most.", "L4", M_, "PE"),
    ("Examine the pattern in the sequence 1, 4, 9, 16, 25 and state the rule.", "L4", E_, "PA"),
    ("Analyze the trade-offs of storing embeddings in memory versus on disk.", "L4", H_, "AE"),
    ("Evaluate whether recursion or iteration is better for computing Fibonacci numbers and justify.", "L5", H_, "EG"),
    ("Critique the use of random splits when the same question appears many times in the dataset.", "L5", H_, "E"),
    ("Assess the risk of promoting a model that has high accuracy but near-zero recall on a rare class.", "L5", H_, "E"),
    ("Justify when a linked list is preferable to an array.", "L5", M_, "EA"),
    ("Judge whether a 25-sample test set is enough to trust an F1 score.", "L5", H_, "E"),
    ("Defend the decision to keep real exam questions out of the training data.", "L5", M_, "E"),
    ("Design a system that recommends questions for a revision test based on a student's weak topics.", "L6", H_, "ADG"),
    ("Create an algorithm to merge k sorted lists efficiently.", "L6", H_, "GD"),
    ("Develop a validation checklist for a new labeled dataset before training.", "L6", M_, "ED"),
    ("Design a schema to store model versions with their metrics and data hashes.", "L6", M_, "AD"),
    ("Construct a procedure to rotate between two model versions during a canary release.", "L6", H_, "GE"),
    ("Formulate a labeling guideline that separates Bloom's Apply from Analyze questions.", "L6", H_, "AP"),
]

BLOOMS_NAME = {"L1": "Remember", "L2": "Understand", "L3": "Apply",
               "L4": "Analyze", "L5": "Evaluate", "L6": "Create"}


def to_bank(rows, prefix, description):
    qs = []
    for i, (text, bloom, diff, ct) in enumerate(rows, 1):
        qs.append(
            {
                "id": f"{prefix}-{i:03d}",
                "subject": "Computer Science (synthetic)",
                "question": text,
                "blooms_level": bloom,
                "blooms_label": BLOOMS_NAME[bloom],
                "difficulty": diff,
                "ct_skills": [CT[c] for c in ct],
                "label_source": {
                    "blooms_level": "synthetic_llm",
                    "difficulty": "synthetic_llm",
                    "ct_skills": "synthetic_llm",
                },
            }
        )
    return {
        "metadata": {"description": description, "label_source_note": "all labels are LLM drafts"},
        "questions": qs,
    }


def main() -> None:
    assert not {t for t, *_ in TRAIN} & {t for t, *_ in HOLDOUT}, "train/holdout overlap"
    out = {
        ROOT / "data/raw/public/synthetic_cs_questions.json": to_bank(
            TRAIN, "SYN-TR", "Synthetic LLM-authored CS questions (train only)"
        ),
        ROOT / "data/raw/synthetic/synthetic_holdout.json": to_bank(
            HOLDOUT, "SYN-TE", "Synthetic LLM-authored CS questions (held-out, never trained on)"
        ),
    }
    for path, bank in out.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bank, indent=2) + "\n")
        print(f"{path.relative_to(ROOT)}: {len(bank['questions'])} questions")


if __name__ == "__main__":
    main()
