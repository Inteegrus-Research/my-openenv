#!/usr/bin/env python3
import json
import random
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
INSTANCES_PER_TASK = 5
CANDIDATES_PER_INSTANCE = 20
MAX_DESC_LEN = 600
MAX_ABS_LEN = 1000
CPC_CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H"]

FIXTURE_ROOT = Path("fixtures")
TASKS = ["task1", "task2", "task3", "task4"]

TASK_BUDGETS = {"task1": 20, "task2": 24, "task3": 28, "task4": 32}

TASK_DESCRIPTIONS = {
    "task1": "Screen candidate patents for relevance to the query patent. Mark the strongest prior-art candidates as RELEVANT and the rest as NOT_RELEVANT.",
    "task2": "Rank candidate patents by novelty risk and provide a quality score from 1 to 4 for each review decision.",
    "task3": "Decide whether each candidate should be INCLUDED, EXCLUDED, or DEFERRED. DEFER counts as EXCLUDE at grading time.",
    "task4": "Produce a ranked top-5 prior-art shortlist with justification strings for INCLUDE decisions. This is the hardest task.",
}

random.seed(SEED)
np.random.seed(SEED)

print("Loading dataset and embedding model...")
dataset = load_dataset("big_patent", "all", split="train[:15000]")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_corpus(ds):
    corpus, meta = [], []
    for i, item in enumerate(ds):
        abstract = item.get("abstract", "")
        description = item.get("description", "")
        if len(abstract) < 200:
            continue
        text = f"{abstract[:500]} {description[:500]}"
        corpus.append(text)
        meta.append({
            "id": f"patent_{i:05d}",
            "title": f"Patent Publication {i:05d}",
            "abstract": abstract[:MAX_ABS_LEN],
            "description": description[:MAX_DESC_LEN],
            "cpc": random.choice(CPC_CLASSES)
        })
    return corpus, meta

corpus, meta = build_corpus(dataset)
print(f"Corpus size: {len(corpus)}. Embedding...")
embeddings = model.encode(corpus, batch_size=64, show_progress_bar=True)

def build_instance(task_id: str):
    query_idx = random.randint(0, len(meta) - 1)
    query_vec = embeddings[query_idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]

    ranked_indices = np.argsort(-sims)
    top_10 = [i for i in ranked_indices if i != query_idx][:10]
    random_pool = [i for i in range(len(meta)) if i not in ranked_indices[:100]]
    neg_10 = random.sample(random_pool, 10)

    final_idxs = list(top_10) + neg_10
    random.shuffle(final_idxs)

    candidates, final_sims = [], []
    for idx in final_idxs:
        s = float(cosine_similarity(query_vec, embeddings[idx].reshape(1, -1))[0][0])
        final_sims.append(s)
        candidates.append(meta[idx])

    s_arr = np.array(final_sims)
    norm_scores = 0.01 + (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-8) * 0.98

    sorted_data = sorted(zip(norm_scores, candidates), key=lambda x: x[0], reverse=True)

    fixture = {
        "task_description": TASK_DESCRIPTIONS[task_id],
        "budget": TASK_BUDGETS[task_id],
        "query_patent": meta[query_idx],
        "candidate_patents": candidates,
        "ground_truth": {
            "ranking": [c["id"] for _, c in sorted_data],
            "relevance_scores": {c["id"]: float(s) for s, c in zip(norm_scores, candidates)},
            "novelty_score": float(random.uniform(0.1, 0.9))
        }
    }
    return fixture

for task in TASKS:
    path = FIXTURE_ROOT / task
    path.mkdir(parents=True, exist_ok=True)
    for i in range(1, INSTANCES_PER_TASK + 1):
        instance = build_instance(task)
        with open(path / f"instance_{i:03d}.json", "w") as f:
            json.dump(instance, f, indent=2)
    print(f"Generated {task}")

print("Fixtures regenerated with correct task descriptions, CPC diversity, and budget fields.")