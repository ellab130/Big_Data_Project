
import re
import numpy as np
import pandas as pd

from collections import defaultdict
from autofj.datasets import load_data
from sentence_transformers import SentenceTransformer
from blocking import generate_candidates


DATASETS = ["Museum", "Drug", "Stadium", "TennisTournament"]
CHAR_TOP_K = [3, 10, 20]
EMBEDDING_TOP_K = 3


def distinct_values(series):
    return list(dict.fromkeys(
        str(v) for v in series if pd.notna(v)
    ))


def char_ngrams(text, n=3):
    cleaned = re.sub(r"[^a-z0-9]", "", str(text).lower())

    if len(cleaned) < n:
        return {cleaned} if cleaned else set()

    return {
        cleaned[i:i+n]
        for i in range(len(cleaned) - n + 1)
    }


def ranked_char_candidates(left_values, right_values, top_k):
    index = defaultdict(set)
    right_grams = []

    for j, value in enumerate(right_values):
        grams = char_ngrams(value)
        right_grams.append(grams)

        for gram in grams:
            index[gram].add(j)

    candidates = set()

    for i, value in enumerate(left_values):
        left_grams = char_ngrams(value)
        counts = defaultdict(int)

        for gram in left_grams:
            for j in index.get(gram, ()):
                counts[j] += 1

        scores = []

        for j, shared in counts.items():
            union = (
                len(left_grams)
                + len(right_grams[j])
                - shared
            )

            score = shared / union if union else 0
            scores.append((score, j))

        scores.sort(reverse=True)

        for score, j in scores[:top_k]:
            candidates.add((i, j))

    return candidates


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


for name in DATASETS:
    print(f"\n{'=' * 55}")
    print(f"DATASET: {name}")
    print("=" * 55)

    left, right, gt = load_data(name)

    left_values = distinct_values(left["title"])
    right_values = distinct_values(right["title"])

    left_index = {v: i for i, v in enumerate(left_values)}
    right_index = {v: j for j, v in enumerate(right_values)}

    left_lookup = dict(zip(left["id"], left["title"]))
    right_lookup = dict(zip(right["id"], right["title"]))

    truth = set()

    for row in gt.itertuples(index=False):
        a = left_lookup[row.id_l]
        b = right_lookup[row.id_r]

        if pd.notna(a) and pd.notna(b):
            truth.add((left_index[str(a)], right_index[str(b)]))

    left_vectors = model.encode(
        left_values, normalize_embeddings=True
    )
    right_vectors = model.encode(
        right_values, normalize_embeddings=True
    )

    similarity = left_vectors @ right_vectors.T

    baseline, _ = generate_candidates(
        left_values, right_values
    )

    for i in range(len(left_values)):
        k = min(EMBEDDING_TOP_K, len(right_values))
        indices = np.argsort(similarity[i])[-k:]

        for j in indices:
            baseline.add((i, int(j)))

    total_pairs = len(left_values) * len(right_values)

    baseline_correct = len(truth & baseline)

    print(
        f"Baseline: {len(baseline)} candidates, "
        f"{baseline_correct}/{len(truth)} correct, "
        f"recall={baseline_correct / len(truth):.1%}"
    )

    for k in CHAR_TOP_K:
        char_candidates = ranked_char_candidates(
            left_values, right_values, k
        )

        combined = baseline | char_candidates
        correct = len(truth & combined)

        print(
            f"Char top-{k}: "
            f"{len(combined)} candidates, "
            f"{correct}/{len(truth)} correct, "
            f"recall={correct / len(truth):.1%}, "
            f"reduction={1 - len(combined)/total_pairs:.1%}"
        )
