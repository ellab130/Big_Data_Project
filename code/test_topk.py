
import numpy as np
import pandas as pd

from autofj.datasets import load_data
from sentence_transformers import SentenceTransformer
from blocking import generate_candidates


DATASETS = ["Museum", "Drug", "Stadium", "TennisTournament"]
TOP_K_VALUES = [3, 10, 25, 50]


def distinct_values(series):
    return list(dict.fromkeys(
        str(value)
        for value in series
        if pd.notna(value)
    ))


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

    left_index = {
        value: i for i, value in enumerate(left_values)
    }
    right_index = {
        value: j for j, value in enumerate(right_values)
    }

    left_lookup = dict(zip(left["id"], left["title"]))
    right_lookup = dict(zip(right["id"], right["title"]))

    # Ground truth expressed as distinct-value index pairs.
    true_pairs = set()

    for row in gt.itertuples(index=False):
        a = left_lookup[row.id_l]
        b = right_lookup[row.id_r]

        if pd.notna(a) and pd.notna(b):
            true_pairs.add((
                left_index[str(a)],
                right_index[str(b)]
            ))

    left_vectors = model.encode(
        left_values,
        normalize_embeddings=True
    )

    right_vectors = model.encode(
        right_values,
        normalize_embeddings=True
    )

    similarity = left_vectors @ right_vectors.T

    token_candidates, _ = generate_candidates(
        left_values, right_values
    )

    total_pairs = len(left_values) * len(right_values)

    print(f"Ground-truth pairs: {len(true_pairs)}")
    print(f"Total possible pairs: {total_pairs}")
    print()
    print(
        f"{'Top-k':<8}"
        f"{'Candidates':<14}"
        f"{'Reduction':<14}"
        f"{'Recall':<12}"
    )

    for k in TOP_K_VALUES:
        candidates = set(token_candidates)

        actual_k = min(k, len(right_values))

        # Find the top-k embedding neighbors per left value.
        for i in range(len(left_values)):
            indices = np.argsort(similarity[i])[-actual_k:]

            for j in indices:
                candidates.add((i, int(j)))

        correct = len(true_pairs & candidates)

        recall = correct / len(true_pairs)
        reduction = 1 - len(candidates) / total_pairs

        print(
            f"{k:<8}"
            f"{len(candidates):<14}"
            f"{reduction:<14.1%}"
            f"{recall:<12.1%}"
        )
