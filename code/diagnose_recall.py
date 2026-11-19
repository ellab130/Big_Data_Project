
import numpy as np
import pandas as pd

from autofj.datasets import load_data
from rapidfuzz.fuzz import ratio
from semantic_join import SemanticJoin
from blocking import generate_candidates


DATASETS = ["Museum", "Drug", "Stadium", "TennisTournament"]

THRESHOLD = 0.75
TOP_K = 3


def normalize(text):
    return " ".join(str(text).lower().split())


def diagnose(name):
    print(f"\n{'=' * 55}")
    print(f"DATASET: {name}")
    print("=" * 55)

    left, right, gt = load_data(name)

    # Match the distinct-value ordering used by SemanticJoin.
    left_values = list(dict.fromkeys(
        str(v) for v in left["title"] if pd.notna(v)
    ))
    right_values = list(dict.fromkeys(
        str(v) for v in right["title"] if pd.notna(v)
    ))

    left_index = {v: i for i, v in enumerate(left_values)}
    right_index = {v: j for j, v in enumerate(right_values)}

    # Translate ground-truth row IDs into value pairs.
    left_lookup = dict(zip(left["id"], left["title"]))
    right_lookup = dict(zip(right["id"], right["title"]))

    true_pairs = set()

    for row in gt.itertuples(index=False):
        a = left_lookup[row.id_l]
        b = right_lookup[row.id_r]

        if pd.notna(a) and pd.notna(b):
            true_pairs.add((str(a), str(b)))

    matcher = SemanticJoin(
        embedding_threshold=THRESHOLD,
        fuzzy_threshold=0.80,
        top_k=TOP_K,
        check_year=False,
        locations=None,
        retrieval="hybrid"
    )

    # Encode distinct values.
    left_vectors = matcher.model.encode(
        left_values, normalize_embeddings=True
    )
    right_vectors = matcher.model.encode(
        right_values, normalize_embeddings=True
    )

    similarities = left_vectors @ right_vectors.T

    # Stage 1: Generate candidates.
    candidates, _ = generate_candidates(
        left_values, right_values
    )

    for i in range(len(left_values)):
        k = min(TOP_K, len(right_values))
        top_indices = np.argsort(similarities[i])[-k:]

        for j in top_indices:
            candidates.add((i, int(j)))

    candidate_pairs = {
        (left_values[i], right_values[j])
        for i, j in candidates
    }

    retrieved_true = true_pairs & candidate_pairs

    # Stage 2: Apply the same acceptance rule as SemanticJoin.
    accepted_pairs = set()

    for i, j in candidates:
        a = left_values[i]
        b = right_values[j]

        embedding_score = float(similarities[i, j])

        if (
            embedding_score >= THRESHOLD
            or normalize(a) == normalize(b)
        ):
            accepted_pairs.add((a, b))

    accepted_true = true_pairs & accepted_pairs

    # Stage 3: Right-unique selection.
    selected = {}

    for a, b in accepted_pairs:
        i = left_index[a]
        j = right_index[b]

        embedding_score = float(similarities[i, j])
        fuzzy_score = ratio(
            normalize(a), normalize(b)
        ) / 100

        score = (embedding_score, fuzzy_score)

        if b not in selected or score > selected[b][1]:
            selected[b] = (a, score)

    selected_pairs = {
        (a, b)
        for b, (a, _) in selected.items()
    }

    selected_true = true_pairs & selected_pairs

    total = len(true_pairs)

    print(f"Ground-truth value pairs: {total}")
    print(f"Total possible pairs: {len(left_values) * len(right_values)}")
    print(f"Generated candidates: {len(candidate_pairs)}")

    print("\n=== RECALL BY STAGE ===")

    print(
        f"Candidate recall: "
        f"{len(retrieved_true)}/{total} "
        f"({len(retrieved_true) / total:.1%})"
    )

    print(
        f"After verification: "
        f"{len(accepted_true)}/{total} "
        f"({len(accepted_true) / total:.1%})"
    )

    print(
        f"After right-unique selection: "
        f"{len(selected_true)}/{total} "
        f"({len(selected_true) / total:.1%})"
    )

    print("\n=== LOST MATCHES ===")
    print(
        "Lost during retrieval:",
        len(true_pairs - candidate_pairs)
    )
    print(
        "Lost during verification:",
        len(retrieved_true - accepted_true)
    )
    print(
        "Lost during selection:",
        len(accepted_true - selected_true)
    )

    print("\n=== EXAMPLE MISSED MATCHES ===")

    missed = sorted(true_pairs - accepted_pairs)

    for a, b in missed[:5]:
        print(f"  LEFT:  {a}")
        print(f"  RIGHT: {b}")
        print()


for dataset in DATASETS:
    diagnose(dataset)
