
import numpy as np
from autofj.datasets import load_data
from sentence_transformers import SentenceTransformer

from blocking import generate_candidates

left, right, ground_truth = load_data("TennisTournament")

left_values = left["title"].tolist()
right_values = right["title"].tolist()

# 1. Token blocking
token_candidates, _ = generate_candidates(
    left_values, right_values
)

# 2. Embedding retrieval
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

left_vectors = model.encode(
    left_values, normalize_embeddings=True
)
right_vectors = model.encode(
    right_values, normalize_embeddings=True
)

similarity = left_vectors @ right_vectors.T

top_k = 3
embedding_candidates = set()

for i in range(len(left_values)):
    top_indices = np.argsort(similarity[i])[
        -min(top_k, len(right_values)):
    ]

    for j in top_indices:
        embedding_candidates.add((i, int(j)))

# 3. Combine both retrieval methods
hybrid_candidates = token_candidates | embedding_candidates

# 4. Evaluate candidate recall
true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

def evaluate(candidates):
    predicted_pairs = {
        (left.iloc[i]["id"], right.iloc[j]["id"])
        for i, j in candidates
    }
    return len(predicted_pairs & true_pairs)

total_pairs = len(left_values) * len(right_values)

print("\n=== HYBRID BLOCKING RESULTS ===")

for name, candidates in [
    ("Token blocking", token_candidates),
    ("Embedding top-k", embedding_candidates),
    ("Hybrid blocking", hybrid_candidates),
]:
    retrieved = evaluate(candidates)

    print(f"\n{name}")
    print(f"Candidate pairs: {len(candidates)}")
    print(
        f"Reduction: "
        f"{1 - len(candidates) / total_pairs:.1%}"
    )
    print(
        f"Candidate recall: "
        f"{retrieved}/{len(true_pairs)} "
        f"({retrieved / len(true_pairs):.3f})"
    )
