
import pandas as pd
import numpy as np
from autofj.datasets import load_data
from sentence_transformers import SentenceTransformer

# Load benchmark
left, right, ground_truth = load_data("TennisTournament")

true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

# Load pretrained embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Convert titles into numerical vectors
left_vectors = model.encode(
    left["title"].astype(str).tolist(),
    normalize_embeddings=True
)

right_vectors = model.encode(
    right["title"].astype(str).tolist(),
    normalize_embeddings=True
)

# Cosine similarity (vectors are normalized)
similarities = left_vectors @ right_vectors.T

# Retrieve top 3 candidates for each left title
top_k = min(3, len(right))

predicted_pairs = set()
candidate_rows = []

for i in range(len(left)):
    top_indices = np.argsort(
        similarities[i]
    )[-top_k:][::-1]

    for j in top_indices:
        score = float(similarities[i, j])

        candidate_rows.append({
            "id_l": left.iloc[i]["id"],
            "title_l": left.iloc[i]["title"],
            "id_r": right.iloc[j]["id"],
            "title_r": right.iloc[j]["title"],
            "similarity": score
        })

        if score >= 0.80:
            predicted_pairs.add((
                left.iloc[i]["id"],
                right.iloc[j]["id"]
            ))

# Evaluate
tp = len(predicted_pairs & true_pairs)
fp = len(predicted_pairs - true_pairs)
fn = len(true_pairs - predicted_pairs)

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\n=== EMBEDDING MATCHING RESULTS ===")
print(f"Predicted matches: {len(predicted_pairs)}")
print(f"Correct matches: {tp}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Save candidate pairs
pd.DataFrame(candidate_rows).to_csv(
    "embedding_candidates.csv",
    index=False
)

print("\nSaved candidates to embedding_candidates.csv")


# Inspect false positives
print("\n=== FALSE POSITIVES ===")

for row in candidate_rows:
    pair = (row["id_l"], row["id_r"])

    if (
        pair in predicted_pairs
        and pair not in true_pairs
    ):
        print(
            f'{row["title_l"]}  -->  '
            f'{row["title_r"]}  '
            f'(similarity={row["similarity"]:.3f})'
        )

# Inspect missed matches
print("\n=== FALSE NEGATIVES ===")

for _, row in ground_truth.iterrows():
    pair = (row["id_l"], row["id_r"])

    if pair not in predicted_pairs:
        print(
            f'{row["title_l"]}  -->  '
            f'{row["title_r"]}'
        )
