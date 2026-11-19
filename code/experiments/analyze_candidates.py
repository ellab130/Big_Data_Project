
import pandas as pd
from autofj.datasets import load_data

# Load candidate pairs and ground truth
candidates = pd.read_csv("hybrid_candidates.csv")
left, right, ground_truth = load_data("TennisTournament")

true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

# Label candidates for development analysis only.
# Ground-truth labels must NOT be used in final inference.
candidates["is_match"] = candidates.apply(
    lambda row: (
        row["id_l"], row["id_r"]
    ) in true_pairs,
    axis=1
)

print("\n=== SCORE DISTRIBUTIONS ===")

for label, group in candidates.groupby("is_match"):
    name = "TRUE MATCHES" if label else "OTHER CANDIDATES"

    print(f"\n{name}: {len(group)}")
    print("\nFuzzy similarity:")
    print(group["fuzzy_score"].describe())

    print("\nEmbedding similarity (available candidates):")
    print(group["embedding_score"].describe())

print("\n=== RETRIEVAL SOURCE COUNTS ===")

print(
    candidates.groupby(["sources", "is_match"])
    .size()
    .unstack(fill_value=0)
)

print("\n=== TRUE MATCHES WITH LOW FUZZY SCORES ===")

true_matches = candidates[candidates["is_match"]]

print(
    true_matches[
        ["title_l", "title_r", "fuzzy_score",
         "embedding_score", "sources"]
    ]
    .sort_values("fuzzy_score")
    .head(15)
    .to_string(index=False)
)

candidates.to_csv(
    "labeled_candidates_dev.csv",
    index=False
)

print("\nSaved to labeled_candidates_dev.csv")
