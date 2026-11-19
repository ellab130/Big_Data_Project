
import pandas as pd
from autofj.datasets import load_data
from sklearn.metrics import precision_score, recall_score, f1_score

# Load benchmark
left, right, ground_truth = load_data("TennisTournament")

# 1. Exact string matching
exact_matches = pd.merge(
    left,
    right,
    on="title",
    how="inner",
    suffixes=("_l", "_r")
)

# Keep the IDs of matched records
exact_pairs = set(
    zip(exact_matches["id_l"], exact_matches["id_r"])
)

# 2. Ground-truth pairs
true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

# 3. Calculate evaluation metrics
true_positives = len(exact_pairs & true_pairs)
false_positives = len(exact_pairs - true_pairs)
false_negatives = len(true_pairs - exact_pairs)

precision = (
    true_positives / (true_positives + false_positives)
    if exact_pairs else 0
)

recall = (
    true_positives / (true_positives + false_negatives)
    if true_pairs else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0
)

print("=== EXACT MATCHING RESULTS ===")
print(f"Total ground-truth pairs: {len(true_pairs)}")
print(f"Predicted matches: {len(exact_pairs)}")
print(f"Correct matches: {true_positives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Save predicted matches
exact_matches.to_csv("exact_matches.csv", index=False)

print("\nResults saved to exact_matches.csv")
