
import pandas as pd
from autofj.datasets import load_data
from rapidfuzz.fuzz import ratio

# Load benchmark
left, right, ground_truth = load_data("TennisTournament")

true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

def evaluate(predicted_pairs, method):
    tp = len(predicted_pairs & true_pairs)
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(f"\n=== {method} ===")
    print(f"Predicted matches: {len(predicted_pairs)}")
    print(f"Correct matches: {tp}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return {
        "Method": method,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# --------------------------------
# 1. Exact matching
# --------------------------------

exact = pd.merge(
    left,
    right,
    on="title",
    suffixes=("_l", "_r")
)

exact_pairs = set(zip(exact["id_l"], exact["id_r"]))

# --------------------------------
# 2. Normalized matching
# --------------------------------

left_norm = left.copy()
right_norm = right.copy()

left_norm["normalized"] = (
    left_norm["title"].str.lower().str.strip()
)

right_norm["normalized"] = (
    right_norm["title"].str.lower().str.strip()
)

normalized = pd.merge(
    left_norm,
    right_norm,
    on="normalized",
    suffixes=("_l", "_r")
)

normalized_pairs = set(
    zip(normalized["id_l"], normalized["id_r"])
)

# --------------------------------
# 3. Fuzzy matching
# --------------------------------

fuzzy_pairs = set()
threshold = 80

for _, lrow in left.iterrows():
    for _, rrow in right.iterrows():
        score = ratio(
            str(lrow["title"]).lower(),
            str(rrow["title"]).lower()
        )

        if score >= threshold:
            fuzzy_pairs.add((lrow["id"], rrow["id"]))

# --------------------------------
# Evaluate and save results
# --------------------------------

results = [
    evaluate(exact_pairs, "Exact"),
    evaluate(normalized_pairs, "Normalized"),
    evaluate(fuzzy_pairs, "Fuzzy (80)")
]

pd.DataFrame(results).to_csv(
    "baseline_results.csv",
    index=False
)

print("\nSaved results to baseline_results.csv")
