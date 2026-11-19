
import re
import pandas as pd
from autofj.datasets import load_data

# Load benchmark
left, right, ground_truth = load_data("TennisTournament")

# Load embedding candidates from previous experiment
candidates = pd.read_csv("embedding_candidates.csv")

true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

def extract_year(text):
    """Extract a four-digit year from a title."""
    match = re.search(r"\b(?:19|20)\d{2}\b", str(text))
    return match.group(0) if match else None

def years_compatible(title_a, title_b):
    """Reject pairs with conflicting explicit years."""
    year_a = extract_year(title_a)
    year_b = extract_year(title_b)

    if year_a is not None and year_b is not None:
        return year_a == year_b

    return True

# Apply original similarity threshold and year filter
predicted_pairs = set()

for _, row in candidates.iterrows():
    if (
        row["similarity"] >= 0.80
        and years_compatible(row["title_l"], row["title_r"])
    ):
        predicted_pairs.add((row["id_l"], row["id_r"]))

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

print("\n=== YEAR-FILTERED EMBEDDINGS ===")
print(f"Predicted matches: {len(predicted_pairs)}")
print(f"Correct matches: {tp}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Save accepted matches
filtered = candidates[
    candidates.apply(
        lambda row:
            row["similarity"] >= 0.80
            and years_compatible(
                row["title_l"], row["title_r"]
            ),
        axis=1
    )
]

filtered.to_csv("year_filtered_matches.csv", index=False)

print("\nSaved to year_filtered_matches.csv")
