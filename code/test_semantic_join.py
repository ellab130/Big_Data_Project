
from autofj.datasets import load_data
from semantic_join import SemanticJoin

left, right, ground_truth = load_data("TennisTournament")

matcher = SemanticJoin(
    embedding_threshold=0.75,
    fuzzy_threshold=0.80,
    top_k=3,
    check_year=True,
    locations={
        "granby",
        "rimouski",
        "saguenay",
        "san benedetto",
        "umbria"
    }
)

results = matcher.join(
    left,
    right,
    left_on="title",
    right_on="title"
)

predicted_pairs = set(
    zip(results["id_l"], results["id_r"])
)

true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

tp = len(predicted_pairs & true_pairs)
fp = len(predicted_pairs - true_pairs)
fn = len(true_pairs - predicted_pairs)

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\n=== SEMANTIC JOIN RESULTS ===")
print(f"Predicted matches: {len(predicted_pairs)}")
print(f"Correct matches: {tp}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")

results.to_csv(
    "../results/semantic_join_results.csv",
    index=False
)

print("\nSaved to results/semantic_join_results.csv")
