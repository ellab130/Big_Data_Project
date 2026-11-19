
import pandas as pd

# Development data only
df = pd.read_csv("labeled_candidates_dev.csv")

# Ensure labels are Boolean
df["is_match"] = (
    df["is_match"].astype(str).str.lower() == "true"
)

# Missing embedding scores get a conservative value
df["embedding_score"] = df["embedding_score"].fillna(0.0)

def evaluate(predictions, labels):
    tp = int((predictions & labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return precision, recall, f1, tp, fp, fn

results = []

# Test different embedding weights and thresholds
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    scores = (
        alpha * df["embedding_score"]
        + (1 - alpha) * df["fuzzy_score"]
    )

    for threshold in [
        0.50, 0.55, 0.60, 0.65,
        0.70, 0.75, 0.80, 0.85, 0.90
    ]:
        predictions = scores >= threshold

        precision, recall, f1, tp, fp, fn = evaluate(
            predictions,
            df["is_match"]
        )

        results.append({
            "alpha": alpha,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    ["f1", "recall"],
    ascending=False
)

print("\n=== TOP 10 HYBRID CONFIGURATIONS ===")
print(results_df.head(10).to_string(index=False))

best = results_df.iloc[0]

print("\n=== BEST CONFIGURATION ===")
print(f"Embedding weight (alpha): {best['alpha']}")
print(f"Threshold: {best['threshold']}")
print(f"Precision: {best['precision']:.3f}")
print(f"Recall: {best['recall']:.3f}")
print(f"F1 Score: {best['f1']:.3f}")
print(f"TP: {int(best['tp'])}")
print(f"FP: {int(best['fp'])}")
print(f"FN: {int(best['fn'])}")

results_df.to_csv(
    "hybrid_scoring_results.csv",
    index=False
)

print("\nSaved to hybrid_scoring_results.csv")
