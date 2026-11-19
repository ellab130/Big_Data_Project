
import pandas as pd

df = pd.read_csv("labeled_candidates_dev.csv")

df["is_match"] = (
    df["is_match"].astype(str).str.lower() == "true"
)

# Development experiment: small tournament-location vocabulary.
# This is NOT a general-purpose location extractor.
locations = {
    "granby",
    "rimouski",
    "saguenay",
    "san benedetto",
    "umbria",
}

def find_locations(title):
    text = str(title).lower()
    return {
        loc for loc in locations
        if loc in text
    }

def locations_compatible(a, b):
    loc_a = find_locations(a)
    loc_b = find_locations(b)

    # Reject only when both titles specify locations
    # and none of those locations agree.
    if loc_a and loc_b and loc_a.isdisjoint(loc_b):
        return False

    return True

# Start from the higher-recall embedding threshold
df["predicted"] = (
    df["embedding_score"].fillna(0.0) >= 0.75
)

# Apply location filter
df["predicted"] = df["predicted"] & df.apply(
    lambda row: locations_compatible(
        row["title_l"],
        row["title_r"]
    ),
    axis=1
)

tp = int((df["predicted"] & df["is_match"]).sum())
fp = int((df["predicted"] & ~df["is_match"]).sum())
fn = int((~df["predicted"] & df["is_match"]).sum())

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\n=== LOCATION-FILTERED MATCHING ===")
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")

print("\n=== REMAINING FALSE POSITIVES ===")
for row in df[df["predicted"] & ~df["is_match"]].itertuples():
    print(f"{row.title_l} --> {row.title_r}")
