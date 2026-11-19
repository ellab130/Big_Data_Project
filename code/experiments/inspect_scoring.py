
import pandas as pd

df = pd.read_csv("labeled_candidates_dev.csv")

df["is_match"] = (
    df["is_match"].astype(str).str.lower() == "true"
)

# Higher-recall configuration
threshold = 0.75

df["predicted"] = (
    df["embedding_score"].fillna(0.0) >= threshold
)

false_positives = df[
    df["predicted"] & ~df["is_match"]
]

false_negatives = df[
    ~df["predicted"] & df["is_match"]
]

print("\n=== FALSE POSITIVES AT 0.75 ===")
for row in false_positives.itertuples():
    print(
        f"{row.title_l} --> {row.title_r} "
        f"(embedding={row.embedding_score:.3f}, "
        f"fuzzy={row.fuzzy_score:.3f})"
    )

print("\n=== FALSE NEGATIVES AT 0.75 ===")
for row in false_negatives.itertuples():
    print(
        f"{row.title_l} --> {row.title_r} "
        f"(embedding={row.embedding_score:.3f}, "
        f"fuzzy={row.fuzzy_score:.3f})"
    )
