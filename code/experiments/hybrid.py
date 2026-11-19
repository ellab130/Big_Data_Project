
import re
import pandas as pd
from rapidfuzz.fuzz import ratio
from autofj.datasets import load_data

left, right, ground_truth = load_data("TennisTournament")
embeddings = pd.read_csv("embedding_candidates.csv")

true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

def normalize(text):
    return " ".join(str(text).lower().split())

def extract_year(text):
    match = re.search(r"\b(?:19|20)\d{2}\b", str(text))
    return match.group(0) if match else None

def years_compatible(a, b):
    year_a = extract_year(a)
    year_b = extract_year(b)
    return not (year_a and year_b and year_a != year_b)

# Look up embedding scores for the previously retrieved
# top-3 candidates.
embedding_scores = {
    (row.id_l, row.id_r): float(row.similarity)
    for row in embeddings.itertuples()
}

candidate_rows = []

# Small benchmark: compare all pairs.
# We will add efficient blocking for larger datasets.
for l in left.itertuples():
    for r in right.itertuples():
        if not years_compatible(l.title, r.title):
            continue

        normalized_l = normalize(l.title)
        normalized_r = normalize(r.title)

        exact_match = normalized_l == normalized_r
        fuzzy_score = ratio(normalized_l, normalized_r) / 100
        embedding_score = embedding_scores.get(
            (l.id, r.id)
        )

        # Candidate retrieval: union of three methods.
        sources = []

        if exact_match:
            sources.append("normalized")

        if fuzzy_score >= 0.80:
            sources.append("fuzzy")

        if embedding_score is not None:
            sources.append("embedding")

        if sources:
            candidate_rows.append({
                "id_l": l.id,
                "title_l": l.title,
                "id_r": r.id,
                "title_r": r.title,
                "fuzzy_score": fuzzy_score,
                "embedding_score": embedding_score,
                "sources": ",".join(sources)
            })

candidates = pd.DataFrame(candidate_rows)

candidate_pairs = set(
    zip(candidates["id_l"], candidates["id_r"])
)

tp = len(candidate_pairs & true_pairs)
fp = len(candidate_pairs - true_pairs)
fn = len(true_pairs - candidate_pairs)

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

print("\n=== HYBRID CANDIDATE RETRIEVAL ===")
print(f"Total candidates: {len(candidate_pairs)}")
print(f"Correct candidates: {tp}")
print(f"Other candidates: {fp}")
print(f"Missed true matches: {fn}")
print(f"Candidate precision: {precision:.3f}")
print(f"Candidate recall: {recall:.3f}")

print("\n=== STILL MISSING ===")
for row in ground_truth.itertuples():
    if (row.id_l, row.id_r) not in candidate_pairs:
        print(f"{row.title_l} --> {row.title_r}")

candidates.to_csv("hybrid_candidates.csv", index=False)
print("\nSaved to hybrid_candidates.csv")
