
from autofj.datasets import load_data
from blocking import generate_candidates

left, right, ground_truth = load_data("TennisTournament")

left_values = left["title"].tolist()
right_values = right["title"].tolist()

candidates, comparisons = generate_candidates(
    left_values, right_values
)

predicted_pairs = {
    (left.iloc[i]["id"], right.iloc[j]["id"])
    for i, j in candidates
}

true_pairs = set(
    zip(ground_truth["id_l"], ground_truth["id_r"])
)

retrieved = len(predicted_pairs & true_pairs)

total_pairs = len(left_values) * len(right_values)

print("\n=== BLOCKING RESULTS ===")
print(f"Total possible pairs: {total_pairs}")
print(f"Candidate pairs: {comparisons}")
print(f"Reduction: {1 - comparisons / total_pairs:.1%}")
print(f"True matches retrieved: {retrieved}/{len(true_pairs)}")
print(f"Candidate recall: {retrieved / len(true_pairs):.3f}")
