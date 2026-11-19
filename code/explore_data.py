
from autofj.datasets import load_data

# Load one benchmark dataset
left, right, ground_truth = load_data("TennisTournament")

print("LEFT TABLE")
print(left.head())
print("Shape:", left.shape)

print("\nRIGHT TABLE")
print(right.head())
print("Shape:", right.shape)

print("\nGROUND TRUTH")
print(ground_truth.head())
print("Shape:", ground_truth.shape)

print("\nCOLUMN NAMES")
print("Left:", left.columns.tolist())
print("Right:", right.columns.tolist())
print("Ground truth:", ground_truth.columns.tolist())
