
from autofj.datasets import load_data

DATASETS = [
    "TennisTournament",
    "Museum",
    "Drug",
    "Stadium"
]

for name in DATASETS:
    left, right, gt = load_data(name)

    left_counts = gt.groupby("id_l")["id_r"].nunique()
    right_counts = gt.groupby("id_r")["id_l"].nunique()

    print(f"\n=== {name} ===")
    print(f"Ground truth pairs: {len(gt)}")
    print(f"Unique left IDs: {gt['id_l'].nunique()}")
    print(f"Unique right IDs: {gt['id_r'].nunique()}")

    print(
        "Left IDs with multiple matches:",
        (left_counts > 1).sum()
    )

    print(
        "Right IDs with multiple matches:",
        (right_counts > 1).sum()
    )

    print(
        "Strictly one-to-one:",
        (left_counts.max() <= 1)
        and (right_counts.max() <= 1)
    )
