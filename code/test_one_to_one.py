
import time
from autofj.datasets import load_data
from semantic_join import SemanticJoin

DATASETS = ["Museum", "Drug", "Stadium"]


def greedy_one_to_one(matches):
    matches = matches.sort_values(
        "embedding_score",
        ascending=False
    )

    used_left = set()
    used_right = set()
    selected = []

    for row in matches.itertuples(index=False):
        if (
            row.left_value in used_left
            or row.right_value in used_right
        ):
            continue

        selected.append({
            "left_value": row.left_value,
            "right_value": row.right_value,
            "embedding_score": row.embedding_score,
            "fuzzy_score": row.fuzzy_score
        })

        used_left.add(row.left_value)
        used_right.add(row.right_value)

    return selected


for name in DATASETS:
    print(f"\n=== {name} ===")

    left, right, gt = load_data(name)

    matcher = SemanticJoin(
        embedding_threshold=0.75,
        fuzzy_threshold=0.80,
        top_k=3,
        check_year=False,
        locations=None,
        retrieval="hybrid"
    )

    start = time.perf_counter()

    matches = matcher.match_values(
        left["title"],
        right["title"]
    )

    selected = greedy_one_to_one(matches)

    # Convert selected value mappings to row-ID pairs.
    left_ids = left.groupby("title")["id"].apply(list)
    right_ids = right.groupby("title")["id"].apply(list)

    predicted = set()

    for row in selected:
        for left_id in left_ids[row["left_value"]]:
            for right_id in right_ids[row["right_value"]]:
                predicted.add((left_id, right_id))

    truth = set(zip(gt["id_l"], gt["id_r"]))

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(f"Selected mappings: {len(selected)}")
    print(f"Runtime: {time.perf_counter() - start:.2f}s")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
