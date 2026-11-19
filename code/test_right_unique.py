
from autofj.datasets import load_data
from semantic_join import SemanticJoin

DATASETS = ["Museum", "Drug", "Stadium", "TennisTournament"]


def right_unique(matches):
    # Highest embedding similarity wins for each right value.
    ranked = matches.sort_values(
        ["embedding_score", "fuzzy_score"],
        ascending=False
    )

    return ranked.drop_duplicates(
        subset=["right_value"],
        keep="first"
    )


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

    matches = matcher.match_values(
        left["title"],
        right["title"]
    )

    selected = right_unique(matches)

    left_ids = left.groupby("title")["id"].apply(list)
    right_ids = right.groupby("title")["id"].apply(list)

    predicted = set()

    for row in selected.itertuples(index=False):
        for left_id in left_ids[row.left_value]:
            for right_id in right_ids[row.right_value]:
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
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
