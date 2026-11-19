
import time
from autofj.datasets import load_data
from semantic_join import SemanticJoin

DATASETS = ["Museum", "Drug", "Stadium"]

for dataset_name in DATASETS:
    print(f"\n{'=' * 50}")
    print(f"DATASET: {dataset_name}")
    print("=" * 50)

    left, right, ground_truth = load_data(dataset_name)

    print(f"Left rows: {len(left)}")
    print(f"Right rows: {len(right)}")
    print(f"Ground truth matches: {len(ground_truth)}")

    # Use generic settings without tennis-specific rules.
    matcher = SemanticJoin(
        embedding_threshold=0.75,
        fuzzy_threshold=0.80,
        top_k=3,
        check_year=False,
        locations=None,
        retrieval="hybrid"
    )

    start = time.perf_counter()

    result = matcher.join(
        left,
        right,
        left_on="title",
        right_on="title"
    )

    elapsed = time.perf_counter() - start

    predicted_pairs = set(
        zip(result["id_l"], result["id_r"])
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

    total_pairs = matcher.last_total_pairs
    candidates = matcher.last_candidate_count

    reduction = (
        1 - candidates / total_pairs
        if total_pairs else 0
    )

    print(f"\nCandidate pairs: {candidates}")
    print(f"Candidate reduction: {reduction:.1%}")
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
