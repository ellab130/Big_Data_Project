
import time
from pathlib import Path

import pandas as pd
from autofj.datasets import load_data

from semantic_join import SemanticJoin


DATASETS = [
    "TennisTournament",
    "Museum",
    "Drug",
    "Stadium",
]

CONFIGURATIONS = {
    "Original Hybrid": {
        "char_top_k": 0,
        "cardinality": "many_to_many",
    },
    "Hybrid + Character": {
        "char_top_k": 10,
        "cardinality": "many_to_many",
    },
    "Hybrid + Cardinality": {
        "char_top_k": 0,
        "cardinality": "right_unique",
    },
    "Full Improved Hybrid": {
        "char_top_k": 10,
        "cardinality": "right_unique",
    },
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "final_benchmark_results.csv"


def evaluate(predicted, truth):
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return tp, fp, fn, precision, recall, f1


results = []

for dataset_name in DATASETS:
    print(f"\n{'=' * 65}")
    print(f"DATASET: {dataset_name}")
    print("=" * 65)

    left, right, ground_truth = load_data(dataset_name)

    truth = set(
        zip(
            ground_truth["id_l"],
            ground_truth["id_r"],
        )
    )

    for config_name, config in CONFIGURATIONS.items():
        print(f"\nRunning: {config_name}")

        matcher = SemanticJoin(
            embedding_threshold=0.75,
            fuzzy_threshold=0.80,
            top_k=3,
            check_year=False,
            locations=None,
            retrieval="hybrid",
            **config,
        )

        start = time.perf_counter()

        joined = matcher.join(
            left,
            right,
            left_on="title",
            right_on="title",
        )

        elapsed = time.perf_counter() - start

        predicted = set(
            zip(
                joined["id_l"],
                joined["id_r"],
            )
        )

        tp, fp, fn, precision, recall, f1 = evaluate(
            predicted,
            truth,
        )

        candidate_count = matcher.last_candidate_count
        total_pairs = matcher.last_total_pairs

        reduction = (
            1 - candidate_count / total_pairs
            if total_pairs else 0.0
        )

        result = {
            "dataset": dataset_name,
            "configuration": config_name,
            "left_rows": len(left),
            "right_rows": len(right),
            "ground_truth_matches": len(truth),
            "candidate_pairs": candidate_count,
            "total_possible_pairs": total_pairs,
            "candidate_reduction": reduction,
            "accepted_mappings": matcher.last_verified_count,
            "selected_mappings": matcher.last_match_count,
            "predicted_matches": len(predicted),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "runtime_seconds": elapsed,
        }

        results.append(result)

        print(f"Candidates: {candidate_count:,}")
        print(f"Candidate reduction: {reduction:.1%}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")
        print(f"Runtime: {elapsed:.2f}s")

        # Save after each configuration so partial results survive
        # if a later run is interrupted.
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False,
        )

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

summary = pd.DataFrame(results)

print(
    summary[
        [
            "dataset",
            "configuration",
            "precision",
            "recall",
            "f1",
            "runtime_seconds",
        ]
    ].to_string(index=False)
)

print(f"\nResults saved to: {OUTPUT_FILE}")
