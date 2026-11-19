
from autofj.datasets import load_data
from pyspark.sql import SparkSession
from spark_semantic_join import SparkSemanticJoin

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("SemanticJoinBenchmark")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    left, right, ground_truth = load_data("TennisTournament")

    left_spark = spark.createDataFrame(left)
    right_spark = spark.createDataFrame(right)

    matcher = SparkSemanticJoin(
        spark,
        embedding_threshold=0.75,
        fuzzy_threshold=0.80,
        top_k=3,
        check_year=True,
        locations={
            "granby",
            "rimouski",
            "saguenay",
            "san benedetto",
            "umbria",
        },
    )

    result = matcher.join(
        left_spark,
        right_spark,
        left_on="title",
        right_on="title",
    )
    print(
    "Candidate pairs:",
    matcher.matcher.last_candidate_count
    )

    print(
        "Total possible pairs:",
        matcher.matcher.last_total_pairs
    )

    print(
        "Candidate reduction:",
        f"{1 - matcher.matcher.last_candidate_count / matcher.matcher.last_total_pairs:.1%}"
    )

    predicted_pairs = {
        (row["id_l"], row["id_r"])
        for row in result.select("id_l", "id_r").collect()
    }

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

    print("\n=== SPARK SEMANTIC JOIN BENCHMARK ===")
    print(f"Predicted matches: {len(predicted_pairs)}")
    print(f"Correct matches: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

finally:
    spark.stop()
