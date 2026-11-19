
from pyspark.sql import SparkSession
from spark_semantic_join import SparkSemanticJoin

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("SparkSemanticJoinTest")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    left = spark.createDataFrame([
        (1, "2012 Granby Challenger"),
        (2, "2012 Granby Challenger"),
        (3, "2013 Rimouski Challenger"),
    ], ["id", "title"])

    right = spark.createDataFrame([
        (10, "2012 Granby Challenger"),
        (11, "2012 Granby Challenger"),
        (12, "2013 Rimouski Challenger"),
    ], ["id", "title"])

    matcher = SparkSemanticJoin(
        spark,
        embedding_threshold=0.75,
        check_year=True,
        top_k=3,
    )

    result = matcher.join(
        left, right,
        left_on="title",
        right_on="title",
    )

    result.select(
        "id_l", "id_r", "title_l", "title_r"
    ).orderBy("id_l", "id_r").show(truncate=False)

    print("Total joined rows:", result.count())

finally:
    spark.stop()
