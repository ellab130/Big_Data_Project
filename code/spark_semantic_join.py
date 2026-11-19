
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType
)

from semantic_join import SemanticJoin


class SparkSemanticJoin:
    def __init__(self, spark, **matcher_kwargs):
        self.spark = spark
        self.matcher = SemanticJoin(**matcher_kwargs)

    def join(self, left, right, left_on, right_on):
        # 1. Extract distinct, non-null values.
        left_values = [
            row["value"]
            for row in (
                left.select(
                    F.col(left_on).cast("string").alias("value")
                )
                .where(F.col("value").isNotNull())
                .distinct()
                .collect()
            )
        ]

        right_values = [
            row["value"]
            for row in (
                right.select(
                    F.col(right_on).cast("string").alias("value")
                )
                .where(F.col("value").isNotNull())
                .distinct()
                .collect()
            )
        ]

        # 2. Use the existing semantic matcher.
        mappings = self.matcher.match_values(
            left_values, right_values
        )

        schema = StructType([
            StructField("left_value", StringType()),
            StructField("right_value", StringType()),
            StructField("fuzzy_score", DoubleType()),
            StructField("embedding_score", DoubleType()),
        ])

        mapping_rows = [
            (
                str(row.left_value),
                str(row.right_value),
                float(row.fuzzy_score),
                float(row.embedding_score),
            )
            for row in mappings.itertuples(index=False)
        ]

        mapping_df = self.spark.createDataFrame(
            mapping_rows, schema=schema
        )

        # 3. Join mappings back to all original rows.
        # Aliases prevent ambiguity when columns share names.
        l = left.alias("l")
        m = F.broadcast(mapping_df).alias("m")
        r = right.alias("r")

        matched_left = l.join(
            m,
            F.col(f"l.{left_on}").cast("string")
            == F.col("m.left_value"),
            "inner",
        )

        result = matched_left.join(
            r,
            F.col("m.right_value")
            == F.col(f"r.{right_on}").cast("string"),
            "inner",
        )

        return result.select(
            *[
                F.col(f"l.`{c}`").alias(f"{c}_l")
                for c in left.columns
            ],
            *[
                F.col(f"r.`{c}`").alias(f"{c}_r")
                for c in right.columns
            ],
            F.col("m.fuzzy_score"),
            F.col("m.embedding_score"),
        )
