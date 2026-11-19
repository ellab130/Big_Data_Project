
import re
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio
from sentence_transformers import SentenceTransformer
from blocking import generate_candidates


class SemanticJoin:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_threshold=0.75,
        fuzzy_threshold=0.80,
        top_k=3,
        check_year=False,
        locations=None,
        retrieval="hybrid"
    ):
        self.embedding_threshold = embedding_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.top_k = top_k
        self.check_year = check_year
        self.locations = locations or set()

        if retrieval not in {"exhaustive", "embedding", "hybrid"}:
            raise ValueError(
                "retrieval must be 'exhaustive', 'embedding', or 'hybrid'"
            )

        self.retrieval = retrieval
        self.model = SentenceTransformer(model_name)

    def normalize(self, text):
        return " ".join(str(text).lower().split())

    def extract_year(self, text):
        match = re.search(r"\b(?:19|20)\d{2}\b", str(text))
        return match.group(0) if match else None

    def extract_locations(self, text):
        normalized = self.normalize(text)

        return {
            location
            for location in self.locations
            if location.lower() in normalized
        }

    def compatible(self, a, b):
        # Reject pairs with conflicting years.
        if self.check_year:
            year_a = self.extract_year(a)
            year_b = self.extract_year(b)

            if year_a and year_b and year_a != year_b:
                return False

        # Reject pairs with conflicting known locations.
        if self.locations:
            loc_a = self.extract_locations(a)
            loc_b = self.extract_locations(b)

            if loc_a and loc_b and loc_a.isdisjoint(loc_b):
                return False

        return True

    def match_values(self, left_values, right_values):
        # Remove duplicates and missing values.
        left_values = list(dict.fromkeys(
            str(value)
            for value in left_values
            if pd.notna(value)
        ))

        right_values = list(dict.fromkeys(
            str(value)
            for value in right_values
            if pd.notna(value)
        ))

        columns = [
            "left_value",
            "right_value",
            "fuzzy_score",
            "embedding_score"
        ]

        if not left_values or not right_values:
            return pd.DataFrame(columns=columns)

        # Generate normalized embedding vectors.
        left_vectors = self.model.encode(
            left_values,
            normalize_embeddings=True
        )

        right_vectors = self.model.encode(
            right_values,
            normalize_embeddings=True
        )

        similarity = left_vectors @ right_vectors.T

        candidates = set()

        # Embedding top-k retrieval.
        if self.retrieval in {"embedding", "hybrid", "exhaustive"}:
            for i in range(len(left_values)):
                k = min(self.top_k, len(right_values))

                top_indices = np.argsort(similarity[i])[-k:]

                for j in top_indices:
                    candidates.add((i, int(j)))

        # Token-based blocking.
        if self.retrieval == "hybrid":
            token_candidates, _ = generate_candidates(
                left_values,
                right_values
            )

            candidates.update(token_candidates)

        # Exhaustive fuzzy candidate generation.
        if self.retrieval == "exhaustive":
            for i, a in enumerate(left_values):
                for j, b in enumerate(right_values):
                    fuzzy = ratio(
                        self.normalize(a),
                        self.normalize(b)
                    ) / 100

                    if (
                        self.normalize(a) == self.normalize(b)
                        or fuzzy >= self.fuzzy_threshold
                    ):
                        candidates.add((i, j))

        self.last_candidate_count = len(candidates)
        self.last_total_pairs = len(left_values) * len(right_values)
        # Verify candidate pairs.
        matches = []

        for i, j in sorted(candidates):
            a = left_values[i]
            b = right_values[j]

            if not self.compatible(a, b):
                continue

            fuzzy = ratio(
                self.normalize(a),
                self.normalize(b)
            ) / 100

            embedding = float(similarity[i, j])

            if (
                embedding >= self.embedding_threshold
                or self.normalize(a) == self.normalize(b)
            ):
                matches.append({
                    "left_value": a,
                    "right_value": b,
                    "fuzzy_score": fuzzy,
                    "embedding_score": embedding
                })

        return pd.DataFrame(matches, columns=columns)

    def join(self, left, right, left_on, right_on):
        # Match distinct values from the two DataFrames.
        mappings = self.match_values(
            left[left_on],
            right[right_on]
        )

        # Preserve original rows, including duplicates.
        left_rows = left.copy()
        right_rows = right.copy()

        left_rows["_join_value"] = left_rows[left_on].astype(str)
        right_rows["_join_value"] = right_rows[right_on].astype(str)

        # Join matching values back to left rows.
        matched_left = left_rows.merge(
            mappings,
            left_on="_join_value",
            right_on="left_value",
            how="inner"
        )

        # Join matched left rows to right rows.
        result = matched_left.merge(
            right_rows,
            left_on="right_value",
            right_on="_join_value",
            how="inner",
            suffixes=("_l", "_r")
        )

        # Remove temporary join columns.
        return result.drop(
            columns=["_join_value_l", "_join_value_r"],
            errors="ignore"
        )
