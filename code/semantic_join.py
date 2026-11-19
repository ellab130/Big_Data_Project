
import re

import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio
from sentence_transformers import SentenceTransformer

from blocking import (
    generate_candidates,
    generate_ranked_char_candidates,
)


class SemanticJoin:
    """
    Semantic join over distinct text values.

    Retrieval modes:
        embedding: embedding top-k only
        hybrid: embedding top-k + token blocking
        exhaustive: embedding top-k + exhaustive fuzzy blocking

    Optional character retrieval:
        char_top_k > 0 adds ranked character n-gram candidates.

    Cardinality modes:
        many_to_many: retain all accepted value mappings
        one_to_one: each left/right value used at most once
        right_unique: each right value used at most once
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_threshold=0.75,
        fuzzy_threshold=0.80,
        top_k=3,
        check_year=False,
        locations=None,
        retrieval="hybrid",
        char_top_k=0,
        cardinality="many_to_many",
    ):
        if retrieval not in {
            "exhaustive",
            "embedding",
            "hybrid",
        }:
            raise ValueError(
                "retrieval must be 'exhaustive', "
                "'embedding', or 'hybrid'"
            )

        if cardinality not in {
            "many_to_many",
            "one_to_one",
            "right_unique",
        }:
            raise ValueError(
                "cardinality must be 'many_to_many', "
                "'one_to_one', or 'right_unique'"
            )

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if char_top_k < 0:
            raise ValueError("char_top_k cannot be negative")

        self.embedding_threshold = embedding_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.top_k = top_k
        self.check_year = check_year
        self.locations = locations or set()
        self.retrieval = retrieval
        self.char_top_k = char_top_k
        self.cardinality = cardinality

        self.model = SentenceTransformer(model_name)

        # Diagnostics for benchmarking.
        self.last_candidate_count = 0
        self.last_total_pairs = 0
        self.last_verified_count = 0
        self.last_match_count = 0

    def normalize(self, text):
        return " ".join(str(text).lower().split())

    def extract_year(self, text):
        match = re.search(
            r"\b(?:19|20)\d{2}\b",
            str(text),
        )
        return match.group(0) if match else None

    def extract_locations(self, text):
        normalized = self.normalize(text)

        return {
            location
            for location in self.locations
            if location.lower() in normalized
        }

    def compatible(self, a, b):
        if self.check_year:
            year_a = self.extract_year(a)
            year_b = self.extract_year(b)

            if year_a and year_b and year_a != year_b:
                return False

        if self.locations:
            loc_a = self.extract_locations(a)
            loc_b = self.extract_locations(b)

            if loc_a and loc_b and loc_a.isdisjoint(loc_b):
                return False

        return True

    def apply_cardinality(self, matches):
        """
        Apply optional cardinality constraints to value mappings.

        Greedy selection by embedding similarity, breaking ties
        with fuzzy similarity and then value ordering.
        """
        if (
            matches.empty
            or self.cardinality == "many_to_many"
        ):
            return matches.reset_index(drop=True)

        ranked = matches.sort_values(
            [
                "embedding_score",
                "fuzzy_score",
                "left_value",
                "right_value",
            ],
            ascending=[False, False, True, True],
            kind="stable",
        )

        if self.cardinality == "right_unique":
            return (
                ranked.drop_duplicates(
                    subset=["right_value"],
                    keep="first",
                )
                .reset_index(drop=True)
            )

        # Greedy one-to-one selection.
        used_left = set()
        used_right = set()
        selected_indices = []

        for index, row in ranked.iterrows():
            a = row["left_value"]
            b = row["right_value"]

            if a in used_left or b in used_right:
                continue

            selected_indices.append(index)
            used_left.add(a)
            used_right.add(b)

        return (
            ranked.loc[selected_indices]
            .reset_index(drop=True)
        )

    def match_values(self, left_values, right_values):
        """
        Match distinct non-null text values.

        Returns a DataFrame containing:
            left_value
            right_value
            fuzzy_score
            embedding_score
        """
        left_values = list(dict.fromkeys(
            str(v)
            for v in left_values
            if pd.notna(v)
        ))

        right_values = list(dict.fromkeys(
            str(v)
            for v in right_values
            if pd.notna(v)
        ))

        columns = [
            "left_value",
            "right_value",
            "fuzzy_score",
            "embedding_score",
        ]

        self.last_candidate_count = 0
        self.last_total_pairs = (
            len(left_values) * len(right_values)
        )
        self.last_verified_count = 0
        self.last_match_count = 0

        if not left_values or not right_values:
            return pd.DataFrame(columns=columns)

        # Generate normalized embeddings.
        left_vectors = self.model.encode(
            left_values,
            normalize_embeddings=True,
        )

        right_vectors = self.model.encode(
            right_values,
            normalize_embeddings=True,
        )

        similarity = left_vectors @ right_vectors.T

        candidates = set()

        # Embedding top-k retrieval.
        k = min(self.top_k, len(right_values))

        for i in range(len(left_values)):
            top_indices = np.argsort(
                similarity[i]
            )[-k:]

            for j in top_indices:
                candidates.add((i, int(j)))

        # Shared-token retrieval.
        if self.retrieval == "hybrid":
            token_candidates, _ = generate_candidates(
                left_values,
                right_values,
            )
            candidates.update(token_candidates)

        # Exhaustive fuzzy retrieval, retained for baselines.
        if self.retrieval == "exhaustive":
            for i, a in enumerate(left_values):
                for j, b in enumerate(right_values):
                    normalized_a = self.normalize(a)
                    normalized_b = self.normalize(b)

                    fuzzy = (
                        ratio(normalized_a, normalized_b)
                        / 100
                    )

                    if (
                        normalized_a == normalized_b
                        or fuzzy >= self.fuzzy_threshold
                    ):
                        candidates.add((i, j))

        # Optional ranked character n-gram retrieval.
        if self.char_top_k > 0:
            char_candidates = (
                generate_ranked_char_candidates(
                    left_values,
                    right_values,
                    top_k=self.char_top_k,
                    n=3,
                )
            )
            candidates.update(char_candidates)

        self.last_candidate_count = len(candidates)

        # Candidate verification.
        matches = []

        for i, j in sorted(candidates):
            a = left_values[i]
            b = right_values[j]

            if not self.compatible(a, b):
                continue

            normalized_a = self.normalize(a)
            normalized_b = self.normalize(b)

            fuzzy = (
                ratio(normalized_a, normalized_b)
                / 100
            )

            embedding = float(similarity[i, j])

            if (
                embedding >= self.embedding_threshold
                or normalized_a == normalized_b
            ):
                matches.append({
                    "left_value": a,
                    "right_value": b,
                    "fuzzy_score": fuzzy,
                    "embedding_score": embedding,
                })

        self.last_verified_count = len(matches)

        matches_df = pd.DataFrame(
            matches,
            columns=columns,
        )

        # Apply cardinality after verification.
        result = self.apply_cardinality(matches_df)

        self.last_match_count = len(result)

        return result

    def join(self, left, right, left_on, right_on):
        """
        Join pandas DataFrames using semantic value mappings.

        Original rows and duplicate occurrences are preserved.
        """
        mappings = self.match_values(
            left[left_on],
            right[right_on],
        )

        left_rows = left.copy()
        right_rows = right.copy()

        left_rows["_join_value"] = (
            left_rows[left_on].astype(str)
        )

        right_rows["_join_value"] = (
            right_rows[right_on].astype(str)
        )

        matched_left = left_rows.merge(
            mappings,
            left_on="_join_value",
            right_on="left_value",
            how="inner",
        )

        result = matched_left.merge(
            right_rows,
            left_on="right_value",
            right_on="_join_value",
            how="inner",
            suffixes=("_l", "_r"),
        )

        return result.drop(
            columns=[
                "_join_value_l",
                "_join_value_r",
            ],
            errors="ignore",
        )
