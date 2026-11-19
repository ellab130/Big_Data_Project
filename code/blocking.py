
import re
from collections import defaultdict


STOPWORDS = {
    "tennis",
    "tournament",
    "championship",
    "championships",
    "open",
    "cup",
    "challenger",
    "international",
    "mens",
    "womens",
    "the",
    "of",
    "at",
    "and",
}


def tokenize(text):
    """Extract meaningful tokens for candidate blocking."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())

    return {
        token
        for token in tokens
        if token not in STOPWORDS
        and not re.fullmatch(r"(19|20)\d{2}", token)
    }


def generate_candidates(left_values, right_values):
    """
    Generate candidate index pairs using shared-token blocking.

    Returns:
        candidates: set of (left_index, right_index) pairs
        count: number of candidate pairs
    """
    index = defaultdict(set)

    for j, value in enumerate(right_values):
        for token in tokenize(value):
            index[token].add(j)

    candidates = set()

    for i, value in enumerate(left_values):
        for token in tokenize(value):
            for j in index.get(token, set()):
                candidates.add((i, j))

    return candidates, len(candidates)


def char_ngrams(text, n=3):
    """Return character n-grams after removing punctuation."""
    if n < 1:
        raise ValueError("n must be at least 1")

    cleaned = re.sub(r"[^a-z0-9]", "", str(text).lower())

    if not cleaned:
        return set()

    if len(cleaned) < n:
        return {cleaned}

    return {
        cleaned[i:i + n]
        for i in range(len(cleaned) - n + 1)
    }


def generate_ranked_char_candidates(
    left_values,
    right_values,
    top_k=10,
    n=3
):
    """
    Retrieve the strongest character-overlap candidates per left value.

    Uses an inverted index to find right values sharing character
    n-grams, then ranks them by Jaccard similarity.

    Returns:
        candidates: set of (left_index, right_index) pairs

    Each left value contributes at most top_k candidate pairs.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    index = defaultdict(set)
    right_grams = []

    # Build an inverted index over right-side character n-grams.
    for j, value in enumerate(right_values):
        grams = char_ngrams(value, n=n)
        right_grams.append(grams)

        for gram in grams:
            index[gram].add(j)

    candidates = set()

    for i, value in enumerate(left_values):
        left_grams = char_ngrams(value, n=n)
        shared_counts = defaultdict(int)

        # Count shared n-grams with each potential right match.
        for gram in left_grams:
            for j in index.get(gram, ()):
                shared_counts[j] += 1

        scored_candidates = []

        for j, shared in shared_counts.items():
            union_size = (
                len(left_grams)
                + len(right_grams[j])
                - shared
            )

            similarity = (
                shared / union_size
                if union_size else 0.0
            )

            scored_candidates.append((similarity, j))

        # Highest Jaccard similarity first.
        # Secondary index ordering makes ties deterministic.
        scored_candidates.sort(
            key=lambda item: (-item[0], item[1])
        )

        for similarity, j in scored_candidates[:top_k]:
            candidates.add((i, j))

    return candidates
