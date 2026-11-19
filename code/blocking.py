
import re
from collections import defaultdict


STOPWORDS = {
    "tennis", "tournament", "championship",
    "championships", "open", "cup",
    "challenger", "international",
    "mens", "womens", "the", "of",
    "at", "and"
}


def tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())

    return {
        token
        for token in tokens
        if token not in STOPWORDS
        and not re.fullmatch(r"(19|20)\d{2}", token)
    }


def generate_candidates(left_values, right_values):
    """
    Generate candidate pairs using shared informative tokens.

    Returns:
        candidates: set of (left_index, right_index)
        comparisons: number of candidate pairs
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
