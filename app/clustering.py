from __future__ import annotations

import math
from typing import Iterable


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def centroid(vectors: Iterable[list[float]]) -> list[float]:
    vectors = list(vectors)
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vector in vectors:
        for i, value in enumerate(vector):
            sums[i] += value
    return [value / len(vectors) for value in sums]


def best_group(embedding: list[float], group_vectors: dict[str, list[list[float]]], threshold: float) -> str | None:
    best_id = None
    best_score = threshold
    for group_id, vectors in group_vectors.items():
        score = cosine_similarity(embedding, centroid(vectors))
        if score >= best_score:
            best_score = score
            best_id = group_id
    return best_id
