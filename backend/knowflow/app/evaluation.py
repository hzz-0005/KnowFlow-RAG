from collections.abc import Iterable, Sequence
from math import log2


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of known relevant items found in the first k results."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the first k result slots that contain a relevant item."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    return len(set(retrieved_ids[:k]) & relevant) / k


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant result, or zero when none is found."""
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1 / rank
    return 0.0


def mean_reciprocal_rank(
    result_sets: Iterable[tuple[Sequence[str], Iterable[str]]],
) -> float:
    """Mean reciprocal rank across (retrieved IDs, relevant IDs) query pairs."""
    scores = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in result_sets]
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary relevance normalized discounted cumulative gain at k."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum(
        1 / log2(rank + 1)
        for rank, item_id in enumerate(retrieved_ids[:k], start=1)
        if item_id in relevant
    )
    ideal_count = min(k, len(relevant))
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0
