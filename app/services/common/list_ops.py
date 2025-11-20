"""
List operations for deduplication and filtering.
"""

from typing import Any, Dict, List, Set, Tuple


def dedupe_list(items: List[Any]) -> List[Any]:
    """
    Deduplicate a list while preserving order.

    Args:
        items: List that may contain duplicates

    Returns:
        List with duplicates removed, order preserved
    """
    seen: Set[Any] = set()
    result = []
    for item in items:
        # Handle unhashable types
        try:
            if item not in seen:
                seen.add(item)
                result.append(item)
        except TypeError:
            # Unhashable type (e.g., dict, list)
            if item not in result:
                result.append(item)
    return result


def dedupe_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """
    Deduplicate list of dicts by a specific key, keeping first occurrence.

    Args:
        items: List of dicts
        key: Key to deduplicate on

    Returns:
        Deduplicated list
    """
    seen: Set[Any] = set()
    result = []
    for item in items:
        value = item.get(key)
        if value not in seen:
            seen.add(value)
            result.append(item)
    return result


def dedupe_by_multiple_keys(items: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    """
    Deduplicate list of dicts by multiple keys.

    Args:
        items: List of dicts
        keys: List of keys to form composite dedup key

    Returns:
        Deduplicated list
    """
    seen: Set[Tuple[Any, ...]] = set()
    result = []
    for item in items:
        # Create tuple of values for composite key
        composite_key = tuple(item.get(k) for k in keys)
        if composite_key not in seen:
            seen.add(composite_key)
            result.append(item)
    return result


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists into a single list.

    Args:
        nested_list: List of lists

    Returns:
        Flattened list
    """
    result = []
    for sublist in nested_list:
        if isinstance(sublist, (list, tuple)):
            result.extend(sublist)
        else:
            result.append(sublist)
    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def merge_dicts_in_list(dicts: List[Dict[str, Any]], key: str, merge_fn: Any = None) -> List[Dict[str, Any]]:
    """
    Merge dicts with the same key value.

    Args:
        dicts: List of dicts
        key: Key to group by
        merge_fn: Function to merge values (default: concatenate lists)

    Returns:
        List with merged dicts
    """

    def default_merge(v1: Any, v2: Any) -> Any:
        """Default merge function that concatenates lists."""
        return (v1 if isinstance(v1, list) else [v1]) + (v2 if isinstance(v2, list) else [v2])

    if merge_fn is None:
        merge_fn = default_merge

    grouped: Dict[Any, Dict[str, Any]] = {}
    for d in dicts:
        group_key = d.get(key)
        if group_key not in grouped:
            grouped[group_key] = d.copy()
        else:
            # Merge fields
            for k, v in d.items():
                if k != key:
                    if k in grouped[group_key]:
                        grouped[group_key][k] = merge_fn(grouped[group_key][k], v)
                    else:
                        grouped[group_key][k] = v

    return list(grouped.values())


def filter_by_score(items: List[Dict[str, Any]], score_key: str, threshold: float) -> List[Dict[str, Any]]:
    """
    Filter dicts by a numeric score field.

    Args:
        items: List of dicts
        score_key: Key containing numeric score
        threshold: Minimum score (inclusive)

    Returns:
        Filtered list
    """
    return [item for item in items if item.get(score_key, 0) >= threshold]
