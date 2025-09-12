from collections.abc import Iterable, Sequence
from typing import Hashable


def _dict_to_tuple(x: dict, keys: Sequence[Hashable]):
    return tuple(x[k] for k in keys)


def get_json_keys(x: Iterable[dict]) -> list[str]:
    """
    Get keys from list of dicts and make sure they are sync'd and all `str`s.
    """
    keys = get_keys(x)
    assert all(isinstance(k, str) for k in keys)
    return keys  # pyright: ignore[reportReturnType]


def get_keys(x: Iterable[dict]) -> list[Hashable]:
    """
    Get keys from list of dicts and make sure they're sync'd.
    """
    xl = list(x)
    all_keys = set(xl[0].keys())
    assert all(set(datum.keys()) == all_keys for datum in x), (
        "Provided data do not all have same keys."
    )
    return [k for k in xl[0].keys()]


def select(x: Iterable[dict], keys: Iterable[Hashable]) -> list[dict]:
    """
    Get a list of dicts with only the specified keys.
    """
    res = []
    for row in x:
        res.append({k: row[k] for k in keys})
    return res


def _tuple_to_dict(x: tuple, keys: Sequence[Hashable]):
    return {k: v for k, v in zip(keys, x)}


def unique(
    x: Iterable[dict], select: Iterable[Hashable] | None = None
) -> list[dict]:
    """
    Get only the rows out of an iterable of dicts that are unique for selected keys (default is all keys).
    """
    keys = get_keys(x)

    if select is not None:
        assert all(k in keys for k in select)
        keys = list(select)

    tuples = [_dict_to_tuple(row, keys) for row in x]
    unique_tuples = list(set(tuples))
    return [_tuple_to_dict(row, keys) for row in unique_tuples]
