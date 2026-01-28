import functools
import itertools
from typing import Literal, Self, Tuple

import pytest


@functools.total_ordering
class Age:
    def __init__(self, x: int, metric: Literal["year", "month", "week"]):
        self.x = x
        self.metric = metric

    def _get_cmp(self, other: Self) -> Tuple[int, int]:
        if self.metric == other.metric:
            return (self.x, other.x)
        elif self.x == 0 and other.x == 0:
            return (0, 0)
        elif self.metric == "year" and other.metric == "month":
            return (self.x * 12, other.x)
        elif self.metric == "month" and other.metric == "year":
            return (self.x, other.x * 12)
        else:
            raise ValueError(f"Cannot compare {self} and {other}")

    def __eq__(self, other: Self) -> bool:
        x1, x2 = self._get_cmp(other)
        return x1 == x2

    def __lt__(self, other: Self) -> bool:
        x1, x2 = self._get_cmp(other)
        return x1 < x2

    def __repr__(self) -> str:
        return f"Age(x={self.x}, metric='{self.metric}')"

    def __hash__(self) -> int:
        return hash((self.x, self.metric))


class AgeGroup:
    def __init__(self, start: Age, end: Age):
        assert start < end
        self.start = start
        self.end = end

    def is_in(self, other: Self) -> bool:
        return other.start <= self.start and self.end <= other.end

    def subdivide(self, cuts: list[Age]) -> list[Self]:
        assert len(cuts) >= 2
        assert cuts[0] == self.start
        assert cuts[-1] == self.end

        return [
            type(self)(start, end) for start, end in itertools.pairwise(cuts)
        ]

    def __repr__(self) -> str:
        return f"AgeGroup({self.start}, {self.end})"

    def __eq__(self, other: Self) -> bool:
        return self.start == other.start and self.end == other.end


with pytest.raises(ValueError):
    _ = Age(1, "year") < Age(52, "week")

assert Age(2, "week") == Age(2, "week")
assert Age(1, "year") == Age(12, "month")

assert AgeGroup(Age(0, "month"), Age(1, "month")).is_in(
    AgeGroup(Age(0, "year"), Age(1, "year"))
)

assert AgeGroup(Age(0, "year"), Age(1, "year")).subdivide(
    [Age(m, "month") for m in range(13)]
) == [AgeGroup(Age(m, "month"), Age(m + 1, "month")) for m in range(12)]

# subdivide into months *and* years
age_groups = AgeGroup(Age(0, "year"), Age(4, "year")).subdivide(
    [Age(m, "month") for m in range(12)]
    + [Age(y, "year") for y in range(1, 5)]
)
# at first, it's month by month
assert age_groups[0] == AgeGroup(Age(0, "month"), Age(1, "month"))
# then there's a transition, where we note 1 year = 12 months
assert age_groups[-4] == AgeGroup(Age(11, "month"), Age(1, "year"))
assert age_groups[-4] == AgeGroup(Age(11, "month"), Age(12, "month"))
# and after that it's in years
assert age_groups[-1] == AgeGroup(Age(3, "year"), Age(4, "year"))
