"""
Submodule for handling variables, whether measurements or quantities used to define subgroups.
"""

from collections.abc import Iterable
from typing import (
    Any,
    Hashable,
    Literal,
    NamedTuple,
    Self,
    get_args,
)

MassMeasurementType = Literal["mass", "mass_from_density"]
DensityMeasurementType = Literal["density", "density_from_mass"]
MeasurementType = Literal[
    "mass", "density", "mass_from_density", "density_from_mass"
]
"""
How a measurement behaves for disaggregation.

Mass-like behavior are things like counts, while density-like measurements are
things like rates or proportions.
"""

ImputeAction = Literal["impute", "copy", "ignore"]
"""
What should be done with this value when disaggregating?
- "impute" means the value will be imputed (must be )
- "copy" means the value from the supergroup will be copied to all subgroups
- "ignore" means this value is not propagated from supergroups to subgroups
"""


class Attribute:
    """
    A class for data we can associate with a subgroup.
    """

    def __init__(
        self,
        value: Any,
        name: Hashable,
        impute_action: ImputeAction,
    ):
        """
        Attribute constructor.

        Parameters
        ----------
        value : Any
            The value of the variable.
        name : Hashable
            What is this variable? E.g., "size" or "vaccination rate"
        impute_action: ImputeAction
            What should we do with this measurement when disaggregating?
            Note that just because we can impute it doesn't mean we will.
        """
        self.value = value
        self.name: Hashable = name
        self.impute_action: ImputeAction = impute_action
        self._validate()

    def __eq__(self, x):
        return (
            self.name == x.name
            and self.value == x.value
            and self.impute_action == x.impute_action
        )

    def __repr__(self):
        return f"Attribute(name={self.name}, impute_action={self.impute_action}, value={self.value})"

    def _validate(self):
        assert isinstance(self.name, Hashable)
        # Can't impute the base class
        assert self.impute_action in ["copy", "ignore"]


class ImputableAttribute(Attribute):
    """
    A class for data we can associate with a subgroup and which can be imputed to subgroups.
    """

    def __init__(
        self,
        value: float,
        name: Hashable,
        impute_action: ImputeAction,
        measurement_type: MeasurementType,
    ):
        """
        ImputableAttribute constructor.

        Parameters
        ----------
        value : float
            The value, e.g. a number of cases.
        name : Hashable
            What is this variable? E.g., "size" or "vaccination rate"
        impute_action: ImputeAction
            What should we do with this measurement when disaggregating?
            Note that just because we can impute it doesn't mean we will.
        type: MeasurementType
            What kind of imputable attribute is this?
        """
        assert value >= 0.0
        super().__init__(value, name, impute_action)
        self.measurement_type: MeasurementType = measurement_type
        assert self.measurement_type in get_args(MeasurementType)

    def _validate(self):
        assert self.impute_action in get_args(ImputeAction)

    def __eq__(self, x):
        return super().__eq__() and self.measurement_type == x.measurement_type

    def __mul__(self, k: float) -> Self:
        return type(self)(
            value=self.value * k,
            name=self.name,
            impute_action=self.impute_action,
            measurement_type=self.measurement_type,
        )

    def to_count(self, size: float) -> Self:
        return type(self)(
            value=self.value * size,
            name=self.name,
            impute_action=self.impute_action,
            measurement_type="mass_from_density",
        )

    def to_rate(self, volume: float) -> Self:
        return type(self)(
            value=self.value / volume,
            name=self.name,
            impute_action=self.impute_action,
            measurement_type="density_from_mass",
        )


class Range:
    """
    A slice of a one-dimensional variable. e.g. [0, 3.14159).

    Parameters
    ----------
    lower : float
        Value at the lower end of the range.
    lower_included: bool
        Is the range inclusive of the lower value?
    upper : float
        Value at the upper end of the range.
    upper_included: bool
        Is the range inclusive of the upper value?
    """

    def __init__(
        self,
        lower: float,
        upper: float,
    ):
        self.lower: float = lower
        self.upper: float = upper

    def __add__(self, x: Self) -> Self:
        # @TODO: should this be less exact?
        assert self.upper == x.lower
        return type(self)(lower=self.lower, upper=x.upper)

    def __contains__(self, x: Self):
        return x.lower >= self.lower and x.upper <= self.upper

    def __gt__(self, x: Self):
        return self.lower >= x.upper

    def __hash__(self):
        return self.to_tuple().__hash__()

    def __lt__(self, x: Self):
        return self.upper <= x.lower

    def __eq__(self, x: Self):
        return self.lower == x.lower and self.upper == x.upper

    def __repr__(self):
        return f"Range({self.lower},{self.upper})"

    @classmethod
    def from_tuple(cls, low_high: tuple[float, float]):
        return cls(low_high[0], low_high[1])

    def to_tuple(self) -> tuple[float, float]:
        return (self.lower, self.upper)


def assert_range_spanned_exactly(
    range: Range, ranges: Iterable[Range]
) -> None:
    """
    Checks that the provided `ranges`, in aggregate, span exactly `range`.

    [Range(0., 1.), Range(1., 10.)] span Range(0., 10.)
    [Range(0., 1.), Range(1., 10.1)] does not span Range(0., 10.)
    [Range(0., 1.), Range(2., 10.)] does not span Range(0., 10.)
    """
    ranges = sorted(ranges)
    lower = range.lower
    assert ranges[0].lower == lower
    cumulative = ranges[0]
    for r in ranges[1:]:
        cumulative += r
    assert cumulative.upper == range.upper


GroupableTypes = Literal["Categorical", "Continuous"]


class GroupingVariable(NamedTuple):
    """
    A class for holding variables that can define subgroups

    Parameters
    name: str
        The name of the variable.
    type: GroupableTypes
        The type of variable.
    ----------
    """

    name: str
    type: GroupableTypes
