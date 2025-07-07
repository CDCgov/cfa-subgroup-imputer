"""
Submodule for handling one-dimensional variables, be they measurements for imputing or axes for splitting groups.
"""

from typing import Literal, NamedTuple, Self

MeasurementType = Literal["mass", "density"]
"""
How a measurement behaves for disaggregation.

Mass-like behavior are things like counts, while density-like measurements are
things like rates or proportions.
"""


class ImputableMeasurement(NamedTuple):
    """
    A class for a measurement we can impute from supergroups to subgroups.

    Parameters
    ----------
    value : float
        The value, e.g. a number of cases.
    name : str
        What is this a measurement of?
    type: MeasurementType
        What kind of measurement is this?
    """

    value: float
    name: str
    type: MeasurementType

    def __mul__(self, k: float) -> Self:
        return type(self)(value=self.value * k, name=self.name, type=self.type)

    def to_mass(self, volume: float) -> Self:
        return type(self)(
            value=self.value * volume, name=self.name, type="mass"
        )

    def to_density(self, volume: float) -> Self:
        return type(self)(
            value=self.value / volume, name=self.name, type="density"
        )


class Range(NamedTuple):
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

    lower: float
    lower_included: bool
    upper: float
    upper_included: bool


class Enumerator:
    """
    A class that assists in making sub to supergroup maps for an underlying
    axis defined by a continuous variable, such as age.

    E.g., something that takes you from "my age subgroups are... and my age
    supergroups are..." to a sub : super group name/string dict.
    """

    pass


class Aligner:
    """
    A class that takes in multiple sets of supergroups and defines the largest
    common denominator set of subgroups which allow things to be aligned among
    the groups.
    """

    pass
