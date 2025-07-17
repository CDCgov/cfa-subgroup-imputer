"""
Submodule for handling variables, whether measurements or quantities used to define subgroups.
"""

from typing import Literal, NamedTuple, Self

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
            value=self.value * volume, name=self.name, type="mass_from_density"
        )

    def to_density(self, volume: float) -> Self:
        return type(self)(
            value=self.value / volume, name=self.name, type="density_from_mass"
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
