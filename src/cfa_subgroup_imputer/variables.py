"""
Submodule for handling variables, whether measurements or quantities used to define subgroups.
"""

from typing import Any, Hashable, Literal, NamedTuple, Self, get_args

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

    def _validate(self):
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
