"""
Submodule for broad-sense handling of supergroups and subgroups.
"""

from collections.abc import Iterable
from typing import Any, Literal, Self, get_args

import polars as pl

from cfa_subgroup_imputer.variables import (
    DensityMeasurementType,
    GroupingVariable,
    ImputableMeasurement,
)

GroupType = Literal["supergroup", "subgroup"]


class Group:
    """
    A class to represent a super or subgroup.
    """

    def __init__(
        self,
        name: str,
        group_vartype: GroupingVariable,
        measurements: Iterable[ImputableMeasurement] = [],
        properties: Iterable[tuple[str, Any]] = [],
    ):
        """
        Group constructor

        Parameters
        ----------
        name : str
            The name defining the group.
        group_vartype: float
            The size of the group, e.g. number of people in it.
        weight_adjustment: float
            A value that accounts for a relative increase or decrease in how much mass a
            subgroup contributes to a measurement compared to its size.
        weight: float | None
            Composite of size and weight_adjustment relative to other groups.
            Should not be touched except by GroupMap.
        measurements: Iterable[ImputableMeasurement] | None
            Optional values that can be imputed from supergroups to subgroups (and back).
        """
        self.name = name
        self.group_vartype = group_vartype
        self.measurements = tuple(measurements)
        self.properties = tuple(properties)
        self._validate()

    def _copy_except_measurements(
        self, measurements: Iterable[ImputableMeasurement]
    ) -> Self:
        return type(self)(
            name=self.name,
            group_vartype=self.group_vartype,
            measurements=measurements,
            properties=self.properties,
        )

    def _validate(self):
        assert all(
            [isinstance(m, ImputableMeasurement) for m in self.measurements]
        ), "All measurements must be ImputableMeasurements"
        measurement_names = [m.name for m in self.measurements]
        assert len(set(measurement_names)) == len(measurement_names), (
            "Found multiple measurements for same variable."
        )
        # TODO: validate properties?

    def add_measurement(self, measurement: ImputableMeasurement) -> Self:
        """
        Give this group a new measurement.
        """
        assert measurement.name not in [m.name for m in self.measurements], (
            f"Cannot add measurement {measurement} to group {self.name} which already has {self.get_measurement(measurement.name)}"
        )
        return self._copy_except_measurements(
            self.measurements + (measurement,)
        )

    def scale_measurements(self, val: float) -> Self:
        """
        Scale all group's measurements by a constant.
        """
        return self._copy_except_measurements(
            [m * val for m in self.measurements]
        )

    def density_to_mass(self, size_from: str = "size") -> Self:
        """
        Make all measurements masses.
        """
        assert self.measurements is not None, (
            f"Cannot convert densities to masses for group {self.name} which has no measurements."
        )
        size = self.get_measurement(size_from).value
        measurements = [
            m.to_mass(size)
            if m.type in get_args(DensityMeasurementType)
            else m
            for m in self.measurements
        ]
        return self._copy_except_measurements(measurements)

    def get_measurement(self, name: str) -> ImputableMeasurement:
        """
        Retrieve stated measurement.
        """
        assert name in self.measurements, (
            f"Group {self.name} has no measurement {name}"
        )
        return [m for m in self.measurements if m.name == name][0]

    def restore_densities(self, size_from: str = "size") -> Self:
        """
        Undo density_to_mass().
        """
        assert self.measurements is not None, (
            f"Cannot convert masses to densities for group {self.name} which has no measurements."
        )
        size = self.get_measurement(size_from).value
        measurements = [
            m.to_density(size) if m.name == "density_from_mass" else m
            for m in self.measurements
        ]
        return self._copy_except_measurements(measurements)


class GroupMap:
    """
    A class that binds supergroups and subgroups together, primarily serving to validate inputs.
    """

    def __init__(self, sub_to_super: dict[str, str], groups: Iterable[Group]):
        """
        Default constructor, takes in a subgroup : supergroup dict.
        """
        # Should probably store one dict of group name to Group, then sub<>super dicts as dict[str, str]
        self.sub_to_super = sub_to_super
        self.super_to_sub = GroupMap.make_one_to_many(sub_to_super)
        self.groups = {group.name: group for group in groups}
        self._validate()

    @classmethod
    def from_supergroups(
        cls, super_to_sub: dict[str, Iterable[str]], groups: Iterable[Group]
    ) -> Self:
        """
        Alternative constructor, takes in a supergroup : [subgroups] dict.
        """
        sub_to_super = GroupMap.make_many_to_one(super_to_sub)
        return cls(sub_to_super, groups)

    def _assert_names_unique(self) -> None:
        """
        Ensure that super and subgroup names are all unique.
        """
        raise NotImplementedError()

    def _assert_no_missing_data(self) -> None:
        """
        Ensure that each supergroup's size is the sum of constituent subgroup sizes.
        """
        raise NotImplementedError()

    def _assert_no_missing_population(self) -> None:
        """
        Ensure that each supergroup's size is the sum of constituent subgroup sizes.
        """
        raise NotImplementedError()

    def _validate(self):
        self._assert_names_unique()
        self._assert_no_missing_population()
        self._assert_no_missing_data()
        assert self.aggregatable ^ self.disaggregatable

    @property
    def aggregatable(self) -> bool:
        """
        We can aggregate the subgroups if we have data for all measurements in all subgroups.
        """
        raise NotImplementedError()

    def density_to_mass(self, sub_or_super: GroupType) -> Self:
        """
        Put all density measurements in super or subgroups on mass scale for ease of downstream manipulation.
        """
        raise NotImplementedError()

    def group(self, name: str) -> Group:
        return self.groups[name]

    def restore_densities(self, sub_or_super: GroupType) -> Self:
        """
        Undo density_to_mass for selected measurements.
        """
        raise NotImplementedError()

    def data_as_polars(self, sub_or_super: GroupType) -> pl.DataFrame:
        """
        Creates a polars dataframe of the measurements in either the supergroups or subgroups.
        """
        raise NotImplementedError()

    @property
    def disaggregatable(self) -> bool:
        """
        We can disaggregate the supergroups if we have data for all measurements in all supergroups.
        """
        raise NotImplementedError()

    @staticmethod
    def make_many_to_one(
        super_to_sub: dict[str, Iterable[str]],
    ) -> dict[str, str]:
        """
        Inverts a supergroup : [subgroups] one to one dict to a subgroup : supergroup one to many dict
        """
        return {v: k for k, v_list in super_to_sub.items() for v in v_list}

    @staticmethod
    def make_one_to_many(
        sub_to_super: dict[str, str],
    ) -> dict[str, list[str]]:
        """
        Inverts a subgroup : supergroup one to one dict to a supergroup : [subgroups] one to many dict
        """
        super_to_sub = {}
        for k, v in sub_to_super.items():
            if v in super_to_sub:
                super_to_sub[v].append(k)
            else:
                super_to_sub[v] = [k]
        return super_to_sub

    def subgroups(self, name: str) -> list[str]:
        assert name in self.super_to_sub.keys()
        return self.super_to_sub[name]

    def supergroup(self, name: str) -> str:
        assert name in self.sub_to_super.keys()
        return self.sub_to_super[name]

    @property
    def supergroups(self) -> list[str]:
        return list(self.super_to_sub.keys())
