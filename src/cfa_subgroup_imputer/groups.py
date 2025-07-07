"""
Submodule for broad-sense handling of supergroups and subgroups.
"""

from collections.abc import Container, Iterable
from typing import Literal, NamedTuple, Self

import polars as pl

from cfa_subgroup_imputer.one_dimensional import ImputableMeasurement

GroupType = Literal["supergroup", "subgroup"]


class Group(NamedTuple):
    """
    A class to represent a super or subgroup.

    Parameters
    ----------
    name : str
        The name defining the group.
    size: float
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

    name: str
    size: float
    weight_adjustment: float | None
    weight: float | None
    measurements: list[ImputableMeasurement] | None

    def alter_measurements(
        self, measurements: list[ImputableMeasurement]
    ) -> Self:
        assert self.measurements is not None
        existing = [(m.name, m.type) for m in self.measurements]
        proposed = [(m.name, m.type) for m in measurements]
        assert set(existing) == set(proposed)
        return type(self)(
            self.name,
            self.size,
            self.weight_adjustment,
            self.weight,
            measurements,
        )

    def density_to_mass(self) -> Self:
        """
        Make all measurements masses.
        """
        assert self.measurements is not None, (
            f"Cannot convert densities to masses for group {self.name} which has no measurements."
        )
        return type(self)(
            name=self.name,
            size=self.size,
            weight_adjustment=self.weight_adjustment,
            weight=self.weight,
            measurements=[
                m.to_mass(self.size) if m.type == "density" else m
                for m in self.measurements
            ],
        )

    def mass_to_density(self, measurements: Container[str]) -> Self:
        """
        Make selected measurements densities.
        """
        assert self.measurements is not None, (
            f"Cannot convert masses to densities for group {self.name} which has no measurements."
        )
        return type(self)(
            name=self.name,
            size=self.size,
            weight_adjustment=self.weight_adjustment,
            weight=self.weight,
            measurements=[
                m.to_density(self.size) if m.name in measurements else m
                for m in self.measurements
            ],
        )


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

    def calculate_normalized_weights(self) -> Self:
        """
        Returns a new GroupMap with computed `.weight` for each Group where,
        for each supergroup, the sum of the weights across subgroups is 1.
        """
        raise NotImplementedError()

    def density_to_mass(self, sub_or_super: GroupType) -> Self:
        """
        Put all density measurements in super or subgroups on mass scale for ease of downstream manipulation.
        """
        raise NotImplementedError()

    def group(self, name: str) -> Group:
        return self.groups[name]

    def mass_to_density(
        self, sub_or_super: GroupType, measurements: Container[str]
    ) -> Self:
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
    def densities(self) -> list[str]:
        """
        Which of the measurements in this GroupMap are densities?
        """
        name = list(self.sub_to_super.values())[0]
        group = self.group(name)
        assert group.measurements is not None
        return [m.name for m in group.measurements if m.type == "density"]

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
