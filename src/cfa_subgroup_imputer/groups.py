"""
Submodule for broad-sense handling of supergroups and subgroups.
"""

from collections.abc import Iterable, Mapping
from typing import Hashable, Literal, Self, get_args

import polars as pl

from cfa_subgroup_imputer.variables import (
    Attribute,
    DensityMeasurementType,
    ImputableAttribute,
)

GroupType = Literal["supergroup", "subgroup"]


class Group:
    """
    A class to represent a super or subgroup.
    """

    def __init__(
        self,
        name: Hashable,
        attributes: Iterable[Attribute] = [],
    ):
        """
        Group constructor

        Parameters
        ----------
        name : Hashable
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
        self.attributes = tuple(attributes)
        self._validate()

    def __eq__(self, x: Self):
        if self.name != x.name:
            return False

        my_attr = set(a.name for a in self.attributes)
        their_attr = set(a.name for a in self.attributes)

        if not my_attr.difference(their_attr) == set():
            return False

        return all(
            self.get_attribute(a) == x.get_attribute(a) for a in my_attr
        )

    def _validate(self):
        assert all([isinstance(a, Attribute) for a in self.attributes]), (
            "All attributes must be of class Attribute"
        )
        measurement_names = [a.name for a in self.attributes]
        assert len(set(measurement_names)) == len(measurement_names), (
            "Found multiple measurements for same variable."
        )
        to_impute = set(
            a.name for a in self.attributes if a.impute_action == "impute"
        )
        imputable = set(
            a.name
            for a in self.attributes
            if isinstance(a, ImputableAttribute)
        )
        assert to_impute.issubset(imputable), (
            f"The following attributes are requested to be imputed but are not imputable: {to_impute.difference(imputable)}"
        )

    def add_attribute(self, attribute: Attribute) -> Self:
        """
        Give this group a new measurement.
        """
        assert attribute.name not in [a.name for a in self.attributes], (
            f"Cannot add measurement {attribute} to group {self.name} which already has {self.get_attribute(attribute.name)}"
        )
        return type(self)(
            name=self.name, attributes=self.attributes + (attribute,)
        )

    def rate_to_count(self, size_from: Hashable = "size") -> Self:
        """
        Make all measurements masses.
        """

        size = self.get_attribute(size_from).value
        assert isinstance(size, float)
        assert size > 0
        attributes = [
            a.to_count(size)
            if a.impute_action == "impute"
            and isinstance(a, ImputableAttribute)
            and a.measurement_type in get_args(DensityMeasurementType)
            else a
            for a in self.attributes
        ]
        return type(self)(name=self.name, attributes=attributes)

    def get_attribute(self, name: Hashable) -> Attribute:
        """
        Retrieve stated measurement.
        """
        assert name in self.attributes, (
            f"Group {self.name} has no attribute {name}"
        )
        return [a for a in self.attributes if a.name == name][0]

    def restore_rates(self, size_from: Hashable = "size") -> Self:
        """
        Undo density_to_mass().
        """
        size = self.get_attribute(size_from).value
        assert isinstance(size, float)
        assert size > 0
        attributes = [
            a.to_rate(size)
            if a.impute_action == "impute"
            and isinstance(a, ImputableAttribute)
            and a.measurement_type == "mass_from_density"
            else a
            for a in self.attributes
        ]
        return type(self)(name=self.name, attributes=attributes)


class GroupMap:
    """
    A class that binds supergroups and subgroups together, primarily serving to validate inputs.
    """

    def __init__(
        self,
        sub_to_super: Mapping[Hashable, Hashable],
        groups: Iterable[Group],
    ):
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
        cls,
        super_to_sub: dict[Hashable, Iterable[Hashable]],
        groups: Iterable[Group],
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

    def add_data_from_polars(self, df: pl.DataFrame) -> Self:
        """
        Add the data found in this dataframe to the relevant groups.
        """
        raise NotImplementedError()

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

    def data_to_polars(self, sub_or_super: GroupType) -> pl.DataFrame:
        """
        Creates a polars dataframe of the measurements in either the supergroups or subgroups.
        """
        raise NotImplementedError()

    def data_from_polars(self, df: pl.DataFrame):
        """
        Populates measurements and attributes for groups found in the dataframe.
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
        super_to_sub: Mapping[Hashable, Iterable[Hashable]],
    ) -> Mapping[Hashable, Hashable]:
        """
        Inverts a supergroup : [subgroups] one to one dict to a subgroup : supergroup one to many dict
        """
        return {v: k for k, v_list in super_to_sub.items() for v in v_list}

    @staticmethod
    def make_one_to_many(
        sub_to_super: Mapping[Hashable, Hashable],
    ) -> Mapping[Hashable, list[Hashable]]:
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

    def subgroups(self, name: Hashable) -> list[Hashable]:
        assert name in self.super_to_sub.keys()
        return self.super_to_sub[name]

    def supergroup(self, name: Hashable) -> Hashable:
        assert name in self.sub_to_super.keys()
        return self.sub_to_super[name]

    @property
    def supergroups(self) -> list[Hashable]:
        return list(self.super_to_sub.keys())
