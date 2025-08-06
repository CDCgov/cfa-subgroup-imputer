"""
Submodule for broad-sense handling of supergroups and subgroups.
"""

from collections import Counter
from collections.abc import Container, Iterable, Mapping
from typing import Any, Hashable, Literal, Self, get_args

import polars as pl

from cfa_subgroup_imputer.variables import (
    Attribute,
    ImputableAttribute,
    ImputeAction,
    MeasurementType,
    RateMeasurementType,
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
        filter_on: Iterable[str] | None = None,
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
        self.filter_on = filter_on
        self._validate()

    def __eq__(self, x: Self):
        if self.name != x.name:
            return False

        my_attr = set(a.name for a in self.attributes)
        their_attr = set(a.name for a in x.attributes)

        if not my_attr == their_attr:
            return False

        return all(
            self.get_attribute(a) == x.get_attribute(a) for a in my_attr
        )

    def __repr__(self):
        return f"Group(name={self.name}, attributes={[a for a in self.attributes]})"

    def _validate(self):
        assert all([isinstance(a, Attribute) for a in self.attributes]), (
            "All attributes must be of class Attribute"
        )
        measurement_names = [a.name for a in self.attributes]
        assert len(set(measurement_names)) == len(measurement_names), (
            "Found multiple measurements for same attribute."
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

    def disaggregate_one_subgroup(
        self,
        subgroup: Self,
        prop: float,
        size_from: Hashable = "size",
        subgroup_size_from: Hashable = "size",
    ) -> Self:
        assert 0.0 <= prop <= 1.0, (
            f"Cannot disaggregate proportion {prop} of {self}."
        )
        disagg_attributes = list(subgroup.attributes)
        for attr in self.rate_to_count(size_from).attributes:
            if attr.impute_action == "copy":
                disagg_attributes.append(attr)
            elif attr.impute_action == "impute":
                assert isinstance(attr, ImputableAttribute)
                disagg_attributes.append(attr * prop)
        return type(self)(subgroup.name, disagg_attributes).restore_rates(
            subgroup_size_from
        )

    def filter(
        self, df: pl.DataFrame, assert_unique: bool = True
    ) -> pl.DataFrame:
        assert self.filter_on is not None, f"{self} has nothing to filter on."
        assert all(isinstance(fo, str) for fo in self.filter_on), (
            f"{self} has non-str elements in `filter_on`."
        )
        filter_attributes = self.get_attributes(self.filter_on)
        this_grp = df.filter(
            pl.col(attr.name) == attr.value  # pyright: ignore[reportArgumentType]
            for attr in filter_attributes
        ).drop(self.filter_on)
        if assert_unique:
            assert this_grp.shape[0] == 1, (
                f"{df} contains multiple rows for {self}"
            )

        return this_grp

    def _get_attribute(self, name: Hashable) -> Attribute | None:
        """
        Get stated attribute, if it exists.
        """
        name_matched = [a for a in self.attributes if a.name == name]
        if len(name_matched) == 0:
            return None
        assert len(name_matched) == 1, (
            f"Malformed group {self} has multiple attributes {name}"
        )
        return name_matched[0]

    def get_attribute(self, name: Hashable) -> Attribute:
        """
        Retrieve stated attribute or die trying.
        """
        attr = self._get_attribute(name)
        assert attr is not None, f"{self} has no attribute {name}"
        return attr

    def get_attributes(self, names: Iterable[Hashable]) -> Iterable[Attribute]:
        """
        Retrieve stated attribute or die trying.
        """
        return [self.get_attribute(name) for name in names]

    def rate_to_count(self, size_from: Hashable = "size") -> Self:
        """
        Make all measurements masses.
        """

        size = self.get_attribute(size_from).value
        assert size > 0
        attributes = [
            a.to_count(size)
            if a.impute_action == "impute"
            and isinstance(a, ImputableAttribute)
            and a.measurement_type in get_args(RateMeasurementType)
            else a
            for a in self.attributes
        ]
        return type(self)(name=self.name, attributes=attributes)

    def restore_rates(self, size_from: Hashable = "size") -> Self:
        """
        Undo density_to_mass().
        """
        size = self.get_attribute(size_from).value
        assert size > 0
        attributes = [
            a.to_rate(size)
            if a.impute_action == "impute"
            and isinstance(a, ImputableAttribute)
            and a.measurement_type == "count_from_rate"
            else a
            for a in self.attributes
        ]
        return type(self)(name=self.name, attributes=attributes)

    def to_dict(self) -> dict[Hashable, Any]:
        assert self.attributes, (
            f"Cannot call to_dict() on {self} which has no attributes."
        )
        return {attr.name: attr.value for attr in self.attributes}

    def to_polars_dict(self) -> dict[str, Any]:
        as_dict = self.to_dict()
        assert (
            nonstr := set(
                nm for nm in as_dict.keys() if not isinstance(nm, str)
            )
        ) == set(), (
            f"Cannot convert {self} to polars dict, some attribute names are not strings: {nonstr}"
        )

        return as_dict  # pyright: ignore[reportReturnType]


class GroupMap:
    """
    A class that binds supergroups and subgroups together.
    """

    def __init__(
        self,
        sub_to_super: Mapping[Hashable, Hashable],
        groups: Iterable[Group] | None,
    ):
        """
        Default constructor, takes in a subgroup : supergroup dict, and, optionally, groups.

        If no groups are provided, empty groups are created.
        """
        # Should probably store one dict of group name to Group, then sub<>super dicts as dict[str, str]
        self.sub_to_super = sub_to_super
        self.super_to_sub = GroupMap.make_one_to_many(sub_to_super)
        if groups is None:
            group_names = set(sub_to_super.values()).union(sub_to_super.keys())
            groups = [Group(name) for name in group_names]
        self.groups = {group.name: group for group in groups}
        self._validate()

    @classmethod
    def from_supergroups(
        cls,
        super_to_sub: dict[Hashable, Iterable[Hashable]],
        groups: Iterable[Group] | None,
    ) -> Self:
        """
        Alternative constructor, takes in a supergroup : [subgroups] dict.
        """
        sub_to_super = GroupMap.make_many_to_one(super_to_sub)
        return cls(sub_to_super, groups)

    def _validate(self):
        # Groups in mapping are in self.groups
        for group in self.sub_to_super.keys():
            assert group in self.groups, (
                f"Subgroup {group} is present in self.sub_to_super but not in self.groups"
            )
        for group in set(self.sub_to_super.values()):
            assert group in self.groups, (
                f"Supergroup {group} is present in self.sub_to_super but not in self.groups"
            )
        # Groups in self.groups are in mapping
        for group in self.groups.keys():
            in_sub = group in self.sub_to_super.keys()
            in_super = group in self.sub_to_super.values()
            assert in_sub or in_super, (
                f"Group {group} is present in self.groups but not in self.sub_to_super"
            )
            if in_sub and in_super:
                assert Counter(self.sub_to_super.items())[group] == 1, (
                    "Group is both a supergroup and a subgroup but is not 1:1."
                )

    def add_attribute(
        self,
        group_type: GroupType,
        attribute_name: Hashable,
        attribute_values: dict[Hashable, object],
        impute_action: ImputeAction,
        attribute_class: type[Attribute] | type[ImputableAttribute],
        measurement_type: MeasurementType | None = None,
    ):
        """
        Bulk addition of attributes to all sub or supergroups.

        Parameters
        ----------
        group_type : GroupType
            Should the attribute be added to supergroups or subgroups?
        attribute_name : Hashable
            The name of the attribute to be added.
        attribute_values : dict[Hashable, object]
            For all groups of the specified type, the values of the attribute to be added.
        impute_action : ImputeAction
            The impute_action for the attribute to be added.
        attribute_class : type[Attribute] | type[ImputableAttribute]
            The class of the attribute to be added.
        measurement_type : MeasurementType | None
            The measurement type of the attribute to be added, if it is an ImputableAttribute.
        """
        if group_type == "supergroup":
            group_names = self.supergroup_names
        elif group_type == "subgroup":
            group_names = [
                k for k in self.groups if k not in self.supergroup_names
            ]
        else:
            raise ValueError(f"Unknown group_type: {group_type}")
        assert set(group_names).issubset(attribute_values.keys()), (
            f"Cannot add attribute {attribute_name} to groups {set(group_names).difference(attribute_values.keys())} which are not found in `attr_values`. "
        )
        kwargs = {"name": attribute_name, "impute_action": impute_action}
        if attribute_class is ImputableAttribute:
            kwargs |= {"measurement_type": measurement_type}
        for group_name in group_names:
            attr = attribute_class(
                **(kwargs | {"value": attribute_values[group_name]})
            )  # pyright: ignore[reportCallIssue]
            self.groups[group_name] = self.groups[group_name].add_attribute(
                attr
            )

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

    def group(self, name: Hashable) -> Group:
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
        if sub_or_super == "subgroup":
            group_names = []
            for supergrp in self.supergroup_names:
                group_names = group_names + self.subgroup_names(supergrp)
        elif sub_or_super == "supergroup":
            group_names = self.supergroup_names
        else:
            raise RuntimeError(f"Unknown group type {sub_or_super}")

        return pl.from_dicts(
            [self.group(grp_name).to_polars_dict() for grp_name in group_names]
        )

    def data_from_polars(
        self,
        df: pl.DataFrame,
        group_type: GroupType,
        exclude: Container[str],
        count: Container[str],
        copy: Container[str],
        rate: Container[str],
    ):
        """
        Populates measurements and attributes for groups found in the dataframe.
        """
        if group_type == "subgroup":
            raise NotImplementedError()
        elif group_type == "supergroup":
            group_names = self.supergroup_names
        else:
            raise RuntimeError(f"Unknown group type {group_type}")

        cols = [col for col in df.columns if col not in exclude]

        all_grps_all_vals: dict[Hashable, dict[str, Any]] = {
            grp_name: self.group(grp_name)
            .filter(df, assert_unique=True)
            .to_dicts()[0]
            for grp_name in group_names
        }

        for col in cols:
            vals = {
                grp_name: all_grps_all_vals[grp_name][col]
                for grp_name in group_names
            }
            impute_action = "copy" if col in copy else "ignore"
            measurement_type = None
            attribute_class = Attribute
            if col in count or col in rate:
                measurement_type = "count" if col in count else "rate"
                attribute_class = ImputableAttribute
            self.add_attribute(
                group_type=group_type,
                attribute_name=col,
                attribute_values=vals,
                impute_action=impute_action,
                attribute_class=attribute_class,
                measurement_type=measurement_type,
            )

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

    def subgroup_names(self, name: Hashable) -> list[Hashable]:
        """
        Get names of subgroups this supergroup contains
        """
        assert name in self.super_to_sub.keys()
        return self.super_to_sub[name]

    @property
    def supergroup_names(self) -> list[Hashable]:
        """
        Get all supergroup names.
        """
        return list(self.super_to_sub.keys())
