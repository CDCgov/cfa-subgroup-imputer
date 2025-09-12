"""
Submodule for broad-sense handling of supergroups and subgroups.
"""

from collections import Counter
from collections.abc import Container, Iterable, Mapping
from typing import Any, Hashable, Literal, Self, get_args

from cfa_subgroup_imputer.utils import get_json_keys
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
        supergroup_filter_on: Iterable[str] | None = None,
        subgroup_filter_on: Iterable[str] | None = None,
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
        self.supergroup_filter_on = supergroup_filter_on
        self.subgroup_filter_on = subgroup_filter_on
        self._validate()

    def __eq__(self, x: Self) -> bool:
        if not self.equals_ignore_filters(x):
            return False

        if self.subgroup_filter_on != x.subgroup_filter_on:
            return False

        return self.supergroup_filter_on == x.supergroup_filter_on

    def __repr__(self):
        return f"Group(name={self.name}, attributes={self.attributes}, supergroup_filter_on={self.supergroup_filter_on}, subgroup_filter_on={self.subgroup_filter_on})"

    def add_attribute(self, attribute: Attribute) -> Self:
        """
        Give this group a new measurement.
        """
        assert attribute.name not in [a.name for a in self.attributes], (
            f"Cannot add measurement {attribute} to group {self.name} which already has {self.get_attribute(attribute.name)}"
        )
        return self._copy_modify(
            **{"attributes": self.attributes + (attribute,)}
        )

    def add_filters(
        self,
        filters: Iterable[str],
        group_type: GroupType,
        overwrite: bool = False,
    ) -> Self:
        """
        Add keys that this group can use to filter an Iterable[dict] to find its row.
        """

        if ((f := self._get_filters(group_type)) is not None) and (
            not overwrite
        ):
            raise RuntimeError(f"{self} already has filters: {f}")

        if group_type == "subgroup":
            key = "subgroup_filter_on"
        elif group_type == "supergroup":
            key = "supergroup_filter_on"
        else:
            raise RuntimeError(f"Invalid GroupType {group_type}.")

        return self._copy_modify(**{key: filters})

    def _copy_modify(self, **kwargs) -> Self:
        copy_kwargs = {
            "name": self.name,
            "attributes": self.attributes,
            "supergroup_filter_on": self.supergroup_filter_on,
            "subgroup_filter_on": self.subgroup_filter_on,
        } | kwargs

        return type(self)(**copy_kwargs)

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
        return subgroup._copy_modify(
            **{"attributes": disagg_attributes}
        ).restore_rates(subgroup_size_from)

    def equals_ignore_filters(self, x) -> bool:
        """
        Aside from filters, are these two Groups the same?
        """
        if self.name != x.name:
            return False

        my_attr = set(a.name for a in self.attributes)
        their_attr = set(a.name for a in x.attributes)

        if not my_attr == their_attr:
            return False

        return all(
            self.get_attribute(a) == x.get_attribute(a) for a in my_attr
        )

    def filter(
        self,
        data: Iterable[dict[str, Any]],
        group_type: GroupType,
        assert_unique: bool = True,
    ) -> list[dict]:
        filters = self.get_filters(group_type)
        assert all(isinstance(fo, str) for fo in filters), (
            f"{self} has non-str elements in `filter_on`."
        )

        filtered_data = list(
            filter(
                lambda row: all(
                    row[filter_key]
                    == self.get_attribute(filter_key).json_value
                    for filter_key in filters
                ),
                data,
            )
        )

        if assert_unique:
            assert len(filtered_data) == 1, (
                f"{data} contains multiple rows for {self}"
            )

        return filtered_data

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

    def _get_filters(
        self,
        group_type: GroupType,
    ) -> Iterable[str] | None:
        """
        Get keys that this group can use to filter an Iterable[dict] to find its row, if there are any.
        """

        if group_type == "subgroup":
            return self.subgroup_filter_on
        elif group_type == "supergroup":
            return self.supergroup_filter_on
        else:
            raise RuntimeError(f"Invalid GroupType {group_type}.")

    def get_filters(
        self,
        group_type: GroupType,
    ) -> Iterable[str]:
        """
        Get keys that this group can use to filter an Iterable[dict] to find its row or die trying.
        """

        filters = self._get_filters(group_type)
        assert filters is not None, f"{self} has no filters."
        return filters

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
        return self._copy_modify(**{"attributes": attributes})

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
        return self._copy_modify(**{"attributes": attributes})

    def to_dict(self, use_json_values=False) -> dict[Hashable, Any]:
        assert self.attributes, (
            f"Cannot call to_dict() on {self} which has no attributes."
        )
        if use_json_values:
            return {attr.name: attr.json_value for attr in self.attributes}
        else:
            return {attr.name: attr.value for attr in self.attributes}

    def to_json_dict(self) -> dict[str, Any]:
        for attr in self.attributes:
            attr._assert_jsonable()

        as_dict = self.to_dict(use_json_values=True)

        return as_dict  # pyright: ignore[reportReturnType]

    def _validate(self):
        assert all([isinstance(a, Attribute) for a in self.attributes]), (
            "All attributes must be of class Attribute"
        )
        measurement_names = [a.name for a in self.attributes]
        assert len(set(measurement_names)) == len(measurement_names), (
            f"Found multiple measurements for same attribute when constructing group named {self.name}: {measurement_names}"
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
                assert Counter(self.sub_to_super.values())[group] == 1, (
                    "Group is both a supergroup and a subgroup but is not 1:1."
                )

    def add_attribute(
        self,
        group_type: GroupType,
        attribute_name: Hashable,
        attribute_values: dict[Hashable, Any],
        impute_action: ImputeAction,
        attribute_class: type[Attribute] | type[ImputableAttribute],
        measurement_type: MeasurementType | None = None,
        attribute_json_values: dict[Hashable, Any] | None = None,
    ) -> None:
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
        attribute_json_values : dict[Hashable, object] | None
            If the `attribute_values` are not something recorded directly in the json,
            this specifies how the values will be compared against json
            values and how they will be exported to json. None means to use the `attribute_values`.
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
            f"Cannot add attribute {attribute_name} to groups {set(group_names).difference(attribute_values.keys())} which are not found in `attr_values`."
        )
        if attribute_json_values is not None:
            assert set(attribute_json_values.keys()).issubset(
                attribute_values.keys()
            ), (
                "If providing distinct filtering values from values, must provide one per group in `attribute_values`."
            )
        kwargs = {"name": attribute_name, "impute_action": impute_action}
        if attribute_class is ImputableAttribute:
            kwargs |= {"measurement_type": measurement_type}
        for group_name in group_names:
            attr = attribute_class(
                **(
                    kwargs
                    | {
                        "value": attribute_values[group_name],
                        "json_value": None
                        if attribute_json_values is None
                        else attribute_json_values[group_name],
                    }
                )
            )  # pyright: ignore[reportCallIssue]
            self.groups[group_name] = self.groups[group_name].add_attribute(
                attr
            )

    def add_filters(self, group_type: GroupType, filters: Iterable[str]):
        if group_type == "subgroup":
            group_names = self.subgroup_names()
        elif group_type == "supergroup":
            group_names = self.supergroup_names
        else:
            raise RuntimeError(f"Unknown group type {group_type}")
        for grp_name in group_names:
            self.groups[grp_name] = self.group(grp_name).add_filters(
                filters, group_type
            )

    def group(self, name: Hashable) -> Group:
        return self.groups[name]

    def to_dicts(self, group_type: GroupType) -> list[dict]:
        """
        Creates a list of dicts of the measurements in either the supergroups or subgroups.
        """
        if group_type == "subgroup":
            group_names = self.subgroup_names()
        elif group_type == "supergroup":
            group_names = self.supergroup_names
        else:
            raise RuntimeError(f"Unknown group type {group_type}")

        return [
            self.group(grp_name).to_json_dict() for grp_name in group_names
        ]

    def data_from_dicts(
        self,
        data: Iterable[dict],
        group_type: GroupType,
        exclude: Container[str],
        count: Container[str],
        copy: Container[str],
        rate: Container[str],
    ):
        """
        Populates measurements and attributes for groups found in the data.
        """
        if group_type == "subgroup":
            group_names = self.subgroup_names()
        elif group_type == "supergroup":
            group_names = self.supergroup_names
        else:
            raise RuntimeError(f"Unknown group type {group_type}")

        filters = self.get_filters(group_type)
        assert filters is not None

        data_list = list(data)

        keys = [
            key
            for key in get_json_keys(data)
            if ((key not in exclude) and (key not in filters))
        ]

        # We can do better than O(n^2)
        all_grps_all_vals: dict[Hashable, dict[str, Any]] = {
            grp_name: self.group(grp_name).filter(
                data_list, group_type, assert_unique=True
            )[0]
            for grp_name in group_names
        }

        for key in keys:
            vals = {
                grp_name: all_grps_all_vals[grp_name][key]
                for grp_name in group_names
            }
            impute_action = "copy" if key in copy else "ignore"
            measurement_type = None
            attribute_class = Attribute
            if key in count or key in rate:
                impute_action = "impute"
                measurement_type = "count" if key in count else "rate"
                attribute_class = ImputableAttribute
            self.add_attribute(
                group_type=group_type,
                attribute_name=key,
                attribute_values=vals,
                impute_action=impute_action,
                attribute_class=attribute_class,
                measurement_type=measurement_type,
            )

    def get_filters(self, group_type: GroupType) -> Iterable[str]:
        if group_type == "subgroup":
            group_names = self.subgroup_names()
        elif group_type == "supergroup":
            group_names = self.supergroup_names
        else:
            raise RuntimeError(f"Unknown group type {group_type}")

        all_filters = []
        for grp_name in group_names:
            grp_filters = self.group(grp_name).get_filters(group_type)
            assert grp_filters is not None, (
                f"Group named {grp_name} has no filter"
            )
            all_filters.append(tuple(grp_filters))

        assert len(set(all_filters)) == 1, (
            f"Not all {group_type}s have same filters."
        )

        return all_filters.pop()

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

    def subgroup_names(self, name: Hashable | None = None) -> list[Hashable]:
        """
        Get names of subgroups this supergroup contains
        """
        if name is None:
            group_names = []
            for supergrp in self.supergroup_names:
                group_names = group_names + self.subgroup_names(supergrp)

            return group_names

        assert name in self.super_to_sub.keys()
        return self.super_to_sub[name]

    @property
    def supergroup_names(self) -> list[Hashable]:
        """
        Get all supergroup names.
        """
        return list(self.super_to_sub.keys())
