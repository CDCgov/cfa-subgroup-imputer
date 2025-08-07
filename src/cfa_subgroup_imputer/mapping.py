"""
Submodule for enumerating subgroup and supergroup maps.
"""

import itertools
import re
from abc import ABC
from collections.abc import Iterable, Sequence
from math import inf
from typing import Hashable, Protocol, runtime_checkable

from cfa_subgroup_imputer.groups import GroupMap
from cfa_subgroup_imputer.variables import (
    Attribute,
    Range,
    assert_range_spanned_exactly,
)


def assert_hashable_sequence(x) -> None:
    assert isinstance(x, Sequence), f"{x} is not a Sequence"
    assert all(isinstance(x_, Hashable) for x_ in x), (
        f"Not all items in {x} are hashable"
    )


@runtime_checkable
class Mapper(Protocol):
    """
    A class that assists in making sub to supergroup maps for an underlying
    axis defined by a continuous variable, such as age.

    E.g., something that takes you from "my age subgroups are... and my age
    supergroups are..." to a sub : super group name/string dict.
    """

    def construct_group_map(self, **kwargs) -> GroupMap:
        """
        Makes a GroupMap.
        """
        ...


class RaggedOuterProductSubgroupHandler(ABC):
    def construct_group_map(self, **kwargs) -> GroupMap:
        """
        Uses a Sequence of Sequences of Hashables to construct a GroupMap. Each inner Sequence
        defines, in order, the category in each variable that defines a group. The last variable
        is taken to be the one which defines the supergroup.

        E.g [["low risk", "child"], ["high risk", "child"], ["low risk", "adult"],]
        defines two supergroups, "child" and "adult", and three subgroups,
        "low risk child", "high risk child", and "low risk adult".

        If provided, a second Sequence (`"variable_names"`) of the names of the variables is used
        when populating the group attributes.
        """
        assert "category_combinations" in kwargs
        cat_combs = kwargs.get("category_combinations")

        assert isinstance(cat_combs, Sequence)
        lens = []
        for cat_comb in cat_combs:
            assert_hashable_sequence(cat_comb)
            lens.append(len(cat_comb))
        nvar = lens[0]
        assert all(len == nvar for len in lens)

        subgroups = [tuple(cat_comb) for cat_comb in cat_combs]
        supergroups = [cat_comb[-1] for cat_comb in cat_combs]

        group_map = GroupMap(
            sub_to_super={
                subgrp: supergrp
                for subgrp, supergrp in zip(subgroups, supergroups)
            },
            groups=None,
        )

        if "variable_names" in kwargs:
            variable_names = kwargs.get("variable_names")
            assert isinstance(variable_names, Sequence)
            assert len(variable_names) == nvar
            assert all(
                isinstance(var_name, str) for var_name in variable_names
            )
        else:
            variable_names = [f"variable_{i}" for i in range(nvar)]

        group_map.add_attribute(
            group_type="supergroup",
            attribute_name=variable_names[-1],
            attribute_values={supergrp: supergrp for supergrp in supergroups},
            impute_action="ignore",
            attribute_class=Attribute,
            measurement_type=None,
        )

        for varname, cats in zip(variable_names, zip(*cat_combs)):
            group_map.add_attribute(
                group_type="subgroup",
                attribute_name=varname,
                attribute_values={
                    subgrp: cat for subgrp, cat in zip(subgroups, cats)
                },
                impute_action="ignore",
                attribute_class=Attribute,
                measurement_type=None,
            )

        group_map.add_filters("supergroup", [variable_names[-1]])
        group_map.add_filters("subgroup", variable_names)

        return group_map


class OuterProductSubgroupHandler:
    """
    A class for handling subgroups based on a categorical variable, where all categories
    (levels) of the subgrouping variable are found in all supergroups.

    For example, if we have age-based supergroups [0-17 years, 18-64 years, 65+ years]
    and want [low, moderate, high]-risk subgroups, this class makes and handles
    creating all "0-17 years, low risk", ..., "65+ years, high risk" subgroups
    and mapping them to the supergroups.
    """

    def construct_group_map(
        self,
        **kwargs,
    ) -> GroupMap:
        """
        Constructs a GroupMap from all subgroups defined by the categories of subgroup
        and supergroup variables.

        Parameters
        ----------
        supergroup_categories: Sequence[Hashable]
            The catgegories of the variable defining the supergroups.
        subgroup_categories: Sequence[Sequence[Hashable]]
            For each variable defining subgroups, the catgegories it can take.
        supergroup_variable_name: str
            What is the variable that defines the supergroup?
        subgroup_variable_names: Sequence[str]
            What are the variables that defines the subgroups?
        """
        assert "supergroup_categories" in kwargs
        super_cats: Sequence[Hashable] = kwargs.get("supergroup_categories")  # type: ignore
        assert_hashable_sequence(super_cats)

        assert "subgroup_categories" in kwargs
        sub_cats: Sequence[Sequence[Hashable]] = kwargs.get(
            "subgroup_categories"
        )  # type: ignore
        assert isinstance(sub_cats, Sequence)
        for sc in sub_cats:
            assert_hashable_sequence(sc)

        sub_super = itertools.product(*[*sub_cats, list(super_cats)])

        supergroup_varname = kwargs.get(
            "supergroup_variable_name", "supergroup_variable"
        )
        if "subgroup_variable_names" in kwargs:
            subgroup_variable_names = kwargs.get("subgroup_variable_names")
            assert isinstance(subgroup_variable_names, Sequence)
            assert len(subgroup_variable_names) == len(sub_cats)
            assert all(
                isinstance(var_name, str)
                for var_name in subgroup_variable_names
            )
        else:
            subgroup_variable_names = [
                f"subgroup_variable_{i}" for i in range(len(sub_cats))
            ]

        return RaggedOuterProductSubgroupHandler().construct_group_map(
            category_combinations=list(sub_super),
            variable_names=list(subgroup_variable_names)
            + [supergroup_varname],
        )


class AgeGroupHandler:
    """
    A class for working with age groups.

    Implements:
        cfa_subgroup_imputer.enumerator.Mapper
    """

    STR_AGE_RANGE_CONVERTERS = (
        (
            re.compile(r"^(\d+) years*"),
            lambda x: (
                float(x[0]),
                float(x[0]) + 1.0,
            ),
        ),
        (
            re.compile(r"^(\d+)\+ years"),
            lambda x: (
                float(x[0]),
                inf,
            ),
        ),
        (
            re.compile(r"^(\d+)-(\d+) years"),
            lambda x: (
                float(x[0]),
                float(x[1]) + 1.0,
            ),
        ),
        (
            re.compile(r"^(\d+)-<(\d+) years*"),
            lambda x: (
                float(x[0]),
                float(x[1]),
            ),
        ),
        (
            re.compile(r"^(\d+) months*-(\d+) years*"),
            lambda x: (
                float(x[0]) / 12.0,
                float(x[1]) + 1.0,
            ),
        ),
        (
            re.compile(r"^(\d+) months*-<(\d+) years*"),
            lambda x: (
                float(x[0]) / 12.0,
                float(x[1]),
            ),
        ),
        (
            re.compile(r"^(\d+)-(\d+) months*"),
            lambda x: (
                float(x[0]) / 12.0,
                (float(x[1]) + 1.0) / 12.0,
            ),
        ),
        (
            re.compile(r"^(\d+)-<(\d+) months*"),
            lambda x: (
                float(x[0]) / 12.0,
                float(x[1]) / 12.0,
            ),
        ),
    )
    """
    The master list of age ranges we can convert.

    Each element is a tuple of
    - A regex which can extract the single age or the low/high ages and
    - A Callable which returns a (low, high,) tuple of floats for ages in years
    """

    def __init__(self, age_max: float | None = None):
        self.age_max = age_max if age_max is not None else inf

    def age_range_from_str(self, x: str) -> Range:
        """
        Takes string defining age group and returns a (low, high,) float defining the range in years.
        """
        for sarc in AgeGroupHandler.STR_AGE_RANGE_CONVERTERS:
            if ages := sarc[0].fullmatch(x):
                low, high = sarc[1](ages.groups())
                if high == inf:
                    high = self.age_max
                return Range(low, high)
        raise RuntimeError(f"Cannot process age range {x}")

    def age_ranges_equivalent(self, x: str, y: str) -> bool:
        """
        True if the age ranges encode the same values, else False.

        E.g., 1-3 years and 1-<4 years imply age group of 1, 2, and 3 year olds.
        """
        return self.age_range_from_str(x) == self.age_range_from_str(y)

    def construct_group_map(
        self,
        supergroups: Iterable[str],
        subgroups: Iterable[str],
        **kwargs,
    ) -> GroupMap:
        age_varname = kwargs.get("continuous_var_name", "age")
        missing_option = kwargs.get("missing_option", "error")

        # Brute force attribution
        super_dict = {grp: self.age_range_from_str(grp) for grp in supergroups}
        sub_to_super = {}
        for sub in subgroups:
            sub_range = self.age_range_from_str(sub)
            super = [
                super_name
                for super_name, super_range in super_dict.items()
                if sub_range in super_range
            ]
            if len(super) == 1:
                sub_to_super[sub] = super[0]
            elif len(super) == 0:
                if missing_option == "add_one_to_one":
                    super_dict[sub] = sub_range
                else:
                    raise RuntimeError(
                        f"Subgroup {sub} has no corresponding supergroup in {supergroups}"
                    )
            else:
                raise RuntimeError(
                    f"Subgroup {sub} is contained by multiple supergroups: {super}"
                )

        grp_map = GroupMap(sub_to_super=sub_to_super, groups=None)
        grp_map.add_attribute(
            group_type="subgroup",
            attribute_name=age_varname,
            attribute_values={
                subgrp: self.age_range_from_str(subgrp) for subgrp in subgroups
            },
            attribute_filter_values={subgrp: subgrp for subgrp in subgroups},
            impute_action="ignore",
            attribute_class=Attribute,
        )
        grp_map.add_attribute(
            group_type="supergroup",
            attribute_name=age_varname,
            attribute_values={
                supergrp: self.age_range_from_str(supergrp)
                for supergrp in supergroups
            },
            attribute_filter_values={
                supergrp: supergrp for supergrp in supergroups
            },
            impute_action="ignore",
            attribute_class=Attribute,
        )
        self.assert_no_missing_subgroups(grp_map, age_varname)
        sorted_super_ranges = sorted(super_dict.values())
        assert_range_spanned_exactly(
            Range(sorted_super_ranges[0].lower, sorted_super_ranges[-1].upper),
            sorted_super_ranges,
        )

        grp_map.add_filters("supergroup", [age_varname])
        grp_map.add_filters("subgroup", [age_varname])

        return grp_map

    def is_valid_age_group(self, x: str) -> bool:
        try:
            _ = self.age_range_from_str(x)
            return True
        except RuntimeError as e:
            if re.fullmatch("Cannot process age range", str(e)):
                return False
            else:
                raise e

    def assert_no_missing_subgroups(self, group_map: GroupMap, age_varname):
        for supergrp_nm in group_map.supergroup_names:
            supergrp = group_map.group(supergrp_nm)
            supergrp_range = supergrp.get_attribute(age_varname).value
            subgrp_ranges = [
                group_map.group(nm).get_attribute(age_varname).value
                for nm in group_map.subgroup_names(supergrp_nm)
            ]
            assert_range_spanned_exactly(supergrp_range, subgrp_ranges)
