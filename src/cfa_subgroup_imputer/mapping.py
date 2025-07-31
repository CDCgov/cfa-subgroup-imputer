"""
Submodule for enumerating subgroup and supergroup maps.
"""

import re
from collections.abc import Iterable
from math import inf
from typing import Hashable, Protocol, runtime_checkable

from cfa_subgroup_imputer.groups import Group, GroupMap
from cfa_subgroup_imputer.variables import (
    Attribute,
    Range,
    assert_range_spanned_exactly,
)


@runtime_checkable
class IdCombiner(Protocol):
    def combine(self, *args) -> Hashable: ...


class StringPaster:
    def combine(self, *args) -> Hashable:
        return "_".join(str(arg) for arg in args)


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


class ArbitraryGroupHandler:
    """
    A class for working with groups given a sub : super group dict
    """

    def __init__(self, id_combiner: IdCombiner):
        self.id_combiner: IdCombiner = id_combiner

    def make_subgroups(
        self,
        sub_super: Iterable[tuple[Hashable, Hashable]],
        supergroup_varname: str,
        subgroup_varname: str,
    ) -> list[Group]:
        return [
            Group(
                name=self.id_combiner.combine(supergrp, subgrp),
                attributes=[
                    Attribute(
                        value=supergrp,
                        name=supergroup_varname,
                        impute_action="ignore",
                    ),
                    Attribute(
                        value=subgrp,
                        name=subgroup_varname,
                        impute_action="ignore",
                    ),
                ],
            )
            for subgrp, supergrp in sub_super
        ]

    def make_supergroups(
        self,
        sub_super: Iterable[tuple[Hashable, Hashable]],
        supergroup_varname: str,
    ) -> list[Group]:
        supergroups = set(sub_sup[1] for sub_sup in sub_super)
        return [
            Group(
                name=supergrp,
                attributes=[
                    Attribute(
                        value=supergrp,
                        name=supergroup_varname,
                        impute_action="ignore",
                    )
                ],
            )
            for supergrp in supergroups
        ]

    def construct_group_map(
        self,
        **kwargs,
    ) -> GroupMap:
        assert "sub_super" in kwargs and isinstance(
            kwargs.get("sub_super"), Iterable
        )
        assert "supergroup_varname" in kwargs and isinstance(
            kwargs.get("supergroup_varname"), str
        )
        assert "subgroup_varname" in kwargs and isinstance(
            kwargs.get("subgroup_varname"), str
        )

        sub_super: Iterable[tuple[Hashable, Hashable]] = kwargs.get(
            "sub_super"
        )  # type: ignore
        supergroup_varname: str = kwargs.get("supergroup_varname")  # type: ignore
        subgroup_varname: str = kwargs.get("subgroup_varname")  # type: ignore

        groups = self.make_supergroups(
            sub_super, supergroup_varname
        ) + self.make_subgroups(
            sub_super, supergroup_varname, subgroup_varname
        )

        sub_to_super = {
            self.id_combiner.combine(supergrp, subgrp): supergrp
            for subgrp, supergrp in sub_super
        }

        return GroupMap(sub_to_super, groups)


class OuterProductSubgroupHandler:
    """
    A class for handling subgroups based on a categorical variable, where all subgroups
    are found in all supergroups.

    For example, if we have age-based supergroups [0-17 years, 18-64 years, 65+ years]
    and want [low, moderate, high]-risk subgroups, this class makes and handles
    creating all "0-17 years, low risk", ..., "65+ years, high risk" subgroups
    and mapping them to the supergroups.
    """

    def construct_group_map(
        self,
        **kwargs,
    ) -> GroupMap:
        assert "subgroups" in kwargs and isinstance(
            kwargs.get("subgroups"), Iterable
        )
        assert "supergroups" in kwargs and isinstance(
            kwargs.get("supergroups"), Iterable
        )

        supergroups: Iterable[Hashable] = kwargs.get("supergroups")  # type: ignore
        subgroups: Iterable[Hashable] = kwargs.get("subgroups")  # type: ignore
        assert all(isinstance(sg, Hashable) for sg in supergroups)
        assert all(isinstance(sg, Hashable) for sg in subgroups)

        id_combiner = kwargs.get("id_combiner", StringPaster())
        pairs = [
            (
                subgrp,
                supergrp,
            )
            for subgrp in subgroups
            for supergrp in supergroups
        ]

        return ArbitraryGroupHandler(
            id_combiner=id_combiner
        ).construct_group_map(sub_super=pairs, **kwargs)


# @TODO: Needs more static methods
class AgeGroupHandler:
    """
    A class for working with age groups.

    Implements:
        cfa_subgroup_imputer.enumerator.Mapper
        cfa_subgroup_imputer.polars.FilterConstructor
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
        age_varname = kwargs.get("variable_name", "age")
        missing_option = kwargs.get("missing_option", "error")

        groups = self.make_groups(supergroups, subgroups, age_varname)

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

        grp_map = GroupMap(sub_to_super=sub_to_super, groups=groups)
        self.assert_no_missing_subgroups(grp_map, age_varname)
        sorted_super_ranges = sorted(super_dict.values())
        assert_range_spanned_exactly(
            Range(sorted_super_ranges[0].lower, sorted_super_ranges[-1].upper),
            sorted_super_ranges,
        )

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

    def make_groups(
        self,
        supergroups: Iterable[str],
        subgroups: Iterable[str],
        age_varname: str,
    ) -> list[Group]:
        return [
            Group(
                name=grp,
                attributes=[
                    Attribute(
                        value=self.age_range_from_str(grp),
                        name=age_varname,
                        impute_action="ignore",
                    )
                ],
            )
            for grp in list(supergroups) + list(subgroups)
        ]

    def assert_no_missing_subgroups(self, group_map: GroupMap, age_varname):
        for supergrp_nm in group_map.supergroup_names:
            supergrp = group_map.group(supergrp_nm)
            supergrp_range = supergrp.get_attribute(age_varname).value
            subgrp_ranges = [
                group_map.group(nm).get_attribute(age_varname).value
                for nm in group_map.subgroup_names(supergrp_nm)
            ]
            assert_range_spanned_exactly(supergrp_range, subgrp_ranges)
