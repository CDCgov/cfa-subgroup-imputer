"""
Submodule for enumerating subgroup and supergroup maps.
"""

import re
from collections.abc import Iterable
from math import inf
from typing import Hashable, Protocol, runtime_checkable

from cfa_subgroup_imputer.groups import Group, GroupMap
from cfa_subgroup_imputer.variables import Attribute, Range


@runtime_checkable
class HashableCombiner(Protocol):
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

    def construct_group_map(
        self, supergroups: Iterable[str], subgroups: Iterable[str], **kwargs
    ) -> GroupMap:
        """
        Takes flat supergroup and subgroup inputs, returns a sub : super
        group map which can be fed to a GroupMap.
        """
        ...


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
                float(x[0]) - 1.0,
                float(x[0]),
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
            re.compile(r"^(\d+)-(\d+) months*"),
            lambda x: (
                float(x[0]) / 12.0,
                (float(x[1]) + 1.0) / 12.0,
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

        return GroupMap(sub_to_super=sub_to_super, groups=groups)

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


class CategoricalSubgroupHandler:
    """
    A class for handling subgroups based on a categorical variable, where all subgroups
    are found in all supergroups.

    For example, if we have age-based supergroups [0-17 years, 18-64 years, 65+ years]
    and want [low, moderate, high]-risk subgroups, this class makes and handles
    creating all "0-17 years, low risk", ..., "65+ years, high risk" subgroups
    and mapping them to the supergroups.
    """

    def __init__(self, hash_combiner: HashableCombiner = StringPaster()):
        self.hash_combiner: HashableCombiner = hash_combiner

    def make_subgroups(
        self,
        supergroups: Iterable[str],
        subgroups: Iterable[str],
        supergroup_varname: str,
        subgroup_varname: str,
    ) -> list[Group]:
        return [
            Group(
                name=self.hash_combiner.combine(supergrp, subgrp),
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
            for subgrp in list(subgroups)
            for supergrp in list(supergroups)
        ]

    def make_supergroups(
        self,
        supergroups: Iterable[str],
        supergroup_varname: str,
    ) -> list[Group]:
        return [
            Group(
                name=grp,
                attributes=[
                    Attribute(
                        value=grp,
                        name=supergroup_varname,
                        impute_action="ignore",
                    )
                ],
            )
            for grp in list(supergroups)
        ]

    def construct_group_map(
        self,
        supergroups: Iterable[str],
        subgroups: Iterable[str],
        supergroup_varname: str,
        subgroup_varname: str,
        **kwargs,
    ) -> GroupMap:
        groups = self.make_supergroups(
            supergroups, supergroup_varname
        ) + self.make_subgroups(
            supergroups, subgroups, supergroup_varname, subgroup_varname
        )

        sub_to_super = {
            self.hash_combiner.combine(
                supergrp,
                subgrp,
            ): supergrp
            for subgrp in subgroups
            for supergrp in supergroups
        }

        return GroupMap(sub_to_super, groups)
