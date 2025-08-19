"""
Module for imputation machinery.
"""

from math import isclose
from typing import Hashable, Protocol

from cfa_subgroup_imputer.groups import (
    Group,
    GroupMap,
)
from cfa_subgroup_imputer.variables import Range, assert_range_spanned_exactly


class ProportionCalculator(Protocol):
    def calculate(
        self, supergroup_name: Hashable, group_map: GroupMap, **kwargs
    ) -> dict[Hashable, float]: ...


class ProportionsFromCategories:
    def __init__(self, size_from: Hashable):
        self.size_from = size_from

    def relative_proportion(self, group: Group):
        rel_prop = group.get_attribute(self.size_from)
        assert rel_prop.value >= 0.0
        return rel_prop.value

    def calculate(
        self, supergroup_name: Hashable, group_map: GroupMap, **kwargs
    ) -> dict[Hashable, float]:
        wt = {
            grp: self.relative_proportion(group_map.group(grp))
            for grp in group_map.subgroup_names(supergroup_name)
        }
        wt_sum = sum(wt.values())

        supergroup_size = (
            group_map.group(supergroup_name)
            .get_attribute(self.size_from)
            .value
        )
        normalize = kwargs.get("normalize", False)
        rel_tol = kwargs.get("rel_tol", 1e-6)

        if (not isclose(wt_sum, supergroup_size, rel_tol=rel_tol)) and (
            not normalize
        ):
            raise RuntimeError(
                f"Subgroup sizes sum to {wt_sum} while supergroup size is {supergroup_size}"
            )

        wt = {k: v / wt_sum for k, v in wt.items()}
        return wt


class ProportionsFromContinuous:
    def __init__(self, continuous_var_name: str):
        self.var_name = continuous_var_name

    def calculate(
        self, supergroup_name: Hashable, group_map: GroupMap, **kwargs
    ) -> dict[Hashable, float]:
        ranges: dict[Hashable, Range] = {
            grp: group_map.group(grp).get_attribute(self.var_name).value
            for grp in group_map.subgroup_names(supergroup_name)
        }
        supergroup_range = (
            group_map.group(supergroup_name).get_attribute(self.var_name).value
        )
        assert all(
            isinstance(r, Range) for r in ranges.values()
        ) and isinstance(supergroup_range, Range), (
            "Cannot disaggregate continuous variables unless the attribute is a `Range` object."
        )
        assert_range_spanned_exactly(supergroup_range, ranges.values())

        wt = {k: v.duration() for k, v in ranges.items()}
        wt_sum = sum(wt.values())

        wt = {k: v / wt_sum for k, v in wt.items()}
        return wt


class Disaggregator:
    """
    A class which imputes and disaggregates subgroups.
    """

    def __init__(self, proportion_calculator: ProportionCalculator):
        self.proportion_calculator = proportion_calculator

    def __call__(self, map: GroupMap) -> GroupMap:
        """
        Impute and disaggregate the given group map.
        """

        sub_to_super = map.sub_to_super
        groups = []

        for supergroup_name in map.supergroup_names:
            supergroup = map.group(supergroup_name)
            groups.append(supergroup)
            props = self.proportion_calculator.calculate(supergroup_name, map)
            for grp_name in map.subgroup_names(supergroup_name):
                groups.append(
                    supergroup.disaggregate_one_subgroup(
                        map.group(grp_name), props[grp_name]
                    )
                )

        return GroupMap(sub_to_super, groups)
