"""
Module for imputation machinery.
"""

from collections.abc import Iterable
from math import isclose
from typing import Hashable, Protocol, get_args

from cfa_subgroup_imputer.groups import (
    Group,
    GroupMap,
)
from cfa_subgroup_imputer.variables import (
    CountMeasurementType,
    ImputableAttribute,
    Range,
    assert_range_spanned_exactly,
)


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

    def __init__(
        self,
        proportion_calculator: ProportionCalculator,
        size_from: Hashable = "size",
    ):
        self.proportion_calculator = proportion_calculator
        self.size_from = size_from

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
                        map.group(grp_name),
                        props[grp_name],
                        size_from=self.size_from,
                        subgroup_size_from=self.size_from,
                    )
                )

        return GroupMap(sub_to_super, groups)


class Aggregator:
    """
    A class which aggregates subgroups.
    """

    def __init__(self, size_from: Hashable = "size"):
        self.size_from = size_from

    def __call__(self, map: GroupMap) -> GroupMap:
        """
        Impute and aggregate the given group map.
        """

        sub_to_super = map.sub_to_super
        groups = []

        for supergroup_name in map.supergroup_names:
            supergroup = map.group(supergroup_name)
            subgroups = [
                map.group(nm).rate_to_count(self.size_from)
                for nm in map.subgroup_names(supergroup_name)
            ]
            attribute_names = [a.name for a in subgroups[0].attributes]

            for nm in attribute_names:
                supergroup = self._aggregate_one_attribute(
                    nm, supergroup, subgroups
                )
            groups.append(supergroup.restore_rates(self.size_from))
            for nm in map.subgroup_names(supergroup_name):
                groups.append(map.group(nm))

        return GroupMap(sub_to_super, groups)

    def _aggregate_one_attribute(
        self,
        attribute_name: Hashable,
        supergroup: Group,
        subgroups: Iterable[Group],
    ) -> Group:
        """
        Aggregate a single attribute from subgroups to supergroup.
        """
        subgroups = list(subgroups)
        assert len(subgroups) > 0, "Cannot aggregate non-existent subgroups."
        attr0 = subgroups[0].get_attribute(attribute_name)
        act0 = attr0.impute_action

        if act0 == "copy":
            vals = set(
                [grp.get_attribute(attribute_name) for grp in subgroups]
            )
            assert len(vals) == 1, (
                f"Found multiple incompatible values for attribute named {attribute_name} in subgroups: {vals}"
            )
            return supergroup.add_attribute(attr0)
        elif act0 == "impute":
            cmt = get_args(CountMeasurementType)
            assert isinstance(attr0, ImputableAttribute)
            assert attr0.measurement_type in cmt, (
                "All subgroups must have been pre-processed with `.rate_to_count()`"
            )
            final_type = attr0.measurement_type
            val = attr0.value
            for grp in subgroups[1:]:
                attr = grp.get_attribute(attribute_name)
                assert isinstance(attr, ImputableAttribute)
                assert attr.measurement_type in cmt, (
                    "All subgroups must have been pre-processed with `.rate_to_count()`"
                )
                if attr.measurement_type == "count_from_rate":
                    final_type = "count_from_rate"
                val += attr.value

            return supergroup.add_attribute(
                ImputableAttribute(
                    value=val,
                    name=attribute_name,
                    impute_action="impute",
                    measurement_type=final_type,
                )
            )
        elif act0 == "ignore":
            return supergroup
        else:
            raise RuntimeError(f"{act0} is not a valid ImputeAction.")
