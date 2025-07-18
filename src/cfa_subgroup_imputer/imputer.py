"""
Module for imputation machinery.
"""

from typing import Protocol

from cfa_subgroup_imputer.groups import (
    Group,
    GroupMap,
)


class ProportionCalculator(Protocol):
    def calculate(self, supergroup_name: str) -> dict[str, float]: ...


class ProportionsFromCategories:
    def __init__(self, proportions_from: str):
        self.varname = proportions_from

    def relative_proportion(self, group: Group):
        rel_prop = group.get_attribute(self.varname)
        assert rel_prop.value >= 0.0
        return rel_prop.value

    def calculate(self, supergroup_name: str) -> dict[str, float]:
        raise NotImplementedError()


class ProportionsFromContinuous:
    def calculate(self, supergroup_name: str) -> dict[str, float]:
        raise NotImplementedError()


class Disaggregator:
    """
    A class which imputes and disaggregates subgroups.
    """

    def __init__(self, weight_calculator: ProportionCalculator):
        self.weight_calculator = weight_calculator

    def __call__(self, map: GroupMap) -> GroupMap:
        """
        Impute and disaggregate the given group map.
        """
        assert map.disaggregatable
        raise NotImplementedError()
        # all_mass = map.density_to_mass("supergroup")
        # imputed_groups = [map.group(name) for name in all_mass.supergroups]
        # for supergroup_name in all_mass.supergroups:
        # weights = self.weight_calculator.calculate(supergroup_name)
        # TODO: use weights to apportion the supergroup measurements to the subgroups
        # TODO: add the new data-inclusive subgroup to the list

        # return GroupMap(
        #     all_mass.sub_to_super, imputed_groups
        # ).restore_densities("subgroup")
