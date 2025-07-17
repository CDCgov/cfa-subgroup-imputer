"""
Module for imputation machinery.
"""

from typing import Protocol, get_args

from cfa_subgroup_imputer.groups import (
    Group,
    GroupMap,
)
from cfa_subgroup_imputer.variables import MassMeasurementType


class WeightCalculator(Protocol):
    def calculate(self, supergroup_name: str) -> dict[str, float]: ...


class CategoricalWeights(WeightCalculator):
    def __init__(self, weight_from: str):
        self.varname = weight_from

    def weight(self, group: Group):
        weight = group.get_measurement(self.varname)
        assert weight.type in get_args(MassMeasurementType), (
            "Weight must derive from a mass measurement."
        )
        return weight.value

    def calculate(self, supergroup_name: str) -> dict[str, float]:
        raise NotImplementedError()


class ContinuousWeights(WeightCalculator):
    def calculate(self, supergroup_name: str) -> dict[str, float]:
        raise NotImplementedError()


class Disaggregator:
    """
    A class which imputes and disaggregates subgroups.
    """

    def __init__(self, weight_calculator: WeightCalculator):
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
