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


class SizeWeights(WeightCalculator):
    def size(self, group: Group):
        size = group.get_measurement("size")
        assert size.type in get_args(MassMeasurementType), (
            "Size must be a mass measurement."
        )
        return size.value

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
        # all_mass = map.density_to_mass("supergroup")
        # imputed_groups = [map.group(name) for name in all_mass.supergroups]
        # for supergroup_name in all_mass.supergroups:
        # weights = self.weight_calculator.calculate(supergroup_name)
        # TODO: use weights to apportion the supergroup measurements to the subgroups
        # TODO: add the new data-inclusive subgroup to the list

        # return GroupMap(
        #     all_mass.sub_to_super, imputed_groups
        # ).restore_densities("subgroup")
