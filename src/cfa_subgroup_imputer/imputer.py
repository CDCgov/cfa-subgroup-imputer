"""
Module for imputation machinery.
"""

from typing import Literal

from cfa_subgroup_imputer.groups import (
    Group,
    GroupMap,
)

DisaggregationMethod = Literal["uniform"]


class Disaggregator:
    """
    A class which imputes and disaggregates subgroups.

    For a GroupMap map and choice of method "uniform", usage is
    Disaggregator(method="uniform")(map).
    """

    def __init__(self, method: DisaggregationMethod):
        if method == "uniform":
            self.weight_calculator = Disaggregator.uniform_weight_calculator
        elif method == "cubic_spline":
            self.method = Disaggregator.cubic_spline_weight_calculator
        else:
            raise RuntimeError(f"Unknown method {method}")

    def __call__(self, map: GroupMap) -> GroupMap:
        """
        Impute and disaggregate the given group map.
        """
        assert map.disaggregatable
        all_mass = map.density_to_mass("supergroup")
        weighted = self.weight_calculator(all_mass)
        imputed_groups = []
        for supergroup_name in weighted.supergroups:
            imputed_groups.extend(
                [
                    Disaggregator.apportion_to_subgroup(
                        weighted, subgroup_name, supergroup_name
                    )
                    for subgroup_name in weighted.subgroups(supergroup_name)
                ]
            )
        return GroupMap(weighted.sub_to_super, imputed_groups).mass_to_density(
            "subgroup", map.densities
        )

    @staticmethod
    def apportion_to_subgroup(
        map: GroupMap, supergroup_name, subgroup_name: str
    ) -> Group:
        supergroup = map.group(supergroup_name)
        subgroup = map.group(subgroup_name)
        assert supergroup.measurements is not None
        assert subgroup.weight is not None
        apportioned = [m * subgroup.weight for m in supergroup.measurements]
        return subgroup.alter_measurements(apportioned)

    @staticmethod
    def uniform_weight_calculator(
        map: GroupMap,
    ) -> GroupMap:
        return map.calculate_normalized_weights()
