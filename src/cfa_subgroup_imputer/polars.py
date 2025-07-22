"""
Module for polars interface.
"""

from collections.abc import Collection

import polars as pl

from cfa_subgroup_imputer.enumerator import CartesianEnumerator
from cfa_subgroup_imputer.groups import Group, GroupMap
from cfa_subgroup_imputer.imputer import (
    Disaggregator,
    ProportionsFromCategories,
    ProportionsFromContinuous,
)
from cfa_subgroup_imputer.variables import (
    GroupableTypes,
    GroupingVariable,
)


def attribute_subgroups(
    supergroup_df: pl.DataFrame,
    subgroup_df: pl.DataFrame,
    group_vartype: GroupingVariable,
    **kwargs,
) -> dict[str, str]:
    supergroup_col = kwargs.get("supergroup_col", "group")
    subgroup_col = kwargs.get("subgroup_col", "group")
    supergroups = supergroup_df[supergroup_col].unique().to_list()
    subgroups = subgroup_df[subgroup_col].unique().to_list()
    if group_vartype.type == "Categorical":
        return CartesianEnumerator().attribute(
            supergroups=supergroups, subgroups=subgroups
        )
    elif group_vartype.type == "Continuous":
        # TODO: need to find some way to match something the user can provide here
        #       to a column in the dataset and a particular Attributor, namely,
        #       for now, the AgeGroupAttributor
        raise NotImplementedError()
    else:
        raise RuntimeError(f"Unknown grouping variable type {group_vartype}")


def disaggregate(
    supergroup_df: pl.DataFrame,
    subgroup_df: pl.DataFrame,
    subgroup_to_supergroup: pl.DataFrame | None,
    subgroups_from: str,
    subgroup_type: GroupableTypes,
    loop_over: Collection[str] = [],
    **kwargs,
) -> pl.DataFrame:
    """
    Takes in a dataframe `df` with measurements for the `supergroups`.
    Imputes values for the subgroups and returns a dataframe with those.

    Parameters
    ----------

    Returns
    -------
    pl.DataFrame
        Dataframe with measurements imputed for the subgroups.
    """

    group_vartype = GroupingVariable(subgroups_from, subgroup_type)

    # Attribute subgroups
    if subgroup_to_supergroup is None:
        sub_to_super = attribute_subgroups(
            supergroup_df, subgroup_df, group_vartype, **kwargs
        )
    else:
        supergroup_col = kwargs.get("supergroup_col", "supergroup")
        subgroup_col = kwargs.get("subgroup_col", "subgroup")
        sub_to_super = dict(
            zip(
                subgroup_to_supergroup[subgroup_col].to_list(),
                subgroup_to_supergroup[supergroup_col].to_list(),
            )
        )

    if loop_over:
        supergroup_df_looper = [
            dfg[1] for dfg in supergroup_df.group_by(loop_over)
        ]
        # Broadcast static subgroup data up, or else make the list
        if all(var in subgroup_df.columns for var in loop_over):
            subgroup_df_list = [
                dfg[1] for dfg in subgroup_df.group_by(loop_over)
            ]
        else:
            subgroup_df_list = [subgroup_df] * len(supergroup_df_looper)
    else:
        supergroup_df_list = [supergroup_df]
        subgroup_df_list = [subgroup_df]

    assert len(supergroup_df_list) == len(subgroup_df_list)
    return pl.concat(
        [
            _disaggregate(
                super_df, sub_df, sub_to_super, group_vartype, **kwargs
            )
            for super_df, sub_df in zip(supergroup_df_list, subgroup_df_list)
        ]
    )


def _disaggregate(
    supergroup_df: pl.DataFrame,
    subgroup_df: pl.DataFrame,
    subgroup_to_supergroup: dict[str, str],
    group_vartype: GroupingVariable,
    **kwargs,
) -> pl.DataFrame:
    """
    Internal disaggregation function for processed inputs.
    """
    all_groups = [
        Group(name=group_name)
        for group_name in list(set(subgroup_to_supergroup.values()))
        + list(subgroup_to_supergroup.keys())
    ]

    map = GroupMap(subgroup_to_supergroup, all_groups)
    map.add_data_from_polars(supergroup_df)
    map.add_data_from_polars(subgroup_df)

    weight_calculator = (
        ProportionsFromCategories(group_vartype.name)
        if group_vartype.type == "Categorical"
        else ProportionsFromContinuous()
    )
    return Disaggregator(weight_calculator)(map).data_to_polars("subgroup")
