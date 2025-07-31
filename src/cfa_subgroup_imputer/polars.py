"""
Module for polars interface.
"""

from collections.abc import Collection

import polars as pl

from cfa_subgroup_imputer.groups import GroupMap
from cfa_subgroup_imputer.mapping import (
    AgeGroupHandler,
    ArbitraryGroupHandler,
    OuterProductSubgroupHandler,
    StringPaster,
)
from cfa_subgroup_imputer.variables import GroupableTypes


def create_group_map(
    supergroup_df: pl.DataFrame | None,
    subgroup_df: pl.DataFrame | None,
    subgroup_to_supergroup: pl.DataFrame | None,
    supergroups_from: str,
    subgroups_from: str,
    group_type: GroupableTypes | None,
    **kwargs,
) -> GroupMap:
    if subgroup_to_supergroup is not None:
        sub_super_pairs = [
            (row_dict[subgroups_from], row_dict[supergroups_from])
            for row_dict in subgroup_to_supergroup.to_dicts()
        ]
        return ArbitraryGroupHandler(
            id_combiner=StringPaster()
        ).construct_group_map(
            sub_super=sub_super_pairs,
            supergroup_varname=supergroups_from,
            subgroup_varname=subgroups_from,
        )

    assert supergroup_df is not None, (
        "If not supplying `subgroup_to_supergroup`, must supply `supergroup_df`."
    )
    assert subgroup_df is not None, (
        "If not supplying `subgroup_to_supergroup`, must supply `subgroup_df`."
    )

    supergroups = supergroup_df[supergroups_from].unique().to_list()
    subgroups = subgroup_df[subgroups_from].unique().to_list()
    if group_type == "categorical":
        return OuterProductSubgroupHandler().construct_group_map(
            supergroups=supergroups, subgroups=subgroups
        )
    elif group_type == "age":
        return AgeGroupHandler().construct_group_map(
            supergroups=supergroups, subgroups=subgroups
        )
    else:
        raise RuntimeError(f"Unknown grouping variable type {group_type}")


def disaggregate(
    supergroup_df: pl.DataFrame,
    subgroup_df: pl.DataFrame,
    subgroup_to_supergroup: pl.DataFrame | None,
    supergroups_from: str,
    subgroups_from: str,
    group_type: GroupableTypes | None,
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

    _ = create_group_map(
        supergroup_df=supergroup_df,
        subgroup_df=subgroup_df,
        subgroup_to_supergroup=subgroup_to_supergroup,
        supergroups_from=supergroups_from,
        subgroups_from=subgroups_from,
        group_type=group_type,
        **kwargs,
    )

    raise NotImplementedError()
