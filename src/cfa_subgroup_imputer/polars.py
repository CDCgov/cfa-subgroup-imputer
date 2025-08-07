"""
Module for polars interface.
"""

from collections.abc import Collection
from copy import deepcopy

import polars as pl

from cfa_subgroup_imputer.groups import GroupMap
from cfa_subgroup_imputer.imputer import (
    Disaggregator,
    ProportionsFromCategories,
    ProportionsFromContinuous,
)
from cfa_subgroup_imputer.mapping import (
    AgeGroupHandler,
    OuterProductSubgroupHandler,
    RaggedOuterProductSubgroupHandler,
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
        return RaggedOuterProductSubgroupHandler().construct_group_map(
            category_combinations=sub_super_pairs,
            variable_names=[subgroups_from, supergroups_from],
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
            supergroups=supergroups, subgroups=subgroups, **kwargs
        )
    elif group_type == "age":
        return AgeGroupHandler(
            age_max=kwargs.get("age_max", 100)
        ).construct_group_map(
            supergroups=supergroups, subgroups=subgroups, **kwargs
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
    rate: Collection[str] = [],
    count: Collection[str] = [],
    exclude: Collection[str] = [],
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

    group_map = create_group_map(
        supergroup_df=supergroup_df,
        subgroup_df=subgroup_df,
        subgroup_to_supergroup=subgroup_to_supergroup,
        supergroups_from=supergroups_from,
        subgroups_from=subgroups_from,
        group_type=group_type,
        **kwargs,
    )

    if subgroup_to_supergroup is not None or group_type == "categorical":
        prop_calc = ProportionsFromCategories(
            size_from=kwargs.get("size_from", "size")
        )
    elif group_type == "age":
        prop_calc = ProportionsFromContinuous(
            continuous_var_name=kwargs.get("continuous_var_name", "age")
        )
    else:
        raise RuntimeError(f"Unknown grouping variable type {group_type}")

    disaggregator = Disaggregator(proportion_calculator=prop_calc)

    safe_loop_over = list(loop_over) + ["dummy"]
    supergroup_df = supergroup_df.with_columns(dummy=pl.lit("dummy"))
    subgroup_df = subgroup_df.with_columns(dummy=pl.lit("dummy"))

    for grp_type, grp_info in {
        "supergroup": {
            "df": supergroup_df,
            "groups_from": supergroups_from,
            "n_groups": len(group_map.supergroup_names),
        },
        "subgroup": {
            "df": subgroup_df,
            "groups_from": subgroups_from,
            "n_groups": len(group_map.subgroup_names()),
        },
    }.items():
        assert (
            missing := set(safe_loop_over).difference(grp_info["df"].columns)
        ) == set(), (
            f"Looping variables are missing from {grp_type} dataframe: {missing}"
        )

        aux = safe_loop_over + [grp_info["groups_from"]]
        for col in set(supergroup_df.columns).difference(aux):
            assert (
                supergroup_df.select([col] + aux).unique().shape[0]
                == grp_info["n_groups"]
            ), (
                f"Column {col} in {grp_type} dataframe is not unique within {grp_type} ({grp_info['n_groups']}) and looping variable ({loop_over}) combinations."
            )

    # If we're not told what to do with the column, copy it
    copy = (
        set(supergroup_df.columns)
        .difference(safe_loop_over)
        .difference(exclude)
        .difference(rate)
        .difference(count)
    )
    disagg_comp = []
    for supergroup_dfg, subgroup_dfg in zip(
        supergroup_df.group_by(safe_loop_over),
        subgroup_df.group_by(safe_loop_over),
    ):
        grp_map = deepcopy(group_map)
        grp_map.data_from_polars(
            supergroup_dfg[1],
            "supergroup",
            copy=copy,
            exclude=exclude,
            count=count,
            rate=rate,
        )
        grp_map.data_from_polars(
            subgroup_dfg[1],
            "subgroup",
            copy=copy,
            exclude=exclude,
            count=count,
            rate=rate,
        )

        disagg_comp.append(disaggregator(grp_map).data_to_polars("subgroup"))

    return pl.concat(disagg_comp).drop("dummy")
