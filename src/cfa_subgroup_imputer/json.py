"""
Module for interfacing with JSON-style inputs.
"""

from collections.abc import Collection, Iterable
from copy import deepcopy
from itertools import groupby
from operator import itemgetter
from typing import Any

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
    supergroup_data: Iterable[dict[str, Any]] | None,
    subgroup_data: Iterable[dict[str, Any]] | None,
    subgroup_to_supergroup: Iterable[dict[str, Any]] | None,
    supergroups_from: str,
    subgroups_from: str,
    group_type: GroupableTypes | None,
    **kwargs,
) -> GroupMap:
    if subgroup_to_supergroup is not None:
        sub_super_pairs = [
            (row[subgroups_from], row[supergroups_from])
            for row in subgroup_to_supergroup
        ]
        return RaggedOuterProductSubgroupHandler().construct_group_map(
            category_combinations=sub_super_pairs,
            variable_names=[subgroups_from, supergroups_from],
        )

    assert supergroup_data is not None, (
        "If not supplying `subgroup_to_supergroup`, must supply `supergroup_data`."
    )
    assert subgroup_data is not None, (
        "If not supplying `subgroup_to_supergroup`, must supply `subgroup_data`."
    )

    supergroup_cats = list(
        set(row[supergroups_from] for row in supergroup_data)
    )
    subgroup_cats = list(set(row[subgroups_from] for row in subgroup_data))
    if group_type == "categorical":
        return OuterProductSubgroupHandler().construct_group_map(
            supergroup_categories=supergroup_cats,
            subgroup_categories=[subgroup_cats],
            supergroup_variable_name=supergroups_from,
            subgroup_variable_names=[subgroups_from],
            **kwargs,
        )
    elif group_type == "age":
        # TODO: we could rename this ourselves, instead of erroring out
        #       though then we'd have to tweak the written output at the end too
        assert supergroups_from == subgroups_from, (
            "Age groups must be named identically in super and subgroup data"
        )
        return AgeGroupHandler(
            age_max=kwargs.get("age_max", 100)
        ).construct_group_map(
            supergroups=supergroup_cats,
            subgroups=subgroup_cats,
            continuous_var_name=subgroups_from,
            **kwargs,
        )
    else:
        raise RuntimeError(f"Unknown grouping variable type {group_type}")


def disaggregate(
    supergroup_data: Iterable[dict[str, Any]],
    subgroup_data: Iterable[dict[str, Any]],
    subgroup_to_supergroup: Iterable[dict[str, Any]] | None,
    supergroups_from: str,
    subgroups_from: str,
    group_type: GroupableTypes | None,
    loop_over: Collection[str] = [],
    rate: Collection[str] = [],
    count: Collection[str] = [],
    exclude: Collection[str] = [],
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Takes in a dataframe `df` with measurements for the `supergroups`.
    Imputes values for the subgroups and returns a dataframe with those.

    Parameters
    ----------

    Returns
    -------
    list[dict[str, Any]]
        Data with measurements imputed for the subgroups.
    """

    group_map = create_group_map(
        supergroup_data=supergroup_data,
        subgroup_data=subgroup_data,
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
        # TODO: as above we could rename this ourselves
        assert supergroups_from == subgroups_from, (
            "Age groups must be named identically in super and subgroup data"
        )
        prop_calc = ProportionsFromContinuous(
            continuous_var_name=subgroups_from
        )
    else:
        raise RuntimeError(f"Unknown grouping variable type {group_type}")

    disaggregator = Disaggregator(proportion_calculator=prop_calc)

    # Add a dummy variable to loop over if none are provided
    safe_loop_over = list(loop_over) if loop_over else ["dummy"]
    supergroup_data = [d | {"dummy": "dummy"} for d in supergroup_data]
    subgroup_data = [d | {"dummy": "dummy"} for d in subgroup_data]

    # Sort data for groupby
    supergroup_data.sort(key=itemgetter(*safe_loop_over))
    subgroup_data.sort(key=itemgetter(*safe_loop_over))

    # If we're not told what to do with the column, and it's not being used to compute proportions, copy it
    if subgroup_to_supergroup is not None or group_type == "categorical":
        ignore = [kwargs.get("size_from", "size")]
    elif group_type == "age":
        ignore = [kwargs.get("continuous_var_name", "age")]
    else:
        ignore = []

    copy = (
        set(supergroup_data[0].keys())
        .difference(safe_loop_over)
        .difference(exclude)
        .difference(rate)
        .difference(count)
        .difference(ignore)
        .difference([supergroups_from])
    )
    disagg_df_comp = []

    super_grouper = groupby(supergroup_data, key=itemgetter(*safe_loop_over))
    sub_grouper = groupby(subgroup_data, key=itemgetter(*safe_loop_over))

    for (super_key, super_grp), (sub_key, sub_grp) in zip(
        super_grouper, sub_grouper
    ):
        assert super_key == sub_key, (
            "Mismatch in looping variables between supergroup and subgroup data"
        )
        grp_map = deepcopy(group_map)

        super_grp_list = list(super_grp)
        sub_grp_list = list(sub_grp)

        grp_map.data_from_dicts(
            super_grp_list,
            "supergroup",
            copy=copy,
            exclude=exclude,
            count=count,
            rate=rate,
        )
        grp_map.data_from_dicts(
            sub_grp_list,
            "subgroup",
            copy=copy,
            exclude=exclude,
            count=count,
            rate=rate,
        )

        disagg_map = disaggregator(grp_map)
        disagg_df_comp.extend(disagg_map.to_dicts("subgroup"))

    # Remove dummy variable if it was added
    if not loop_over:
        for row in disagg_df_comp:
            del row["dummy"]

    return disagg_df_comp
