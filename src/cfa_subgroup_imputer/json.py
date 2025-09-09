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
    Aggregator,
    Disaggregator,
    ProportionsFromCategories,
    ProportionsFromContinuous,
)
from cfa_subgroup_imputer.mapping import (
    AgeGroupHandler,
    OuterProductSubgroupHandler,
    RaggedOuterProductSubgroupHandler,
)
from cfa_subgroup_imputer.utils import get_json_keys, select, unique
from cfa_subgroup_imputer.variables import GroupableTypes


def create_group_map(
    supergroup_data: Iterable[dict[str, Any]] | None,
    subgroup_defs: Iterable[dict[str, Any]] | None,
    subgroup_to_supergroup: Iterable[dict[str, Any]] | None,
    supergroups_from: str,
    subgroups_from: str,
    group_type: GroupableTypes | None,
    **kwargs,
) -> GroupMap:
    """
    GroupMap construction utility for `disaggregate`. See there for more details.
    """
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
    assert subgroup_defs is not None, (
        "If not supplying `subgroup_to_supergroup`, must supply `subgroup_defs`."
    )

    supergroup_cats = sorted(
        list(set(row[supergroups_from] for row in supergroup_data))
    )
    subgroup_cats = sorted(
        list(set(row[subgroups_from] for row in subgroup_defs))
    )
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


def aggregate(
    supergroup_data: Iterable[dict[str, Any]],
    subgroup_defs: Iterable[dict[str, Any]],
    subgroup_to_supergroup: Iterable[dict[str, Any]] | None,
    supergroups_from: str,
    subgroups_from: str,
    group_type: GroupableTypes | None,
    loop_over: Collection[str] = [],
    rate: Collection[str] = [],
    count: Collection[str] = [],
    exclude: Collection[str] = [],
    size_from: str = "size",
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Takes in data for supergroups, imputes and returns values for the subgroups.

    Parameters
    ----------
    supergroup_data: Iterable[dict[str, Any]]
        Data for supergroups as list of dicts.
    subgroup_defs: Iterable[dict[str, Any]]
        Information defining the subgroups, including data to aggregate.
    subgroup_to_supergroup: Iterable[dict[str, Any]] | None
        Optional mapping defining all subgroup : supergroup.
    supergroups_from: str
        Name of key in `supergroup_data` defining supergroups.
    subgroups_from: str
        Name of key in `subgroup_defs` defining subgroups.
    group_type: GroupableTypes | None
        What kind of groups are these, categorical or age? Can only
        be None if providing `subgroup_to_supergroup`.
    loop_over: Collection[str] = []
        A collection of covariates, within each combination of which
        we will separately disaggregate. For example, if we wanted
        to disaggregate age groups separately in every state and county
        in a dataset, this would be ["state", "county"].
    rate: Collection[str] = []
        A list of the keys in `supergroup_data` which define rate measurements.
    count: Collection[str] = []
        A list of the keys in `supergroup_data` which define count measurements.
    exclude: Collection[str] = []
        A list the keys in `supergroup_data` which define variables
        which are to be excluded from imputation and which will not
        be present in the output.
    **kwargs
        Passed to internals.

    Returns
    -------
    list[dict[str, Any]]
        Data with measurements imputed for the subgroups.
    """

    group_map = create_group_map(
        supergroup_data=supergroup_data,
        subgroup_defs=subgroup_defs,
        subgroup_to_supergroup=subgroup_to_supergroup,
        supergroups_from=supergroups_from,
        subgroups_from=subgroups_from,
        group_type=group_type,
        **kwargs,
    )

    aggregator = Aggregator(size_from)

    # Add a dummy variable to loop over if none are provided
    safe_loop_over = list(loop_over) if loop_over else ["dummy"]
    supergroup_data = [d | {"dummy": "dummy"} for d in supergroup_data]
    subgroup_defs = [d | {"dummy": "dummy"} for d in subgroup_defs]

    for grp_type, grp_info in {
        "supergroup": {
            "data": supergroup_data,
            "groups_from": [supergroups_from],
            "n_groups": len(group_map.supergroup_names),
        },
        "subgroup": {
            "data": subgroup_defs,
            "groups_from": [subgroups_from] + [supergroups_from]
            if group_type == "categorical"
            else [subgroups_from],
            "n_groups": len(group_map.subgroup_names()),
        },
    }.items():
        assert (
            missing := set(safe_loop_over).difference(
                get_json_keys(grp_info["data"])
            )
        ) == set(), (
            f"Looping variables are missing from {grp_type} data: {missing}"
        )

        assert len(
            unique(
                select(
                    grp_info["data"], safe_loop_over + grp_info["groups_from"]
                )
            )
        ) == len(grp_info["data"]), (
            f"Provided data has multiple entries for at least one combination of group-defining variables ({grp_info['groups_from']}) and variables to loop over ({loop_over}).\n{grp_info['data']}"
        )

    # Sort data for groupby
    supergroup_data.sort(key=itemgetter(*safe_loop_over))
    subgroup_defs.sort(key=itemgetter(*safe_loop_over))

    # If we're not told what to do with the column, and it's not being used to compute proportions, copy it
    if subgroup_to_supergroup is not None or group_type == "categorical":
        ignore = [size_from]
    elif group_type == "age":
        ignore = [kwargs.get("continuous_var_name", "age")]
    else:
        ignore = []

    copy = (
        set(get_json_keys(subgroup_defs))
        .difference(safe_loop_over)
        .difference(exclude)
        .difference(rate)
        .difference(count)
        .difference(ignore)
        # TODO: this is somewhat redundant with data_from_json knowing not to copy group-defining variables
        .difference([supergroups_from])
    )
    agg_comp = []

    super_grouper = groupby(supergroup_data, key=itemgetter(*safe_loop_over))
    sub_grouper = groupby(subgroup_defs, key=itemgetter(*safe_loop_over))

    for (super_key, super_grp), (sub_key, sub_grp) in zip(
        super_grouper, sub_grouper
    ):
        assert super_key == sub_key, (
            "Mismatch in looping variables between supergroup and subgroup data"
        )
        grp_map = deepcopy(group_map)

        grp_map.data_from_dicts(
            list(super_grp),
            "supergroup",
            copy=copy,
            exclude=exclude,
            count=count,
            rate=rate,
        )
        grp_map.data_from_dicts(
            list(sub_grp),
            "subgroup",
            copy=copy,
            exclude=exclude,
            count=count,
            rate=rate,
        )

        agg_map = aggregator(grp_map)
        agg_comp.extend(agg_map.to_dicts("supergroup"))

    # Remove dummy variable if it was added
    if not loop_over:
        for row in agg_comp:
            del row["dummy"]

    return agg_comp


def disaggregate(
    supergroup_data: Iterable[dict[str, Any]],
    # TODO: we should perhaps let this be just a list of values for splitting on age?
    subgroup_defs: Iterable[dict[str, Any]],
    subgroup_to_supergroup: Iterable[dict[str, Any]] | None,
    supergroups_from: str,
    subgroups_from: str,
    group_type: GroupableTypes | None,
    loop_over: Collection[str] = [],
    rate: Collection[str] = [],
    count: Collection[str] = [],
    exclude: Collection[str] = [],
    size_from: str = "size",
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Takes in data for supergroups, imputes and returns values for the subgroups.

    Parameters
    ----------
    supergroup_data: Iterable[dict[str, Any]]
        Data for supergroups as list of dicts.
    subgroup_defs: Iterable[dict[str, Any]]
        Information defining the subgroups.
    subgroup_to_supergroup: Iterable[dict[str, Any]] | None
        Optional mapping defining all subgroup : supergroup.
    supergroups_from: str
        Name of key in `supergroup_data` defining supergroups.
    subgroups_from: str
        Name of key in `subgroup_defs` defining subgroups.
    group_type: GroupableTypes | None
        What kind of groups are these, categorical or age? Can only
        be None if providing `subgroup_to_supergroup`.
    loop_over: Collection[str] = []
        A collection of covariates, within each combination of which
        we will separately disaggregate. For example, if we wanted
        to disaggregate age groups separately in every state and county
        in a dataset, this would be ["state", "county"].
    rate: Collection[str] = []
        A list of the keys in `supergroup_data` which define rate measurements.
    count: Collection[str] = []
        A list of the keys in `supergroup_data` which define count measurements.
    exclude: Collection[str] = []
        A list the keys in `supergroup_data` which define variables
        which are to be excluded from imputation and which will not
        be present in the output.
    **kwargs
        Passed to internals.

    Returns
    -------
    list[dict[str, Any]]
        Data with measurements imputed for the subgroups.
    """

    group_map = create_group_map(
        supergroup_data=supergroup_data,
        subgroup_defs=subgroup_defs,
        subgroup_to_supergroup=subgroup_to_supergroup,
        supergroups_from=supergroups_from,
        subgroups_from=subgroups_from,
        group_type=group_type,
        **kwargs,
    )

    if subgroup_to_supergroup is not None or group_type == "categorical":
        prop_calc = ProportionsFromCategories(size_from=size_from)
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
    subgroup_defs = [d | {"dummy": "dummy"} for d in subgroup_defs]

    for grp_type, grp_info in {
        "supergroup": {
            "data": supergroup_data,
            "groups_from": [supergroups_from],
            "n_groups": len(group_map.supergroup_names),
        },
        "subgroup": {
            "data": subgroup_defs,
            "groups_from": [subgroups_from] + [supergroups_from]
            if group_type == "categorical"
            else [subgroups_from],
            "n_groups": len(group_map.subgroup_names()),
        },
    }.items():
        assert (
            missing := set(safe_loop_over).difference(
                get_json_keys(grp_info["data"])
            )
        ) == set(), (
            f"Looping variables are missing from {grp_type} data: {missing}"
        )

        assert len(
            unique(
                select(
                    grp_info["data"], safe_loop_over + grp_info["groups_from"]
                )
            )
        ) == len(grp_info["data"]), (
            f"Provided data has multiple entries for at least one combination of group-defining variables ({grp_info['groups_from']}) and variables to loop over ({loop_over}).\n{grp_info['data']}"
        )

    # Sort data for groupby
    supergroup_data.sort(key=itemgetter(*safe_loop_over))
    subgroup_defs.sort(key=itemgetter(*safe_loop_over))

    # If we're not told what to do with the column, and it's not being used to compute proportions, copy it
    if subgroup_to_supergroup is not None or group_type == "categorical":
        ignore = [size_from]
    elif group_type == "age":
        ignore = [kwargs.get("continuous_var_name", "age")]
    else:
        ignore = []

    copy = (
        set(get_json_keys(supergroup_data))
        .difference(safe_loop_over)
        .difference(exclude)
        .difference(rate)
        .difference(count)
        .difference(ignore)
        # TODO: this is somewhat redundant with data_from_json knowing not to copy group-defining variables
        .difference([supergroups_from])
    )
    disagg_comp = []

    super_grouper = groupby(supergroup_data, key=itemgetter(*safe_loop_over))
    sub_grouper = groupby(subgroup_defs, key=itemgetter(*safe_loop_over))

    for (super_key, super_grp), (sub_key, sub_grp) in zip(
        super_grouper, sub_grouper
    ):
        assert super_key == sub_key, (
            "Mismatch in looping variables between supergroup and subgroup data"
        )
        grp_map = deepcopy(group_map)

        grp_map.data_from_dicts(
            list(super_grp),
            "supergroup",
            copy=copy,
            exclude=exclude,
            count=count,
            rate=rate,
        )
        grp_map.data_from_dicts(
            list(sub_grp),
            "subgroup",
            copy=copy,
            exclude=exclude,
            count=count,
            rate=rate,
        )

        disagg_map = disaggregator(grp_map)
        disagg_comp.extend(disagg_map.to_dicts("subgroup"))

    # Remove dummy variable if it was added
    if not loop_over:
        for row in disagg_comp:
            del row["dummy"]

    return disagg_comp
