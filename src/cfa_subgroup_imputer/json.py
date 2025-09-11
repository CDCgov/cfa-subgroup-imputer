"""
Module for interfacing with JSON-style inputs.
"""

from collections.abc import Collection, Iterable
from copy import deepcopy
from itertools import groupby
from operator import itemgetter
from typing import Any, Literal

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


def _assert_levels_match(
    supergroup_data: Iterable[dict[str, Any]],
    subgroup_data: Iterable[dict[str, Any]],
    variable: str,
):
    """
    Check that levels of the relevant variable are the same in both datasets.
    """
    super_lvls = set([row[variable] for row in supergroup_data])
    sub_lvls = set([row[variable] for row in subgroup_data])

    assert sub_lvls == super_lvls, (
        f"Supergroup levels for variable `{variable}` are {super_lvls} but subgroup levels are {sub_lvls}."
    )


def create_group_map(
    supergroup_data: Iterable[dict[str, Any]] | None,
    subgroup_data: Iterable[dict[str, Any]] | None,
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
    assert subgroup_data is not None, (
        "If not supplying `subgroup_to_supergroup`, must supply `subgroup_data`."
    )

    supergroup_cats = sorted(
        list(set(row[supergroups_from] for row in supergroup_data))
    )
    subgroup_cats = sorted(
        list(set(row[subgroups_from] for row in subgroup_data))
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


def expand_categorical_subgroups(
    supergroup_data: Iterable[dict[str, Any]],
    subgroup_data: Iterable[dict[str, Any]],
    supergroups_from: str,
) -> Iterable[dict[str, Any]]:
    """
    Ensure both supergrouping and subgrouping variables are present in `subgroup_data` as required by imputer.Disaggregator.
    If the subgroup data has both supergrouping and subgrouping variables already, it is left alone.
    Otherwise, takes an independent and shared categorical subgrouping variable and creates a dataframe for all
    (supergroup level, subgroup level) pairs by recycling the subgrouping variable levels.

    For example, if disaggregating per-county data into per-county-by-sex data, when assuming the same proportion of sexes in
    all counties, the `subgroup_data` fed in might be [{"male": 0.5}, {"female": 0.5}]. This will expand it to
    [{"male": 0.5, "county": "Adams"}, {"female": 0.5, "county": "Adams"}, ...,{"male": 0.5, "county": "Yakima"},
    {"female": 0.5, "county": "Yakima"}].
    """
    if supergroups_from in get_json_keys(subgroup_data):
        return subgroup_data

    supergroup_lvls = set([row[supergroups_from] for row in supergroup_data])

    expanded = []
    for row in subgroup_data:
        expanded += [row | {supergroups_from: lvl} for lvl in supergroup_lvls]
    return expanded


def impute(
    action: Literal["aggregate", "disaggregate"],
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
    size_from: str = "size",
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Takes in data for supergroups/subgroups, imputes and returns values for the subgroups/supergroups.

    Parameters
    ----------
    action: Literal["aggregate", "disaggregate"]
        Whether to aggregate or disaggregate.
    supergroup_data: Iterable[dict[str, Any]]
        Information defining supergroups, including any data to disaggregate.
    subgroup_data: Iterable[dict[str, Any]]
        Information defining the subgroups, including any data to aggregate.
    subgroup_to_supergroup: Iterable[dict[str, Any]] | None
        Optional mapping defining all subgroup : supergroup.
    supergroups_from: str
        Name of key in `supergroup_data` defining supergroups.
    subgroups_from: str
        Name of key in `subgroup_data` defining subgroups.
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

    if loop_over:
        raise NotImplementedError(
            "Looping over covariates is not yet supported."
        )

    if group_type == "categorical":
        subgroup_data = expand_categorical_subgroups(
            supergroup_data, subgroup_data, supergroups_from
        )
        _assert_levels_match(supergroup_data, subgroup_data, supergroups_from)

    group_map = create_group_map(
        supergroup_data=supergroup_data,
        subgroup_data=subgroup_data,
        subgroup_to_supergroup=subgroup_to_supergroup,
        supergroups_from=supergroups_from,
        subgroups_from=subgroups_from,
        group_type=group_type,
        **kwargs,
    )

    if action == "aggregate":
        imputer = Aggregator(size_from)
        output_level = "supergroup"
    elif action == "disaggregate":
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
        imputer = Disaggregator(
            proportion_calculator=prop_calc, size_from=size_from
        )
        output_level = "subgroup"
    else:
        raise ValueError(f"Unknown action {action}")

    # Add a dummy variable to loop over if none are provided
    if not loop_over:
        safe_loop_over = ["dummy"]
        supergroup_data = [d | {"dummy": "dummy"} for d in supergroup_data]
        subgroup_data = [d | {"dummy": "dummy"} for d in subgroup_data]
    else:
        safe_loop_over = list(loop_over)
        supergroup_data = [d for d in supergroup_data]
        subgroup_data = [d for d in subgroup_data]

    for grp_type, grp_info in {
        "supergroup": {
            "data": supergroup_data,
            "groups_from": [supergroups_from],
            "n_groups": len(group_map.supergroup_names),
        },
        "subgroup": {
            "data": subgroup_data,
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

    supergroup_data.sort(key=itemgetter(*safe_loop_over))
    subgroup_data.sort(key=itemgetter(*safe_loop_over))

    # If we're not told what to do with the column, and it's not being used to compute proportions, copy it
    if subgroup_to_supergroup is not None or group_type == "categorical":
        ignore = [size_from]
    elif group_type == "age":
        ignore = [kwargs.get("continuous_var_name", "age")]
    else:
        ignore = []

    if action == "aggregate":
        copy_from = subgroup_data
        groups_from = subgroups_from
    else:
        copy_from = supergroup_data
        groups_from = supergroups_from

    copy = (
        set(get_json_keys(copy_from))
        .difference(safe_loop_over)
        .difference(exclude)
        .difference(rate)
        .difference(count)
        .difference(ignore)
        # TODO: this is somewhat redundant with data_from_json knowing not to copy group-defining variables
        .difference([groups_from])
    )
    imputed_comp = []

    super_grouper = groupby(supergroup_data, key=itemgetter(*safe_loop_over))
    sub_grouper = groupby(subgroup_data, key=itemgetter(*safe_loop_over))

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

        imputed_map = imputer(grp_map)
        imputed_comp.extend(imputed_map.to_dicts(output_level))

    # Remove dummy variable if it was added
    if not loop_over:
        for row in imputed_comp:
            del row["dummy"]

    return imputed_comp


def aggregate(
    # TODO: we should perhaps let this be just a list of values for aggregating on age, or some simple categorical cases
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
    size_from: str = "size",
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Wrapper for `impute` with `action="aggregate"`.
    """
    return impute(
        action="aggregate",
        supergroup_data=supergroup_data,
        subgroup_data=subgroup_data,
        subgroup_to_supergroup=subgroup_to_supergroup,
        supergroups_from=supergroups_from,
        subgroups_from=subgroups_from,
        group_type=group_type,
        loop_over=loop_over,
        rate=rate,
        count=count,
        exclude=exclude,
        size_from=size_from,
        **kwargs,
    )


def disaggregate(
    supergroup_data: Iterable[dict[str, Any]],
    # TODO: we should perhaps let this be just a list of values for splitting on age
    subgroup_data: Iterable[dict[str, Any]],
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
    Wrapper for `impute` with `action="disaggregate"`.
    """
    return impute(
        action="disaggregate",
        supergroup_data=supergroup_data,
        subgroup_data=subgroup_data,
        subgroup_to_supergroup=subgroup_to_supergroup,
        supergroups_from=supergroups_from,
        subgroups_from=subgroups_from,
        group_type=group_type,
        loop_over=loop_over,
        rate=rate,
        count=count,
        exclude=exclude,
        size_from=size_from,
        **kwargs,
    )
