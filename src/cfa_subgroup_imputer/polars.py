"""
Module for polars interface.
"""

import polars as pl

from cfa_subgroup_imputer.groups import GroupMap
from cfa_subgroup_imputer.imputer import DisaggregationMethod, Disaggregator
from cfa_subgroup_imputer.one_dimensional import MeasurementType


def make_group_map(
    subgroup_to_supergroup: dict[str, str],
    subgroup_df: pl.DataFrame,
) -> GroupMap:
    """
    Makes a GroupMap for the given supergroups and subgroups
    """
    raise NotImplementedError()


def populate_supergroup_data(
    df: pl.DataFrame,
    map: GroupMap,
    measurements: dict[str, MeasurementType],
) -> GroupMap:
    """
    Adds data from `df` for specified `measurements` to the supergroups in the `map`.
    """
    # Maybe this should be a GroupMap method?
    raise NotImplementedError()


def disaggregate(
    supergroup_df: pl.DataFrame,
    subgroup_df: pl.DataFrame,
    subgroup_to_supergroup: dict[str, str],
    measurements: dict[str, MeasurementType],
    method: DisaggregationMethod = "uniform",
):
    """
    Takes in a dataframe `df` with measurements for the `supergroups`.
    Imputes values for the subgroups and returns a dataframe with those.

    Parameters
    ----------
    supergroup_df : pl.DataFrame
        Dataframe with measurements at the supergroup level.
        Must only contain: one column named "group" (pl.String) and the columns specified by `measurements` (numeric columns).
        Must include exactly the supergroups included in `subgroup_to_supergroup`.
    subgroup_df : pl.DataFrame
        Dataframe with subgroup sizes and, optionally, weight adjustments and the variable defining the supergroup to subgroup relationships.
        Must contain: one column named "group" (pl.String) and one column named "size" (a numeric column).
        May contain: one column named "relative_weight" (a numeric column) and a set of axis columns
        ("lower" and  "upper" (numeric columns), "lower_included" and "upper_included" (pl.Boolean))
    subgroup_to_supergroup: dict[str, str]
        Dict mapping names of all subgroups to their encompassing supergroups.
    measurements: dict[str, MeasurementType]
        The names of the columns in `supergroup_df` which have the data to be disaggregated, and the type thereof (see one_dimensional.MeasurementType).
    method: DisaggregationMethod
        The method to use for imputing subgroup values, see Disaggregator.

    Returns
    -------
    pl.DataFrame
        Dataframe with measurements imputed for the subgroups.
    """
    # TODO: lots of correctness checking of the dataframes
    map = populate_supergroup_data(
        supergroup_df,
        make_group_map(
            subgroup_to_supergroup,
            subgroup_df,
        ),
        measurements,
    )
    return Disaggregator(method)(map).data_as_polars("subgroup")
