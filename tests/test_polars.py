import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.polars import create_group_map, disaggregate
from cfa_subgroup_imputer.variables import Attribute


@pytest.fixture
def three_counties():
    state_counties = pl.DataFrame(
        {
            "state": [
                "California",
                "Washington",
                "Washington",
            ],
            "county": ["Sutter", "Skagit", "San Juan"],
        }
    )

    return create_group_map(
        supergroup_df=None,
        subgroup_df=None,
        subgroup_to_supergroup=state_counties,
        supergroups_from="state",
        subgroups_from="county",
        group_type=None,
    )


@pytest.fixture
def state_data():
    return pl.DataFrame(
        {
            "state": ["California", "Washington"],
            "size": [40, 8],
            "flower": [
                "Eschscholzia californica",
                "Rhododendron macrophyllum",
            ],
            "some_rate": [1.2, 1.3],
            "some_count": [10, 20],
            "to_exclude": ["wont", "see"],
            "to_ignore": ["willbe", "ignored"],
        }
    )


@pytest.fixture
def age_group_data():
    return pl.DataFrame(
        {
            "age_group": ["0-17 years", "18+ years"],
            "size": [1800, 8200],
            "cases": [180, 820],
            "vaccination_rate": [0.4, 0.8],
            "collection_date": ["2024-01-01", "2024-01-01"],
            "notes": ["young", "adult"],
            "to_exclude": ["skip1", "skip2"],
        }
    )


@pytest.fixture
def age_subgroups():
    return pl.DataFrame(
        {
            "age_group": [
                "0-4 years",
                "5-17 years",
                "18-64 years",
                "65+ years",
            ],
        }
    )


def test_groups_from_df(three_counties):
    map_expected = {
        ("Sutter", "California"): "California",
        ("Skagit", "Washington"): "Washington",
        ("San Juan", "Washington"): "Washington",
    }

    groups_expected = {
        "California": Group(
            name="California",
            attributes=[
                Attribute(
                    value="California",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
        "Washington": Group(
            name="Washington",
            attributes=[
                Attribute(
                    value="Washington",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
        ("Sutter", "California"): Group(
            name=("Sutter", "California"),
            attributes=[
                Attribute(
                    value="Sutter",
                    name="county",
                    impute_action="ignore",
                ),
                Attribute(
                    value="California",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
        ("Skagit", "Washington"): Group(
            name=("Skagit", "Washington"),
            attributes=[
                Attribute(
                    value="Skagit",
                    name="county",
                    impute_action="ignore",
                ),
                Attribute(
                    value="Washington",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
        ("San Juan", "Washington"): Group(
            name=("San Juan", "Washington"),
            attributes=[
                Attribute(
                    value="San Juan",
                    name="county",
                    impute_action="ignore",
                ),
                Attribute(
                    value="Washington",
                    name="state",
                    impute_action="ignore",
                ),
            ],
        ),
    }

    assert three_counties.groups == groups_expected
    assert three_counties.sub_to_super == map_expected


def test_data_io_categorical(three_counties, state_data):
    three_counties.data_from_polars(
        state_data,
        "supergroup",
        exclude=["to_exclude", "size"],
        count=["some_count"],
        rate=["some_rate"],
        copy=["flower"],
    )

    df = three_counties.data_to_polars("supergroup")

    assert_frame_equal(
        state_data.drop(["to_exclude", "size"]), df, check_row_order=False
    )


def test_data_io_age_groups(age_subgroups, age_group_data):
    age_group_map = create_group_map(
        supergroup_df=age_group_data,
        subgroup_df=age_subgroups,
        subgroup_to_supergroup=None,
        supergroups_from="age_group",
        subgroups_from="age_group",
        group_type="age",
    )

    age_group_map.data_from_polars(
        age_group_data,
        "supergroup",
        exclude=["to_exclude", "notes"],
        count=["cases", "size"],
        rate=["vaccination_rate"],
        copy=["collection_date"],
    )

    df = age_group_map.data_to_polars("supergroup")

    assert_frame_equal(
        age_group_data.drop(["to_exclude", "notes"]), df, check_row_order=False
    )


def test_disagg_ragged_categorical(state_data):
    subgroup_df = pl.DataFrame(
        {
            "state": ["California", "California", "Washington", "Washington"],
            "splitvar": ["cat1", "cat2", "cat1", "cat2"],
            "size": [20, 20, 2, 6],
        }
    )

    disagg = disaggregate(
        supergroup_df=state_data,
        subgroup_df=subgroup_df,
        subgroup_to_supergroup=None,
        supergroups_from="state",
        subgroups_from="splitvar",
        group_type="categorical",
        loop_over=[],
        rate=["some_rate"],
        count=["some_count"],
        exclude=["to_exclude", "to_ignore"],
    )

    expected_disagg = pl.DataFrame(
        {
            "state": ["California", "California", "Washington", "Washington"],
            "splitvar": ["cat1", "cat2", "cat1", "cat2"],
            "size": [20, 20, 2, 6],
            "flower": [
                "Eschscholzia californica",
                "Eschscholzia californica",
                "Rhododendron macrophyllum",
                "Rhododendron macrophyllum",
            ],
            "some_rate": [1.2, 1.2, 1.3, 1.3],
            "some_count": [5.0, 5.0, 5.0, 15.0],
        }
    )

    assert_frame_equal(
        disagg,
        expected_disagg,
        check_row_order=False,
        check_column_order=False,
    )


# def test_disagg_continuous_age(age_group_data, age_subgroups):
#     disagg = disaggregate(
#         supergroup_df=age_group_data,
#         subgroup_df=age_subgroups,
#         subgroup_to_supergroup=None,
#         supergroups_from="age_group",
#         subgroups_from="age_group",
#         group_type="age",
#         loop_over=[],
#         rate=["vaccination_rate"],
#         count=["cases", "size"],
#         copy=["collection_date"],
#         exclude=["notes", "to_exclude"],
#     )

#     expected_disagg = pl.DataFrame(
#         {
#             "age_group": [
#                 "0-4 years",
#                 "5-17 years",
#                 "18-64 years",
#                 "65+ years",
#             ],
#             "size": [500.0, 1300.0, 4700.0, 3500.0],
#             "cases": [50.0, 130.0, 470.0, 350.0],
#             "vaccination_rate": [0.4, 0.4, 0.8, 0.8],
#             "collection_date": [
#                 "2024-01-01",
#                 "2024-01-01",
#                 "2024-01-01",
#                 "2024-01-01",
#             ],
#         }
#     )

#     assert_frame_equal(
#         disagg,
#         expected_disagg,
#         check_row_order=False,
#         check_column_order=False,
#     )
