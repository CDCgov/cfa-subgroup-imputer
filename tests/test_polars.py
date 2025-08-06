import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.polars import create_group_map
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


def test_data_io(three_counties):
    state_data = pl.DataFrame(
        {
            "state": ["California", "Washington"],
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

    three_counties.data_from_polars(
        state_data,
        "supergroup",
        exclude=["to_exclude"],
        count=["some_count"],
        rate=["some_rate"],
        copy=["flower"],
    )

    df = three_counties.data_to_polars("supergroup")

    assert_frame_equal(
        state_data.drop(["to_exclude"]), df, check_row_order=False
    )
