import polars as pl

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.polars import create_group_map
from cfa_subgroup_imputer.variables import Attribute


def test_groups_from_df():
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

    group_map = create_group_map(
        supergroup_df=None,
        subgroup_df=None,
        subgroup_to_supergroup=state_counties,
        supergroups_from="state",
        subgroups_from="county",
        group_type=None,
    )

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

    assert group_map.groups == groups_expected
    assert group_map.sub_to_super == map_expected
