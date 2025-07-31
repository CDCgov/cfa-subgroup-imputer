import polars as pl

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.polars import create_group_map
from cfa_subgroup_imputer.variables import Attribute


def test_arbitrary_groups():
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
        "California_Sutter": "California",
        "Washington_Skagit": "Washington",
        "Washington_San Juan": "Washington",
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
        "California_Sutter": Group(
            name="California_Sutter",
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
        "Washington_Skagit": Group(
            name="Washington_Skagit",
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
        "Washington_San Juan": Group(
            name="Washington_San Juan",
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
