import math

import pytest

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.mapping import (
    AgeGroupHandler,
    CategoricalSubgroupHandler,
)
from cfa_subgroup_imputer.variables import Attribute


class TestAgeGroups:
    def test_ranges(self):
        assert AgeGroupHandler().age_range_from_str(
            "6 months-4 years"
        ).to_tuple() == (
            0.5,
            5.0,
        )

        assert AgeGroupHandler().age_range_from_str(
            "6-23 months"
        ).to_tuple() == (
            0.5,
            2.0,
        )

        assert AgeGroupHandler().age_range_from_str("42 years").to_tuple() == (
            42.0,
            43.0,
        )

        assert AgeGroupHandler().age_range_from_str(
            "60+ years"
        ).to_tuple() == (
            60,
            math.inf,
        )

        assert AgeGroupHandler().age_range_from_str(
            "50-64 years"
        ).to_tuple() == (
            50.0,
            65.0,
        )

        assert AgeGroupHandler().age_range_from_str(
            "0-<1 year"
        ).to_tuple() == (
            0.0,
            1.0,
        )

        assert AgeGroupHandler().age_range_from_str(
            "2-<3 years"
        ).to_tuple() == (
            2.0,
            3.0,
        )

        assert AgeGroupHandler().age_range_from_str(
            "1-<3 months"
        ).to_tuple() == (
            1.0 / 12.0,
            3.0 / 12.0,
        )

    def test_constructor(self):
        supergroups = ["0 years", "1-<2 years"]
        subgroups = ["0-<6 months", "6 months-<1 year", "1 years"]
        AgeGroupHandler().construct_group_map(
            supergroups=supergroups, subgroups=subgroups
        )
        # groups_expected = {}
        # map_expected = {}

        # Subgroup missing supergroup
        with pytest.raises(Exception):
            AgeGroupHandler().construct_group_map(
                supergroups=supergroups,
                subgroups=["0-<6 months", "6 months-<1 year", "2 years"],
            )

        # Supergroup missing subgroup
        with pytest.raises(Exception):
            AgeGroupHandler().construct_group_map(
                supergroups=supergroups,
                subgroups=["0-<6 months", "2 years"],
            )

        # Supergroups noncontiguous
        with pytest.raises(Exception):
            AgeGroupHandler().construct_group_map(
                supergroups=["0 years", "2 years"],
                subgroups=["0 years", "2 years"],
            )


class TestCategoroical:
    def test_constructor(self):
        supergroups = ["festina", "festimus materia"]
        subgroups = ["lente", "velociter"]

        map_expected = {
            "festina_lente": "festina",
            "festina_velociter": "festina",
            "festimus materia_lente": "festimus materia",
            "festimus materia_velociter": "festimus materia",
        }

        groups_expected = {
            "festina": Group(
                name="festina",
                attributes=[
                    Attribute(
                        value="festina",
                        name="phrase",
                        impute_action="ignore",
                    ),
                ],
            ),
            "festimus materia": Group(
                name="festimus materia",
                attributes=[
                    Attribute(
                        value="festimus materia",
                        name="phrase",
                        impute_action="ignore",
                    ),
                ],
            ),
            "festina_lente": Group(
                name="festina_lente",
                attributes=[
                    Attribute(
                        value="lente",
                        name="speed",
                        impute_action="ignore",
                    ),
                    Attribute(
                        value="festina",
                        name="phrase",
                        impute_action="ignore",
                    ),
                ],
            ),
            "festina_velociter": Group(
                name="festina_velociter",
                attributes=[
                    Attribute(
                        value="velociter",
                        name="speed",
                        impute_action="ignore",
                    ),
                    Attribute(
                        value="festina",
                        name="phrase",
                        impute_action="ignore",
                    ),
                ],
            ),
            "festimus materia_lente": Group(
                name="festimus materia_lente",
                attributes=[
                    Attribute(
                        value="lente",
                        name="speed",
                        impute_action="ignore",
                    ),
                    Attribute(
                        value="festimus materia",
                        name="phrase",
                        impute_action="ignore",
                    ),
                ],
            ),
            "festimus materia_velociter": Group(
                name="festimus materia_velociter",
                attributes=[
                    Attribute(
                        value="velociter",
                        name="speed",
                        impute_action="ignore",
                    ),
                    Attribute(
                        value="festimus materia",
                        name="phrase",
                        impute_action="ignore",
                    ),
                ],
            ),
        }

        gmap = CategoricalSubgroupHandler().construct_group_map(
            supergroups=supergroups,
            subgroups=subgroups,
            supergroup_varname="phrase",
            subgroup_varname="speed",
        )

        assert gmap.groups == groups_expected
        print(">>>>>> map_expected <<<<<<")
        print(map_expected)
        print(">>>>>> gmap.sub_to_super <<<<<<")
        print(gmap.sub_to_super)
        assert gmap.sub_to_super == map_expected
