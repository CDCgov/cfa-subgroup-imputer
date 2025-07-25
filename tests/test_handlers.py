import math

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
            41.0,
            42.0,
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
