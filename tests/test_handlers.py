import math

import pytest

from cfa_subgroup_imputer.groups import Group
from cfa_subgroup_imputer.mapping import (
    AgeGroupHandler,
    OuterProductSubgroupHandler,
)
from cfa_subgroup_imputer.variables import Attribute, Range


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
        subgroups = ["0-<6 months", "6 months-<1 year", "1 year"]
        group_map = AgeGroupHandler().construct_group_map(
            supergroups=supergroups, subgroups=subgroups
        )
        groups_expected = {
            "0 years": Group(
                name="0 years",
                attributes=[
                    Attribute(
                        name="age", value=Range(0, 1), impute_action="ignore"
                    )
                ],
            ),
            "1-<2 years": Group(
                name="1-<2 years",
                attributes=[
                    Attribute(
                        name="age", value=Range(1, 2), impute_action="ignore"
                    )
                ],
            ),
            "0-<6 months": Group(
                name="0-<6 months",
                attributes=[
                    Attribute(
                        name="age",
                        value=Range(0, 6.0 / 12.0),
                        impute_action="ignore",
                    )
                ],
            ),
            "6 months-<1 year": Group(
                name="6 months-<1 year",
                attributes=[
                    Attribute(
                        name="age",
                        value=Range(6.0 / 12.0, 1.0),
                        impute_action="ignore",
                    )
                ],
            ),
            "1 year": Group(
                name="1 year",
                attributes=[
                    Attribute(
                        name="age", value=Range(1, 2), impute_action="ignore"
                    )
                ],
            ),
        }
        map_expected = {
            "0-<6 months": "0 years",
            "6 months-<1 year": "0 years",
            "1 year": "1-<2 years",
        }

        assert group_map.groups == groups_expected
        assert group_map.sub_to_super == map_expected

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
        supergroup_cats = ["festina", "festimus materia"]
        subgroup_cats = [["lente", "velociter"]]

        map_expected = {
            (
                "lente",
                "festina",
            ): "festina",
            (
                "velociter",
                "festina",
            ): "festina",
            (
                "lente",
                "festimus materia",
            ): "festimus materia",
            (
                "velociter",
                "festimus materia",
            ): "festimus materia",
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
            ("lente", "festina"): Group(
                name=("lente", "festina"),
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
            ("velociter", "festina"): Group(
                name=("velociter", "festina"),
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
            ("lente", "festimus materia"): Group(
                name=("lente", "festimus materia"),
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
            ("velociter", "festimus materia"): Group(
                name=("velociter", "festimus materia"),
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

        group_map = OuterProductSubgroupHandler().construct_group_map(
            supergroup_categories=supergroup_cats,
            subgroup_categories=subgroup_cats,
            supergroup_variable_name="phrase",
            subgroup_variable_names=["speed"],
        )

        assert group_map.groups == groups_expected
        assert group_map.sub_to_super == map_expected
